"""Train, calibrate, evaluate, and publish ClairicForExtraction on GPU."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from generate_dataset import DATA_DIR, main as generate_data_main
from modeling_clairic import (
    ClairicConfig,
    ClairicForExtraction,
    extract_numeric_context_features,
    format_field_query,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts" / "clairic-extract-v1"


class ClairicFieldDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer: Any, max_length: int = 128) -> None:
        self.rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded = [self._encode_row(r) for r in self.rows]

    def _encode_row(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        state: str = row["state"]
        fname: str = row["field_name"]
        spec: dict[str, Any] = row["field_spec"]
        target: dict[str, Any] = row["target"]
        query = format_field_query(fname, spec)

        enc = self.tokenizer(
            query,
            state,
            padding="max_length",
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]
        seq_ids = enc.sequence_ids()
        L = len(enc["input_ids"])

        state_mask = [0] * L
        for i in range(L):
            if seq_ids[i] == 1 and offsets[i][1] > offsets[i][0]:
                state_mask[i] = 1

        is_null = int(target.get("is_null", 0))
        op_id = int(target.get("op", 0))

        # Map character span [char_s, char_e] to token start_pos and end_pos
        start_pos = -100
        end_pos = -100
        if "span" in target and target["span"]:
            cs, ce = int(target["span"][0]), int(target["span"][1])
            overlapping = [
                i for i in range(L)
                if state_mask[i] == 1 and offsets[i][0] < ce and offsets[i][1] > cs
            ]
            if overlapping:
                start_pos = overlapping[0]
                end_pos = overlapping[-1]

        # Map multi-span list targets to BIO tags (0=O, 1=B, 2=I, -100=ignored)
        bio_labels = [-100] * L
        if spec.get("type") == "list" and "spans" in target:
            for i in range(L):
                if state_mask[i] == 1:
                    bio_labels[i] = 0
            for sp_pair in target["spans"]:
                cs, ce = int(sp_pair[0]), int(sp_pair[1])
                overlapping = [
                    i for i in range(L)
                    if state_mask[i] == 1 and offsets[i][0] < ce and offsets[i][1] > cs
                ]
                for idx_pos, tok_i in enumerate(overlapping):
                    bio_labels[tok_i] = 1 if idx_pos == 0 else 2

        num_feats = extract_numeric_context_features(state, spec)
        has_q = 1 if "quantiles_norm" in target else 0
        q_labels = target.get("quantiles_norm", [0.5, 0.5, 0.5])

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "state_mask": torch.tensor(state_mask, dtype=torch.long),
            "num_feats": torch.tensor(num_feats, dtype=torch.float32),
            "is_null_labels": torch.tensor(is_null, dtype=torch.long),
            "start_positions": torch.tensor(start_pos, dtype=torch.long),
            "end_positions": torch.tensor(end_pos, dtype=torch.long),
            "bio_labels": torch.tensor(bio_labels, dtype=torch.long),
            "op_labels": torch.tensor(op_id, dtype=torch.long),
            "quantile_labels": torch.tensor(q_labels, dtype=torch.float32),
            "has_quantile_mask": torch.tensor(has_q, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encoded[idx]


@torch.no_grad()
def evaluate_validation(
    model: ClairicForExtraction,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    null_correct, null_total = 0, 0
    span_em_correct, span_total = 0, 0
    op_correct, op_total = 0, 0
    bio_tok_correct, bio_tok_total = 0, 0
    q_errors: list[float] = []

    for batch in val_loader:
        b_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(**b_dev)

        null_preds = (torch.sigmoid(out["null_logits"]) > 0.5).long()
        null_correct += int((null_preds == b_dev["is_null_labels"]).sum().item())
        null_total += b_dev["is_null_labels"].numel()

        op_preds = out["op_logits"].argmax(dim=-1)
        op_correct += int((op_preds == b_dev["op_labels"]).sum().item())
        op_total += b_dev["op_labels"].numel()

        s_preds = out["start_logits"].argmax(dim=-1)
        e_preds = out["end_logits"].argmax(dim=-1)
        valid_sp = b_dev["start_positions"] != -100
        if valid_sp.any():
            exact = (s_preds[valid_sp] == b_dev["start_positions"][valid_sp]) & (
                e_preds[valid_sp] == b_dev["end_positions"][valid_sp]
            )
            span_em_correct += int(exact.sum().item())
            span_total += int(valid_sp.sum().item())

        bio_p = out["bio_logits"].argmax(dim=-1)
        valid_bio = b_dev["bio_labels"] != -100
        if valid_bio.any():
            bio_tok_correct += int((bio_p[valid_bio] == b_dev["bio_labels"][valid_bio]).sum().item())
            bio_tok_total += int(valid_bio.sum().item())

        q_mask = b_dev["has_quantile_mask"] == 1
        if q_mask.any():
            err = (out["quantile_preds"][q_mask] - b_dev["quantile_labels"][q_mask]).abs().mean(dim=-1)
            q_errors.extend(err.cpu().tolist())

    return {
        "null_accuracy": round(100.0 * null_correct / max(1, null_total), 2),
        "span_exact_match": round(100.0 * span_em_correct / max(1, span_total), 2),
        "op_accuracy": round(100.0 * op_correct / max(1, op_total), 2),
        "list_bio_token_accuracy": round(100.0 * bio_tok_correct / max(1, bio_tok_total), 2),
        "quantile_mae_norm": round(sum(q_errors) / max(1, len(q_errors)), 4),
    }


def evaluate_canonical_examples(model: ClairicForExtraction, tokenizer: Any) -> dict[str, Any]:
    golden = json.loads((DATA_DIR / "golden_eval.json").read_text())
    canonical_docs = golden["canonical"]
    results = {}
    latencies_ms = []
    field_hits = 0
    field_total = 0

    # Warmup pass to exclude one-time kernel dispatch overhead
    if canonical_docs:
        w_doc = canonical_docs[0]
        model.extract(w_doc["state"], {k: v["spec"] for k, v in w_doc["fields"].items()}, tokenizer)

    for doc in canonical_docs:
        state = doc["state"]
        fields_spec = {k: v["spec"] for k, v in doc["fields"].items()}
        t0 = time.perf_counter()
        out = model.extract(state, fields_spec, tokenizer)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        results[doc["id"]] = {
            "latency_ms": round(dt_ms, 2),
            "num_fields": len(fields_spec),
            "answers": out["answers"],
        }

        for fname, fbundle in doc["fields"].items():
            field_total += 1
            pred = out["answers"][fname]
            tgt = fbundle["target"]
            ftype = fbundle["spec"]["type"]
            if tgt.get("is_null") == 1:
                if pred.get("value") is None:
                    field_hits += 1
            elif ftype in ("estimate", "forecast"):
                p10 = pred["p10"][0] if isinstance(pred["p10"], list) else pred["p10"]
                p90 = pred["p90"][0] if isinstance(pred["p90"], list) else pred["p90"]
                tgt_med = tgt["median"][0] if isinstance(tgt["median"], list) else tgt["median"]
                if p10 <= tgt_med <= p90:
                    field_hits += 1
            else:
                if pred.get("value") == tgt.get("value"):
                    field_hits += 1

    return {
        "canonical_field_accuracy": round(100.0 * field_hits / max(1, field_total), 2),
        "mean_doc_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 2),
        "examples": results,
    }


def write_model_card(
    artifact_dir: Path,
    repo_id: str,
    val_metrics: dict[str, float],
    canonical_eval: dict[str, Any],
    device_name: str,
    train_seconds: float,
) -> None:
    support_out = json.dumps(canonical_eval["examples"]["clairic_blog_support_claim"]["answers"], indent=2)
    card = f"""---
license: mit
library_name: transformers
tags:
  - clairic
  - system-one
  - structured-extraction
  - span-extraction
  - calibrated-estimation
  - time-series-forecasting
  - distilbert
datasets:
  - gnumanth/clairic-extraction-dataset
metrics:
  - accuracy
  - exact_match
pipeline_tag: question-answering
---

# {repo_id}

Lightweight **System One extraction model** inspired by [Cloudflare Clairic](https://blog.cloudflare.com/clairic-extraction-model/). Processes an unstructured `state` and a map of typed `fields` (`text`, `number`, `date`, `datetime`, `duration`, `list`, `estimate`, `forecast`) in a **single parallel encoder forward pass** with zero output token generation.

## Architecture

Built on `distilbert/distilbert-base-uncased` (67M parameters) with 5 parallel System One heads:
1. **`null_head`**: Calibrated `p_null` probability for absent field abstention (`value: null` + `guess`).
2. **`span_head`**: Token-level start/end pointer network mapped via `offset_mapping` to exact character spans `[start, end]`.
3. **`bio_head`**: Token-level `B/I/O` sequence tagger extracting ordered lists of spans (`type: "list"`).
4. **`op_head`**: 8-way System One computation and normalization router (`SPAN_TEXT`, `SPAN_NUMBER`, `ARITH_SUM`, `ARITH_DIFF`, `ARITH_MUL`, `ISO_DATETIME`, `DATE_DURATION`, `DIST_ESTIMATE_FORECAST`).
5. **`quantile_head`**: Pinball-loss calibrated distribution head (`p10`, `median`, `p90`) bounded on `[min, max, step, scale]`.

## Empirical Benchmarks (Fine-Tuned on GCP `{device_name}`)

- **Training time**: `{train_seconds:.1f}s`
- **Null Abstention Accuracy (`p_null`)**: `{val_metrics['null_accuracy']}%`
- **Grounded Span Exact Match (`[start, end]`)**: `{val_metrics['span_exact_match']}%`
- **Operation Router Accuracy**: `{val_metrics['op_accuracy']}%`
- **Multi-Span List BIO Accuracy**: `{val_metrics['list_bio_token_accuracy']}%`
- **Quantile Distribution MAE (normalized `[0, 1]`)**: `{val_metrics['quantile_mae_norm']}`
- **Canonical Cloudflare Blog Suite Accuracy**: `{canonical_eval['canonical_field_accuracy']}%` (`{canonical_eval['mean_doc_latency_ms']} ms` per multi-field pass)

## Quick Start

```python
from transformers import AutoModel, AutoTokenizer

repo_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

email = (
    "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented corner. "
    "I paid $649.00 plus $52.10 tax. It's now Oct 9, 2026 and I'd like this fixed or refunded. Thanks, Dana"
)

result = model.extract(
    state=email,
    fields={{
        "order_id": {{"type": "text", "instructions": "Order number"}},
        "delivered_on": {{"type": "date", "instructions": "Delivery date"}},
        "amount_paid": {{"type": "number", "decimals": 2, "unit": "USD", "instructions": "Total paid including tax"}},
        "days_since_delivery": {{"type": "duration", "unit": "days", "integer": True, "instructions": "Days from delivery to the date the customer wrote"}},
        "damage": {{"type": "list", "items": "text", "instructions": "Each kind of damage described"}},
        "tracking_number": {{"type": "text", "instructions": "Carrier tracking number"}},
        "repair_cost": {{"type": "estimate", "unit": "USD", "min": 10, "max": 5000, "scale": "log", "step": 1, "instructions": "Cost to repair the damage described"}},
    }},
    tokenizer=tokenizer,
)
print(result["answers"])
```

### Output on Customer Claim Example

```json
{support_out}
```

## License

MIT © [Hemanth.HM](https://h3manth.com)
"""
    (artifact_dir / "README.md").write_text(card)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--repo-id", type=str, default="gnumanth/clairic-extract-v1")
    parser.add_argument("--dataset-repo-id", type=str, default="gnumanth/clairic-extraction-dataset")
    parser.add_argument("--push-hf", action="store_true")
    args = parser.parse_args()

    if not (DATA_DIR / "train.jsonl").exists():
        generate_data_main()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon MPS"
    else:
        device = torch.device("cpu")
        ncpu = os.cpu_count() or 16
        torch.set_num_threads(ncpu)
        device_name = f"GCP Compute Engine ({ncpu} vCPUs)"

    print(f"Using device: {device} ({device_name})", flush=True)

    backbone = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    train_ds = ClairicFieldDataset(DATA_DIR / "train.jsonl", tokenizer)
    val_ds = ClairicFieldDataset(DATA_DIR / "val.jsonl", tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    cfg = ClairicConfig(
        backbone_name=backbone,
        auto_map={
            "AutoConfig": "modeling_clairic.ClairicConfig",
            "AutoModel": "modeling_clairic.ClairicForExtraction",
        },
    )
    model = ClairicForExtraction(cfg, pretrained_backbone=True).to(device)

    head_params = (
        list(model.null_head.parameters())
        + list(model.span_head.parameters())
        + list(model.bio_head.parameters())
        + list(model.op_head.parameters())
        + list(model.quantile_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": head_params, "lr": args.lr * 6.0},
        ],
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(8, int(total_steps * 0.08)),
        num_training_steps=total_steps,
    )

    t_start = time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            b_dev = {k: v.to(device) for k, v in batch.items()}
            out = model(**b_dev)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += float(loss.item())

        avg_loss = ep_loss / max(1, len(train_loader))
        val_metrics = evaluate_validation(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": round(avg_loss, 4), **val_metrics})
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | loss={avg_loss:.4f} | "
            f"null_acc={val_metrics['null_accuracy']}% | span_em={val_metrics['span_exact_match']}% | "
            f"op_acc={val_metrics['op_accuracy']}% | bio_acc={val_metrics['list_bio_token_accuracy']}% | "
            f"q_mae={val_metrics['quantile_mae_norm']}",
            flush=True,
        )

    train_seconds = round(time.time() - t_start, 2)
    val_metrics = evaluate_validation(model, val_loader, device)
    canonical_eval = evaluate_canonical_examples(model, tokenizer)
    print("Canonical Eval Summary:", json.dumps({
        "canonical_field_accuracy": canonical_eval["canonical_field_accuracy"],
        "mean_doc_latency_ms": canonical_eval["mean_doc_latency_ms"],
    }, indent=2))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ARTIFACT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(ARTIFACT_DIR)
    shutil.copyfile(Path(__file__).resolve().parent / "modeling_clairic.py", ARTIFACT_DIR / "modeling_clairic.py")

    summary_payload = {
        "repo_id": args.repo_id,
        "device": device_name,
        "train_seconds": train_seconds,
        "epochs": args.epochs,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "val_metrics": val_metrics,
        "canonical_eval": canonical_eval,
        "history": history,
    }
    (ARTIFACT_DIR / "eval_results.json").write_text(json.dumps(summary_payload, indent=2))
    write_model_card(ARTIFACT_DIR, args.repo_id, val_metrics, canonical_eval, device_name, train_seconds)
    print(f"Saved model artifacts to {ARTIFACT_DIR}")

    if args.push_hf:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            tok_path = Path.home() / ".cache" / "huggingface" / "token"
            if tok_path.exists():
                token = tok_path.read_text().strip()
        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(ARTIFACT_DIR),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload fine-tuned Clairic System One extraction model ({device_name})",
        )
        print(f"Pushed model to https://huggingface.co/{args.repo_id}")

        api.create_repo(repo_id=args.dataset_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(DATA_DIR),
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
            commit_message="Upload multi-task Clairic System One extraction dataset",
        )
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.dataset_repo_id}")


if __name__ == "__main__":
    main()
