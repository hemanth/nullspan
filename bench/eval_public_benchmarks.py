#!/usr/bin/env python3
"""Official External Public Benchmark Evaluation Harness (Evaluation ONLY).

Suites:
  - `typed_decisions`: Evaluates `LocalLLaMA/typed-decisions` `test` (400 cases, 2,000 decisions).
                       Strictly gated by `bench/TEST_LEDGER.md`: requires `--ledger-entry <id>` with
                       `Status = READY`, `Test Acc = PENDING`, and `Git SHA` matching `HEAD`.
  - `jevbench`       : Evaluates JevBench public items via `/v1/systemone`. Requires `JEVBENCH_DIR`.
  - `squad_v2`       : Evaluates `rajpurkar/squad_v2` zero-overlap validation (`--n-eval 500` or `0` for full).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "training") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "training"))


def score_typed_decision_one(pred: dict[str, Any], gold: dict[str, Any]) -> tuple[bool, float, float, float]:
    opts = list(gold["probabilities"].keys())
    g = [float(gold["probabilities"][o]) for o in opts]
    if gold["type"] == "noul":
        pred = dict(pred)
        pred["probabilities"] = {
            {"yes": "true", "no": "false"}.get(k, k): v for k, v in pred["probabilities"].items()
        }
    p = [float(pred["probabilities"].get(o, 0.0)) for o in opts]
    s = sum(p)
    p = [v / s for v in p] if s > 0 else [1.0 / len(opts)] * len(opts)
    am_p = p.index(max(p))
    am_g = g.index(max(g))
    kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)
    brier = sum((pi - gi) ** 2 for pi, gi in zip(p, g))
    return am_p == am_g, kl, brier, max(p)


def compute_ece_10bin(accs: list[bool], confs: list[float]) -> float:
    n = len(accs)
    if not n:
        return 0.0
    n_bins = 10
    bin_hits = [0] * n_bins
    bin_total = [0] * n_bins
    bin_conf = [0.0] * n_bins
    for ok, conf in zip(accs, confs):
        b = min(int(conf * n_bins), n_bins - 1)
        bin_total[b] += 1
        bin_hits[b] += int(ok)
        bin_conf[b] += conf
    ece = 0.0
    for b in range(n_bins):
        if bin_total[b]:
            ece += bin_total[b] / n * abs(bin_hits[b] / bin_total[b] - bin_conf[b] / bin_total[b])
    return ece


def _current_git_sha() -> str:
    env_sha = os.environ.get("NULLSPAN_GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def parse_test_ledger(ledger_path: str | Path) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    text = Path(ledger_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [c.strip() for c in line.strip("|").split("|")]
        if len(parts) < 7 or parts[0] in ("ID", "---") or set(parts[0]) == {"-"}:
            continue
        entries[parts[0]] = {
            "id": parts[0],
            "git_sha": parts[1],
            "status": parts[2].upper(),
            "oof_acc": parts[3],
            "oof_kl": parts[4],
            "test_acc": parts[5],
            "notes": parts[6],
        }
    return entries


def verify_ledger_entry(
    entry_id: str | None,
    ledger_path: str | Path | None = None,
    expected_sha: str | None = None,
) -> dict[str, str]:
    if not entry_id:
        raise SystemExit(
            "REFUSED: --suite typed_decisions requires --ledger-entry <id> registered in bench/TEST_LEDGER.md."
        )
    resolved_ledger = Path(ledger_path or os.path.join(ROOT, "bench", "TEST_LEDGER.md"))
    if not resolved_ledger.exists():
        raise SystemExit(f"REFUSED: Ledger file not found at {resolved_ledger}.")

    entries = parse_test_ledger(resolved_ledger)
    if entry_id not in entries:
        raise SystemExit(f"REFUSED: Unknown ledger entry '{entry_id}' in {resolved_ledger}.")

    entry = entries[entry_id]
    if entry["status"] != "READY":
        raise SystemExit(
            f"REFUSED: Ledger entry '{entry_id}' has status '{entry['status']}' (expected 'READY')."
        )
    if entry["test_acc"].upper() not in ("PENDING", ""):
        raise SystemExit(
            f"REFUSED: Ledger entry '{entry_id}' already records test_acc='{entry['test_acc']}'."
        )

    rec_sha = entry["git_sha"].strip()
    if len(rec_sha) < 7 or rec_sha.upper() in ("PENDING", "EMPTY", "N/A"):
        raise SystemExit(f"REFUSED: Ledger entry '{entry_id}' has unset git SHA '{rec_sha}'.")

    head_sha = (expected_sha or _current_git_sha()).strip()
    if not (head_sha.startswith(rec_sha) or rec_sha.startswith(head_sha)):
        raise SystemExit(
            f"REFUSED: Ledger entry '{entry_id}' SHA '{rec_sha}' does not match HEAD '{head_sha}'."
        )
    return entry


def eval_typed_decisions_test(model_dir: str, device: str, ledger_entry: str) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from systemone_decision import NullspanSystemOneEngine

    print(f"=== Evaluating LocalLLaMA/typed-decisions `test` (ledger={ledger_entry}) ===", flush=True)
    engine = NullspanSystemOneEngine.load_from_dir(model_dir, device=device)
    ds_test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")

    engine.decide(state={"warmup": True}, questions={"q": {"type": "noul", "instructions": "Warmup?"}})
    if device == "cuda":
        torch.cuda.synchronize()

    accs, kls, briers, confs = [], [], [], []
    by_wf_acc: dict[str, list[bool]] = collections.defaultdict(list)
    by_type_acc: dict[str, list[bool]] = collections.defaultdict(list)
    preds_jsonl_path = os.path.join(model_dir, "predictions_typed_decisions_test.jsonl")

    t0 = time.perf_counter()
    with open(preds_jsonl_path, "w", encoding="utf-8") as pred_f:
        for raw in ds_test:
            state = json.loads(raw["state"])
            questions = json.loads(raw["questions"])
            gold = json.loads(raw["gold"])
            resp = engine.decide(state=state, questions=questions)
            answers = resp["answers"]
            pred_f.write(json.dumps({"id": raw["id"], "workflow": raw["workflow"], "answers": answers}) + "\n")
            for qid, gq in gold.items():
                ok, kl, brier, conf = score_typed_decision_one(answers.get(qid, {"probabilities": {}}), gq)
                accs.append(ok)
                kls.append(kl)
                briers.append(brier)
                confs.append(conf)
                by_wf_acc[raw["workflow"]].append(ok)
                by_type_acc[gq["type"]].append(ok)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n = len(accs)
    ece = compute_ece_10bin(accs, confs)
    summary = {
        "model": "gnumanth/nullspan",
        "backbone": getattr(engine, "backbone_id", "unknown"),
        "arch_version": 3,
        "mode": "specialist",
        "ledger_entry": ledger_entry,
        "n_cases": len(ds_test),
        "n_decisions": n,
        "accuracy": round(sum(accs) / n, 4),
        "kl_from_gold": round(sum(kls) / n, 4),
        "brier": round(sum(briers) / n, 4),
        "ece": round(ece, 4),
        "latency_ms_per_case": round(1000.0 * elapsed / len(ds_test), 2),
        "latency_ms_per_decision": round(1000.0 * elapsed / n, 2),
        "by_workflow_accuracy": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_wf_acc.items())},
        "by_type_accuracy": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_type_acc.items())},
    }
    print("Typed Decisions Official Test Summary:", json.dumps(summary, indent=2), flush=True)
    return summary


def _ece_val(x: Any) -> float:
    if isinstance(x, dict):
        return float(x.get("ece") or 0.0)
    return float(x or 0.0)


def eval_jevbench_public(model_dir: str, jevbench_dir: str, device: str) -> dict[str, Any]:
    print("=== Evaluating on fstandhartinger/jevbench (Public Suite) ===", flush=True)
    sys.path.insert(0, jevbench_dir)
    from jevbench import composite_v13, composite_v14
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize
    from jevbench.tasks import load_jsonl

    port = 18011
    proc = subprocess.Popen(
        [
            sys.executable,
            os.path.join(ROOT, "training", "serve_systemone.py"),
            "--model-dir",
            model_dir,
            "--port",
            str(port),
            "--device",
            device,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ready = False
        boot_lines: list[str] = []
        for _ in range(180):
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                boot_lines.append(line.strip())
            if "serving" in line:
                ready = True
                break
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        if not ready:
            raise RuntimeError("serve_systemone.py did not become ready:\n" + "\n".join(boot_lines[-20:]))

        adapter = TypeSafeAdapter(
            endpoint=f"http://127.0.0.1:{port}",
            model="gnumanth/nullspan",
            key_env="",
            price_input_per_m=0.01,
            price_output_per_m=0.0,
        )
        warm_tasks = load_jsonl(os.path.join(jevbench_dir, "datasets", "public", "easy.jsonl"))
        for wt in warm_tasks[:3]:
            adapter.run(wt)

        tier_summaries: dict[str, Any] = {}
        all_tasks = []
        all_records = []
        hard_ece = 0.0

        for tier in ("original", "easy", "hard"):
            tpath = os.path.join(jevbench_dir, "datasets", "public", f"{tier}.jsonl")
            tasks = load_jsonl(tpath)
            records = []
            for t in tasks:
                dr = adapter.run(t)
                sc = (
                    score_task(dr.probs, t)
                    if dr.ok and dr.probs is not None
                    else {"valid": False, "strict_valid": False, "correct": False, "probs": None}
                )
                in_toks = (dr.usage or {}).get("input_tokens", 250)
                records.append(
                    {
                        "task_id": t.id,
                        "ok": dr.ok,
                        "valid": sc.get("valid", False),
                        "strict_valid": sc.get("strict_valid", False),
                        "renormalized": sc.get("renormalized", False),
                        "correct": sc.get("correct", False),
                        "predicted": sc.get("predicted"),
                        "probs": sc.get("probs"),
                        "probs_source": "native",
                        "model": "gnumanth/nullspan",
                        "latency_s": dr.latency_s,
                        "cost_usd": in_toks * 0.01 / 1e6,
                        "cost_basis": "encoder_classifier_<=0.6B",
                    }
                )
            tier_sum = summarize(tasks, records)
            t_ece = _ece_val(tier_sum["ece"])
            if tier == "hard":
                hard_ece = t_ece
            tier_summaries[tier] = {
                "n": tier_sum["n_scorable"],
                "correct": tier_sum["n_correct"],
                "accuracy": round(float(tier_sum["accuracy"]), 4),
                "strict_valid_rate": round(float(tier_sum["schema_validity_strict"]), 4),
                "brier": round(float(tier_sum["brier_mean"]), 4),
                "ece": round(t_ece, 4),
                "p50_ms": round(float(tier_sum["latency"]["p50_s"]) * 1000.0, 2),
                "p95_ms": round(float(tier_sum["latency"]["p95_s"]) * 1000.0, 2),
            }
            all_tasks.extend(tasks)
            all_records.extend(records)

        full_sum = summarize(all_tasks, all_records)
        intel_v13 = composite_v13.intelligence(
            {
                "easy": tier_summaries["easy"]["accuracy"],
                "standard": tier_summaries["original"]["accuracy"],
                "hard": tier_summaries["hard"]["accuracy"],
            }
        )
        cal_v13 = composite_v13.calibration(hard_ece)
        p50_s = float(full_sum["latency"]["p50_s"])
        p95_s = float(full_sum["latency"]["p95_s"])
        spd_v13 = composite_v13.speed(p50_s, p95_s, "gpu")
        usd_1k = float(full_sum["price_per_1000_decisions_usd"])
        cst_v13 = composite_v13.cost(usd_1k)
        axes_public = {
            "intelligence": round(intel_v13, 2),
            "calibration": round(cal_v13, 2),
            "speed": round(spd_v13, 2),
            "cost": round(cst_v13, 2),
        }
        overall: dict[str, Any] = {
            "total_public_items": full_sum["n_scorable"],
            "total_correct": full_sum["n_correct"],
            "overall_accuracy": round(float(full_sum["accuracy"]), 4),
            "macro_accuracy": round(float(full_sum["macro_accuracy"]), 4),
            "brier_mean": round(float(full_sum["brier_mean"]), 4),
            "ece": round(_ece_val(full_sum["ece"]), 4),
            "strict_valid_rate": round(float(full_sum["schema_validity_strict"]), 4),
            "p50_ms": round(p50_s * 1000.0, 2),
            "p95_ms": round(p95_s * 1000.0, 2),
            "price_per_1000_decisions_usd": round(usd_1k, 5),
            "public_axes_v13": axes_public,
            "public_jevbench_v13_geometric": round(composite_v13.jevbench_score(axes_public), 2),
            "public_jevbench_v14_harmonic": round(composite_v14.harmonic(axes_public), 2),
            "per_family_accuracy": {
                fk: round(float(fv["accuracy"]), 4)
                for fk, fv in (full_sum.get("per_family") or {}).items()
                if fv.get("accuracy") is not None
            },
            "tiers": tier_summaries,
        }
        # Also attach v1.6 scorer readout if present in the JevBench clone
        try:
            from jevbench import score_v16  # type: ignore

            if hasattr(score_v16, "score_public_records"):
                overall["public_v16"] = score_v16.score_public_records(all_tasks, all_records)
        except Exception:
            pass
        print("JevBench Public Suite Summary:", json.dumps(overall, indent=2), flush=True)
        return overall
    finally:
        proc.terminate()


def _normalize_squad_answer(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _squad_f1(pred: str, gold: str) -> float:
    p_toks = _normalize_squad_answer(pred).split()
    g_toks = _normalize_squad_answer(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = collections.Counter(p_toks) & collections.Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_toks)
    rec = num_same / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def _predict_squad_batch(
    model: Any,
    tok: Any,
    rows: list[dict[str, Any]],
    batch_size: int = 8,
) -> list[str]:
    import torch
    import torch.nn.functional as F
    from modeling_nullspan import (
        apply_qa_precondition_guard,
        extract_numeric_context_features,
        format_field_query,
        select_context_window,
    )

    device = next(model.parameters()).device
    preds_out: list[str] = []
    for start_idx in range(0, len(rows), batch_size):
        chunk = rows[start_idx : start_idx + batch_size]
        specs = [{"type": "text", "instructions": r["question"]} for r in chunk]
        queries = [format_field_query("answer", sp) for sp in specs]
        win_pairs = [select_context_window(r["context"], sp, "answer") for r, sp in zip(chunk, specs)]
        states = [wp[0] for wp in win_pairs]
        win_offsets = [wp[1] for wp in win_pairs]
        num_feats_list = [extract_numeric_context_features(r["context"], sp) for r, sp in zip(chunk, specs)]

        enc = tok(
            queries,
            states,
            padding=True,
            truncation="only_second",
            max_length=512,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mappings = enc.pop("offset_mapping")
        B, L = enc["input_ids"].shape
        state_mask = torch.zeros((B, L), dtype=torch.long)
        for b_i in range(B):
            seq_ids = enc.sequence_ids(b_i)
            for t_i, s_id in enumerate(seq_ids):
                if s_id == 1 and offset_mappings[b_i, t_i, 1] > offset_mappings[b_i, t_i, 0]:
                    state_mask[b_i, t_i] = 1

        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                state_mask=state_mask.to(device),
                num_feats=torch.tensor(num_feats_list, dtype=torch.float32, device=device),
            )
            p_nulls = torch.sigmoid(out["null_logits"]).cpu().tolist()
            start_probs = F.softmax(out["start_logits"], dim=-1).cpu()
            end_probs = F.softmax(out["end_logits"], dim=-1).cpu()

        for b_i, row in enumerate(chunk):
            state = row["context"]
            p_null = round(float(p_nulls[b_i]), 8)
            p_null = apply_qa_precondition_guard(queries[b_i], state, p_null)
            if p_null > 0.5:
                preds_out.append("")
                continue
            offsets = offset_mappings[b_i].tolist()
            mask_list = state_mask[b_i].tolist()
            valid_indices = [t_i for t_i, m_v in enumerate(mask_list) if m_v == 1]
            best_s, best_e, best_score = (valid_indices[0], valid_indices[0], 0.0) if valid_indices else (0, 0, 0.0)
            sp_list = start_probs[b_i].tolist()
            ep_list = end_probs[b_i].tolist()
            for idx_a, s_i in enumerate(valid_indices):
                sp_v = sp_list[s_i]
                for e_i in valid_indices[idx_a:]:
                    if (e_i - s_i) > 24:
                        break
                    sc = sp_v * ep_list[e_i]
                    if sc > best_score:
                        best_score = sc
                        best_s, best_e = s_i, e_i
            w_off = win_offsets[b_i]
            char_s, char_e = int(offsets[best_s][0]) + w_off, int(offsets[best_e][1]) + w_off
            span_text = state[char_s:char_e].strip(" .,;:\n")
            preds_out.append(span_text)
        if len(preds_out) % 512 == 0 or len(preds_out) == len(rows):
            print(f"  [squad_v2] {len(preds_out)}/{len(rows)} questions scored", flush=True)
    return preds_out


def _score_squad_rows(rows: list[dict[str, Any]], pred_strs: list[str]) -> dict[str, Any]:
    em_all, f1_all, em_has, f1_has, em_no, f1_no = [], [], [], [], [], []
    for row, pred_str in zip(rows, pred_strs):
        golds = row["answers"]["text"]
        if not golds:
            em = 1.0 if pred_str == "" else 0.0
            f1 = em
            em_no.append(em)
            f1_no.append(f1)
        else:
            em = max(1.0 if _normalize_squad_answer(pred_str) == _normalize_squad_answer(g) else 0.0 for g in golds)
            f1 = max(_squad_f1(pred_str, g) for g in golds)
            em_has.append(em)
            f1_has.append(f1)
        em_all.append(em)
        f1_all.append(f1)
    return {
        "n_evaluated": len(rows),
        "exact": round(100.0 * sum(em_all) / len(em_all), 2),
        "f1": round(100.0 * sum(f1_all) / len(f1_all), 2),
        "HasAns_exact": round(100.0 * sum(em_has) / len(em_has), 2),
        "HasAns_f1": round(100.0 * sum(f1_has) / len(f1_has), 2),
        "NoAns_exact": round(100.0 * sum(em_no) / len(em_no), 2),
        "NoAns_f1": round(100.0 * sum(f1_no) / len(f1_no), 2),
    }


def eval_squad_v2_public(model_dir: str, device: str, n_eval: int = 500) -> dict[str, Any]:
    from datasets import load_dataset
    from modeling_nullspan import NullspanForExtraction
    from transformers import AutoTokenizer

    print(f"=== Evaluating on Stanford rajpurkar/squad_v2 (n_eval={n_eval or 'FULL 11,873'}) ===", flush=True)
    seen_contexts: set[str] = set()
    for fname in ("train.jsonl", "cal_dev.jsonl", "val.jsonl"):
        fpath = os.path.join(ROOT, "data", fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        st = obj.get("state") or obj.get("context") or ""
                        if st:
                            seen_contexts.add(st[:120])

    ds_sq = load_dataset("rajpurkar/squad_v2", split="validation")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = NullspanForExtraction.from_pretrained(model_dir).to(device).eval()

    if n_eval <= 0:
        sample = list(ds_sq)
        pred_strs = _predict_squad_batch(model, tok, sample, batch_size=8)
        full_metrics = _score_squad_rows(sample, pred_strs)
        zo_rows = [r for r in sample if r["context"][:120] not in seen_contexts]
        zo_preds = [p for r, p in zip(sample, pred_strs) if r["context"][:120] not in seen_contexts]
        summary = {
            "dataset": "rajpurkar/squad_v2",
            **full_metrics,
            "zero_overlap_subset": _score_squad_rows(zo_rows, zo_preds),
        }
    else:
        has_ans_pool, no_ans_pool = [], []
        for row in ds_sq:
            if row["context"][:120] in seen_contexts:
                continue
            if row["answers"]["text"]:
                has_ans_pool.append(row)
            else:
                no_ans_pool.append(row)
        rng = random.Random(20261030)
        rng.shuffle(has_ans_pool)
        rng.shuffle(no_ans_pool)
        sample = has_ans_pool[: n_eval // 2] + no_ans_pool[: n_eval // 2]
        pred_strs = _predict_squad_batch(model, tok, sample, batch_size=8)
        summary = {
            "dataset": "rajpurkar/squad_v2",
            **_score_squad_rows(sample, pred_strs),
        }

    print("SQuAD v2.0 Validation Summary:", json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        required=True,
        choices=["typed_decisions", "jevbench", "squad_v2"],
        help="Benchmark suite to evaluate",
    )
    parser.add_argument("--ledger-entry", default=None, help="Required ledger ID for --suite typed_decisions")
    parser.add_argument("--ledger-path", default=None, help="Optional override path to TEST_LEDGER.md")
    parser.add_argument("--jevbench-dir", default=None, help="Path to JevBench repo (or set JEVBENCH_DIR)")
    parser.add_argument("--model-dir", default=os.path.join(ROOT, "artifacts", "nullspan"))
    parser.add_argument("--n-eval", type=int, default=500, help="Subsample size for squad_v2 (0 = full validation)")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "bench", "public_benchmarks_results.json"),
        help="Output JSON file to update",
    )
    args = parser.parse_args(argv)

    # Gate checks BEFORE any torch/datasets/model imports or loads
    if args.suite == "typed_decisions":
        verify_ledger_entry(args.ledger_entry, ledger_path=args.ledger_path)
    elif args.suite == "jevbench":
        jb_dir = (args.jevbench_dir or os.environ.get("JEVBENCH_DIR") or "").strip()
        if not jb_dir or not os.path.isdir(jb_dir):
            raise SystemExit("REFUSED: JEVBENCH_DIR environment variable (or --jevbench-dir) must point to a valid JevBench checkout.")

    import torch

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    hw_name = torch.cuda.get_device_name(0) if device == "cuda" else device

    existing: dict[str, Any] = {}
    if os.path.exists(args.out):
        try:
            existing = json.loads(Path(args.out).read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.setdefault("hardware", hw_name)

    if args.suite == "typed_decisions":
        assert args.ledger_entry is not None
        res = eval_typed_decisions_test(args.model_dir, device, args.ledger_entry)
        res["hardware"] = hw_name
        existing["typed_decisions"] = res
    elif args.suite == "jevbench":
        jb_dir = (args.jevbench_dir or os.environ.get("JEVBENCH_DIR") or "").strip()
        res = eval_jevbench_public(args.model_dir, jb_dir, device)
        res["hardware"] = hw_name
        existing["jevbench_public"] = res
    elif args.suite == "squad_v2":
        key = "squad_v2_full" if args.n_eval <= 0 else "squad_v2"
        res = eval_squad_v2_public(args.model_dir, device, n_eval=args.n_eval)
        res["hardware"] = hw_name
        existing[key] = res

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    print(f"Updated {args.out} ({args.suite})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
