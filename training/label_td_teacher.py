#!/usr/bin/env python3
"""Label Typed Decisions cases with a ~4B-class teacher (3 samples at T=0.7, averaged).

Interfaces (`plan.md` Task 4):
  # Step 2: Teacher selection & KL < 0.10 gate on real `train` (1,200 cases)
  python training/label_td_teacher.py --teacher <model-id> --samples 3 --calibrate-train \
      --report-out data/td_synth/teacher_calibration.json

  # Step 3: Label synthetic cases in exact `LocalLLaMA/typed-decisions` `gold` schema
  python training/label_td_teacher.py --teacher <model-id> --samples 3 \
      --in data/td_synth/cases.jsonl --out data/td_synth/cases_labelled.jsonl
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.systemone_decision import canonical_options
from training.train_systemone import score_typed_decision_one


def normalize_option_probs(
    raw_probs: dict[str, Any] | None,
    option_keys: list[str],
    laplace_eps: float = 0.0,
) -> dict[str, float]:
    """Normalize an option->probability mapping onto `option_keys`."""
    K = max(1, len(option_keys))
    if not raw_probs or not isinstance(raw_probs, dict):
        return {k: 1.0 / K for k in option_keys}

    # Handle case-insensitive / yes-no aliases cleanly
    lowered = {str(k).strip().lower(): v for k, v in raw_probs.items()}
    vals: list[float] = []
    for k in option_keys:
        v = raw_probs.get(k)
        if v is None:
            kl = k.lower()
            if kl == "true":
                v = lowered.get("true", lowered.get("yes", 0.0))
            elif kl == "false":
                v = lowered.get("false", lowered.get("no", 0.0))
            else:
                v = lowered.get(kl, 0.0)
        try:
            fv = max(0.0, float(v))
        except (TypeError, ValueError):
            fv = 0.0
        vals.append(fv)

    tot = sum(vals)
    if tot <= 0.0:
        probs = [1.0 / K] * K
    else:
        probs = [x / tot for x in vals]

    if laplace_eps > 0.0:
        u = 1.0 / K
        probs = [(1.0 - laplace_eps) * p + laplace_eps * u for p in probs]
        s2 = sum(probs)
        probs = [p / s2 for p in probs]

    return {k: probs[i] for i, k in enumerate(option_keys)}


def aggregate_teacher_samples(
    questions: dict[str, dict[str, Any]],
    samples: list[dict[str, dict[str, float]]],
    eps_floor: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Average K sampled question probability maps into Typed Decisions `gold` format."""
    if not samples:
        raise ValueError("`samples` must contain at least one teacher sample.")

    gold: dict[str, dict[str, Any]] = {}
    n_s = float(len(samples))
    for qid, qspec in questions.items():
        qtype = str(qspec.get("type", "choice"))
        opt_keys = [k for k, _ in canonical_options(qspec)]
        K = len(opt_keys)

        norm_samples = [
            normalize_option_probs(s.get(qid), opt_keys, laplace_eps=eps_floor)
            for s in samples
        ]
        mean_vals = [sum(ns[k] for ns in norm_samples) / n_s for k in opt_keys]
        s_tot = sum(mean_vals)
        mean_vals = [v / s_tot for v in mean_vals] if s_tot > 0 else [1.0 / K] * K

        prob_map = {opt_keys[i]: round(mean_vals[i], 6) for i in range(K)}
        best_idx = max(range(K), key=lambda idx: mean_vals[idx])
        best_key = opt_keys[best_idx]
        sorted_p = sorted(mean_vals, reverse=True)
        margin = sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0)

        entry: dict[str, Any] = {
            "type": qtype,
            "label": best_key,
            "confidence": round(float(margin), 6),
            "probabilities": prob_map,
        }
        if qtype == "noul":
            p_true = float(prob_map.get("true", mean_vals[-1]))
            entry["noul"] = round(p_true, 6)
        elif qtype == "score":
            exp_score = sum(float(opt_keys[i]) * mean_vals[i] for i in range(K))
            entry["score"] = round(exp_score, 6)
        gold[qid] = entry

    return gold


def evaluate_teacher_kl_against_gold(
    pred_gold_rows: list[dict[str, dict[str, Any]]],
    ref_gold_rows: list[dict[str, dict[str, Any]]],
    max_kl: float = 0.10,
) -> dict[str, Any]:
    """Compare teacher 3-sample mean distributions against reference `train` gold."""
    if len(pred_gold_rows) != len(ref_gold_rows):
        raise ValueError(f"Row count mismatch: {len(pred_gold_rows)} vs {len(ref_gold_rows)}")

    accs: list[bool] = []
    kls: list[float] = []
    briers: list[float] = []
    by_type: dict[str, list[float]] = {"noul": [], "choice": [], "score": []}

    for p_case, g_case in zip(pred_gold_rows, ref_gold_rows):
        for qid, g_q in g_case.items():
            p_q = p_case[qid]
            ok, kl, br, _ = score_typed_decision_one(p_q, g_q)
            accs.append(ok)
            kls.append(kl)
            briers.append(br)
            qt = str(g_q.get("type", "choice"))
            by_type.setdefault(qt, []).append(kl)

    mean_kl = sum(kls) / max(1, len(kls))
    mean_acc = sum(accs) / max(1, len(accs))
    mean_brier = sum(briers) / max(1, len(briers))
    return {
        "n_cases": len(ref_gold_rows),
        "n_decisions": len(kls),
        "accuracy": round(mean_acc, 6),
        "mean_kl": round(mean_kl, 6),
        "mean_brier": round(mean_brier, 6),
        "per_type_kl": {k: round(sum(v) / len(v), 6) for k, v in sorted(by_type.items()) if v},
        "max_kl_threshold": max_kl,
        "accepted": bool(mean_kl < max_kl),
    }


def build_teacher_prompt(state: Any, questions: dict[str, dict[str, Any]]) -> str:
    """Prompt a general instruct LLM to output calibrated probabilities across all questions."""
    state_json = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    q_lines: list[str] = []
    schema_example: dict[str, dict[str, float]] = {}

    for qid, qspec in questions.items():
        qtype = qspec.get("type", "choice")
        inst = qspec.get("instructions", "")
        opts = canonical_options(qspec)
        opt_desc = "; ".join(f'"{k}": {desc}' for k, desc in opts)
        q_lines.append(f'- "{qid}" (type={qtype}): {inst}\n  Options: {opt_desc}')
        u = round(1.0 / max(1, len(opts)), 2)
        schema_example[qid] = {k: u for k, _ in opts}

    return (
        "You are an calibrated probabilistic workflow judge. Given the input `state` and "
        "five typed questions, estimate a realistic, well-calibrated probability distribution "
        "over every option for each question.\n"
        "Rules:\n"
        "1. Probabilities for each question MUST sum to 1.0.\n"
        "2. Reflect genuine ambiguity in the state using soft probabilities (e.g. 0.55/0.35/0.10 "
        "when uncertain rather than extreme 1.0/0.0).\n"
        "3. Output ONLY a single valid JSON object mapping each question id to its option probability map.\n\n"
        f"State:\n{state_json}\n\n"
        "Questions:\n" + "\n".join(q_lines) + "\n\n"
        f"Required JSON shape:\n{json.dumps(schema_example)}"
    )


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_teacher_response_json(raw_text: str, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Extract per-question probability maps from an LLM's JSON completion."""
    cleaned = raw_text.strip()
    # Strip optional <think>...</think> blocks emitted by Qwen3 models if thinking slipped in
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1].strip()
    m = JSON_BLOCK_RE.search(cleaned)
    parsed: dict[str, Any] = {}
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                parsed = obj.get("probabilities") if isinstance(obj.get("probabilities"), dict) and set(questions).issubset(obj["probabilities"]) else obj
        except json.JSONDecodeError:
            parsed = {}

    out: dict[str, dict[str, float]] = {}
    for qid, qspec in questions.items():
        opt_keys = [k for k, _ in canonical_options(qspec)]
        q_val = parsed.get(qid)
        if isinstance(q_val, dict) and "probabilities" in q_val and isinstance(q_val["probabilities"], dict):
            q_val = q_val["probabilities"]
        # Small 0.015 Laplace smoothing prevents single rounded zeros from blowing up KL
        out[qid] = normalize_option_probs(q_val if isinstance(q_val, dict) else None, opt_keys, laplace_eps=0.015)
    return out


def _sample_anthropic_case(
    state: Any,
    questions: dict[str, dict[str, Any]],
    model: str,
    api_key: str,
    n_samples: int = 3,
    temperature: float = 0.7,
) -> dict[str, dict[str, Any]]:
    props: dict[str, Any] = {}
    for qid, qspec in questions.items():
        opt_keys = [k for k, _ in canonical_options(qspec)]
        props[qid] = {
            "type": "object",
            "properties": {k: {"type": "number"} for k in opt_keys},
            "required": opt_keys,
        }

    prompt = build_teacher_prompt(state, questions)
    body = {
        "model": model,
        "max_tokens": 1024,
        "temperature": temperature,
        "tools": [
            {
                "name": "emit_probabilities",
                "description": "Emit calibrated option probabilities for every question.",
                "input_schema": {
                    "type": "object",
                    "properties": props,
                    "required": list(questions.keys()),
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "emit_probabilities"},
        "messages": [{"role": "user", "content": prompt}],
    }
    encoded = json.dumps(body).encode("utf-8")
    samples: list[dict[str, dict[str, float]]] = []

    for _ in range(n_samples):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=encoded,
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        raw_input: dict[str, Any] = {}
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                for blk in payload.get("content", []):
                    if blk.get("type") == "tool_use":
                        raw_input = blk.get("input") or {}
                        break
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 529) and attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except Exception:
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        samp_map = {
            qid: normalize_option_probs(
                raw_input.get(qid) if isinstance(raw_input, dict) else None,
                [k for k, _ in canonical_options(qspec)],
                laplace_eps=0.015,
            )
            for qid, qspec in questions.items()
        }
        samples.append(samp_map)

    return aggregate_teacher_samples(questions, samples)


class PersistentHFTeacherLabeller:
    """Persistent GPU teacher runner (`vLLM` continuous batching or `transformers` fallback)."""

    def __init__(
        self,
        teacher_id: str,
        n_samples: int = 3,
        temperature: float = 0.7,
        batch_size: int = 16,
    ) -> None:
        self.teacher_id = teacher_id
        self.n_samples = n_samples
        self.temperature = temperature
        self.batch_size = batch_size
        self.backend = "transformers"
        self.llm: Any = None
        self.sp: Any = None
        self.tok: Any = None
        self.model: Any = None
        self.device: str = "cpu"

        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams  # type: ignore

            print(f"[Teacher vLLM] Loading {teacher_id} with continuous batching...", flush=True)
            self.tok = AutoTokenizer.from_pretrained(teacher_id)
            v_dtype = "auto" if "fp8" in teacher_id.lower() else "bfloat16"
            self.llm = LLM(model=teacher_id, dtype=v_dtype, max_model_len=8192, gpu_memory_utilization=0.88)
            self.sp = SamplingParams(
                n=n_samples,
                temperature=temperature,
                top_p=0.92,
                max_tokens=384,
            )
            self.backend = "vllm"
            return
        except ImportError:
            pass

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8) else (
            torch.float16 if self.device == "cuda" else torch.float32
        )
        print(f"[Teacher HF] Loading {teacher_id} on {self.device} ({dtype})...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(teacher_id, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(teacher_id, torch_dtype=dtype).to(self.device)
        self.model.eval()

    def label_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
        prompts: list[str] = []
        q_parsed_list: list[dict[str, dict[str, Any]]] = []
        for r in rows:
            qs = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
            q_parsed_list.append(qs)
            user_msg = build_teacher_prompt(r["state"], qs)
            if len(user_msg) > 22000:
                user_msg = user_msg[:11000] + "\n...\n" + user_msg[-11000:]
            msgs = [{"role": "user", "content": user_msg}]
            try:
                chat_txt = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                chat_txt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(chat_txt)

        if self.backend == "vllm":
            v_outs = self.llm.generate(prompts, self.sp)
            results: list[dict[str, dict[str, Any]]] = []
            for v_o, qs in zip(v_outs, q_parsed_list):
                samples = [parse_teacher_response_json(o.text, qs) for o in v_o.outputs]
                results.append(aggregate_teacher_samples(qs, samples))
            return results

        import torch

        all_samples_per_row: list[list[dict[str, dict[str, float]]]] = [[] for _ in rows]
        for s_idx in range(self.n_samples):
            for start in range(0, len(prompts), self.batch_size):
                end = min(len(prompts), start + self.batch_size)
                enc = self.tok(
                    prompts[start:end],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=3072,
                ).to(self.device)
                in_len = enc["input_ids"].shape[1]
                with torch.no_grad():
                    gen_ids = self.model.generate(
                        **enc,
                        max_new_tokens=384,
                        do_sample=(self.temperature > 0.0),
                        temperature=max(self.temperature, 1e-3),
                        top_p=0.92,
                        pad_token_id=self.tok.pad_token_id,
                    )
                for b_i, row_idx in enumerate(range(start, end)):
                    out_txt = self.tok.decode(gen_ids[b_i, in_len:], skip_special_tokens=True)
                    all_samples_per_row[row_idx].append(parse_teacher_response_json(out_txt, q_parsed_list[row_idx]))
                if (end % (self.batch_size * 10) == 0) or end == len(prompts):
                    print(f"  [Sample {s_idx+1}/{self.n_samples}] {end}/{len(prompts)} cases", flush=True)

        return [
            aggregate_teacher_samples(q_parsed_list[i], all_samples_per_row[i])
            for i in range(len(rows))
        ]


_DEFAULT_RUNNER_CACHE: dict[tuple[str, int, float], Any] = {}


def label_rows_systemone_http(
    rows: list[dict[str, Any]],
    api_url: str,
    n_samples: int = 3,
    workers: int = 8,
) -> list[dict[str, dict[str, Any]]]:
    """Label cases via a `/v1/systemone` endpoint (e.g. vLLM System One server)."""
    def _one(idx_row: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, dict[str, Any]]]:
        idx, r = idx_row
        st = json.loads(r["state"]) if isinstance(r["state"], str) else r["state"]
        qs = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        samples = []
        payload = json.dumps({"state": st, "questions": qs}).encode("utf-8")
        for _ in range(n_samples):
            req = urllib.request.Request(
                api_url,
                data=payload,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                ans = json.loads(resp.read().decode("utf-8")).get("answers", {})
            s_map = {}
            for qid, qspec in qs.items():
                opt_keys = [k for k, _ in canonical_options(qspec)]
                a_q = ans.get(qid, {})
                probs = a_q.get("probabilities")
                if not probs and qspec.get("type") == "noul" and "noul" in a_q:
                    pt = float(a_q["noul"])
                    probs = {"true": pt, "false": 1.0 - pt}
                s_map[qid] = normalize_option_probs(probs, opt_keys, laplace_eps=0.005)
            samples.append(s_map)
        return idx, aggregate_teacher_samples(qs, samples)

    results: list[dict[str, dict[str, Any]]] = [{} for _ in rows]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, g_map in ex.map(_one, enumerate(rows)):
            results[idx] = g_map
    return results


def label_rows(
    rows: list[dict[str, Any]],
    teacher: str,
    n_samples: int = 3,
    temperature: float = 0.7,
    batch_size: int = 16,
    workers: int = 8,
    systemone_url: str | None = None,
    runner_cache: dict[Any, Any] | None = None,
    runner_cls: Any = PersistentHFTeacherLabeller,
) -> list[dict[str, dict[str, Any]]]:
    if systemone_url:
        return label_rows_systemone_http(rows, systemone_url, n_samples=n_samples, workers=workers)
    if teacher.startswith("claude-") or teacher.startswith("anthropic:"):
        model_name = teacher.split(":", 1)[1] if ":" in teacher else teacher
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        results: list[dict[str, dict[str, Any]]] = [{} for _ in rows]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {
                ex.submit(
                    _sample_anthropic_case,
                    r["state"],
                    json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"],
                    model_name,
                    api_key,
                    n_samples,
                    temperature,
                ): i
                for i, r in enumerate(rows)
            }
            done_cnt = 0
            for fut in as_completed(fut_map):
                idx = fut_map[fut]
                results[idx] = fut.result()
                done_cnt += 1
                if done_cnt % 100 == 0 or done_cnt == len(rows):
                    print(f"  [Anthropic {model_name}] {done_cnt}/{len(rows)} cases labelled", flush=True)
        return results

    hf_id = teacher.split(":", 1)[1] if teacher.startswith("hf:") else teacher
    cache = _DEFAULT_RUNNER_CACHE if runner_cache is None else runner_cache
    key = (hf_id, n_samples, round(float(temperature), 4))
    if key not in cache:
        cache[key] = runner_cls(hf_id, n_samples=n_samples, temperature=temperature, batch_size=batch_size)
    return cache[key].label_batch(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, help="Teacher model ID (HF repo ID or claude-*)")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples at T=0.7 per case")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--systemone-url", default=None, help="Optional /v1/systemone HTTP endpoint")
    parser.add_argument("--calibrate-train", action="store_true", help="Evaluate KL agreement on real `train`")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for calibration smoke check")
    parser.add_argument("--report-out", default="data/td_synth/teacher_calibration.json")
    parser.add_argument("--in", dest="in_path", default="data/td_synth/cases.jsonl")
    parser.add_argument("--out", dest="out_path", default="data/td_synth/cases_labelled.jsonl")
    args = parser.parse_args(argv)

    if args.calibrate_train:
        from datasets import load_dataset

        ds_train = list(load_dataset("LocalLLaMA/typed-decisions", "all", split="train"))
        if args.limit > 0:
            ds_train = ds_train[: args.limit]
        ref_gold = [json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"] for r in ds_train]
        preds = label_rows(
            ds_train,
            teacher=args.teacher,
            n_samples=args.samples,
            temperature=args.temperature,
            batch_size=args.batch_size,
            workers=args.workers,
            systemone_url=args.systemone_url,
        )
        metrics = evaluate_teacher_kl_against_gold(preds, ref_gold, max_kl=0.10)
        metrics["teacher"] = args.teacher
        metrics["samples"] = args.samples
        metrics["temperature"] = args.temperature
        rep_p = Path(args.report_out)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        existing_cal: dict[str, Any] = {}
        if rep_p.exists():
            try:
                existing_cal = json.loads(rep_p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_cal = {}
        existing_cal.setdefault("candidates", {})[args.teacher] = metrics
        rep_p.write_text(json.dumps(existing_cal, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        return 0

    in_p = Path(args.in_path)
    out_p = Path(args.out_path)
    raw_rows = [json.loads(line) for line in in_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Support clean resume if interrupted
    done_rows: list[dict[str, Any]] = []
    if out_p.exists():
        done_rows = [json.loads(line) for line in out_p.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Resuming labelling from {len(done_rows):,}/{len(raw_rows):,} rows in {out_p}", flush=True)

    todo_rows = raw_rows[len(done_rows) :]
    chunk_sz = 240
    with out_p.open("a", encoding="utf-8") as fh:
        for c_start in range(0, len(todo_rows), chunk_sz):
            sub = todo_rows[c_start : c_start + chunk_sz]
            golds = label_rows(
                sub,
                teacher=args.teacher,
                n_samples=args.samples,
                temperature=args.temperature,
                batch_size=args.batch_size,
                workers=args.workers,
                systemone_url=args.systemone_url,
            )
            for r, g in zip(sub, golds):
                rec = dict(r)
                rec["gold"] = json.dumps(g, ensure_ascii=False)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    manifest_p = out_p.parent / "MANIFEST.json"
    manifest: dict[str, Any] = {}
    if manifest_p.exists():
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    manifest["teacher_id"] = args.teacher
    manifest["teacher_samples"] = args.samples
    manifest["teacher_temperature"] = args.temperature
    manifest["labelled_cases"] = len(raw_rows)
    manifest["labelled_sha256"] = hashlib.sha256(out_p.read_bytes()).hexdigest()
    manifest_p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
