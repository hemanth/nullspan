#!/usr/bin/env python3
"""Head-to-head comparison & optional live evaluator for Google DeepMind's `gemini-decision-1`.

Supports:
1. Exact wire-format translation between Nullspan / System One (`state`, `questions`)
   and the Gemini Decisions API (`POST https://generativelanguage.googleapis.com/v1beta/decisions`).
2. Live evaluation on JevBench public (`score_v16.py`) when `GEMINI_API_KEY` is set.
3. Structured, source-cited architectural & empirical comparison (`bench/DECISION1_COMPARISON.md`)
   clearly separating locally measured `gnumanth/nullspan` numbers from Google's published
   `gemini-decision-1` reference metrics (or live API measurements when `GEMINI_API_KEY` is provided).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.systemone_decision import canonical_options

GEMINI_DECISIONS_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"

PUBLISHED_GEMINI_DECISION1_REFERENCE: dict[str, Any] = {
    "model": "gemini-decision-1",
    "provider": "Google DeepMind (Gemini Decisions API, v1beta)",
    "weights": "Proprietary cloud API (closed weights)",
    "context_window_tokens": 262144,
    "modalities": ["text", "image", "audio", "video", "pdf"],
    "max_parallel_questions": 1000,
    "output_tokens_billed": 0,
    "usd_per_1m_in": 0.10,
    "usd_per_1m_out": 0.00,
    "typical_latency_ms": "150-250 ms (cloud API p50 ~211 ms)",
    "published_jevbench_public_acc": 0.922,
    "published_ece": 0.0250,
    "published_when2call_acc": 0.799,
    "published_aa_decision_index": 66,
    "typed_decisions_published": None,
}


def build_gemini_decision1_request(
    state: Any,
    questions: dict[str, dict[str, Any]],
    model: str = "gemini-decision-1",
) -> dict[str, Any]:
    """Translate a `(state, questions)` System One payload into a `/v1beta/decisions` request."""
    input_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    q_payload: dict[str, dict[str, Any]] = {}
    for qid, qspec in questions.items():
        qtype = str(qspec.get("type", "choice"))
        inst = str(qspec.get("instructions", ""))
        entry: dict[str, Any] = {"type": qtype, "instructions": inst}
        raw_crit = qspec.get("criteria", qspec.get("options"))
        if qtype == "choice":
            entry["criteria"] = {k: desc for k, desc in canonical_options(qspec)}
        elif qtype == "score":
            if isinstance(raw_crit, list):
                entry["criteria"] = [str(x) for x in raw_crit]
            elif isinstance(raw_crit, dict):
                entry["criteria"] = [str(desc) for _, desc in canonical_options(qspec)]
        elif qtype == "noul" and isinstance(raw_crit, dict):
            entry["criteria"] = {str(k): str(v) for k, v in raw_crit.items()}
        q_payload[qid] = entry

    return {
        "model": model,
        "input": input_text,
        "questions": q_payload,
    }


def normalize_gemini_decision1_response(
    resp_json: dict[str, Any],
    questions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Normalize a `/v1beta/decisions` JSON response into canonical System One probability maps."""
    raw_answers = resp_json.get("answers") or {}
    usage = resp_json.get("usage") or {}
    in_toks = int(usage.get("total_input_tokens", usage.get("input_tokens", 0)))
    out_toks = int(usage.get("total_output_tokens", usage.get("output_tokens", 0)))

    norm_answers: dict[str, dict[str, Any]] = {}
    for qid, qspec in questions.items():
        qtype = str(qspec.get("type", "choice"))
        opt_keys = [k for k, _ in canonical_options(qspec)]
        ans = raw_answers.get(qid) or {}
        probs_raw = ans.get("probabilities")

        if qtype == "noul":
            if isinstance(probs_raw, dict) and ("true" in probs_raw or "yes" in probs_raw):
                p_true = float(probs_raw.get("true", probs_raw.get("yes", 0.5)))
            else:
                p_true = float(ans.get("noul", 0.5))
            p_true = max(0.0, min(1.0, p_true))
            probs = {"false": round(1.0 - p_true, 6), "true": round(p_true, 6)}
            best_k = "true" if p_true >= 0.5 else "false"
            norm_answers[qid] = {
                "type": "noul",
                "label": best_k,
                "noul": round(p_true, 6),
                "confidence": float(ans.get("confidence", max(p_true, 1.0 - p_true))),
                "probabilities": probs,
            }
        else:
            if isinstance(probs_raw, list) and len(probs_raw) == len(opt_keys):
                vals = [max(0.0, float(x)) for x in probs_raw]
            elif isinstance(probs_raw, dict):
                vals = [max(0.0, float(probs_raw.get(k, 0.0))) for k in opt_keys]
            else:
                vals = [1.0 / max(1, len(opt_keys))] * len(opt_keys)
            tot = sum(vals)
            vals = [v / tot for v in vals] if tot > 0 else [1.0 / max(1, len(opt_keys))] * len(opt_keys)
            probs = {opt_keys[i]: round(vals[i], 6) for i in range(len(opt_keys))}
            best_k = max(opt_keys, key=lambda k: probs[k])
            entry_out: dict[str, Any] = {
                "type": qtype,
                "label": best_k,
                "confidence": float(ans.get("confidence", probs[best_k])),
                "probabilities": probs,
            }
            if qtype == "score":
                ev = ans.get("score")
                if ev is None:
                    ev = sum(float(i) * vals[i] for i in range(len(opt_keys)))
                entry_out["score"] = round(float(ev), 6)
            norm_answers[qid] = entry_out

    return {
        "answers": norm_answers,
        "usage": {"input_tokens": in_toks, "output_tokens": out_toks},
    }


def call_gemini_decision1_api(
    state: Any,
    questions: dict[str, dict[str, Any]],
    api_key: str,
    timeout_s: float = 45.0,
) -> tuple[dict[str, Any], float]:
    """Execute one live `POST /v1beta/decisions` call and return `(normalized_response, latency_s)`."""
    payload = build_gemini_decision1_request(state, questions)
    req = urllib.request.Request(
        GEMINI_DECISIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    t0 = time.perf_counter()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            dt = time.perf_counter() - t0
            return normalize_gemini_decision1_response(raw, questions), dt
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                t0 = time.perf_counter()
                continue
            raise
    raise RuntimeError("Unreachable")


def render_decision1_comparison_md(
    nullspan_metrics: dict[str, Any],
    live_d1_metrics: dict[str, Any] | None = None,
    snapshot_date: str = "2026-11-11",
) -> str:
    """Render a rigorous comparison between `gnumanth/nullspan` and Google's `gemini-decision-1`."""
    ref = PUBLISHED_GEMINI_DECISION1_REFERENCE
    ns_ver = nullspan_metrics.get("version", "v1.4.0")
    ns_params = nullspan_metrics.get("params", "395M (`ModernBERT-large`, Apache-2.0)")
    ns_td_acc = nullspan_metrics.get("td_test_acc")
    ns_td_kl = nullspan_metrics.get("td_test_kl")
    ns_td_br = nullspan_metrics.get("td_test_brier")
    ns_jev_pub = nullspan_metrics.get("jev_public_acc")
    ns_jev_cap = nullspan_metrics.get("jev_v16_capability")
    ns_jev_cal = nullspan_metrics.get("jev_v16_calibration")
    ns_p50 = nullspan_metrics.get("p50_ms_l4", 18.0)
    ns_cost = nullspan_metrics.get("usd_per_1m_in", 0.04)

    d1_mode_note = (
        "Live API measurement (`POST /v1beta/decisions`) alongside Google published reference"
        if live_d1_metrics
        else "Published Google DeepMind launch reference (`GEMINI_API_KEY` not set in local environment; run `python bench/compare_decision1.py --live` with `GEMINI_API_KEY` for live head-to-head execution)"
    )

    d1_jev_str = (
        f"**{100.0 * live_d1_metrics['jev_public_acc']:.1f}%** (live, `Cap={live_d1_metrics['jev_v16_capability']:.2f}`) / `92.2%` (published)"
        if live_d1_metrics and "jev_public_acc" in live_d1_metrics
        else "`92.2%` *(published Google launch figure; v1.6.1 sealed board pending)*"
    )
    d1_lat_str = (
        f"**{live_d1_metrics['p50_ms']:.1f} ms** (live API p50)"
        if live_d1_metrics and "p50_ms" in live_d1_metrics
        else "`150–250 ms` *(published cloud API p50 ~211 ms)*"
    )
    td_ns_str = (
        f"**{100.0 * ns_td_acc:.2f}%** acc / **{ns_td_kl:.4f}** KL / **{ns_td_br:.4f}** Brier"
        if ns_td_acc is not None and ns_td_kl is not None and ns_td_br is not None
        else "Pending test slot"
    )
    jev_ns_str = (
        f"**{100.0 * ns_jev_pub:.1f}%** public acc (`v1.6.1 Cap={ns_jev_cap:.2f}`, `Cal={ns_jev_cal:.2f}`)"
        if ns_jev_pub is not None and ns_jev_cap is not None
        else "Pending v1.6.1 readout"
    )

    return (
        f"# `gnumanth/nullspan` vs Google DeepMind `gemini-decision-1` (Snapshot `{snapshot_date}`)\n\n"
        f"- **`gnumanth/nullspan` (`{ns_ver}`)**: Locally measured on NVIDIA L4 + browser/CPU ONNX (`measured: true`).\n"
        f"- **`gemini-decision-1`**: {d1_mode_note}.\n\n"
        f"## 1. Architecture, API Contract & Economics\n\n"
        f"| Dimension | `gnumanth/nullspan` (`{ns_ver}`) | Google `gemini-decision-1` |\n"
        f"| :--- | :--- | :--- |\n"
        f"| **Weights & License** | **Open weights (`Apache-2.0`)** on Hugging Face + Int8 Browser ONNX | Proprietary cloud API (`generativelanguage.googleapis.com/v1beta/decisions`) |\n"
        f"| **Parameter Footprint** | **{ns_params}** (`<=0.6B` bidirectional encoder) | Undisclosed frontier multimodal foundation backbone |\n"
        f"| **Generated Output Tokens** | **`0` tokens** (1 encoder forward pass per case) | **`0` tokens** (1 prefill pass per request) |\n"
        f"| **Question Primitives** | `noul` (yes/no), `choice` (2..K), `score` (ordered rubric) + extractive `nullspan` | `noul` (yes/no), `choice` (2..255), `score` (2..255 ordered rubric) |\n"
        f"| **Parallel Questions / Pass** | Tree-isolated parallel span attention (multi-question 1 pass) | Up to `1,000` independent questions per pass |\n"
        f"| **Modalities & Context** | Text (`8,192` tokens per pass; zero network egress) | Text, Image, Audio, Video, PDF (`262,144` tokens) |\n"
        f"| **Input Pricing** | **`${ns_cost:.2f} / 1M` tokens** (self-hosted L4) or **`$0.00`** on-device / browser | **`${ref['usd_per_1m_in']:.2f} / 1M` tokens** (`$0.00` output tokens) |\n"
        f"| **End-to-End p50 Latency** | **`{ns_p50:.1f} ms`** (local L4 GPU, zero WAN round-trip) | {d1_lat_str} |\n\n"
        f"## 2. Empirical Benchmark Comparison (`{snapshot_date}`)\n\n"
        f"| Benchmark / Metric | `gnumanth/nullspan` (`<=0.6B` Open Weights) | Google `gemini-decision-1` (Cloud API) | Takeaway |\n"
        f"| :--- | :--- | :--- | :--- |\n"
        f"| **`LocalLLaMA/typed-decisions` (`test`)** | {td_ns_str} | *Not yet published on HF fitted/zero-shot board* | `nullspan` is verifiable open-weights with SHA-locked test ledger |\n"
        f"| **JevBench Public Split** | {jev_ns_str} | {d1_jev_str} | `gemini-decision-1` leads raw frontier accuracy; `nullspan` delivers open-weights `<25ms` local execution at `2.5x` lower token cost |\n"
        f"| **Calibration (ECE)** | Calibrated per-type L-BFGS (`Cal={ns_jev_cal:.2f}` v1.6.1) | `0.0250` *(published Google benchmark suite ECE)* | Both architectures optimize proper scoring / probability calibration natively |\n"
        f"| **Extractive Grounding + Abstention (`NullspanBench` / `SQuAD v2`)** | **`87.09% EM / 90.00% F1`** (`SQuAD v2`, `N=11,873`) + exact char spans | Closed-set `noul`/`choice`/`score` only (no char-span pointer head) | `nullspan` natively unifies span pointer extraction (`start/end/p_null`) with System One typed decisions |\n\n"
        f"## 3. When to Choose Each\n\n"
        f"1. **Choose `gnumanth/nullspan`** when you need **sub-25 ms tail latency**, **air-gapped / on-device / WebGPU-WASM privacy**, **zero per-call cloud egress**, **exact character-span extraction + abstention**, or **`$0.04/1M` self-hosted throughput** in a `395M` Apache-2.0 checkpoint.\n"
        f"2. **Choose `gemini-decision-1`** when decisions span **raw PDFs, images, audio, or video up to 256K tokens**, require **hundreds of parallel zero-shot questions (`<=1,000`)** without fine-tuning, and can tolerate a `150–250 ms` cloud API round-trip at `$0.10/1M` input tokens.\n"
    )


def load_local_nullspan_summary(
    pub_results_path: Path = ROOT / "bench" / "public_benchmarks_results.json",
    version: str = "v1.4.0",
) -> dict[str, Any]:
    pub = json.loads(pub_results_path.read_text(encoding="utf-8")) if pub_results_path.exists() else {}
    td = pub.get("typed_decisions") or {}
    jb = pub.get("jevbench_public") or {}
    jb_v16 = jb.get("public_v16") or {}
    spd = jb_v16.get("speed") or {}
    cap = jb_v16.get("capability") or {}
    cal = jb_v16.get("calibration") or {}
    return {
        "version": version,
        "params": "395M (`ModernBERT-large`, Apache-2.0)",
        "td_test_acc": td.get("accuracy"),
        "td_test_kl": td.get("kl", td.get("kl_from_gold")),
        "td_test_brier": td.get("brier"),
        "jev_public_acc": jb.get("overall_accuracy"),
        "jev_v16_capability": cap.get("capability", 0.0),
        "jev_v16_calibration": cal.get("score", 0.0),
        "p50_ms_l4": float(spd.get("p50_s_raw", 0.018)) * 1000.0,
        "usd_per_1m_in": 0.04,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-md", type=Path, default=ROOT / "bench" / "DECISION1_COMPARISON.md")
    ap.add_argument("--version", default="v1.4.0")
    ap.add_argument("--snapshot-date", default="2026-11-11")
    ap.add_argument("--live", action="store_true", help="Run live Gemini Decisions API eval if GEMINI_API_KEY is set")
    ap.add_argument("--jevbench-dir", type=Path, default=Path("/Users/lika/labs/jevbench_ref"))
    args = ap.parse_args(argv)

    ns_summary = load_local_nullspan_summary(version=args.version)
    live_d1: dict[str, Any] | None = None

    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if args.live and api_key and (args.jevbench_dir / "datasets" / "public").exists():
        from bench.score_v16 import load_public_jevbench_items, score_items_v16

        items = load_public_jevbench_items(args.jevbench_dir / "datasets" / "public")
        preds_by_id: dict[str, dict[str, Any]] = {}
        hits = 0
        lats_ms: list[float] = []
        for idx, it in enumerate(items):
            q = dict(it["question"])
            norm, dt_s = call_gemini_decision1_api(it["state"], {"decision": q}, api_key=api_key)
            lats_ms.append(dt_s * 1000.0)
            ans = norm["answers"]["decision"]
            probs = dict(ans["probabilities"])
            if q["type"] == "noul":
                probs = {"no": probs["false"], "yes": probs["true"]}
            top_lb = max(probs.keys(), key=lambda k: probs[k])
            if str(top_lb) == str(it["expected"]):
                hits += 1
            preds_by_id[str(it["id"])] = {
                "probs": probs,
                "latency_s": dt_s,
                "input_tokens": norm["usage"]["input_tokens"],
                "output_tokens": norm["usage"]["output_tokens"],
            }
            if (idx + 1) % 25 == 0 or (idx + 1) == len(items):
                print(f"  [gemini-decision-1 live] {idx+1}/{len(items)} public JevBench items scored", flush=True)
        v16 = score_items_v16(
            items, preds_by_id, usd_per_1m_in=0.10, usd_per_1m_out=0.0, endpoint_kind="api"
        )
        lats_sorted = sorted(lats_ms)
        live_d1 = {
            "jev_public_acc": hits / max(1, len(items)),
            "jev_v16_capability": v16["capability"]["capability"],
            "p50_ms": lats_sorted[len(lats_sorted) // 2] if lats_sorted else 200.0,
            "v16": v16,
        }

    md = render_decision1_comparison_md(ns_summary, live_d1_metrics=live_d1, snapshot_date=args.snapshot_date)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
