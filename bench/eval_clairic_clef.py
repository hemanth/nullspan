"""Live Empirical Evaluation of Cloudflare Workers AI `@cf/cloudflare/clairic` + `@cf/cloudflare/clef`
on the exact same N=500 NullspanBench items (138 unique documents: 336 Open Structured + 60 Stanford SQuAD v2.0 + 96 Sealed Multi-Numeric Contracts + 8 Adversarial).

Caches raw Cloudflare Workers AI responses to `bench/clairic_clef_cache.json` for reproducibility
and computes identical JevBench/NullspanBench metrics (Intelligence, SQuAD v2.0, Multi-Numeric Contracts,
Structured/Math, Calibration ECE/Brier/AUROC, Latency p50/p95, and Token/Pass usage).
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

from run_bench import (
    bootstrap_ci,
    classify_item_metadata,
    compute_auroc,
    compute_brier,
    compute_ece,
    cost_axis_score,
    harmonic_mean_4,
    load_benchmark_pool,
    percentiles_ms,
    score_single_prediction,
    speed_axis_score,
)

CACHE_PATH = ROOT / "bench" / "clairic_clef_cache.json"
ACCOUNT_ID = "0f6fb593710f5180ab2426f06c0eaa56"


def get_cf_token() -> str:
    cfg_path = Path.home() / ".wrangler" / "config" / "default.toml"
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)["oauth_token"]


def call_cf_api(model: str, payload: dict[str, Any], token: str, retries: int = 3) -> tuple[dict[str, Any], float]:
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{model}"
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        t0 = time.perf_counter()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                ms = (time.perf_counter() - t0) * 1000.0
                data = json.loads(resp.read().decode("utf-8"))
                return data, ms
        except urllib.error.HTTPError as e:
            err_txt = e.read().decode("utf-8")
            if attempt == retries - 1:
                raise RuntimeError(f"HTTP {e.code} from {model}: {err_txt}") from e
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("Unreachable")


def build_doc_payloads(doc_rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    state = doc_rows[0]["state"]
    clairic_fields: dict[str, Any] = {}
    clef_questions: dict[str, Any] = {}

    for r in doc_rows:
        fn = r["field_name"]
        sp = dict(r["field_spec"])
        ft = sp.get("type", "text")

        if ft == "boolean":
            clef_questions[fn] = {
                "type": "noul",
                "instructions": sp.get("instructions", fn),
            }
        elif ft == "enum":
            clef_questions[fn] = {
                "type": "choice",
                "instructions": sp.get("instructions", fn),
                "criteria": {opt: opt for opt in sp["options"]},
            }
        else:
            c_sp: dict[str, Any] = {"type": ft, "instructions": sp.get("instructions", fn)}
            if ft == "number":
                if "decimals" in sp:
                    c_sp["decimals"] = sp["decimals"]
                if "integer" in sp:
                    c_sp["integer"] = sp["integer"]
                if "unit" in sp:
                    c_sp["unit"] = sp["unit"]
            elif ft == "duration":
                c_sp["unit"] = sp.get("unit", "days")
                if "integer" in sp:
                    c_sp["integer"] = sp["integer"]
            elif ft == "list":
                c_sp["items"] = sp.get("items", "text")
            elif ft == "estimate":
                c_sp.update({
                    "min": sp["min"],
                    "max": sp["max"],
                    "step": sp.get("step", 10),
                    "scale": sp.get("scale", "linear"),
                })
                if "unit" in sp:
                    c_sp["unit"] = sp["unit"]
            elif ft == "forecast":
                c_sp.update({
                    "horizon": sp.get("horizon", 1),
                    "interval": sp.get("interval", "day"),
                    "min": sp["min"],
                    "max": sp["max"],
                    "scale": sp.get("scale", "log"),
                })
            clairic_fields[fn] = c_sp

            if "constraint" in sp:
                c_expr = sp["constraint"]
                clef_questions[f"{fn}__verified"] = {
                    "type": "noul",
                    "instructions": f"Does the document satisfy the mathematical constraint: {c_expr}?",
                }

    clairic_req = {"model": "clairic", "state": state, "fields": clairic_fields} if clairic_fields else None
    clef_req = {"model": "clef", "state": state, "questions": clef_questions} if clef_questions else None
    return clairic_req, clef_req


def main() -> None:
    rows = load_benchmark_pool()
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)

    cache: dict[str, Any] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())

    token = get_cf_token()
    missing_docs = [doc_id for doc_id in by_doc if doc_id not in cache]
    print(f"Total unique docs: {len(by_doc)} | Cached: {len(cache)} | To fetch live: {len(missing_docs)}")

    def fetch_one(doc_id: str) -> tuple[str, dict[str, Any]]:
        doc_rows = by_doc[doc_id]
        clairic_req, clef_req = build_doc_payloads(doc_rows)
        entry: dict[str, Any] = {"clairic": None, "clairic_ms": 0.0, "clef": None, "clef_ms": 0.0}
        if clairic_req is not None:
            resp, ms = call_cf_api("@cf/cloudflare/clairic", clairic_req, token)
            entry["clairic"] = resp
            entry["clairic_ms"] = round(ms, 2)
        if clef_req is not None:
            resp, ms = call_cf_api("@cf/cloudflare/clef", clef_req, token)
            entry["clef"] = resp
            entry["clef_ms"] = round(ms, 2)
        return doc_id, entry

    if missing_docs:
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(fetch_one, did): did for did in missing_docs}
            for fut in concurrent.futures.as_completed(futures):
                did, entry = fut.result()
                cache[did] = entry
                done += 1
                if done % 20 == 0 or done == len(missing_docs):
                    print(f"  Fetched {done}/{len(missing_docs)} docs...")
                    CACHE_PATH.write_text(json.dumps(cache, indent=2))

    # Now evaluate all 500 items from the live responses
    scores_all: list[float] = []
    scores_squad: list[float] = []
    scores_contracts: list[float] = []
    scores_struct: list[float] = []
    scores_adv: list[float] = []
    scores_by_type: dict[str, list[float]] = {}
    p_null_preds: list[float] = []
    p_null_golds: list[int] = []
    squad_p_null_preds: list[float] = []
    squad_p_null_golds: list[int] = []
    item_latencies_ms: list[float] = []
    doc_latencies_ms: list[float] = []
    total_input_tokens = 0
    total_passes = 0
    total_rpc_calls = 0

    for doc_id, doc_rows in by_doc.items():
        entry = cache[doc_id]
        c_ms = float(entry.get("clairic_ms", 0.0))
        k_ms = float(entry.get("clef_ms", 0.0))
        doc_total_ms = c_ms + k_ms
        doc_latencies_ms.append(doc_total_ms)
        per_field_ms = doc_total_ms / max(1, len(doc_rows))

        c_ans_map = ((entry.get("clairic") or {}).get("result") or {}).get("answers") or {}
        k_ans_map = ((entry.get("clef") or {}).get("result") or {}).get("answers") or {}
        c_usage = ((entry.get("clairic") or {}).get("result") or {}).get("usage") or {}
        k_usage = ((entry.get("clef") or {}).get("result") or {}).get("usage") or {}
        total_input_tokens += int(c_usage.get("input_tokens", 0)) + int(k_usage.get("input_tokens", 0))
        total_passes += int(c_usage.get("passes", 0)) + (1 if entry.get("clef") else 0)
        total_rpc_calls += (1 if entry.get("clairic") else 0) + (1 if entry.get("clef") else 0)

        for r in doc_rows:
            fn = r["field_name"]
            sp = r["field_spec"]
            ft = sp.get("type", "text")
            meta = classify_item_metadata(r)

            if ft == "boolean":
                k_item = k_ans_map.get(fn, {})
                noul_p = float(k_item.get("noul", 0.5))
                pred_ans = {
                    "value": bool(noul_p >= 0.5),
                    "p_null": 0.01,
                    "p_true": noul_p,
                }
            elif ft == "enum":
                k_item = k_ans_map.get(fn, {})
                pred_ans = {
                    "value": k_item.get("choice"),
                    "p_null": 0.01,
                    "probabilities": k_item.get("probabilities", {}),
                }
            else:
                c_item = dict(c_ans_map.get(fn, {}))
                pred_ans = {
                    "value": c_item.get("value"),
                    "p_null": float(c_item.get("p_null", 0.99 if c_item.get("value") is None else 0.01)),
                    "guess": c_item.get("guess"),
                    "span": c_item.get("span"),
                    "p10": c_item.get("p10"),
                    "p90": c_item.get("p90"),
                }
                if "constraint" in sp:
                    ver_item = k_ans_map.get(f"{fn}__verified", {})
                    ver_noul = float(ver_item.get("noul", 0.5))
                    pred_ans["verified"] = bool(ver_noul >= 0.5)

            sc, p_null_val, gold_null = score_single_prediction(pred_ans, r)
            scores_all.append(sc)
            scores_by_type.setdefault(meta["req_type"], []).append(sc)
            p_null_preds.append(p_null_val)
            p_null_golds.append(gold_null)
            item_latencies_ms.append(per_field_ms)

            if r["domain"] == "squad_v2_open_qa":
                scores_squad.append(sc)
                squad_p_null_preds.append(p_null_val)
                squad_p_null_golds.append(gold_null)
            elif r["domain"] == "contract_extraction":
                scores_contracts.append(sc)
            elif r["domain"] == "adversarial_stress":
                scores_adv.append(sc)
                scores_struct.append(sc)
            else:
                scores_struct.append(sc)

    overall_mean, overall_lo, overall_hi = bootstrap_ci(scores_all)
    squad_mean, squad_lo, squad_hi = bootstrap_ci(scores_squad)
    contracts_mean, contracts_lo, contracts_hi = bootstrap_ci(scores_contracts)
    struct_mean, struct_lo, struct_hi = bootstrap_ci(scores_struct)
    adv_mean, _, _ = bootstrap_ci(scores_adv)

    ece = compute_ece(p_null_preds, p_null_golds)
    brier = compute_brier(p_null_preds, p_null_golds)
    auroc_all = compute_auroc(p_null_preds, p_null_golds)
    auroc_squad = compute_auroc(squad_p_null_preds, squad_p_null_golds)
    cal_score = round(100.0 * (1.0 - min(1.0, 0.65 * ece + 0.35 * brier)), 1)

    lat_item = percentiles_ms(item_latencies_ms)
    lat_doc = percentiles_ms(doc_latencies_ms)

    summary = {
        "system": "@cf/cloudflare/clairic + @cf/cloudflare/clef (Live Workers AI)",
        "n_items": len(scores_all),
        "n_docs": len(by_doc),
        "total_rpc_calls": total_rpc_calls,
        "total_passes": total_passes,
        "total_input_tokens": total_input_tokens,
        "overall_acc_pct": round(overall_mean * 100, 1),
        "overall_ci95_pct": [round(overall_lo * 100, 1), round(overall_hi * 100, 1)],
        "squad_v2_acc_pct": round(squad_mean * 100, 1),
        "squad_v2_ci95_pct": [round(squad_lo * 100, 1), round(squad_hi * 100, 1)],
        "squad_v2_auroc": round(auroc_squad, 3),
        "contracts_sealed_acc_pct": round(contracts_mean * 100, 1),
        "contracts_sealed_ci95_pct": [round(contracts_lo * 100, 1), round(contracts_hi * 100, 1)],
        "structured_math_acc_pct": round(struct_mean * 100, 1),
        "structured_math_ci95_pct": [round(struct_lo * 100, 1), round(struct_hi * 100, 1)],
        "adversarial_acc_pct": round(adv_mean * 100, 1),
        "auroc_all": round(auroc_all, 3),
        "ece": ece,
        "brier": brier,
        "calibration_score": cal_score,
        "latency_per_item_ms": lat_item,
        "latency_per_doc_ms": lat_doc,
        "by_req_type_pct": {k: round(sum(v) / len(v) * 100, 1) for k, v in scores_by_type.items()},
    }
    print("\n=== LIVE CLOUDFLARE CLAIRIC + CLEF BENCHMARK SUMMARY ===")
    print(json.dumps(summary, indent=2))
    (ROOT / "bench" / "clairic_clef_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
