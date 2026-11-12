#!/usr/bin/env python3
"""Out-of-Fold Error & Calibration Diagnostic Analyzer for LocalLLaMA/typed-decisions (Task 3).

Consumes `oof_report.json` produced by `training/train_systemone.py` and writes `bench/td_error_report.json` with:
  (a) accuracy, KL, Brier, and recoverable error count by `(workflow, qid)` schema,
  (b) share of model errors where gold top-1 probability <= 2/3,
  (c) accuracy of a leave-one-case-out per-schema train-prior predictor (plus schemas where prior >= model),
  (d) 10-bin ECE of the exact gold-matching predictor (predicts exact gold distribution, scored against gold argmax),
  (e) error rates by state token-length bucket + 99th-percentile state token length & % cases > 2,560 tokens.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any


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
            ece += (bin_total[b] / n) * abs((bin_hits[b] / bin_total[b]) - (bin_conf[b] / bin_total[b]))
    return ece


def _bucket_label(tokens: int) -> str:
    if tokens <= 512:
        return "0000_0512"
    if tokens <= 1024:
        return "0513_1024"
    if tokens <= 2048:
        return "1025_2048"
    if tokens <= 2560:
        return "2049_2560"
    if tokens <= 3072:
        return "2561_3072"
    return "3073_plus"


def _percentile(sorted_vals: list[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(math.ceil((pct / 100.0) * len(sorted_vals))) - 1))
    return int(sorted_vals[idx])


def analyze_oof_report(oof_report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = oof_report.get("per_decision", [])
    if not rows:
        raise ValueError("oof_report['per_decision'] is empty.")

    two_thirds_eps = (2.0 / 3.0) + 1e-5

    # (a) Per-(workflow, qid) schema breakdown
    schema_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        key = f"{r['workflow']}::{r['qid']}"
        schema_groups[key].append(r)

    by_schema: dict[str, dict[str, Any]] = {}
    prior_hits_all: list[bool] = []
    schemas_where_prior_beats_model: list[str] = []

    for schema_key, srows in sorted(schema_groups.items()):
        n_s = len(srows)
        n_correct = sum(1 for r in srows if r["correct"])
        n_errors = n_s - n_correct
        recov_errors = sum(
            1
            for r in srows
            if (not r["correct"]) and max(float(v) for v in r["gold_probs"].values()) > two_thirds_eps
        )
        mean_kl = sum(float(r["kl"]) for r in srows) / n_s
        mean_brier = sum(float(r["brier"]) for r in srows) / n_s

        # (c) Leave-one-row-out per-schema train prior accuracy
        sum_gold: dict[str, float] = collections.defaultdict(float)
        for r in srows:
            for opt_k, opt_v in r["gold_probs"].items():
                sum_gold[opt_k] += float(opt_v)

        prior_hits_schema = 0
        for r in srows:
            opts = list(r["gold_probs"].keys())
            if n_s > 1:
                loo_scores = [sum_gold[o] - float(r["gold_probs"][o]) for o in opts]
            else:
                loo_scores = [sum_gold[o] for o in opts]
            gold_vec = [float(r["gold_probs"][o]) for o in opts]
            hit = loo_scores.index(max(loo_scores)) == gold_vec.index(max(gold_vec))
            prior_hits_schema += int(hit)
            prior_hits_all.append(hit)

        prior_acc_s = prior_hits_schema / n_s
        model_acc_s = n_correct / n_s
        if prior_acc_s > model_acc_s + 1e-9:
            schemas_where_prior_beats_model.append(schema_key)

        by_schema[schema_key] = {
            "n": n_s,
            "qtype": srows[0].get("qtype", ""),
            "accuracy": round(model_acc_s, 6),
            "prior_accuracy": round(prior_acc_s, 6),
            "kl": round(mean_kl, 6),
            "brier": round(mean_brier, 6),
            "errors": n_errors,
            "recoverable_errors": recov_errors,
        }

    top_recoverable_schemas = sorted(
        [{"schema": k, **v} for k, v in by_schema.items()],
        key=lambda x: (-x["recoverable_errors"], -x["errors"], x["schema"]),
    )

    # (b) Share of errors where gold top-1 <= 2/3
    error_rows = [r for r in rows if not r["correct"]]
    noisy_errors = [
        r for r in error_rows if max(float(v) for v in r["gold_probs"].values()) <= two_thirds_eps
    ]
    share_noisy_errors = (len(noisy_errors) / len(error_rows)) if error_rows else 0.0

    # Also compute overall share of ALL decisions where gold top-1 <= 2/3 (3-sample Bayes ceiling context)
    n_split_vote = sum(1 for r in rows if max(float(v) for v in r["gold_probs"].values()) <= two_thirds_eps)

    # (d) ECE of the exact gold-matching predictor (predicts exact gold_probs, scored against gold argmax)
    # By definition, argmax(gold_probs) == argmax(gold_probs), so top-1 accuracy is True on every row,
    # while confidence is max(gold_probs).
    gold_confs = [max(float(v) for v in r["gold_probs"].values()) for r in rows]
    gold_accs = [True] * len(rows)
    gold_matching_ece = compute_ece_10bin(gold_accs, gold_confs)

    # (e) Errors by state token-length bucket + case-level token length distribution
    bucket_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    case_tokens: dict[int, int] = {}
    for r in rows:
        st_tok = int(r.get("state_tokens", 0))
        bucket_groups[_bucket_label(st_tok)].append(r)
        case_tokens[int(r["case_idx"])] = st_tok

    by_token_bucket: dict[str, dict[str, Any]] = {}
    for b_key, brows in sorted(bucket_groups.items()):
        nb = len(brows)
        n_err_b = sum(1 for r in brows if not r["correct"])
        by_token_bucket[b_key] = {
            "n_decisions": nb,
            "accuracy": round((nb - n_err_b) / nb, 6),
            "error_rate": round(n_err_b / nb, 6),
            "errors": n_err_b,
            "kl": round(sum(float(r["kl"]) for r in brows) / nb, 6),
        }

    sorted_case_toks = sorted(case_tokens.values())
    n_cases = max(1, len(sorted_case_toks))
    n_gt_2560 = sum(1 for t in sorted_case_toks if t > 2560)

    return {
        "n_cases": len(sorted_case_toks),
        "n_decisions": len(rows),
        "oof_accuracy": oof_report.get("oof_accuracy"),
        "oof_kl": oof_report.get("oof_kl"),
        "oof_brier": oof_report.get("oof_brier"),
        "oof_ece": oof_report.get("oof_ece"),
        "schema_prior_oof_accuracy": round(sum(prior_hits_all) / len(prior_hits_all), 6),
        "schemas_where_prior_beats_model": schemas_where_prior_beats_model,
        "total_errors": len(error_rows),
        "errors_gold_top1_le_two_thirds": len(noisy_errors),
        "share_errors_gold_top1_le_two_thirds": round(share_noisy_errors, 6),
        "share_all_decisions_gold_top1_le_two_thirds": round(n_split_vote / len(rows), 6),
        "gold_matching_predictor_ece": round(gold_matching_ece, 6),
        "ece_stays_off_target_list": bool(gold_matching_ece > 0.10),
        "state_token_stats": {
            "min": sorted_case_toks[0] if sorted_case_toks else 0,
            "p50": _percentile(sorted_case_toks, 50.0),
            "p90": _percentile(sorted_case_toks, 90.0),
            "p95": _percentile(sorted_case_toks, 95.0),
            "p98": _percentile(sorted_case_toks, 98.0),
            "p99": _percentile(sorted_case_toks, 99.0),
            "max": sorted_case_toks[-1] if sorted_case_toks else 0,
            "cases_gt_2560": n_gt_2560,
            "pct_cases_gt_2560": round(100.0 * n_gt_2560 / n_cases, 2),
        },
        "by_token_bucket": by_token_bucket,
        "top_recoverable_schemas": top_recoverable_schemas[:10],
        "by_schema": by_schema,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof",
        default="artifacts/nullspan/oof_report.json",
        help="Path to oof_report.json produced by training/train_systemone.py",
    )
    parser.add_argument(
        "--out",
        default="bench/td_error_report.json",
        help="Output diagnostic JSON path",
    )
    args = parser.parse_args(argv)

    oof_data = json.loads(Path(args.oof).read_text(encoding="utf-8"))
    report = analyze_oof_report(oof_data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"TD Error Report -> {out_path}\n"
        f"  OOF Acc: {report['oof_accuracy']} | Prior Acc: {report['schema_prior_oof_accuracy']}\n"
        f"  Share of errors with gold top-1 <= 2/3: {report['share_errors_gold_top1_le_two_thirds']:.2%}\n"
        f"  Gold-matching predictor ECE: {report['gold_matching_predictor_ece']:.4f} "
        f"(ECE stays off target list: {report['ece_stays_off_target_list']})\n"
        f"  Cases > 2560 tokens: {report['state_token_stats']['pct_cases_gt_2560']}% "
        f"(p99={report['state_token_stats']['p99']})\n"
        f"  Top-5 recoverable schemas: {[s['schema'] + ':' + str(s['recoverable_errors']) for s in report['top_recoverable_schemas'][:5]]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
