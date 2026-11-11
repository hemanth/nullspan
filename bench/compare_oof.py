#!/usr/bin/env python3
"""Paired case-clustered bootstrap comparison (B=2,000) for System One OOF reports."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any


def _group_by_case(per_decision: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in per_decision:
        c_idx = int(row["case_idx"])
        qid = str(row["qid"])
        grouped[c_idx][qid] = row
    return dict(grouped)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def compare_oof_reports(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    cases_a = _group_by_case(report_a["per_decision"])
    cases_b = _group_by_case(report_b["per_decision"])
    common_cases = sorted(set(cases_a.keys()) & set(cases_b.keys()))
    if not common_cases:
        raise ValueError("No overlapping case_idx entries between OOF reports.")

    # Precompute per-case sums and decision counts for fast exact cluster resampling
    case_stats: list[tuple[int, float, float, float, float, float, float]] = []
    tot_n = 0
    sum_acc_a = sum_acc_b = 0.0
    sum_kl_a = sum_kl_b = 0.0
    sum_br_a = sum_br_b = 0.0

    for c_idx in common_cases:
        qa = cases_a[c_idx]
        qb = cases_b[c_idx]
        qids = sorted(set(qa.keys()) & set(qb.keys()))
        if not qids:
            continue
        k_q = len(qids)
        ca_acc = sum(1.0 if qa[q]["correct"] else 0.0 for q in qids)
        cb_acc = sum(1.0 if qb[q]["correct"] else 0.0 for q in qids)
        ca_kl = sum(float(qa[q]["kl"]) for q in qids)
        cb_kl = sum(float(qb[q]["kl"]) for q in qids)
        ca_br = sum(float(qa[q]["brier"]) for q in qids)
        cb_br = sum(float(qb[q]["brier"]) for q in qids)

        case_stats.append((k_q, ca_acc, cb_acc, ca_kl, cb_kl, ca_br, cb_br))
        tot_n += k_q
        sum_acc_a += ca_acc
        sum_acc_b += cb_acc
        sum_kl_a += ca_kl
        sum_kl_b += cb_kl
        sum_br_a += ca_br
        sum_br_b += cb_br

    if tot_n == 0:
        raise ValueError("Zero matched (case_idx, qid) pairs across OOF reports.")

    acc_a, acc_b = sum_acc_a / tot_n, sum_acc_b / tot_n
    kl_a, kl_b = sum_kl_a / tot_n, sum_kl_b / tot_n
    br_a, br_b = sum_br_a / tot_n, sum_br_b / tot_n

    rng = random.Random(seed)
    n_clusters = len(case_stats)
    boot_d_acc: list[float] = []
    boot_d_kl: list[float] = []
    boot_d_br: list[float] = []

    for _ in range(n_boot):
        b_n = 0
        b_da = b_dk = b_db = 0.0
        for _c in range(n_clusters):
            k_q, ca_acc, cb_acc, ca_kl, cb_kl, ca_br, cb_br = case_stats[rng.randrange(n_clusters)]
            b_n += k_q
            b_da += cb_acc - ca_acc
            b_dk += cb_kl - ca_kl
            b_db += cb_br - ca_br
        boot_d_acc.append(b_da / b_n)
        boot_d_kl.append(b_dk / b_n)
        boot_d_br.append(b_db / b_n)

    boot_d_acc.sort()
    boot_d_kl.sort()
    boot_d_br.sort()

    acc_ci = [_percentile(boot_d_acc, 0.025), _percentile(boot_d_acc, 0.975)]
    kl_ci = [_percentile(boot_d_kl, 0.025), _percentile(boot_d_kl, 0.975)]
    br_ci = [_percentile(boot_d_br, 0.025), _percentile(boot_d_br, 0.975)]

    # Review Focus #5 + Task 2 Acceptance Gate:
    # Reject any candidate that significantly regresses KL or Brier (95% CI strictly > 0),
    # or regresses accuracy significantly. Keep iff:
    #   (accuracy 95% CI > 0) OR (KL 95% CI < 0 and accuracy diff >= 0),
    #   AND neither KL nor Brier 95% CI is strictly > 0.
    kl_regressed = kl_ci[0] > 0.0
    brier_regressed = br_ci[0] > 0.0
    acc_sig_up = acc_ci[0] > 0.0
    kl_sig_down_no_acc_drop = (kl_ci[1] < 0.0) and ((acc_b - acc_a) >= -1e-12)
    accepted = bool((acc_sig_up or kl_sig_down_no_acc_drop) and not kl_regressed and not brier_regressed)

    return {
        "n_cases": n_clusters,
        "n_decisions": tot_n,
        "bootstraps": n_boot,
        "accuracy": {
            "a": round(acc_a, 6),
            "b": round(acc_b, 6),
            "diff": round(acc_b - acc_a, 6),
            "ci95": [round(acc_ci[0], 6), round(acc_ci[1], 6)],
        },
        "kl": {
            "a": round(kl_a, 6),
            "b": round(kl_b, 6),
            "diff": round(kl_b - kl_a, 6),
            "ci95": [round(kl_ci[0], 6), round(kl_ci[1], 6)],
        },
        "brier": {
            "a": round(br_a, 6),
            "b": round(br_b, 6),
            "diff": round(br_b - br_a, 6),
            "ci95": [round(br_ci[0], 6), round(br_ci[1], 6)],
        },
        "accepted": accepted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_a", help="Baseline oof_report.json path")
    parser.add_argument("report_b", help="Candidate oof_report.json path")
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Print full JSON comparison")
    args = parser.parse_args(argv)

    rep_a = json.loads(Path(args.report_a).read_text(encoding="utf-8"))
    rep_b = json.loads(Path(args.report_b).read_text(encoding="utf-8"))
    res = compare_oof_reports(rep_a, rep_b, n_boot=args.bootstraps, seed=args.seed)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(
            f"Paired Case-Clustered Bootstrap (cases={res['n_cases']}, decisions={res['n_decisions']}, B={res['bootstraps']}):"
        )
        for metric in ("accuracy", "kl", "brier"):
            m = res[metric]
            lo, hi = m["ci95"]
            print(
                f"  {metric:8s}: {m['a']:.4f} -> {m['b']:.4f} | diff={m['diff']:+.4f} | 95% CI [{lo:+.4f}, {hi:+.4f}]"
            )
        print(f"GATE: {'PASS' if res['accepted'] else 'REJECT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
