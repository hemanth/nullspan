#!/usr/bin/env python3
"""Render Phase 1 Typed Decisions out-of-fold ablation table & Task 7 RC gate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_phase1_rc_gate(
    baseline_oof: dict[str, Any],
    candidate_oof: dict[str, Any],
    *,
    min_acc: float = 0.815,
    min_kl_drop: float = 0.004,
) -> dict[str, Any]:
    """Task 7 Step 1 gate (plan.md:186):
    Proceed to Typed Decisions test ONLY if:
      - candidate oof_accuracy >= 0.815
      - candidate oof_kl <= baseline oof_kl - 0.004
    """
    base_acc = float(baseline_oof["oof_accuracy"])
    base_kl = float(baseline_oof["oof_kl"])
    base_brier = float(baseline_oof.get("oof_brier", 0.0))

    cand_acc = float(candidate_oof["oof_accuracy"])
    cand_kl = float(candidate_oof["oof_kl"])
    cand_brier = float(candidate_oof.get("oof_brier", 0.0))

    max_allowed_kl = base_kl - min_kl_drop
    acc_ok = cand_acc >= min_acc - 1e-12
    kl_ok = cand_kl <= max_allowed_kl + 1e-12
    passed = bool(acc_ok and kl_ok)

    return {
        "passed": passed,
        "acc_ok": acc_ok,
        "kl_ok": kl_ok,
        "min_acc": min_acc,
        "max_allowed_kl": round(max_allowed_kl, 6),
        "baseline_acc": round(base_acc, 6),
        "baseline_kl": round(base_kl, 6),
        "baseline_brier": round(base_brier, 6),
        "candidate_acc": round(cand_acc, 6),
        "candidate_kl": round(cand_kl, 6),
        "candidate_brier": round(cand_brier, 6),
        "delta_acc": round(cand_acc - base_acc, 6),
        "delta_kl": round(cand_kl - base_kl, 6),
        "delta_brier": round(cand_brier - base_brier, 6),
    }


def render_td_ablation_md(
    *,
    baseline_oof: dict[str, Any],
    teacher_cal: dict[str, Any] | None = None,
    phase1_runs: list[dict[str, Any]] | None = None,
    snapshot_date: str = "2026-11-11",
) -> str:
    """Render `bench/td_ablation.md` summarizing Tasks 4, 5, 6, and the Task 7 RC gate."""
    phase1_runs = phase1_runs or []
    lines: list[str] = [
        f"# Phase 1 Typed Decisions Out-of-Fold Ablation Report (Snapshot {snapshot_date})",
        "",
        "All numbers below are 5-fold case-clustered out-of-fold (`train`, `N=1,200` cases / `6,000` decisions) metrics.",
        "`Typed Decisions test` (`N=400` cases / `2,000` decisions) was **not** touched during model selection.",
        "",
        "## 1. Baseline OOF (Task 3 — v1.3.0 Recipe)",
        "",
        "| Run / Family | Backbone | OOF Accuracy | OOF KL | OOF Brier | OOF ECE |",
        "|---|---|---:|---:|---:|---:|",
        (
            f"| **v1.3.0 Calibrated Ensemble (Baseline)** | ModernBERT-large-zs + Ettin-400M "
            f"| **{float(baseline_oof['oof_accuracy']):.4f}** "
            f"| **{float(baseline_oof['oof_kl']):.4f}** "
            f"| **{float(baseline_oof.get('oof_brier', 0.0)):.4f}** "
            f"| {float(baseline_oof.get('oof_ece', 0.0)):.4f} |"
        ),
    ]
    for fam, info in (baseline_oof.get("per_family_raw_oof") or {}).items():
        lines.append(
            f"| `{fam}` (raw 5-fold) | `{info.get('backbone_id', '')}` "
            f"| {float(info.get('oof_accuracy', 0.0)):.4f} "
            f"| {float(info.get('oof_kl', 0.0)):.4f} "
            f"| {float(info.get('oof_brier', 0.0)):.4f} "
            f"| {float(info.get('oof_ece', 0.0)):.4f} |"
        )

    if teacher_cal and teacher_cal.get("candidates"):
        lines.extend(
            [
                "",
                "## 2. Task 4 Teacher Agreement Gate (`mean_kl < 0.10` on Real `train`)",
                "",
                "| Teacher Candidate | Train 3-Sample Accuracy | Train Mean KL vs Gold | Gate (`KL < 0.10`) | Notes |",
                "|---|---:|---:|---|---|",
            ]
        )
        for model_id, row in teacher_cal["candidates"].items():
            acc_str = f"{float(row['accuracy']):.4f}" if row.get("accuracy") is not None else "n/a"
            kl_str = f"{float(row['mean_kl']):.4f}" if row.get("mean_kl") is not None else "n/a"
            gate_str = "PASS" if row.get("accepted") else "STOP (`>= 0.10`)"
            note = str(row.get("note", ""))
            lines.append(f"| `{model_id}` | {acc_str} | {kl_str} | {gate_str} | {note} |")

    if phase1_runs:
        lines.extend(
            [
                "",
                "## 3. Tasks 5 & 6 Backbone, Masking, Holdout Sweep & Calibration Ablations",
                "",
                "| Run | Winner Backbone | OOF Accuracy | Δ Acc vs v1.3.0 | OOF KL | Δ KL vs v1.3.0 | OOF Brier | Task 7 RC Gate |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for run in phase1_runs:
            oof = run["oof"]
            gate = check_phase1_rc_gate(baseline_oof, oof)
            gate_badge = "**PASS**" if gate["passed"] else "FAIL"
            lines.append(
                f"| `{run['name']}` | `{oof.get('winner_family', 'ensemble')}` "
                f"| **{gate['candidate_acc']:.4f}** | {gate['delta_acc']:+.4f} "
                f"| **{gate['candidate_kl']:.4f}** | {gate['delta_kl']:+.4f} "
                f"| {gate['candidate_brier']:.4f} | {gate_badge} |"
            )

        for run in phase1_runs:
            oof = run["oof"]
            per_fam = oof.get("per_family_raw_oof") or {}
            if per_fam:
                lines.extend(
                    [
                        "",
                        f"### Per-Backbone Raw 5-Fold OOF (`{run['name']}`)",
                        "",
                        "| Candidate Spec | Hugging Face ID | Raw OOF Acc | Raw OOF KL | Raw OOF Brier | Raw OOF ECE |",
                        "|---|---|---:|---:|---:|---:|",
                    ]
                )
                for fam, info in per_fam.items():
                    lines.append(
                        f"| `{fam}` | `{info.get('backbone_id', '')}` "
                        f"| {float(info.get('oof_accuracy', 0.0)):.4f} "
                        f"| {float(info.get('oof_kl', 0.0)):.4f} "
                        f"| {float(info.get('oof_brier', 0.0)):.4f} "
                        f"| {float(info.get('oof_ece', 0.0)):.4f} |"
                    )

            sweep = oof.get("student_holdout_sweep") or []
            if sweep:
                lines.extend(
                    [
                        "",
                        f"### Task 6 Carved Real-Train Holdout Sweep (`{run['name']}`)",
                        "",
                        "| `beta_oof` | `alpha_wise` | Holdout Acc | Holdout KL | Holdout Brier |",
                        "|---:|---:|---:|---:|---:|",
                    ]
                )
                for s in sweep:
                    lines.append(
                        f"| {float(s['beta_oof']):.2f} | {float(s['alpha_wise']):.2f} "
                        f"| {float(s['holdout_acc']):.4f} | {float(s['holdout_kl']):.4f} | {float(s['holdout_brier']):.4f} |"
                    )

            temps = oof.get("temperatures")
            soup = oof.get("soup_selection")
            if temps or soup:
                lines.append("")
                if temps:
                    t_parts = ", ".join(f"`{k}={float(v):.4f}`" for k, v in temps.items())
                    lines.append(f"- **L-BFGS Per-Type Temperatures (`{run['name']}`)**: {t_parts}")
                if soup:
                    lines.append(
                        f"- **3-Seed Weight Soup Gate (`{run['name']}`)**: selected `{soup.get('selected')}` "
                        f"(`Δacc={float(soup.get('delta_acc', 0.0)):+.4f}`, `Δkl={float(soup.get('delta_kl', 0.0)):+.4f}`)"
                    )

        final_oof = phase1_runs[-1]["oof"]
        final_gate = check_phase1_rc_gate(baseline_oof, final_oof)
        lines.extend(
            [
                "",
                "## 4. Task 7 Release Candidate Gate Summary",
                "",
                f"- **Required OOF Accuracy**: `>= {final_gate['min_acc']:.4f}` (Observed: `{final_gate['candidate_acc']:.4f}` -> `{'PASS' if final_gate['acc_ok'] else 'FAIL'}`)",
                f"- **Required OOF KL**: `<= {final_gate['max_allowed_kl']:.4f}` (Observed: `{final_gate['candidate_kl']:.4f}` -> `{'PASS' if final_gate['kl_ok'] else 'FAIL'}`)",
                f"- **Overall Task 7 Gate**: **{'PASS — Eligible for `slot-1` on Typed Decisions `test`' if final_gate['passed'] else 'FAIL — Do NOT spend `slot-1`'}**",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Phase 1 Typed Decisions OOF ablation report.")
    ap.add_argument("--baseline", required=True, help="Path to Task 3 baseline oof_report.json")
    ap.add_argument("--teacher-cal", default="", help="Optional path to data/td_synth/teacher_calibration.json")
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="Named Phase 1 OOF JSON in the form name=path/to/oof_report.json (repeatable)",
    )
    ap.add_argument("--snapshot-date", default="2026-11-11")
    ap.add_argument("--out", default="bench/td_ablation.md")
    args = ap.parse_args()

    baseline_oof = json.loads(Path(args.baseline).read_text())
    teacher_cal = json.loads(Path(args.teacher_cal).read_text()) if args.teacher_cal and Path(args.teacher_cal).exists() else None
    runs: list[dict[str, Any]] = []
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run expects name=path.json, got: {item}")
        name, path_str = item.split("=", 1)
        runs.append({"name": name, "oof": json.loads(Path(path_str).read_text())})

    md = render_td_ablation_md(
        baseline_oof=baseline_oof,
        teacher_cal=teacher_cal,
        phase1_runs=runs,
        snapshot_date=args.snapshot_date,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Wrote {out_path} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
