#!/usr/bin/env python3
"""Dated leaderboard standing and rank calculator for nullspan."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TYPED_DECISIONS_METRICS: list[tuple[str, bool]] = [
    ("accuracy", True),
    ("kl", False),
    ("brier", False),
    ("ece", False),
]

JEVBENCH_METRICS: list[tuple[str, bool]] = [
    ("capability", True),
    ("composite", True),
    ("intelligence", True),
    ("calibration", True),
    ("speed", True),
    ("cost_axis", True),
]

KNOWN_TYPED_DECISIONS_PARAMS_M: dict[str, float | None] = {
    "OpenDecider-small": 4000.0,
    "Bongard-mini": 4000.0,
    "Jeff-Gemma4-E2B": 2000.0,
    "Jeff-Qwen3.5-2B": 2000.0,
    "Jeff-Qwen3.5-0.8B": 800.0,
    "Prior": None,
    "Uniform": None,
    "Arbiter-4B": 4000.0,
    "OpenDecider-large-td": 80000.0,
    "OpenDecider-medium-td": 30000.0,
    "OpenDecider-small-td": 4000.0,
    "OpenDecider-nano": 400.0,
    "Phocinae-Largha-150M-v1": 144.3,
    "soft-decider-421m": 421.0,
    "Laya typed-decisions": 421.3,
    "SAP RPT 1.6": None,
    "Bekko System One v0 400M": 400.0,
    "Bekko System One v0 68M": 68.0,
    "Bekko System One v0 17M": 17.0,
}


def _clean_td_model_name(cell: str) -> str:
    stripped = cell.replace("**", "").strip()
    if stripped.startswith("SAP RPT 1.6"):
        return "SAP RPT 1.6"
    # Match a leading markdown link [Name](url) only when it starts the cell and isn't [#NN]
    lead_link = re.match(r"^\[([^#\]]\S*?[^\]]*)\]\(https?://[^)]+\)", stripped)
    if lead_link:
        return lead_link.group(1).strip()
    return re.split(r"\s*\(|\s*\[#|\s*,|\s+via\s+", stripped)[0].strip()


def _parse_td_params_m(name: str, cell: str) -> float | None:
    if name in KNOWN_TYPED_DECISIONS_PARAMS_M:
        return KNOWN_TYPED_DECISIONS_PARAMS_M[name]
    for group in re.findall(r"\(([^)]+)\)", cell):
        m_m = re.search(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*M\b", group)
        m_b = re.search(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*B\b", group)
        if m_m and not m_b:
            return float(m_m.group(1))
        if m_b:
            return float(m_b.group(1)) * 1000.0
    m_b = re.search(r"(?:^|[-_\s])(\d+(?:\.\d+)?)[bB]\b", name)
    if m_b:
        return float(m_b.group(1)) * 1000.0
    m_m = re.search(r"(?:^|[-_\s])(\d+(?:\.\d+)?)[mM]\b", name)
    if m_m:
        return float(m_m.group(1))
    return None


def _parse_td_metric(raw: str) -> float | None:
    cleaned = raw.replace("*", "").strip()
    if cleaned in ("–", "-", "", "n/a", "N/A"):
        return None
    return float(cleaned)


def parse_typed_decisions_card(markdown_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mode: str | None = None
    for line in markdown_text.splitlines():
        if line.startswith("## Leaderboard"):
            mode = "zero_shot"
            continue
        if line.startswith("### Fitted or fine-tuned on `train`"):
            mode = "fitted"
            continue
        if line.startswith("### Benchmarking") or line.startswith("## Reading"):
            mode = None
            continue
        if mode and line.startswith("|"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 7 or parts[0] in ("#", "---") or set(parts[0]) == {"-"}:
                continue
            model_cell = parts[1]
            name = _clean_td_model_name(model_cell)
            rows.append(
                {
                    "model": name,
                    "params_m": _parse_td_params_m(name, model_cell),
                    "mode": mode,
                    "accuracy": _parse_td_metric(parts[3]),
                    "kl": _parse_td_metric(parts[4]),
                    "brier": _parse_td_metric(parts[5]),
                    "ece": _parse_td_metric(parts[6]),
                }
            )
    return rows


def load(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list of rows in {path}")
    return data


def rank(
    rows: list[dict[str, Any]],
    metric: str,
    higher_is_better: bool,
    max_params_m: float | None = None,
) -> list[tuple[int, str, float]]:
    eligible: list[tuple[str, float]] = []
    for row in rows:
        val = row.get(metric)
        if val is None:
            continue
        if max_params_m is not None:
            params_m = row.get("params_m")
            if params_m is None or float(params_m) > max_params_m:
                continue
        eligible.append((str(row["model"]), float(val)))

    eligible.sort(key=lambda item: (-item[1] if higher_is_better else item[1]))

    ranked: list[tuple[int, str, float]] = []
    prev_val: float | None = None
    current_rank = 0
    for idx, (model_name, val) in enumerate(eligible, start=1):
        if prev_val is None or abs(val - prev_val) > 1e-12:
            current_rank = idx
            prev_val = val
        ranked.append((current_rank, model_name, val))
    return ranked


def _extract_snapshot_date(snapshot_path: str | Path) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", Path(snapshot_path).name)
    return match.group(1) if match else "undated"


def _match_entry(
    ranked: list[tuple[int, str, float]], model: str
) -> tuple[int, str, float] | None:
    needle = model.lower()
    for entry in ranked:
        if entry[1].lower() == needle:
            return entry
    for entry in ranked:
        if needle in entry[1].lower():
            return entry
    return None


def _format_gap(value: float, leader_value: float, higher_is_better: bool) -> str:
    delta = (value - leader_value) if higher_is_better else (leader_value - value)
    if abs(delta) <= 1e-12:
        return "0.0000 (leader)"
    return f"{delta:+.4f} vs leader"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Path to dated leaderboard JSON")
    parser.add_argument("--model", default="nullspan", help="Model name to look up")
    parser.add_argument(
        "--mode",
        default=None,
        help="Optional mode filter for Typed Decisions ('fitted' or 'zero_shot')",
    )
    parser.add_argument(
        "--max-params-m",
        type=float,
        default=1000.0,
        help="Category parameter cap in millions (default: 1000.0 for <=1B)",
    )
    args = parser.parse_args(argv)

    rows = load(args.snapshot)
    snap_date = _extract_snapshot_date(args.snapshot)

    is_jevbench = bool(rows and "capability" in rows[0] and "board" in rows[0])
    if is_jevbench:
        scoped_rows = [
            r for r in rows if r.get("board") == "open" and r.get("capability_eligible", True)
        ]
        metrics = JEVBENCH_METRICS
        scope_label = "JevBench open-weights capability-eligible"
    else:
        target_mode = args.mode
        if target_mode is None:
            matched_modes = [
                r.get("mode")
                for r in rows
                if r.get("mode") and args.model.lower() in str(r.get("model", "")).lower()
            ]
            target_mode = matched_modes[0] if matched_modes else "fitted"
        scoped_rows = [r for r in rows if r.get("mode", target_mode) == target_mode]
        metrics = TYPED_DECISIONS_METRICS
        scope_label = f"Typed Decisions ({target_mode})"

    print(f"Snapshot: {Path(args.snapshot).name} (date: {snap_date}) | Scope: {scope_label}")
    for metric, higher_is_better in metrics:
        sub_ranked = rank(scoped_rows, metric, higher_is_better, max_params_m=args.max_params_m)
        all_ranked = rank(scoped_rows, metric, higher_is_better, max_params_m=None)
        if not all_ranked:
            continue

        sub_entry = _match_entry(sub_ranked, args.model)
        all_entry = _match_entry(all_ranked, args.model)
        sub_leader = sub_ranked[0] if sub_ranked else None
        all_leader = all_ranked[0]

        if all_entry is None:
            sub_str = (
                f"sub-1B leader: {sub_leader[1]} ({sub_leader[2]:.4f}, n={len(sub_ranked)})"
                if sub_leader
                else "sub-1B: none"
            )
            print(
                f"  {metric:12s}: unlisted | {sub_str} | overall leader: {all_leader[1]} ({all_leader[2]:.4f}, n={len(all_ranked)})"
            )
            continue

        val = all_entry[2]
        if sub_entry is not None and sub_leader is not None:
            sub_part = (
                f"<=1B #{sub_entry[0]}/{len(sub_ranked)} "
                f"({_format_gap(val, sub_leader[2], higher_is_better)}; leader {sub_leader[1]} {sub_leader[2]:.4f})"
            )
        else:
            sub_part = "<=1B n/a"

        all_part = (
            f"overall #{all_entry[0]}/{len(all_ranked)} "
            f"({_format_gap(val, all_leader[2], higher_is_better)}; leader {all_leader[1]} {all_leader[2]:.4f})"
        )
        print(f"  {metric:12s}: {val:.4f} | {sub_part} | {all_part}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
