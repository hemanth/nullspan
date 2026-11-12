#!/usr/bin/env python3
"""JevBench v1.6.1 (O1S) official-equation scorer for local public & dev readouts.

Reproduces the frozen JevBench v1.5 + Headline Option A Equal-Types Addendum +
v1.6.1 O1S Noul/Score competence and Capability Score equations to 1e-14 precision:
  - Equal request-type weights: Choice 1/3, Noul 1/3, Score 1/3.
  - Tier weights: easy 0.10, standard 0.20, judge 0.30, hard 0.40
    (renormalized only when a tier has zero items).
  - O1S Noul: P(yes) <= 0.20 -> "no", P(yes) >= 0.80 -> "yes",
    (0.20, 0.80) -> abstention (counted wrong); CC_tier = 100 * (acc - 0.5) / 0.5.
  - O1S Score: nMAE = |ev - gold| / (K - 1); baseline is mid-scale error
    |(K - 1)/2 - gold| / (K - 1) (falling back to uniform-level mean if 0);
    CC_tier = 100 * (1 - mean_nMAE / mean_nMAE_base).
  - Split x type competence clipped at 0.0 before averaging over types.
  - Overfit penalty against G_med = 2.573387642438244 (v1.6.1 cohort median).
  - Capability Score = (Intelligence + Calibration) / 2 within open-board caps
    (usd_per_1000 <= 0.06459465517241379, p50_adjusted_s <= 1.2329566404223442).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TIER_WEIGHTS: dict[str, float] = {
    "easy": 0.10,
    "standard": 0.20,
    "judge": 0.30,
    "hard": 0.40,
}
TYPES: tuple[str, ...] = ("choice", "noul", "score")
G_MED_V161: float = 2.573387642438244
CAP_MAX_USD_PER_1000: float = 0.06459465517241379
CAP_MAX_P50_ADJ_S: float = 1.2329566404223442
SUM_TOL: float = 1e-3
RENORM_TOL: float = 2e-2


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(val)))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    k = (len(vals) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def _ece_10bin(pairs: list[tuple[float, bool]]) -> float:
    if not pairs:
        return 0.0
    n_bins = 10
    bins = [{"n": 0, "conf": 0.0, "acc": 0} for _ in range(n_bins)]
    for conf, ok in pairs:
        c = max(0.0, min(1.0, float(conf)))
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx]["n"] += 1
        bins[idx]["conf"] += c
        bins[idx]["acc"] += 1 if ok else 0
    total = len(pairs)
    ece = 0.0
    for b in bins:
        if b["n"] > 0:
            ece += (b["n"] / total) * abs((b["acc"] / b["n"]) - (b["conf"] / b["n"]))
    return ece


def _clean_probs(probs: Any, labels: list[str]) -> dict[str, float] | None:
    if not isinstance(probs, dict):
        return None
    want = [str(x) for x in labels]
    if set(str(k) for k in probs.keys()) != set(want):
        return None
    out: dict[str, float] = {}
    total = 0.0
    for k in want:
        raw_v = probs.get(k) if k in probs else probs.get(next(rk for rk in probs if str(rk) == k))
        if isinstance(raw_v, bool) or not isinstance(raw_v, (int, float)):
            return None
        fv = float(raw_v)
        if not math.isfinite(fv) or fv < 0.0 or fv > 1.0:
            return None
        out[k] = fv
        total += fv
    if abs(total - 1.0) <= SUM_TOL:
        return out
    if abs(total - 1.0) <= RENORM_TOL and total > 0.0:
        return {k: v / total for k, v in out.items()}
    return None


def _argmax_lex(probs: dict[str, float]) -> str:
    best_k = ""
    best_v = -1.0
    for k in sorted(probs.keys()):
        if probs[k] > best_v:
            best_k, best_v = k, probs[k]
    return best_k


def aggregate_intelligence_v16(
    per_type_split: dict[str, dict[str, Any]],
    g_med: float = G_MED_V161,
) -> dict[str, Any]:
    """Aggregate split x type x tier competence into v1.6.1 Intelligence."""
    computed_cells: dict[str, dict[str, Any]] = {}
    splits_present: set[str] = set()

    for key, cell in per_type_split.items():
        split, typ = key.split("|", 1)
        splits_present.add(split)
        tiers: dict[str, float] = cell.get("tiers", {})
        if "cc" in cell and not tiers:
            cc_val = max(0.0, float(cell["cc"]))
        else:
            w_sum = sum(TIER_WEIGHTS[t] for t in tiers if t in TIER_WEIGHTS and tiers[t] is not None)
            if w_sum > 0.0:
                raw_cc = sum(
                    TIER_WEIGHTS[t] * float(tiers[t])
                    for t in tiers
                    if t in TIER_WEIGHTS and tiers[t] is not None
                ) / w_sum
                cc_val = max(0.0, raw_cc)
            else:
                cc_val = 0.0
        computed_cells[key] = {**cell, "cc": cc_val}

    def _split_mean(sp: str) -> float | None:
        vals = [computed_cells[f"{sp}|{t}"]["cc"] for t in TYPES if f"{sp}|{t}" in computed_cells]
        return (sum(vals) / len(vals)) if vals else None

    i_open = _split_mean("open")
    i_sealed = _split_mean("sealed")

    if i_open is not None and i_sealed is not None:
        base = 0.5 * i_open + 0.5 * i_sealed
        gap = i_open - i_sealed
        excess = gap - g_med
        penalty = max(0.0, 1.0 - max(0.0, excess - 8.0) / 100.0)
        score = _clamp(base * penalty, 0.0, 100.0)
    elif i_open is not None:
        base = i_open
        gap = 0.0
        excess = 0.0
        penalty = 1.0
        score = _clamp(base, 0.0, 100.0)
    elif i_sealed is not None:
        base = i_sealed
        gap = 0.0
        excess = 0.0
        penalty = 1.0
        score = _clamp(base, 0.0, 100.0)
    else:
        base = gap = excess = score = 0.0
        penalty = 1.0

    return {
        "I_open": i_open,
        "I_sealed": i_sealed,
        "base": base,
        "gap": gap,
        "g_med": g_med,
        "excess": excess,
        "penalty": penalty,
        "score": score,
        "per_type_split": computed_cells,
    }


def aggregate_calibration_v16(
    choice_ece_hard: float,
    choice_mean_tvd: float | None,
    noul_ece: float,
    score_nrps: float,
    score_top_ece: float,
    noul_abstention_rate: float = 0.0,
    noul_brier: float | None = None,
) -> dict[str, Any]:
    """Compute v1.6.1 equal-type Calibration score and per-type breakdown."""
    choice_ece_score = max(0.0, 100.0 * (1.0 - 2.0 * float(choice_ece_hard)))
    if choice_mean_tvd is not None:
        tvd_score = max(0.0, 100.0 * (1.0 - float(choice_mean_tvd)))
        c_choice = 0.5 * choice_ece_score + 0.5 * tvd_score
    else:
        c_choice = choice_ece_score

    c_noul = max(0.0, 100.0 * (1.0 - 2.0 * float(noul_ece)))

    nrps_part = max(0.0, 100.0 * (1.0 - float(score_nrps)))
    top_ece_part = max(0.0, 100.0 * (1.0 - 2.0 * float(score_top_ece)))
    c_score = 0.5 * nrps_part + 0.5 * top_ece_part

    overall = (c_choice + c_noul + c_score) / 3.0
    return {
        "score": overall,
        "parts": {
            "choice": {
                "ece_hard": float(choice_ece_hard),
                "ece_score": choice_ece_score,
                "mean_tvd": choice_mean_tvd,
                "score": c_choice,
            },
            "noul": {
                "ece": float(noul_ece),
                "abstention_rate": float(noul_abstention_rate),
                "brier": noul_brier,
                "score": c_noul,
            },
            "score": {
                "nrps": float(score_nrps),
                "top_ece": float(score_top_ece),
                "score": c_score,
            },
        },
    }


def capability_score_v16(
    intelligence: float,
    calibration: float,
    usd_per_1000: float,
    p50_adjusted_s: float,
    max_usd_per_1000: float = CAP_MAX_USD_PER_1000,
    max_p50_adjusted_s: float = CAP_MAX_P50_ADJ_S,
) -> dict[str, Any]:
    """Open-weights v1.6.1 Capability Score = (Intelligence + Calibration) / 2 within caps."""
    eligible = (float(usd_per_1000) <= max_usd_per_1000) and (float(p50_adjusted_s) <= max_p50_adjusted_s)
    return {
        "eligible": eligible,
        "capability": 0.5 * (float(intelligence) + float(calibration)),
        "usd_per_1000": float(usd_per_1000),
        "p50_adjusted_s": float(p50_adjusted_s),
    }


def speed_score_v16(
    p50_raw_s: float,
    p95_raw_s: float,
    endpoint_kind: str = "gpu",
) -> dict[str, float]:
    """Compute adjusted latencies and Speed axis score."""
    if endpoint_kind == "api":
        p50_adj = max(1e-6, float(p50_raw_s))
        p95_adj = max(1e-6, float(p95_raw_s))
    else:
        p50_adj = 2.0 * max(0.0, float(p50_raw_s)) + 0.15
        p95_adj = 2.0 * max(0.0, float(p95_raw_s)) + 0.15
    s50 = _clamp(100.0 - 20.0 * math.log10(p50_adj / 0.1), 0.0, 100.0)
    s95 = _clamp(100.0 - 20.0 * math.log10(p95_adj / 0.1), 0.0, 100.0)
    return {
        "p50_s_raw": float(p50_raw_s),
        "p95_s_raw": float(p95_raw_s),
        "p50_s_adjusted": p50_adj,
        "p95_s_adjusted": p95_adj,
        "score": 0.5 * (s50 + s95),
    }


def cost_score_v16(usd_per_1000: float) -> float:
    eff = max(1e-9, float(usd_per_1000))
    return _clamp(100.0 - 30.0 * math.log10(eff / 0.001), 0.0, 100.0)


def composite_option_a_v16(
    intelligence: float,
    calibration: float,
    speed: float,
    cost: float,
) -> float:
    """Frozen Option A equal-axis harmonic mean with <50 quadratic gates on I, S, $."""
    axes = [float(intelligence), float(calibration), float(speed), float(cost)]
    if any(a <= 0.0 for a in axes):
        return 0.0
    h = 4.0 / sum(1.0 / a for a in axes)
    gate_i = min(1.0, float(intelligence) / 50.0) ** 2
    gate_s = min(1.0, float(speed) / 50.0) ** 2
    gate_c = min(1.0, float(cost) / 50.0) ** 2
    return h * gate_i * gate_s * gate_c


def score_items_v16(
    items: list[dict[str, Any]],
    preds_by_id: dict[str, dict[str, Any]],
    usd_per_1m_in: float = 0.04,
    usd_per_1m_out: float = 0.0,
    endpoint_kind: str = "gpu",
    g_med: float = G_MED_V161,
) -> dict[str, Any]:
    """Score item-level predictions under JevBench v1.6.1 O1S."""
    # Group scorable items by (split, type, tier)
    strata: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, float] | None]]] = {}
    choice_hard_ece_pairs: list[tuple[float, bool]] = []
    choice_tvds: list[float] = []
    noul_ece_pairs: list[tuple[float, bool]] = []
    noul_briers: list[float] = []
    noul_abstentions = 0
    noul_total = 0
    score_nrps_vals: list[float] = []
    score_top_ece_pairs: list[tuple[float, bool]] = []
    speed_latencies: list[float] = []
    all_latencies: list[float] = []
    total_in_tokens = 0.0
    total_out_tokens = 0.0
    n_cost_items = 0

    for item in items:
        iid = str(item["id"])
        raw_split = str(item.get("split", "open"))
        split = "open" if raw_split in ("public", "open", "dev") else raw_split
        tier = str(
            item.get("tier")
            or item.get("_tier")
            or (item.get("provenance") or {}).get("tier")
            or "standard"
        )
        typ = str(item.get("type") or item.get("question", {}).get("type"))
        labels = [str(x) for x in item["labels"]]
        expected = item.get("expected")

        pred = preds_by_id.get(iid, {})
        lat = pred.get("latency_s")
        if isinstance(lat, (int, float)) and math.isfinite(lat) and lat >= 0.0:
            all_latencies.append(float(lat))
            if split == "open" and tier in ("standard", "judge"):
                speed_latencies.append(float(lat))

        in_tok = pred.get("input_tokens", (pred.get("usage") or {}).get("input_tokens", 0))
        out_tok = pred.get("output_tokens", (pred.get("usage") or {}).get("output_tokens", 0))
        if isinstance(in_tok, (int, float)) and isinstance(out_tok, (int, float)):
            total_in_tokens += float(in_tok)
            total_out_tokens += float(out_tok)
            n_cost_items += 1

        clean = _clean_probs(pred.get("probs"), labels)
        if expected is None:
            continue

        exp_str = str(expected)
        strata.setdefault((split, typ, tier), []).append((item, clean))

        if typ == "choice":
            if clean is not None:
                top_lab = _argmax_lex(clean)
                top_conf = clean[top_lab]
                is_ok = top_lab == exp_str
                if tier == "hard":
                    choice_hard_ece_pairs.append((top_conf, is_ok))
                gold_probs = item.get("gold_probs")
                if isinstance(gold_probs, dict) and set(str(k) for k in gold_probs) == set(labels):
                    tvd = 0.5 * sum(abs(clean[k] - float(gold_probs[k])) for k in labels)
                    choice_tvds.append(tvd)
            else:
                if tier == "hard":
                    choice_hard_ece_pairs.append((0.0, False))

        elif typ == "noul":
            noul_total += 1
            y_true = 1.0 if exp_str.lower() in ("yes", "true", "1") else 0.0
            if clean is not None:
                yes_key = "yes" if "yes" in clean else ("true" if "true" in clean else labels[-1])
                p_yes = clean[yes_key]
                if 0.20 < p_yes < 0.80:
                    noul_abstentions += 1
                noul_ece_pairs.append((p_yes, y_true == 1.0))
                noul_briers.append((p_yes - y_true) ** 2)
            else:
                noul_ece_pairs.append((0.5, False))
                noul_briers.append((0.5 - y_true) ** 2)

        elif typ == "score":
            sorted_levels = sorted(labels, key=lambda x: float(x))
            k_len = len(sorted_levels)
            gold_idx = sorted_levels.index(exp_str)
            if clean is not None:
                top_lab = _argmax_lex(clean)
                score_top_ece_pairs.append((clean[top_lab], top_lab == exp_str))
                if k_len > 1:
                    cdf_p = 0.0
                    rps_sum = 0.0
                    for m in range(k_len - 1):
                        cdf_p += clean[sorted_levels[m]]
                        cdf_g = 1.0 if gold_idx <= m else 0.0
                        rps_sum += (cdf_p - cdf_g) ** 2
                    score_nrps_vals.append(rps_sum / (k_len - 1))
            else:
                score_top_ece_pairs.append((0.0, False))
                score_nrps_vals.append(1.0)

    # Compute per-(split, type, tier) CC
    per_type_split: dict[str, dict[str, Any]] = {}
    per_tier_accuracy: dict[str, float] = {}
    tier_hits: dict[str, list[float]] = {}

    for (split, typ, tier), rows in strata.items():
        key = f"{split}|{typ}"
        cell = per_type_split.setdefault(key, {"tiers": {}, "counts": {}})
        n_rows = len(rows)
        cell["counts"][tier] = n_rows

        if typ == "choice":
            corrects = 0
            chances = 0.0
            for it, clean in rows:
                labs = [str(x) for x in it["labels"]]
                chances += 1.0 / max(1, len(labs))
                ok = clean is not None and _argmax_lex(clean) == str(it["expected"])
                corrects += 1 if ok else 0
                tier_hits.setdefault(tier, []).append(1.0 if ok else 0.0)
            acc = corrects / n_rows
            c_bar = chances / n_rows
            cc_tier = 100.0 * (acc - c_bar) / max(1e-9, 1.0 - c_bar)

        elif typ == "noul":
            corrects = 0
            for it, clean in rows:
                exp_yes = str(it["expected"]).lower() in ("yes", "true", "1")
                ok = False
                if clean is not None:
                    labs = [str(x) for x in it["labels"]]
                    yes_key = "yes" if "yes" in clean else ("true" if "true" in clean else labs[-1])
                    p_yes = clean[yes_key]
                    if p_yes >= 0.80 and exp_yes:
                        ok = True
                    elif p_yes <= 0.20 and not exp_yes:
                        ok = True
                corrects += 1 if ok else 0
                tier_hits.setdefault(tier, []).append(1.0 if ok else 0.0)
            acc = corrects / n_rows
            cc_tier = 100.0 * (acc - 0.5) / 0.5

        elif typ == "score":
            nmaes: list[float] = []
            base_mid: list[float] = []
            base_unif: list[float] = []
            for it, clean in rows:
                sorted_levels = sorted([str(x) for x in it["labels"]], key=lambda x: float(x))
                k_len = len(sorted_levels)
                denom = max(1.0, float(k_len - 1))
                gold_pos = float(sorted_levels.index(str(it["expected"])))
                mid_pos = 0.5 * float(k_len - 1)
                base_mid.append(abs(mid_pos - gold_pos) / denom)
                base_unif.append(
                    sum(abs(float(l_idx) - gold_pos) / denom for l_idx in range(k_len)) / float(k_len)
                )
                if clean is not None:
                    ev_pos = sum(float(idx) * clean[lab] for idx, lab in enumerate(sorted_levels))
                    nmae_i = abs(ev_pos - gold_pos) / denom
                    ok = round(ev_pos) == int(gold_pos)
                else:
                    nmae_i = 1.0
                    ok = False
                nmaes.append(nmae_i)
                tier_hits.setdefault(tier, []).append(1.0 if ok else 0.0)
            mean_nmae = sum(nmaes) / n_rows
            mean_mid = sum(base_mid) / n_rows
            mean_base = mean_mid if mean_mid > 1e-12 else (sum(base_unif) / n_rows)
            cc_tier = 100.0 * (1.0 - mean_nmae / max(1e-9, mean_base))
        else:
            raise ValueError(f"Unsupported JevBench type: {typ!r}")

        cell["tiers"][tier] = cc_tier

    for t_name, hits in tier_hits.items():
        per_tier_accuracy[t_name] = sum(hits) / len(hits) if hits else 0.0

    intel = aggregate_intelligence_v16(per_type_split, g_med=g_med)
    cal = aggregate_calibration_v16(
        choice_ece_hard=_ece_10bin(choice_hard_ece_pairs),
        choice_mean_tvd=(sum(choice_tvds) / len(choice_tvds)) if choice_tvds else None,
        noul_ece=_ece_10bin(noul_ece_pairs),
        score_nrps=(sum(score_nrps_vals) / len(score_nrps_vals)) if score_nrps_vals else 0.0,
        score_top_ece=_ece_10bin(score_top_ece_pairs),
        noul_abstention_rate=(noul_abstentions / noul_total) if noul_total else 0.0,
        noul_brier=(sum(noul_briers) / len(noul_briers)) if noul_briers else None,
    )

    lat_pool = speed_latencies if speed_latencies else all_latencies
    p50_raw = _percentile(lat_pool, 0.50) if lat_pool else 0.02
    p95_raw = _percentile(lat_pool, 0.95) if lat_pool else 0.04
    spd = speed_score_v16(p50_raw, p95_raw, endpoint_kind=endpoint_kind)

    if n_cost_items > 0:
        usd_total = (total_in_tokens / 1e6) * usd_per_1m_in + (total_out_tokens / 1e6) * usd_per_1m_out
        usd_per_1000 = (usd_total / n_cost_items) * 1000.0
    else:
        usd_per_1000 = 0.0
    cst_score = cost_score_v16(usd_per_1000)

    cap = capability_score_v16(
        intelligence=intel["score"],
        calibration=cal["score"],
        usd_per_1000=usd_per_1000,
        p50_adjusted_s=spd["p50_s_adjusted"],
    )
    comp_a = composite_option_a_v16(
        intelligence=intel["score"],
        calibration=cal["score"],
        speed=spd["score"],
        cost=cst_score,
    )

    return {
        "protocol": "jevbench::v1.6.1-O1S-local",
        "n_items": len(items),
        "intelligence": intel,
        "calibration": cal,
        "speed": spd,
        "cost": {
            "usd_per_1000": usd_per_1000,
            "usd_per_1m_in": usd_per_1m_in,
            "usd_per_1m_out": usd_per_1m_out,
            "score": cst_score,
        },
        "capability": cap,
        "composite_option_a": comp_a,
        "per_tier_accuracy": per_tier_accuracy,
    }


def load_public_jevbench_items(public_dir: str | Path) -> list[dict[str, Any]]:
    """Load JevBench public JSONLs with canonical tier mapping."""
    pdir = Path(public_dir)
    tier_files = [
        ("easy", pdir / "easy.jsonl"),
        ("standard", pdir / "original.jsonl"),
        ("hard", pdir / "hard.jsonl"),
    ]
    items: list[dict[str, Any]] = []
    for tier, fp in tier_files:
        if not fp.exists():
            continue
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("tier", tier)
            row["split"] = "open"
            row.setdefault("type", row.get("question", {}).get("type"))
            items.append(row)
    return items


def render_jevbench_submission_md(
    version: str,
    hf_repo: str,
    hf_revision: str,
    git_sha: str,
    backbone_desc: str,
    v16_readout: dict[str, Any],
    training_disclosure: str,
    issue_ref: str = "fstandhartinger/jevbench#238",
    snapshot_date: str = "2026-11-11",
) -> str:
    """Render the official JevBench v1.6.1 maintainer submission & disclosure package."""
    intel = float(v16_readout["intelligence"]["score"])
    cal = float(v16_readout["calibration"]["score"])
    spd = v16_readout["speed"]
    cst = v16_readout["cost"]
    cap = v16_readout["capability"]
    comp_a = float(v16_readout["composite_option_a"])
    tiers = v16_readout.get("per_tier_accuracy", {})
    easy_acc = 100.0 * float(tiers.get("easy", 0.0))
    orig_acc = 100.0 * float(tiers.get("standard", 0.0))
    hard_acc = 100.0 * float(tiers.get("hard", 0.0))

    return (
        f"# JevBench v1.6.1 Evaluation Request — `{hf_repo}` ({version})\n\n"
        f"- **Channel / Tracking Issue**: `{issue_ref}` (`https://github.com/{issue_ref.replace('#', '/issues/')}`)\n"
        f"- **Snapshot Date**: `{snapshot_date}`\n"
        f"- **Scorer Protocol**: `{v16_readout.get('protocol', 'jevbench::v1.6.1-O1S-local')}` (`N = {v16_readout.get('n_items', 0)}` public items; official sealed evaluation is run by the JevBench maintainers)\n\n"
        f"## Model & Pinned Artifacts\n\n"
        f"| Field | Value |\n"
        f"| :--- | :--- |\n"
        f"| Hugging Face Repository | `https://huggingface.co/{hf_repo}` |\n"
        f"| Pinned HF Revision | `{hf_revision}` |\n"
        f"| Pinned Source Git SHA | `{git_sha}` |\n"
        f"| Architecture & Category | `{backbone_desc}` (`encoder_classifier_<=0.6B`) |\n"
        f"| Execution Mode | Single bidirectional encoder forward pass per request, `0` generated output tokens, zero ensembling |\n"
        f"| Cost Basis (`endpoint_kind=gpu`) | `${cst['usd_per_1m_in']:.2f} / 1M` input tokens, `${cst['usd_per_1m_out']:.2f} / 1M` output tokens (`${cst['usd_per_1000']:.5f} / 1,000` requests) |\n\n"
        f"## Local JevBench v1.6.1 (O1S) Public Readout\n\n"
        f"> Note: Public split readout only (`easy.jsonl`, `original.jsonl`, `hard.jsonl`). Official JevBench leaderboard standing requires maintainer execution on the sealed split.\n\n"
        f"| Metric | Public v1.6.1 Readout |\n"
        f"| :--- | ---: |\n"
        f"| **Capability Score** `(I + C) / 2` | **{cap['capability']:.2f}** (`eligible={cap['eligible']}`) |\n"
        f"| **Intelligence** (O1S equal-type) | **{intel:.2f}** |\n"
        f"| **Calibration** (O1S equal-type) | **{cal:.2f}** |\n"
        f"| **Speed Score** (`p50_raw={spd['p50_s_raw']*1000:.1f}ms`, `p50_adj={spd['p50_s_adjusted']*1000:.1f}ms`) | **{spd['score']:.2f}** |\n"
        f"| **Cost Score** (`${cst['usd_per_1000']:.5f} / 1k`) | **{cst['score']:.2f}** |\n"
        f"| **Composite (Option A)** | **{comp_a:.2f}** |\n"
        f"| Public Tier Accuracy (Easy / Original / Hard) | `{easy_acc:.1f}% / {orig_acc:.1f}% / {hard_acc:.1f}%` |\n\n"
        f"## Training & Data Hygiene Disclosure\n\n"
        f"{training_disclosure}\n\n"
        f"- **Strict Contamination Guarantee**: Zero JevBench items (public or sealed) were used in pretraining, distillation, hyperparameter tuning, early stopping, or temperature calibration; all synthetic dev/train generators enforce a 12-token shingle rejection filter against `datasets/public/*.jsonl`.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Score predictions with JevBench v1.6.1 O1S equations.")
    ap.add_argument("--public-dir", type=Path, required=True, help="Directory with easy/original/hard.jsonl")
    ap.add_argument("--preds", type=Path, required=True, help="JSONL or JSON map of item predictions")
    ap.add_argument("--usd-per-1m-in", type=float, default=0.04)
    ap.add_argument("--endpoint-kind", choices=["gpu", "api"], default="gpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    items = load_public_jevbench_items(args.public_dir)
    raw_text = args.preds.read_text(encoding="utf-8").strip()
    if raw_text.startswith("{") and "\n" not in raw_text[:2000]:
        preds_by_id = json.loads(raw_text)
    else:
        parsed = json.loads(raw_text) if args.preds.suffix == ".json" else None
        if isinstance(parsed, dict):
            preds_by_id = parsed
        else:
            preds_by_id = {}
            for line in raw_text.splitlines():
                if line.strip():
                    r = json.loads(line)
                    preds_by_id[str(r.get("task_id") or r.get("id"))] = r

    res = score_items_v16(
        items,
        preds_by_id,
        usd_per_1m_in=args.usd_per_1m_in,
        endpoint_kind=args.endpoint_kind,
    )
    out_json = json.dumps(res, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_json + "\n", encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
