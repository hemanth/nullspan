#!/usr/bin/env python3
"""System One Training & 5-Fold Out-of-Fold Cross-Validation Pipeline (Zero Test Access).

Trains `NullspanSystemOneEngine` strictly on public `train` splits and `LocalLLaMA/typed-decisions` `train`,
writing `systemone/`, `systemone_decision.pt`, and `oof_report.json`.
Zero held-out test split access.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "training") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "training"))

from systemone_decision import (  # noqa: E402
    ParallelSpanDecisionHead,
    canonical_options,
    configure_modernbert_full_bidir,
    encode_case_parallel,
    forward_parallel_backbone,
)

CANDIDATE_BACKBONES: list[tuple[str, str]] = [
    ("modernbert_zs", "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"),
    ("ettin_400m", "jhu-clsp/ettin-encoder-400m"),
]

PHASE1_CANDIDATE_SPECS: list[dict[str, Any]] = [
    {"key": "modernbert_zs", "backbone_id": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "seed_offset": 0},
    {"key": "modernbert_base", "backbone_id": "answerdotai/ModernBERT-large", "seed_offset": 34},
    {"key": "ettin_400m_s0", "backbone_id": "jhu-clsp/ettin-encoder-400m", "seed_offset": 17},
    {"key": "ettin_400m_s1", "backbone_id": "jhu-clsp/ettin-encoder-400m", "seed_offset": 51},
    {"key": "ettin_400m_s2", "backbone_id": "jhu-clsp/ettin-encoder-400m", "seed_offset": 68},
    {
        "key": "deberta_v3_large_512",
        "backbone_id": "microsoft/deberta-v3-large",
        "seed_offset": 85,
        "max_total_tokens": 512,
        "max_state_tokens": 384,
    },
]
MAX_STATE_TRAIN = 2560
MAX_STATE_INFER = 3072


def choose_max_state_train(td_error_report: dict[str, Any], default_max: int = 2560) -> int:
    """Task 5 Step 2: if < 2% of train states exceed 2,560 tokens, leave 2,560; else cover p99."""
    sl = td_error_report.get("state_length") or {}
    pct = float(sl.get("pct_cases_gt_2560", 0.0))
    if pct < 0.02:
        return default_max
    p99 = int(math.ceil(float(sl.get("p99", default_max))))
    rounded = ((max(default_max, p99) + 63) // 64) * 64
    return min(8192, rounded)


def resolve_candidate_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if cfg.get("candidate_specs"):
        return [dict(s) for s in cfg["candidate_specs"]]
    if cfg.get("use_phase1_specs"):
        return [dict(s) for s in PHASE1_CANDIDATE_SPECS]
    backbones = [tuple(x) for x in cfg.get("candidate_backbones", CANDIDATE_BACKBONES)]
    return [
        {"key": str(k), "backbone_id": str(b), "seed_offset": idx * 17}
        for idx, (k, b) in enumerate(backbones)
    ]


def _file_sha256_or_empty(path_str: str | None) -> str:
    if not path_str or not os.path.exists(path_str):
        return ""
    h = hashlib.sha256()
    with open(path_str, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_family_cache_key(
    spec: dict[str, Any],
    *,
    max_state_train: int = MAX_STATE_TRAIN,
    header_attends_options: bool = False,
    seed_base: int = 42,
    fold_seed: int = 42,
    stage1_epochs: int = 2,
    fold_epochs: int = 6,
    synth_td_jsonl: str | None = None,
    extra_gen_jsonl: list[str] | None = None,
) -> str:
    spec_max_state = int(spec.get("max_state_tokens", max_state_train))
    spec_max_total = int(spec["max_total_tokens"]) if spec.get("max_total_tokens") is not None else None
    key_obj = {
        "backbone_id": str(spec["backbone_id"]),
        "seed_offset": int(spec.get("seed_offset", 0)),
        "seed_base": int(seed_base),
        "fold_seed": int(fold_seed),
        "max_state_tokens": spec_max_state,
        "max_total_tokens": spec_max_total,
        "header_attends_options": bool(header_attends_options),
        "stage1_epochs": int(stage1_epochs),
        "fold_epochs": int(fold_epochs),
        "synth_sha": _file_sha256_or_empty(synth_td_jsonl),
        "extra_gen_shas": [_file_sha256_or_empty(p) for p in (extra_gen_jsonl or [])],
    }
    digest = hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    safe_bb = str(spec["backbone_id"]).replace("/", "__")
    return f"{safe_bb}_s{int(spec.get('seed_offset', 0))}_{digest}"


def save_cached_family_result(cache_dir: str | Path, cache_key: str, payload: dict[str, Any]) -> Path:
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    final_p = cdir / f"{cache_key}.pt"
    tmp_p = cdir / f".{cache_key}.tmp.pt"
    torch.save(payload, tmp_p)
    os.replace(tmp_p, final_p)
    return final_p


def load_cached_family_result(cache_dir: str | Path | None, cache_key: str) -> dict[str, Any] | None:
    if not cache_dir:
        return None
    fp = Path(cache_dir) / f"{cache_key}.pt"
    if not fp.exists():
        return None
    return torch.load(fp, map_location="cpu", weights_only=False)


def select_best_holdout_grid(grid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not grid_rows:
        raise ValueError("grid_rows must be non-empty")
    return min(
        grid_rows,
        key=lambda r: (
            float(r["holdout_kl"])
            + 0.35 * float(r["holdout_brier"])
            - 0.18 * float(r["holdout_acc"])
        ),
    )


def normalize_extra_gen_item(item: dict[str, Any]) -> tuple[Any, str, dict[str, Any], dict[str, float]]:
    """Normalize either legacy `(state, question, probabilities)` or JevBench task JSONL into a Stage 1 tuple."""
    state = item["state"]
    qid = str(item.get("qid") or "decision")
    q = dict(item["question"])
    qtype = str(q["type"])

    raw_probs = item.get("probabilities") or item.get("gold_probs") or item.get("soft_probs")
    if isinstance(raw_probs, dict) and raw_probs:
        if qtype == "noul":
            mapped = {
                ("true" if str(k).lower() in ("yes", "true", "1") else "false"): float(v)
                for k, v in raw_probs.items()
            }
            s = sum(mapped.values())
            probs = {k: v / s for k, v in mapped.items()} if s > 0 else {"false": 0.5, "true": 0.5}
        else:
            s = sum(float(v) for v in raw_probs.values())
            probs = {str(k): float(v) / s for k, v in raw_probs.items()} if s > 0 else {str(k): 1.0 / len(raw_probs) for k in raw_probs}
        return state, qid, q, probs

    labels = [str(x) for x in (item.get("labels") or [k for k, _ in canonical_options(q)])]
    exp = str(item.get("expected", labels[0]))
    if qtype == "noul":
        is_yes = exp.lower() in ("yes", "true", "1")
        probs = {"true": 0.92 if is_yes else 0.08, "false": 0.08 if is_yes else 0.92}
    elif qtype == "score":
        sorted_lvls = sorted(labels, key=lambda x: float(x))
        g_idx = sorted_lvls.index(exp) if exp in sorted_lvls else 0
        sigma = 0.55
        raw_s = {lv: math.exp(-((idx - g_idx) ** 2) / (2 * sigma * sigma)) for idx, lv in enumerate(sorted_lvls)}
        tot = sum(raw_s.values())
        probs = {k: v / tot for k, v in raw_s.items()}
    else:
        rem = 0.08 / max(1, len(labels) - 1)
        probs = {lab: (0.92 if lab == exp else rem) for lab in labels}
    return state, qid, q, probs


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


def _summarize_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": 0.0, "kl": 0.0, "brier": 0.0, "ece": 0.0}
    accs = [bool(r["correct"]) for r in rows]
    kls = [float(r["kl"]) for r in rows]
    brs = [float(r["brier"]) for r in rows]
    confs = [
        float(r.get("confidence", max(r["probs"].values()) if r.get("probs") else 0.0))
        for r in rows
    ]
    return {
        "n": len(rows),
        "accuracy": round(sum(accs) / len(accs), 6),
        "kl": round(sum(kls) / len(kls), 6),
        "brier": round(sum(brs) / len(brs), 6),
        "ece": round(compute_ece_10bin(accs, confs), 6),
    }


def build_oof_report(
    per_decision: list[dict[str, Any]],
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overall = _summarize_slice(per_decision)
    by_type_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_wf_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in per_decision:
        if "qtype" in r:
            by_type_rows[str(r["qtype"])].append(r)
        if "workflow" in r:
            by_wf_rows[str(r["workflow"])].append(r)

    report: dict[str, Any] = {
        "oof_accuracy": overall["accuracy"],
        "oof_kl": overall["kl"],
        "oof_brier": overall["brier"],
        "oof_ece": overall["ece"],
        "n_decisions": overall["n"],
        "per_type": {k: _summarize_slice(v) for k, v in sorted(by_type_rows.items())},
        "per_workflow": {k: _summarize_slice(v) for k, v in sorted(by_wf_rows.items())},
        "per_decision": per_decision,
    }
    if extra_meta:
        report.update(extra_meta)
    return report


class LatentFactorAuxHead(nn.Module):
    """Training-only multi-task head predicting discrete generator `factors` on `Typed Decisions train`."""

    def __init__(self, hidden_size: int, factor_vocabs: dict[str, list[str]]) -> None:
        super().__init__()
        self.factor_vocabs = factor_vocabs
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.heads = nn.ModuleDict(
            {fk: nn.Linear(hidden_size // 2, len(vals)) for fk, vals in sorted(factor_vocabs.items())}
        )

    def compute_loss(self, seq_out: torch.Tensor, packed_cases: list[dict[str, Any]]) -> torch.Tensor:
        device = seq_out.device
        losses: list[torch.Tensor] = []
        for c_idx, pc in enumerate(packed_cases):
            ftargs: dict[str, int] | None = pc.get("factor_targets")
            if not ftargs:
                continue
            cls_v = seq_out[c_idx, 0]
            ss, se = pc["state_span"]
            st_mean = seq_out[c_idx, ss : max(ss + 1, se)].mean(dim=0)
            h = self.proj(torch.cat([cls_v, st_mean], dim=-1))
            for fk, tgt_idx in ftargs.items():
                if fk in self.heads:
                    lg = self.heads[fk](h).unsqueeze(0)
                    tgt = torch.tensor([tgt_idx], dtype=torch.long, device=device)
                    losses.append(F.cross_entropy(lg, tgt))
        if not losses:
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(losses).mean()


def extract_factor_vocabs(ds_train: Any) -> dict[str, list[str]]:
    """Build discrete factor vocabularies strictly from `Typed Decisions` `train` split."""
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for r in ds_train:
        fc = json.loads(r["factors"]) if isinstance(r["factors"], str) else r["factors"]
        for k, v in fc.items():
            if v is None or isinstance(v, (list, dict, int, float)) and not isinstance(v, bool):
                continue
            sv = str(v)
            if len(sv) <= 48:
                counts[k][sv] += 1
    vocabs: dict[str, list[str]] = {}
    for k, cnt in sorted(counts.items()):
        if 2 <= len(cnt) <= 16 and sum(cnt.values()) >= 150:
            vocabs[k] = sorted(cnt.keys())
    return vocabs


def _pack_single_case(
    tokenizer: Any,
    state: Any,
    questions: dict[str, dict[str, Any]],
    gold_map: dict[str, dict[str, Any]],
    case_id: int,
    source: str = "gen",
    workflow: str = "general",
    factor_targets: dict[str, int] | None = None,
    max_state_tokens: int = MAX_STATE_TRAIN,
    max_total_tokens: int | None = None,
) -> dict[str, Any]:
    pc = encode_case_parallel(
        tokenizer,
        state,
        questions,
        max_state_tokens=max_state_tokens,
        case_id=case_id,
        max_total_tokens=max_total_tokens,
    )
    for qm in pc["questions"]:
        qid = qm["qid"]
        keys = qm["option_keys"]
        g_raw_probs = gold_map[qid]["probabilities"]
        gp = [float(g_raw_probs.get(k, 0.0)) for k in keys]
        s = sum(gp)
        gp = [v / s for v in gp] if s > 0 else [1.0 / len(keys)] * len(keys)
        qm["gold_probs"] = gp
        qm["train_probs"] = list(gp)
        qm["gold_raw"] = {"type": qm["qtype"], "probabilities": {k: gp[i] for i, k in enumerate(keys)}}
    pc["source"] = source
    pc["workflow"] = workflow
    pc["factor_targets"] = factor_targets
    return pc


def prepare_td_cases(
    rows: Any,
    tokenizer: Any,
    factor_vocabs: dict[str, list[str]],
    base_case_id: int = 100000,
    max_state_tokens: int = MAX_STATE_TRAIN,
    source: str = "td",
    max_total_tokens: int | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    v_lookup = {fk: {v: i for i, v in enumerate(vals)} for fk, vals in factor_vocabs.items()}
    for r_idx, row in enumerate(rows):
        state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
        qs = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        gs = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        raw_fc = row.get("factors", {}) if isinstance(row, dict) else row["factors"]
        fc = json.loads(raw_fc) if isinstance(raw_fc, str) else (raw_fc or {})
        ftargs: dict[str, int] = {}
        for fk, lmap in v_lookup.items():
            val = fc.get(fk)
            if val is not None and str(val) in lmap:
                ftargs[fk] = lmap[str(val)]
        pc = _pack_single_case(
            tokenizer,
            state,
            qs,
            gs,
            case_id=base_case_id + r_idx,
            source=source,
            workflow=str(row["workflow"]),
            factor_targets=ftargs,
            max_state_tokens=max_state_tokens,
            max_total_tokens=max_total_tokens,
        )
        pc["orig_train_idx"] = r_idx
        cases.append(pc)
    return cases


def build_fold_splits_with_synth(
    td_all_cases: list[dict[str, Any]],
    folds: list[Any],
    synth_cases: list[dict[str, Any]] | None = None,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Build (fold_train, fold_val) pairs where synthetic cases enter training ONLY."""
    extra = list(synth_cases or [])
    out: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    n_real = len(td_all_cases)
    for item in folds:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list) and (
            not item[0] or not item[1] or set(item[0]).isdisjoint(set(item[1]))
        ) and len(item[0]) + len(item[1]) == n_real:
            tr_idx, va_idx = item
        else:
            va_idx = list(item)
            va_set = set(va_idx)
            tr_idx = [i for i in range(n_real) if i not in va_set]
        fold_tr = [td_all_cases[i] for i in tr_idx] + extra
        fold_va = [td_all_cases[i] for i in va_idx]
        out.append((fold_tr, fold_va))
    return out


def generate_synthetic_systemone_rubrics(
    rng: random.Random, n_total: int = 2200
) -> list[tuple[Any, str, dict[str, Any], dict[str, float]]]:
    """Generate 10 zero-contamination System One reasoning families."""
    out: list[tuple[Any, str, dict[str, Any], dict[str, float]]] = []

    for i in range(n_total):
        kind = i % 10
        if kind == 0:
            n_levels = 9 if (i % 2 == 0) else 5
            err_rate = round(rng.uniform(0.0, 18.0), 2)
            p99_ms = rng.randint(45, 2400)
            data_loss = rng.random() < 0.14
            sev_raw = (err_rate / 18.0) * 0.55 + (min(p99_ms, 2000) / 2000.0) * 0.30 + (0.35 if data_loss else 0.0)
            gold_lvl = max(0, min(n_levels - 1, int(round(sev_raw * (n_levels - 1)))))
            state = {
                "service": f"svc-core-{i % 17}",
                "window_minutes": rng.choice([5, 15, 30, 60]),
                "error_rate_pct": err_rate,
                "p99_latency_ms": p99_ms,
                "confirmed_data_loss": data_loss,
                "customer_facing": rng.choice([True, True, False]),
            }
            if n_levels == 9:
                crit = [
                    "0 - Nominal: error_rate < 0.5%, p99 < 150ms, no data loss.",
                    "1 - Trace noise: minor latency jitter, error_rate < 1.5%, no data loss.",
                    "2 - Low: error_rate 1.5-3.5% or p99 300-600ms, no data loss.",
                    "3 - Guarded: error_rate 3.5-6% or p99 600-900ms, no data loss.",
                    "4 - Moderate: error_rate 6-9% or p99 900-1250ms, degraded UX.",
                    "5 - Elevated: error_rate 9-12% or p99 > 1250ms, partial outage.",
                    "6 - High: error_rate 12-15% or sustained p99 > 1600ms, major impact.",
                    "7 - Severe: error_rate > 15% without data loss, or moderate errors with data loss.",
                    "8 - Critical catastrophe: high error rate / latency combined with confirmed data loss.",
                ]
            else:
                crit = [
                    "0 - Healthy: within all SLO bounds and no data loss.",
                    "1 - Minor: slight latency or <3.5% errors, no data loss.",
                    "2 - Moderate: 3.5-9% errors or elevated latency, no data loss.",
                    "3 - Major: >9% errors or severe latency, or isolated data loss.",
                    "4 - Critical: severe error rate paired with confirmed data loss.",
                ]
            q = {"type": "score", "instructions": "Rate the operational incident severity strictly on the provided rubric scale.", "criteria": crit}
            sigma = 0.65 if n_levels == 9 else 0.50
            raw_p = {str(lv): math.exp(-((lv - gold_lvl) ** 2) / (2 * sigma * sigma)) for lv in range(n_levels)}
            out.append((state, "decision", q, raw_p))

        elif kind == 1:
            amt = rng.choice([120, 480, 950, 2400, 6800, 14500])
            days_ago = rng.choice([3, 12, 28, 45, 78, 95])
            tier = rng.choice(["free", "pro", "enterprise"])
            defect_verified = rng.choice([True, False])
            custom_carveout = rng.random() < 0.25
            sanctions_hold = rng.random() < 0.18

            if sanctions_hold:
                gold_k = "freeze_compliance_hold"
            elif defect_verified and days_ago <= 90:
                gold_k = "full_refund"
            elif custom_carveout and tier == "enterprise" and days_ago <= 60:
                gold_k = "full_refund"
            elif days_ago <= 30 and amt <= 1000:
                gold_k = "full_refund"
            elif days_ago <= 30 and amt > 1000:
                gold_k = "manager_escalation"
            elif days_ago <= 60 and tier in ("pro", "enterprise"):
                gold_k = "account_credit_only"
            else:
                gold_k = "deny_outside_window"

            extra_sec = "\n".join(
                f"Appendix {sec_i}: Standard operational logging, regional VAT archival, and SOC-2 telemetry retention apply across all jurisdictions without altering Section 4 priority rules."
                for sec_i in range(1, rng.randint(3, 8))
            )
            filler = (
                "Section 1 (Scope & Definitions): All commercial disbursements are logged in UTC and reconciled monthly across regional ledgers.\n"
                "Section 2 (General Commercial Terms): Standard requests within 30 calendar days qualify for full_refund up to $1,000 USD, "
                "or manager_escalation when exceeding $1,000 USD. Requests between 31 and 60 days for pro or enterprise tiers qualify for account_credit_only. "
                "Requests beyond 60 days (or free tier >30 days) default to deny_outside_window.\n"
                f"{extra_sec}\n"
                "Section 4 (Priority Exceptions & Overriding Carveouts): Notwithstanding Section 2: "
                "(a) Verified service defects within 90 days qualify for full_refund regardless of dollar amount; "
                "(b) Enterprise accounts with enterprise_60d_addendum=true qualify for full_refund within 60 days; "
                "(c) OVERRIDING COMPLIANCE RULE: If compliance_sanctions_flag=true, ALL disbursements and credits are strictly prohibited and must route to freeze_compliance_hold, overriding every refund or defect rule.\n"
                f"Case Facts: amount_usd={amt}, days_since_charge={days_ago}, plan_tier={tier}, "
                f"verified_service_defect={str(defect_verified).lower()}, enterprise_60d_addendum={str(custom_carveout).lower()}, "
                f"compliance_sanctions_flag={str(sanctions_hold).lower()}."
            )
            crit_map = {
                "full_refund": "Issue 100% refund under standard <=30d/<=1k rule or applicable defect/addendum exception (when no compliance hold applies).",
                "manager_escalation": "Within 30 days and >$1,000 without defect/addendum exception or compliance hold.",
                "account_credit_only": "31-60 days for pro/enterprise without defect/addendum exception or compliance hold.",
                "deny_outside_window": "Ineligible for refund or credit and not under compliance hold.",
                "freeze_compliance_hold": "Mandatory compliance hold triggered by compliance_sanctions_flag=true, overriding all other rules.",
            }
            q = {"type": "choice", "instructions": "Determine the governing policy disposition after applying all sections and overriding exceptions.", "criteria": crit_map}
            probs = {k: (0.91 if k == gold_k else 0.09 / 4.0) for k in crit_map}
            out.append((filler, "decision", q, probs))

        elif kind == 2:
            has_income = rng.random() > 0.30
            has_debt = rng.random() > 0.30
            has_fico = rng.random() > 0.25
            inc = rng.randint(55000, 185000) if has_income else None
            debt = rng.randint(400, 4800) if has_debt else None
            fico = rng.randint(620, 815) if has_fico else None
            sufficient = has_income and has_debt and has_fico

            state_s = (
                "Underwriting Rule: Approve iff ALL THREE verified metrics are present AND annual DTI (12 * monthly_debt / annual_income) <= 36% AND fico >= 680. "
                "Deny if present metrics already violate FICO < 680 or DTI > 36%. Request missing documentation ONLY when eligibility cannot yet be determined.\n"
                f"Applicant File: annual_income_usd={inc if inc is not None else 'MISSING'}, "
                f"monthly_debt_usd={debt if debt is not None else 'MISSING'}, "
                f"fico_score={fico if fico is not None else 'UNVERIFIED'}."
            )
            if fico is not None and fico < 680:
                gold_c = "deny_ineligible"
            elif inc is not None and debt is not None and (debt * 12.0 / inc) > 0.36:
                gold_c = "deny_ineligible"
            elif sufficient:
                gold_c = "approve_application"
            else:
                gold_c = "insufficient_info_request_docs"

            crit_map = {
                "approve_application": "All required metrics are present and satisfy DTI <= 36% and FICO >= 680.",
                "deny_ineligible": "Known metrics already prove DTI > 36% or FICO < 680.",
                "insufficient_info_request_docs": "Cannot approve or deny yet because one or more required inputs are missing/unverified while known inputs do not disqualify.",
            }
            q = {"type": "choice", "instructions": "Select the epistemically valid underwriting decision without guessing missing values.", "criteria": crit_map}
            probs = {k: (0.91 if k == gold_c else 0.045) for k in crit_map}
            out.append((state_s, "decision", q, probs))

        elif kind == 3:
            env = rng.choice(["staging-sandbox", "production", "qa-replay"])
            negated = rng.choice([True, False])
            if env != "production":
                txt = (
                    f"Environment: {env} (NON-PRODUCTION synthetic drill). "
                    "Simulated alert: 14,000 test records exported during scheduled chaos exercise; zero customer production systems touched."
                )
                is_page = False
            elif negated:
                txt = (
                    "Environment: production. Post-scan verification confirms ZERO unauthorized token usage, "
                    "NO exfiltration of customer PII, and normal 99.99% API availability; earlier scanner warning was a false positive."
                )
                is_page = False
            else:
                txt = (
                    "Environment: production. Confirmed unauthorized exfiltration of 420 customer records via compromised service key; "
                    "active mitigation required."
                )
                is_page = True
            q = {
                "type": "noul",
                "instructions": "Should we trigger a SEV-1 production customer breach notification right now?",
                "criteria": {
                    "false": "No — either non-production drill/test environment or confirmed false positive with zero production breach.",
                    "true": "Yes — confirmed real breach or data exfiltration in production.",
                },
            }
            probs = {"true": 0.93 if is_page else 0.07, "false": 0.07 if is_page else 0.93}
            out.append((txt, "decision", q, probs))

        elif kind == 4:
            start_day = rng.randint(1, 12)
            delta_days = rng.choice([5, 13, 14, 15, 29, 30, 31, 45])
            window_days = rng.choice([14, 30])
            d1 = f"2026-03-{start_day:02d}"
            end_dt_num = start_day + delta_days
            d2 = f"2026-03-{end_dt_num:02d}" if end_dt_num <= 31 else f"2026-04-{end_dt_num - 31:02d}"
            within = delta_days <= window_days
            txt = (
                f"SLA Return Policy: Claims must be submitted within {window_days} calendar days of delivery (elapsed_days <= {window_days}).\n"
                f"Shipment delivered on {d1}. Customer claim submitted on {d2}."
            )
            q = {
                "type": "noul",
                "instructions": "Was the customer claim submitted within the allowed calendar-day window?",
                "criteria": {
                    "false": f"Elapsed days between delivery and submission exceed {window_days} days.",
                    "true": f"Elapsed days between delivery and submission are <= {window_days} days.",
                },
            }
            probs = {"true": 0.93 if within else 0.07, "false": 0.07 if within else 0.93}
            out.append((txt, "decision", q, probs))

        elif kind == 5:
            actor = rng.choice(["alice_chen", "marco_silva", "priya_nair"])
            other = "devon_brooks"
            actor_team = rng.choice(["payments-prod", "ml-research", "marketing-web"])
            other_team = "payments-prod" if actor_team != "payments-prod" else "internal-tools"
            team_kms = {
                "payments-prod": "pci-restricted-kms",
                "ml-research": "gpu-artifact-kms",
                "marketing-web": "public-cdn-kms",
                "internal-tools": "corp-it-kms",
            }
            req_kms = rng.choice(["pci-restricted-kms", "gpu-artifact-kms"])
            allowed = team_kms[actor_team] == req_kms
            facts = [
                f"Directory: user '{other}' belongs to team '{other_team}'.",
                f"Directory: user '{actor}' belongs to team '{actor_team}'.",
                "KMS Policy: team 'payments-prod' may only access 'pci-restricted-kms'.",
                "KMS Policy: team 'ml-research' may only access 'gpu-artifact-kms'.",
                "KMS Policy: team 'marketing-web' may only access 'public-cdn-kms'.",
                "KMS Policy: team 'internal-tools' may only access 'corp-it-kms'.",
            ]
            rng.shuffle(facts)
            txt = "\n".join(facts) + f"\nAccess Request: user '{actor}' requests decrypt permission on key '{req_kms}'."
            q = {
                "type": "noul",
                "instructions": f"Is user '{actor}' authorized to access '{req_kms}' under the team KMS policy?",
                "criteria": {
                    "false": "No — tracing user -> team -> allowed KMS key shows a mismatch.",
                    "true": "Yes — the user's assigned team matches the exact KMS key permitted by policy.",
                },
            }
            probs = {"true": 0.92 if allowed else 0.08, "false": 0.08 if allowed else 0.92}
            out.append((txt, "decision", q, probs))

        elif kind == 6:
            has_sec = rng.random() < 0.25
            has_outage = rng.random() < 0.30
            has_billing = rng.random() < 0.50
            conflicting_acct = rng.random() < 0.22
            msg_parts = []
            if has_billing:
                msg_parts.append("We noticed an unexpected $420 overage line on our March invoice.")
            if has_outage:
                msg_parts.append("Our production webhook endpoint has been returning HTTP 503 for 20 minutes.")
            if has_sec:
                msg_parts.append("We found an exposed API token in a contractor repo needing immediate revocation.")
            if not msg_parts:
                msg_parts.append("How do we configure SSO SAML group mappings?")
            if conflicting_acct and not has_sec:
                msg_parts.append("Note: Ticket header lists org_id=ORG-104 (Enterprise), while signed payload asserts org_id=ORG-992 (Suspended).")
            rng.shuffle(msg_parts)

            if has_sec:
                gold_r = "security_incident_response"
            elif conflicting_acct:
                gold_r = "manual_identity_triage"
            elif has_outage:
                gold_r = "sre_production_oncall"
            elif has_billing:
                gold_r = "billing_operations"
            else:
                gold_r = "product_support_docs"

            state = (
                "Routing Policy: (1) Active credential/security exposure ALWAYS routes to security_incident_response. "
                "(2) Otherwise, if ticket metadata has conflicting org_id identities, route to manual_identity_triage before operational queues. "
                "(3) Otherwise prioritize: sre_production_oncall > billing_operations > product_support_docs.\n"
                "Ticket: " + " ".join(msg_parts)
            )
            crit_map = {
                "security_incident_response": "Active credential exposure or security breach (highest priority).",
                "manual_identity_triage": "Conflicting org_id identity assertions when no active security breach is present.",
                "sre_production_oncall": "Production 5xx outage with consistent identity and no security breach.",
                "billing_operations": "Billing/invoice inquiry with consistent identity and no security or outage issue.",
                "product_support_docs": "General product documentation question.",
            }
            q = {"type": "choice", "instructions": "Select the exact routing queue strictly following priority rules (1)-(3).", "criteria": crit_map}
            probs = {k: (0.91 if k == gold_r else 0.09 / 4.0) for k in crit_map}
            out.append((state, "decision", q, probs))

        elif kind == 7:
            p_fail = rng.choice([0.02, 0.08, 0.25, 0.45])
            outage_cost = rng.choice([10000, 50000, 200000])
            exp_loss = p_fail * outage_cost
            if exp_loss > 15000:
                gold_t = "full_rollback_and_block"
            elif exp_loss >= 2500:
                gold_t = "staged_canary_mitigation"
            else:
                gold_t = "monitor_next_window"
            txt = (
                f"Release Telemetry: failure_probability={p_fail:.2f}, outage_impact_usd=${outage_cost:,}, "
                "canary_cost_usd=$900, full_rollback_cost_usd=$3,500.\n"
                "Decision Rule: Compute expected_unmitigated_loss = failure_probability * outage_impact_usd. "
                "Choose full_rollback_and_block if expected_unmitigated_loss > $15,000; "
                "choose staged_canary_mitigation if between $2,500 and $15,000 inclusive; "
                "otherwise choose monitor_next_window."
            )
            crit_map = {
                "full_rollback_and_block": "Expected unmitigated loss exceeds $15,000.",
                "staged_canary_mitigation": "Expected unmitigated loss is between $2,500 and $15,000 inclusive.",
                "monitor_next_window": "Expected unmitigated loss is strictly below $2,500.",
            }
            q = {"type": "choice", "instructions": "Select the optimal mitigation action under the expected-loss decision rule.", "criteria": crit_map}
            probs = {k: (0.91 if k == gold_t else 0.045) for k in crit_map}
            out.append((txt, "decision", q, probs))

        elif kind == 8:
            mode_amb = rng.choice(["defer", "base_rate"])
            if mode_amb == "defer":
                txt = (
                    "Contract Clause A (Master MSA §4.2): Customer data residency must remain exclusively in eu-central-1.\n"
                    "Contract Clause B (Signed Order Form Addendum §4.2, same effective date, no precedence clause): "
                    "Failover replication into us-east-1 is mandated for disaster recovery.\n"
                    "Action Requested: Provision automated cross-region failover to us-east-1."
                )
                crit_map = {
                    "provision_immediately": "Unambiguously permitted by all governing clauses.",
                    "reject_permanently": "Unambiguously forbidden with clear precedence.",
                    "escalate_ambiguous_conflict": "Governing clauses directly conflict without a precedence rule; escalate for legal/human resolution.",
                }
                q = {"type": "choice", "instructions": "Choose the appropriate action when governing documents contain an unresolved conflict.", "criteria": crit_map}
                probs = {"provision_immediately": 0.06, "reject_permanently": 0.06, "escalate_ambiguous_conflict": 0.88}
                out.append((txt, "decision", q, probs))
            else:
                rate = rng.choice([0.18, 0.35, 0.65, 0.82])
                n_obs = 100
                hits = int(round(rate * n_obs))
                txt = (
                    f"Historical Cohort Calibration: Out of {n_obs} identical hardware replacements in Q3 with matching SMART warning codes, "
                    f"exactly {hits} failed within 14 days ({int(rate*100)}%) and {n_obs - hits} survived without issue."
                )
                q = {
                    "type": "noul",
                    "instructions": "Will a newly flagged drive from this exact cohort fail within 14 days?",
                    "criteria": {
                        "false": "Drive survives the 14-day window.",
                        "true": "Drive fails within the 14-day window.",
                    },
                }
                probs = {"true": rate, "false": round(1.0 - rate, 4)}
                out.append((txt, "decision", q, probs))

        else:
            has_cite = rng.random() > 0.28
            has_neutral = rng.random() > 0.25
            leaked_pii = rng.random() < 0.18
            draft = (
                f"Draft Response Audit: cites_verified_kb_article={str(has_cite).lower()}, "
                f"maintains_neutral_non_promissory_tone={str(has_neutral).lower()}, "
                f"exposes_internal_customer_pii={str(leaked_pii).lower()}."
            )
            if leaked_pii:
                lvl = 0
            elif has_cite and has_neutral:
                lvl = 3
            elif has_cite or has_neutral:
                lvl = 2
            else:
                lvl = 1
            crit = [
                "0 - Critical Failure: exposes PII or violates mandatory safety constraint.",
                "1 - Inadequate: safe (no PII leak) but misses both citation and tone requirements.",
                "2 - Partial Pass: safe (no PII leak) and satisfies exactly one of citation or tone.",
                "3 - Full Pass: safe (no PII leak) and satisfies both verified KB citation and neutral tone.",
            ]
            q = {"type": "score", "instructions": "Score the assistant draft strictly against all rubric constraints.", "criteria": crit}
            probs = {str(k): (0.90 if k == lvl else 0.10 / 3.0) for k in range(4)}
            out.append((draft, "decision", q, probs))

    return out


def build_general_systemone_pool(
    tokenizer: Any,
    rng: random.Random,
    max_state_tokens: int = MAX_STATE_TRAIN,
    extra_jsonl_paths: list[str] | None = None,
    max_total_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load strictly `split='train'` public reasoning datasets + synthetic System One families."""
    raw_tuples: list[tuple[Any, str, dict[str, Any], dict[str, float]]] = []

    boolq = load_dataset("google/boolq", split="train")
    bq_idx = list(range(len(boolq)))
    rng.shuffle(bq_idx)
    for idx in bq_idx[:600]:
        r = boolq[idx]
        ans = bool(r["answer"])
        q = {
            "type": "noul",
            "instructions": str(r["question"]).strip().capitalize() + "?",
            "criteria": {
                "false": "No, the passage indicates this is false or does not support yes.",
                "true": "Yes, the passage supports that this is true.",
            },
        }
        probs = {"true": 0.91 if ans else 0.09, "false": 0.09 if ans else 0.91}
        raw_tuples.append((str(r["passage"]), "decision", q, probs))

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_idx = list(range(len(arc)))
    rng.shuffle(arc_idx)
    for idx in arc_idx[:550]:
        r = arc[idx]
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        ak = str(r["answerKey"])
        if ak not in labels or len(labels) < 3:
            continue
        crit = {lab: txt for lab, txt in zip(labels, texts)}
        q = {"type": "choice", "instructions": "Select the most scientifically and logically sound answer.", "criteria": crit}
        rem = 0.12 / max(1, len(labels) - 1)
        probs = {lab: (0.88 if lab == ak else rem) for lab in labels}
        raw_tuples.append((str(r["question"]), "decision", q, probs))

    sciq = load_dataset("allenai/sciq", split="train")
    sq_idx = list(range(len(sciq)))
    rng.shuffle(sq_idx)
    for idx in sq_idx[:500]:
        r = sciq[idx]
        opts_txt = [str(r["correct_answer"]), str(r["distractor1"]), str(r["distractor2"]), str(r["distractor3"])]
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        labs = ["A", "B", "C", "D"]
        crit = {labs[pos]: opts_txt[orig_i] for pos, orig_i in enumerate(order)}
        gold_lab = labs[order.index(0)]
        support = str(r.get("support") or "").strip()
        state_s = f"Reference Context: {support}\nQuestion: {r['question']}" if support else str(r["question"])
        q = {"type": "choice", "instructions": "Choose the single accurate answer supported by the context and scientific principles.", "criteria": crit}
        probs = {lab: (0.90 if lab == gold_lab else 0.10 / 3.0) for lab in labs}
        raw_tuples.append((state_s, "decision", q, probs))

    raw_tuples.extend(generate_synthetic_systemone_rubrics(rng, n_total=2200))

    for jpath in extra_jsonl_paths or []:
        if os.path.exists(jpath):
            with open(jpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    raw_tuples.append(normalize_extra_gen_item(item))

    rng.shuffle(raw_tuples)
    n_val = min(600, max(200, len(raw_tuples) // 10))
    val_tups = raw_tuples[:n_val]
    tr_tups = raw_tuples[n_val:]

    tr_cases = [
        _pack_single_case(
            tokenizer,
            st,
            {qid: q},
            {qid: {"probabilities": gp}},
            case_id=500000 + idx,
            source="gen",
            max_state_tokens=max_state_tokens,
            max_total_tokens=max_total_tokens,
        )
        for idx, (st, qid, q, gp) in enumerate(tr_tups)
    ]
    va_cases = [
        _pack_single_case(
            tokenizer,
            st,
            {qid: q},
            {qid: {"probabilities": gp}},
            case_id=900000 + idx,
            source="gen",
            max_state_tokens=max_state_tokens,
            max_total_tokens=max_total_tokens,
        )
        for idx, (st, qid, q, gp) in enumerate(val_tups)
    ]
    return tr_cases, va_cases


def compute_batch_decision_loss(
    logits: torch.Tensor,
    flat_q_meta: list[dict[str, Any]],
    device: str,
    use_train_probs: bool = True,
) -> torch.Tensor:
    Q_tot, max_K = logits.shape
    gp_t = torch.zeros((Q_tot, max_K), dtype=torch.float32, device=device)
    vk_t = torch.zeros((Q_tot, max_K), dtype=torch.bool, device=device)

    for q_i, qm in enumerate(flat_q_meta):
        K = len(qm["option_keys"])
        target_p = qm["train_probs"] if use_train_probs else qm["gold_probs"]
        gp_t[q_i, :K] = torch.tensor(target_p, dtype=torch.float32, device=device)
        vk_t[q_i, :K] = True

    masked_logits = logits.masked_fill(~vk_t, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    probs = log_probs.exp()

    ce_loss = -(gp_t * log_probs).sum(dim=-1).mean()
    brier_loss = ((probs - gp_t).pow(2) * vk_t.float()).sum(dim=-1).mean()
    return ce_loss + 0.22 * brier_loss


def split_long_subbatch(
    sub_c: list[dict[str, Any]],
    medium_tok_threshold: int = 1800,
    long_tok_threshold: int = 3000,
) -> list[list[dict[str, Any]]]:
    """Keep peak O(L^2) tree-mask VRAM bounded on 24 GB L4 when long_policy cases enter a batch."""
    if len(sub_c) <= 1:
        return [sub_c]
    max_len = max(len(c["input_ids"]) for c in sub_c)
    if max_len > long_tok_threshold:
        return [[c] for c in sub_c]
    if max_len > medium_tok_threshold and len(sub_c) > 2:
        return [sub_c[i : i + 2] for i in range(0, len(sub_c), 2)]
    return [sub_c]


def load_jev_dev_hard_cases(
    jev_dev_jsonl: str,
    tok: Any,
    max_state_tokens: int,
    max_total_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Load only hard-tier items from `data/jev_dev/dev.jsonl` (supports top-level `tier` or `provenance.tier`)."""
    jev_hard_cases: list[dict[str, Any]] = []
    if not jev_dev_jsonl or not os.path.exists(jev_dev_jsonl):
        return jev_hard_cases
    with open(jev_dev_jsonl, "r", encoding="utf-8") as jhf:
        for j_idx, line in enumerate(jhf):
            if not line.strip():
                continue
            raw_j = json.loads(line)
            tier_j = str(raw_j.get("tier") or (raw_j.get("provenance") or {}).get("tier") or "hard")
            if tier_j != "hard":
                continue
            st_j, qid_j, q_j, gp_j = normalize_extra_gen_item(raw_j)
            jev_hard_cases.append(
                _pack_single_case(
                    tok,
                    st_j,
                    {qid_j: q_j},
                    {qid_j: {"probabilities": gp_j}},
                    case_id=850000 + j_idx,
                    source="jev_dev_hard",
                    max_state_tokens=max_state_tokens,
                    max_total_tokens=max_total_tokens,
                )
            )
    return jev_hard_cases


@torch.no_grad()
def eval_packed_cases(
    backbone: Any,
    head: ParallelSpanDecisionHead,
    cases: list[dict[str, Any]],
    pad_id: int,
    device: str,
    amp_dtype: torch.dtype,
    batch_cases: int = 8,
    header_attends_options: bool = False,
    max_seq_pad: int | None = None,
) -> tuple[float, float, float, float, list[list[torch.Tensor]]]:
    backbone.eval()
    head.eval()
    accs, kls, briers, confs = [], [], [], []
    per_case_logits: list[list[torch.Tensor]] = []

    for s_i in range(0, len(cases), batch_cases):
        raw_sub = cases[s_i : s_i + batch_cases]
        for sub_c in split_long_subbatch(raw_sub):
            seq_f, zs_fn = forward_parallel_backbone(
                backbone,
                sub_c,
                pad_id=pad_id,
                device=device,
                amp_dtype=amp_dtype,
                header_attends_options=header_attends_options,
                max_seq_pad=max_seq_pad,
            )
            logits, _ = head.forward_packed_cases(seq_f, sub_c, zs_head_fn=zs_fn, apply_temp=False)

            cursor = 0
            for pc in sub_c:
                c_lgs: list[torch.Tensor] = []
                for qm in pc["questions"]:
                    K = len(qm["option_keys"])
                    lv = logits[cursor, :K].detach().cpu()
                    c_lgs.append(lv)
                    ps = F.softmax(lv, dim=-1).tolist()
                    pred_obj = {"type": qm["qtype"], "probabilities": {k: ps[idx] for idx, k in enumerate(qm["option_keys"])}}
                    ok, kl, br, cf = score_typed_decision_one(pred_obj, qm["gold_raw"])
                    accs.append(ok)
                    kls.append(kl)
                    briers.append(br)
                    confs.append(cf)
                    cursor += 1
                per_case_logits.append(c_lgs)

    return (
        sum(accs) / len(accs),
        sum(kls) / len(kls),
        sum(briers) / len(briers),
        compute_ece_10bin(accs, confs),
        per_case_logits,
    )


def make_case_batches(cases: list[dict[str, Any]], batch_size: int, rng: random.Random) -> list[list[dict[str, Any]]]:
    shuf = list(cases)
    rng.shuffle(shuf)
    batches: list[list[dict[str, Any]]] = []
    pool_sz = batch_size * 16
    for i in range(0, len(shuf), pool_sz):
        window = sorted(shuf[i : i + pool_sz], key=lambda x: len(x["input_ids"]))
        for j in range(0, len(window), batch_size):
            batches.append(window[j : j + batch_size])
    rng.shuffle(batches)
    return batches


def run_training_loop(
    backbone: Any,
    head: ParallelSpanDecisionHead,
    aux_head: LatentFactorAuxHead | None,
    train_cases: list[dict[str, Any]],
    val_cases: list[dict[str, Any]],
    pad_id: int,
    device: str,
    amp_dtype: torch.dtype,
    epochs: int,
    backbone_lr: float,
    head_lr: float,
    batch_cases: int,
    grad_accum: int,
    seed: int,
    tag: str,
    swa_start_epoch: int | None = None,
    header_attends_options: bool = False,
    max_seq_pad: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], tuple[float, float, float, float, list[list[torch.Tensor]]]]:
    rng = random.Random(seed)
    bb_params = [p for p in backbone.parameters() if p.requires_grad]
    hd_params = [p for p in head.parameters() if p.requires_grad]
    if aux_head is not None:
        hd_params.extend([p for p in aux_head.parameters() if p.requires_grad])

    opt = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": backbone_lr, "weight_decay": 0.01},
            {"params": hd_params, "lr": head_lr, "weight_decay": 0.01},
        ]
    )
    sample_batches = make_case_batches(train_cases, batch_cases, rng)
    steps_per_ep = math.ceil(len(sample_batches) / grad_accum)
    total_steps = max(1, epochs * steps_per_ep)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[backbone_lr, head_lr],
        total_steps=total_steps,
        pct_start=0.10,
    )
    use_scaler = (device == "cuda") and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_score = float("inf")
    best_bb_sd: dict[str, torch.Tensor] = {}
    best_hd_sd: dict[str, torch.Tensor] = {}
    best_val_tuple: tuple[float, float, float, float, list[list[torch.Tensor]]] | None = None
    swa_bb_sum: dict[str, torch.Tensor] = {}
    swa_hd_sum: dict[str, torch.Tensor] = {}
    swa_count = 0
    all_params = bb_params + hd_params

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        backbone.train()
        head.train()
        if aux_head is not None:
            aux_head.train()
        batches = make_case_batches(train_cases, batch_cases, rng)
        opt.zero_grad(set_to_none=True)
        losses_ep: list[float] = []

        for b_i, sub_c in enumerate(batches):
            micro_slices = split_long_subbatch(sub_c)
            if len(micro_slices) == 1:
                seq_f, zs_fn = forward_parallel_backbone(
                    backbone,
                    sub_c,
                    pad_id=pad_id,
                    device=device,
                    amp_dtype=amp_dtype,
                    header_attends_options=header_attends_options,
                    max_seq_pad=max_seq_pad,
                )
                logits, flat_q_meta = head.forward_packed_cases(seq_f, sub_c, zs_head_fn=zs_fn, apply_temp=False)
                dec_loss = compute_batch_decision_loss(logits, flat_q_meta, device=device, use_train_probs=True)
                if aux_head is not None:
                    aux_loss = aux_head.compute_loss(seq_f, sub_c)
                    raw_loss = dec_loss + 0.16 * aux_loss
                else:
                    raw_loss = dec_loss

                loss = raw_loss / grad_accum
                scaler.scale(loss).backward()
                losses_ep.append(float(raw_loss.item()))
            else:
                tot_q = max(1, sum(len(pc["questions"]) for pc in sub_c))
                tot_c = max(1, len(sub_c))
                step_loss_val = 0.0
                for mc in micro_slices:
                    seq_f, zs_fn = forward_parallel_backbone(
                        backbone,
                        mc,
                        pad_id=pad_id,
                        device=device,
                        amp_dtype=amp_dtype,
                        header_attends_options=header_attends_options,
                        max_seq_pad=max_seq_pad,
                    )
                    logits, flat_q_meta = head.forward_packed_cases(seq_f, mc, zs_head_fn=zs_fn, apply_temp=False)
                    q_w = len(flat_q_meta) / tot_q
                    dec_part = q_w * compute_batch_decision_loss(logits, flat_q_meta, device=device, use_train_probs=True)
                    if aux_head is not None:
                        c_w = len(mc) / tot_c
                        mc_loss = dec_part + 0.16 * c_w * aux_head.compute_loss(seq_f, mc)
                    else:
                        mc_loss = dec_part
                    scaler.scale(mc_loss / grad_accum).backward()
                    step_loss_val += float(mc_loss.item())
                losses_ep.append(step_loss_val)

            if (b_i + 1) % grad_accum == 0 or (b_i + 1) == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        v_acc, v_kl, v_br, v_ece, v_lgs = eval_packed_cases(
            backbone,
            head,
            val_cases,
            pad_id,
            device,
            amp_dtype,
            header_attends_options=header_attends_options,
            max_seq_pad=max_seq_pad,
        )
        obj = v_kl + 0.40 * v_br - 0.14 * v_acc
        print(
            f"    [{tag} · ep {ep}/{epochs} · {time.perf_counter()-t0:.1f}s] "
            f"loss={sum(losses_ep)/len(losses_ep):.4f} | val_acc={v_acc:.4f} | val_kl={v_kl:.4f} | val_brier={v_br:.4f} | val_ece={v_ece:.4f}",
            flush=True,
        )
        if swa_start_epoch is not None and ep >= swa_start_epoch:
            for k, v in backbone.state_dict().items():
                vt = v.detach().cpu().float()
                swa_bb_sum[k] = swa_bb_sum[k] + vt if k in swa_bb_sum else vt.clone()
            for k, v in head.state_dict().items():
                vt = v.detach().cpu().float()
                swa_hd_sum[k] = swa_hd_sum[k] + vt if k in swa_hd_sum else vt.clone()
            swa_count += 1
        if obj < best_score:
            best_score = obj
            best_bb_sd = {k: v.detach().cpu().float().clone() for k, v in backbone.state_dict().items()}
            best_hd_sd = {k: v.detach().cpu().float().clone() for k, v in head.state_dict().items()}
            best_val_tuple = (v_acc, v_kl, v_br, v_ece, v_lgs)

    assert best_val_tuple is not None
    if swa_start_epoch is not None and swa_count > 0:
        inv_c = 1.0 / swa_count
        swa_bb_out = {k: v * inv_c for k, v in swa_bb_sum.items()}
        swa_hd_out = {k: v * inv_c for k, v in swa_hd_sum.items()}
        backbone.load_state_dict({k: v.to(device=device, dtype=next(backbone.parameters()).dtype) for k, v in swa_bb_out.items()})
        head.load_state_dict({k: v.to(device=device) for k, v in swa_hd_out.items()})
        swa_val_tuple = eval_packed_cases(
            backbone,
            head,
            val_cases,
            pad_id,
            device,
            amp_dtype,
            header_attends_options=header_attends_options,
            max_seq_pad=max_seq_pad,
        )
        return swa_bb_out, swa_hd_out, swa_val_tuple
    return best_bb_sd, best_hd_sd, best_val_tuple


def build_stratified_5folds(ds_train: Any, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    rng = random.Random(seed)
    strata: dict[str, list[int]] = collections.defaultdict(list)
    for idx, r in enumerate(ds_train):
        wf = str(r["workflow"])
        fc = json.loads(r["factors"]) if isinstance(r["factors"], str) else r["factors"]
        sub_k = str(fc.get("issue") or fc.get("completion") or fc.get("rule") or fc.get("topic") or "default")
        strata[f"{wf}::{sub_k}"].append(idx)

    fold_buckets: list[list[int]] = [[] for _ in range(5)]
    cursor = 0
    for key in sorted(strata.keys()):
        arr = list(strata[key])
        rng.shuffle(arr)
        for item_idx in arr:
            fold_buckets[cursor % 5].append(item_idx)
            cursor += 1

    folds: list[tuple[list[int], list[int]]] = []
    for f in range(5):
        va_i = sorted(fold_buckets[f])
        va_set = set(va_i)
        tr_i = [i for i in range(len(ds_train)) if i not in va_set]
        folds.append((tr_i, va_i))
    return folds


def carve_student_holdout_and_folds(
    ds_train: Any,
    seed: int = 42,
) -> tuple[list[int], list[int], list[tuple[list[int], list[int]]]]:
    """Task 6 Step 1: carve out 1/5 of `train` BEFORE teacher folds are built so student hyperparams
    (`beta_oof`, `alpha_wise`) are selected on real unseen `train` data with zero teacher leakage.
    """
    outer_folds = build_stratified_5folds(ds_train, seed=seed)
    holdout_idx = sorted(outer_folds[0][1])
    holdout_set = set(holdout_idx)
    teacher_pool_idx = [i for i in range(len(ds_train)) if i not in holdout_set]

    rng = random.Random(seed + 101)
    strata: dict[str, list[int]] = collections.defaultdict(list)
    for orig_idx in teacher_pool_idx:
        r = ds_train[orig_idx]
        wf = str(r["workflow"])
        fc = json.loads(r["factors"]) if isinstance(r["factors"], str) else r["factors"]
        sub_k = str(fc.get("issue") or fc.get("completion") or fc.get("rule") or fc.get("topic") or "default")
        strata[f"{wf}::{sub_k}"].append(orig_idx)

    inner_buckets: list[list[int]] = [[] for _ in range(5)]
    cursor = 0
    for key in sorted(strata.keys()):
        arr = list(strata[key])
        rng.shuffle(arr)
        for orig_idx in arr:
            inner_buckets[cursor % 5].append(orig_idx)
            cursor += 1

    teacher_folds: list[tuple[list[int], list[int]]] = []
    for f in range(5):
        va_i = sorted(inner_buckets[f])
        va_set = set(va_i)
        tr_i = [idx for idx in teacher_pool_idx if idx not in va_set]
        teacher_folds.append((tr_i, va_i))
    return holdout_idx, teacher_pool_idx, teacher_folds


def fit_per_type_temperatures_lbfgs(
    oof_flat_pairs: list[tuple[torch.Tensor, dict[str, Any]]],
    min_temp: float = 0.15,
    max_temp: float = 5.0,
) -> list[float]:
    """Task 6 Step 2: fit one scalar temperature per `qtype_id` (0=noul, 1=choice, 2=score) by L-BFGS on OOF KL."""
    fitted: list[float] = [1.0, 1.0, 1.0]
    for qt_id in (0, 1, 2):
        sub = [(lv.detach().float().cpu(), qm) for lv, qm in oof_flat_pairs if int(qm["qtype_id"]) == qt_id]
        if not sub:
            continue

        max_k = max(int(lv.numel()) for lv, _ in sub)
        n_q = len(sub)
        logits_pad = torch.full((n_q, max_k), -1e9, dtype=torch.float32)
        gold_pad = torch.zeros((n_q, max_k), dtype=torch.float32)
        valid_mask = torch.zeros((n_q, max_k), dtype=torch.bool)
        for i, (lv, qm) in enumerate(sub):
            k = int(lv.numel())
            logits_pad[i, :k] = lv
            gp = qm.get("gold_probs")
            if gp is None:
                raw_g = qm["gold_raw"]["probabilities"]
                gp = [float(raw_g.get(ok, 0.0)) for ok in qm["option_keys"]]
            gold_pad[i, :k] = torch.tensor(gp[:k], dtype=torch.float32)
            valid_mask[i, :k] = True

        def kl_at_log_t(lt: torch.Tensor) -> torch.Tensor:
            t_val = torch.exp(lt).clamp(min_temp, max_temp)
            scaled = torch.where(valid_mask, logits_pad / t_val, torch.full_like(logits_pad, -1e9))
            log_p = F.log_softmax(scaled, dim=-1)
            pos_g = gold_pad > 0
            safe_log_g = torch.where(pos_g, torch.log(gold_pad.clamp(min=1e-12)), torch.zeros_like(gold_pad))
            kl_vec = torch.where(pos_g, gold_pad * (safe_log_g - log_p), torch.zeros_like(gold_pad)).sum(dim=-1)
            return kl_vec.mean()

        with torch.no_grad():
            base_kl = float(kl_at_log_t(torch.tensor(0.0)).item())

        log_t_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        opt = torch.optim.LBFGS([log_t_param], lr=0.5, max_iter=60, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = kl_at_log_t(log_t_param)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            cand_kl = float(kl_at_log_t(log_t_param).item())
            cand_t = float(torch.exp(log_t_param).clamp(min_temp, max_temp).item())
        if math.isfinite(cand_kl) and cand_kl <= base_kl + 1e-7:
            fitted[qt_id] = round(cand_t, 5)
        else:
            fitted[qt_id] = 1.0
    return fitted


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Average `N >= 1` PyTorch state dicts in float32."""
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    inv_n = 1.0 / len(state_dicts)
    acc: dict[str, torch.Tensor] = {}
    for sd in state_dicts:
        for k, v in sd.items():
            vf = v.detach().cpu().float()
            acc[k] = acc[k] + vf if k in acc else vf.clone()
    return {k: v * inv_n for k, v in acc.items()}


def merge_wise_and_seed_soup(
    seed_state_dicts: list[dict[str, torch.Tensor]],
    gen_state_dict: dict[str, torch.Tensor],
    alpha_wise: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Collapse `K >= 1` seed state dicts and Stage 1 general weights into ONE single encoder/head state dict."""
    soup = average_state_dicts(seed_state_dicts)
    if alpha_wise <= 0.0:
        return soup
    return {
        k: (1.0 - alpha_wise) * soup[k].float() + alpha_wise * gen_state_dict[k].detach().cpu().float()
        for k in soup.keys()
    }


def fit_nested_fold_temperatures_lbfgs(
    oof_logits_by_case: dict[int, list[torch.Tensor]],
    td_cases: list[dict[str, Any]],
    folds: list[tuple[list[int], list[int]]],
    min_temp: float = 0.15,
    max_temp: float = 5.0,
) -> dict[int, list[float]]:
    """Fit per-type L-BFGS temperatures for each fold's `va_idx` strictly using `tr_idx` OOF logits."""
    temps_by_case: dict[int, list[float]] = {}
    for tr_idx, va_idx in folds:
        tr_pairs: list[tuple[torch.Tensor, dict[str, Any]]] = []
        for i in tr_idx:
            pc = td_cases[i]
            for q_i, qm in enumerate(pc["questions"]):
                tr_pairs.append((oof_logits_by_case[i][q_i], qm))
        fold_temps = fit_per_type_temperatures_lbfgs(tr_pairs, min_temp=min_temp, max_temp=max_temp)
        for i in va_idx:
            temps_by_case[i] = fold_temps
    return temps_by_case


def select_soup_or_best_seed(
    seed_reports: list[dict[str, Any]],
    soup_report: dict[str, Any],
) -> dict[str, Any]:
    """Task 6 Step 3: keep 3-seed weight soup ONLY if it passes the Task 2 OOF gate against the best single seed."""
    if not seed_reports:
        raise ValueError("seed_reports must be non-empty")
    best_idx = max(
        range(len(seed_reports)),
        key=lambda i: (float(seed_reports[i]["oof_accuracy"]), -float(seed_reports[i]["oof_kl"])),
    )
    best = seed_reports[best_idx]
    d_acc = float(soup_report["oof_accuracy"]) - float(best["oof_accuracy"])
    d_kl = float(soup_report["oof_kl"]) - float(best["oof_kl"])
    rule1 = (d_acc >= 0.005 - 1e-12) and (d_kl <= 0.002 + 1e-12)
    rule2 = (d_kl <= -0.002 + 1e-12) and (d_acc >= -1e-12)
    if rule1 or rule2:
        return {"selected": "soup", "best_seed_idx": best_idx, "delta_acc": round(d_acc, 6), "delta_kl": round(d_kl, 6)}
    return {"selected": "best_seed", "best_seed_idx": best_idx, "delta_acc": round(d_acc, 6), "delta_kl": round(d_kl, 6)}


def instantiate_family(backbone_id: str, device: str, amp_dtype: torch.dtype) -> tuple[Any, Any, ParallelSpanDecisionHead]:
    from transformers import AutoConfig

    orig_cfg = AutoConfig.from_pretrained(backbone_id)
    orig_i2l = {int(k): str(v).lower() for k, v in (getattr(orig_cfg, "id2label", None) or {}).items()}
    has_nli_head = any("entail" in v for v in orig_i2l.values()) and len(orig_i2l) in (2, 3)

    orig_cfg.reference_compile = False
    if getattr(orig_cfg, "model_type", "") == "modernbert":
        orig_cfg._attn_implementation = "sdpa"
        orig_cfg.local_attention = 16384
        orig_cfg.sliding_window = 16384
        orig_cfg.global_attn_every_n_layers = 1
    else:
        orig_cfg._attn_implementation = "eager"
    if not has_nli_head:
        orig_cfg.num_labels = 2

    tok = AutoTokenizer.from_pretrained(backbone_id)
    if has_nli_head:
        nli_model = AutoModelForSequenceClassification.from_pretrained(backbone_id, config=orig_cfg).float()
    else:
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            backbone_id, config=orig_cfg, ignore_mismatched_sizes=True
        ).float()
    configure_modernbert_full_bidir(nli_model, tok)
    for p in nli_model.parameters():
        p.requires_grad = True
    zs_dim = int(getattr(nli_model.config, "num_labels", 2))
    head = ParallelSpanDecisionHead(
        hidden_size=int(nli_model.config.hidden_size),
        comp_dim=512,
        zs_dim=zs_dim,
        dropout=0.10,
    )
    with torch.no_grad():
        if has_nli_head and zs_dim == 2:
            if "not" in orig_i2l.get(0, "") and "entail" in orig_i2l.get(1, ""):
                head.zs_weights.copy_(torch.tensor([-0.55, 0.55], dtype=torch.float32))
            else:
                head.zs_weights.copy_(torch.tensor([0.55, -0.55], dtype=torch.float32))
        elif not has_nli_head:
            head.zs_weights.zero_()
    return tok, nli_model.to(device), head.to(device)


def train_systemone(model_dir: str, device: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(config or {})
    specs = resolve_candidate_specs(cfg)
    max_state_train = int(cfg.get("max_state_train", MAX_STATE_TRAIN))
    max_state_infer = int(cfg.get("max_state_infer", MAX_STATE_INFER))
    header_attends_options = bool(cfg.get("header_attends_options", False))
    beta_oof = float(cfg.get("beta_oof", 0.32))
    alpha_wise = float(cfg.get("alpha_wise", 0.04))
    sweep_student_holdout = bool(cfg.get("sweep_student_holdout", False))
    student_seeds = [int(s) for s in cfg.get("student_seeds", [303])]
    synth_td_jsonl = cfg.get("synth_td_jsonl")
    extra_gen_jsonl = list(cfg.get("extra_gen_jsonl") or [])

    print("=== Training System One (5-Fold Stratified OOF, Zero Test Access) ===", flush=True)
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32)
    print(
        f"Device={device} | Precision={amp_dtype} | max_state_train={max_state_train} | header_attends_options={header_attends_options}",
        flush=True,
    )

    ds_train_full = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    factor_vocabs = extract_factor_vocabs(ds_train_full)
    folds = build_stratified_5folds(ds_train_full, seed=int(cfg.get("fold_seed", 42)))

    synth_rows: list[dict[str, Any]] = []
    if synth_td_jsonl and os.path.exists(synth_td_jsonl):
        with open(synth_td_jsonl, "r", encoding="utf-8") as sfh:
            synth_rows = [json.loads(line) for line in sfh if line.strip()]
        print(f"Loaded {len(synth_rows):,} synthetic teacher-labelled Typed Decisions cases from {synth_td_jsonl}", flush=True)

    family_results: dict[str, dict[str, Any]] = {}
    family_cache_dir = cfg.get("family_cache_dir")

    for fam_idx, spec in enumerate(specs):
        fam_key = str(spec["key"])
        bb_id = str(spec["backbone_id"])
        spec_seed_offset = int(spec.get("seed_offset", fam_idx * 17))
        spec_max_total = int(spec["max_total_tokens"]) if spec.get("max_total_tokens") is not None else None
        spec_max_state = int(spec.get("max_state_tokens", max_state_train))
        fam_seed_base = int(cfg.get("seed_base", 42)) + spec_seed_offset
        spec_with_offset = {**spec, "seed_offset": spec_seed_offset}
        cache_key = compute_family_cache_key(
            spec_with_offset,
            max_state_train=max_state_train,
            header_attends_options=header_attends_options,
            seed_base=int(cfg.get("seed_base", 42)),
            fold_seed=int(cfg.get("fold_seed", 42)),
            stage1_epochs=int(cfg.get("stage1_epochs", 2)),
            fold_epochs=int(cfg.get("fold_epochs", 6)),
            synth_td_jsonl=synth_td_jsonl,
            extra_gen_jsonl=extra_gen_jsonl,
        )
        cached_fam = load_cached_family_result(family_cache_dir, cache_key)
        if cached_fam is not None:
            print(
                f"\n--- Candidate Family `{fam_key}` LOADED FROM CACHE ({cache_key}) -> "
                f"OOF_acc={cached_fam['oof_acc']:.4f} | OOF_KL={cached_fam['oof_kl']:.4f} ---",
                flush=True,
            )
            family_results[fam_key] = cached_fam
            continue

        print(
            f"\n--- Candidate Family `{fam_key}` ({bb_id} | max_state={spec_max_state} | max_total={spec_max_total}) ---",
            flush=True,
        )
        tok, bb_model, head = instantiate_family(bb_id, device, amp_dtype)
        pad_id = tok.pad_token_id or 0
        rng_fam = random.Random(fam_seed_base)

        gen_tr, gen_va = build_general_systemone_pool(
            tok,
            rng_fam,
            max_state_tokens=spec_max_state,
            extra_jsonl_paths=extra_gen_jsonl,
            max_total_tokens=spec_max_total,
        )
        td_all_cases = prepare_td_cases(
            ds_train_full,
            tok,
            factor_vocabs,
            base_case_id=100000,
            max_state_tokens=spec_max_state,
            max_total_tokens=spec_max_total,
        )
        synth_cases = (
            prepare_td_cases(
                synth_rows,
                tok,
                factor_vocabs,
                base_case_id=300000,
                max_state_tokens=spec_max_state,
                source="td_synth",
                max_total_tokens=spec_max_total,
            )
            if synth_rows
            else []
        )

        print(f"  [Stage 1 · {fam_key}] General System One Pretraining on {len(gen_tr):,} cases (val={len(gen_va):,})...", flush=True)
        gen_bb_sd, gen_hd_sd, (g_acc, g_kl, g_br, g_ece, _) = run_training_loop(
            backbone=bb_model,
            head=head,
            aux_head=None,
            train_cases=gen_tr,
            val_cases=gen_va,
            pad_id=pad_id,
            device=device,
            amp_dtype=amp_dtype,
            epochs=int(cfg.get("stage1_epochs", 2)),
            backbone_lr=2.5e-5,
            head_lr=2.2e-4,
            batch_cases=int(cfg.get("stage1_batch_cases", 4)),
            grad_accum=int(cfg.get("stage1_grad_accum", 4)),
            seed=100 + spec_seed_offset,
            tag=f"{fam_key}:stage1_gen",
            header_attends_options=header_attends_options,
            max_seq_pad=spec_max_total,
        )
        print(f"  [Stage 1 · {fam_key} Best General] gen_val_acc={g_acc:.4f} | gen_val_kl={g_kl:.4f} | gen_val_ece={g_ece:.4f}", flush=True)

        oof_logits_by_case: dict[int, list[torch.Tensor]] = {}
        fold_case_splits = build_fold_splits_with_synth(td_all_cases, folds, synth_cases)
        for f_idx, ((_, va_idx), (fold_tr, fold_va)) in enumerate(zip(folds, fold_case_splits)):
            bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in gen_bb_sd.items()})
            head.load_state_dict({k: v.to(device=device) for k, v in gen_hd_sd.items()})
            aux_head = LatentFactorAuxHead(int(bb_model.config.hidden_size), factor_vocabs).to(device)
            _, _, (_, _, _, _, f_va_logits) = run_training_loop(
                backbone=bb_model,
                head=head,
                aux_head=aux_head,
                train_cases=fold_tr,
                val_cases=fold_va,
                pad_id=pad_id,
                device=device,
                amp_dtype=amp_dtype,
                epochs=int(cfg.get("fold_epochs", 6)),
                backbone_lr=2.8e-5,
                head_lr=2.2e-4,
                batch_cases=4,
                grad_accum=4,
                seed=200 + spec_seed_offset + f_idx,
                tag=f"{fam_key}:fold{f_idx+1}/5",
                header_attends_options=header_attends_options,
                max_seq_pad=spec_max_total,
            )
            for local_i, orig_i in enumerate(va_idx):
                oof_logits_by_case[orig_i] = f_va_logits[local_i]
            del aux_head

        oof_accs, oof_kls, oof_brs, oof_cfs = [], [], [], []
        for orig_i in range(len(td_all_cases)):
            pc = td_all_cases[orig_i]
            c_lgs = oof_logits_by_case[orig_i]
            for q_i, qm in enumerate(pc["questions"]):
                ps = F.softmax(c_lgs[q_i], dim=-1).tolist()
                pred_obj = {"type": qm["qtype"], "probabilities": {k: ps[idx] for idx, k in enumerate(qm["option_keys"])}}
                ok, kl, br, cf = score_typed_decision_one(pred_obj, qm["gold_raw"])
                oof_accs.append(ok)
                oof_kls.append(kl)
                oof_brs.append(br)
                oof_cfs.append(cf)

        fam_oof_acc = sum(oof_accs) / len(oof_accs)
        fam_oof_kl = sum(oof_kls) / len(oof_kls)
        fam_oof_br = sum(oof_brs) / len(oof_brs)
        fam_oof_ece = compute_ece_10bin(oof_accs, oof_cfs)
        print(
            f"  >>> [{fam_key} 5-FOLD OOF (6,000 train decisions)] "
            f"OOF_acc={fam_oof_acc:.4f} | OOF_KL={fam_oof_kl:.4f} | OOF_Brier={fam_oof_br:.4f} | OOF_ECE={fam_oof_ece:.4f} <<<",
            flush=True,
        )

        family_results[fam_key] = {
            "backbone_id": bb_id,
            "max_total_tokens": spec_max_total,
            "max_state_tokens": spec_max_state,
            "gen_bb_sd": gen_bb_sd,
            "gen_hd_sd": gen_hd_sd,
            "gen_acc": g_acc,
            "gen_kl": g_kl,
            "oof_acc": fam_oof_acc,
            "oof_kl": fam_oof_kl,
            "oof_br": fam_oof_br,
            "oof_ece": fam_oof_ece,
            "oof_logits_by_case": oof_logits_by_case,
        }
        if family_cache_dir:
            saved_p = save_cached_family_result(family_cache_dir, cache_key, family_results[fam_key])
            print(f"  [Cache] Saved `{fam_key}` -> {saved_p}", flush=True)
        del bb_model, head, tok, gen_tr, gen_va, td_all_cases, synth_cases
        if device == "cuda":
            torch.cuda.empty_cache()

    ds_check_qs = [json.loads(r["questions"]) for r in ds_train_full]
    ds_check_gs = [json.loads(r["gold"]) for r in ds_train_full]
    ens_accs, ens_kls, ens_brs = [], [], []
    oof_teacher_probs: list[dict[str, list[float]]] = []

    for i in range(len(ds_train_full)):
        q_probs_case: dict[str, list[float]] = {}
        qids_list = list(ds_check_qs[i].keys())
        for q_i, qid in enumerate(qids_list):
            gq = ds_check_gs[i][qid]
            p_list_fams = [F.softmax(fres["oof_logits_by_case"][i][q_i], dim=-1) for fres in family_results.values()]
            p_ens = torch.stack(p_list_fams, dim=0).mean(dim=0).tolist()
            q_probs_case[qid] = p_ens

            opt_keys = [k for k, _ in canonical_options(ds_check_qs[i][qid])]
            pred_obj = {"type": gq["type"], "probabilities": {k: p_ens[idx] for idx, k in enumerate(opt_keys)}}
            ok, kl, br, _ = score_typed_decision_one(pred_obj, gq)
            ens_accs.append(ok)
            ens_kls.append(kl)
            ens_brs.append(br)
        oof_teacher_probs.append(q_probs_case)

    print(
        f"\n>>> [CROSS-BACKBONE OOF TEACHER ON TRAIN (6,000 decisions)] "
        f"acc={sum(ens_accs)/len(ens_accs):.4f} | KL={sum(ens_kls)/len(ens_kls):.4f} | Brier={sum(ens_brs)/len(ens_brs):.4f} <<<",
        flush=True,
    )

    winner_key = min(
        family_results.keys(),
        key=lambda fk: (
            family_results[fk]["oof_kl"]
            + 0.35 * family_results[fk]["oof_br"]
            - 0.18 * family_results[fk]["oof_acc"]
            + 0.15 * family_results[fk]["gen_kl"]
        ),
    )
    win_info = family_results[winner_key]
    win_bb_id = win_info["backbone_id"]
    win_max_total = win_info.get("max_total_tokens")
    win_max_state = int(win_info.get("max_state_tokens", max_state_train))
    print(f"Selected winning single-student backbone `{winner_key}` ({win_bb_id}) for Stage 2B!", flush=True)

    tok, bb_model, head = instantiate_family(win_bb_id, device, amp_dtype)
    pad_id = tok.pad_token_id or 0
    rng_final = random.Random(777)
    gen_tr_final, gen_va_final = build_general_systemone_pool(
        tok,
        rng_final,
        max_state_tokens=win_max_state,
        extra_jsonl_paths=extra_gen_jsonl,
        max_total_tokens=win_max_total,
    )
    td_full_cases = prepare_td_cases(
        ds_train_full,
        tok,
        factor_vocabs,
        base_case_id=100000,
        max_state_tokens=win_max_state,
        max_total_tokens=win_max_total,
    )
    synth_final_cases = (
        prepare_td_cases(
            synth_rows,
            tok,
            factor_vocabs,
            base_case_id=300000,
            max_state_tokens=win_max_state,
            source="td_synth",
            max_total_tokens=win_max_total,
        )
        if synth_rows
        else []
    )

    holdout_sweep_log: list[dict[str, Any]] = []
    soup_selection_meta: dict[str, Any] = {"selected": "single_seed", "seeds": student_seeds}

    # Task 6 Step 1 & Step 3: Carve 1/5 holdout BEFORE teacher folds to tune (beta_oof, alpha_wise) + soup gate on real data
    if sweep_student_holdout:
        print("\n=== [Task 6] Running Carved 1/5 Holdout Sweep for (beta_oof, alpha_wise) & 3-Seed Soup Gate ===", flush=True)
        holdout_idx, teacher_pool_idx, inner_teacher_folds = carve_student_holdout_and_folds(
            ds_train_full, seed=int(cfg.get("fold_seed", 42))
        )
        inner_oof_logits: dict[int, list[torch.Tensor]] = {}
        for f_idx, (in_tr_idx, in_va_idx) in enumerate(inner_teacher_folds):
            bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in win_info["gen_bb_sd"].items()})
            head.load_state_dict({k: v.to(device=device) for k, v in win_info["gen_hd_sd"].items()})
            aux_h = LatentFactorAuxHead(int(bb_model.config.hidden_size), factor_vocabs).to(device)
            in_tr_cases = [td_full_cases[i] for i in in_tr_idx] + synth_final_cases
            in_va_cases = [td_full_cases[i] for i in in_va_idx]
            _, _, (_, _, _, _, in_va_lgs) = run_training_loop(
                backbone=bb_model,
                head=head,
                aux_head=aux_h,
                train_cases=in_tr_cases,
                val_cases=in_va_cases,
                pad_id=pad_id,
                device=device,
                amp_dtype=amp_dtype,
                epochs=int(cfg.get("fold_epochs", 6)),
                backbone_lr=2.8e-5,
                head_lr=2.2e-4,
                batch_cases=4,
                grad_accum=4,
                seed=600 + f_idx,
                tag=f"holdout_inner_teacher:fold{f_idx+1}/5",
                header_attends_options=header_attends_options,
                max_seq_pad=win_max_total,
            )
            for loc_i, orig_i in enumerate(in_va_idx):
                inner_oof_logits[orig_i] = in_va_lgs[loc_i]
            del aux_h

        holdout_cases = [td_full_cases[i] for i in holdout_idx]
        beta_grid = [float(x) for x in cfg.get("beta_oof_grid", [0.0, 0.16, 0.32, 0.50])]
        alpha_grid = [float(x) for x in cfg.get("alpha_wise_grid", [0.0, 0.04, 0.10])]

        for b_cand in beta_grid:
            pool_cases_b = []
            for orig_i in teacher_pool_idx:
                pc = td_full_cases[orig_i]
                for q_i, qm in enumerate(pc["questions"]):
                    gp = qm["gold_probs"]
                    tp = F.softmax(inner_oof_logits[orig_i][q_i], dim=-1).tolist()
                    blended = [(1.0 - b_cand) * g_v + b_cand * t_v for g_v, t_v in zip(gp, tp)]
                    sb = sum(blended)
                    qm["train_probs"] = [v / sb for v in blended]
                pool_cases_b.append(pc)

            rng_hb = random.Random(303)
            rep_hb = list(gen_tr_final)
            rng_hb.shuffle(rep_hb)
            tr_hb = pool_cases_b + synth_final_cases + rep_hb[: int(cfg.get("replay_cases", 320))]

            bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in win_info["gen_bb_sd"].items()})
            head.load_state_dict({k: v.to(device=device) for k, v in win_info["gen_hd_sd"].items()})
            aux_hb = LatentFactorAuxHead(int(bb_model.config.hidden_size), factor_vocabs).to(device)
            swa_b_bb, swa_b_hd, _ = run_training_loop(
                backbone=bb_model,
                head=head,
                aux_head=aux_hb,
                train_cases=tr_hb,
                val_cases=gen_va_final,
                pad_id=pad_id,
                device=device,
                amp_dtype=amp_dtype,
                epochs=int(cfg.get("student_epochs", 8)),
                backbone_lr=2.8e-5,
                head_lr=2.2e-4,
                batch_cases=4,
                grad_accum=4,
                seed=303,
                tag=f"holdout_student_beta_{b_cand:.2f}",
                swa_start_epoch=6,
                header_attends_options=header_attends_options,
                max_seq_pad=win_max_total,
            )
            del aux_hb

            for a_cand in alpha_grid:
                m_bb = {k: (1.0 - a_cand) * swa_b_bb[k].float() + a_cand * win_info["gen_bb_sd"][k].float() for k in swa_b_bb}
                m_hd = {k: (1.0 - a_cand) * swa_b_hd[k].float() + a_cand * win_info["gen_hd_sd"][k].float() for k in swa_b_hd}
                bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in m_bb.items()})
                head.load_state_dict({k: v.to(device=device) for k, v in m_hd.items()})
                h_acc, h_kl, h_br, h_ece, _ = eval_packed_cases(
                    bb_model,
                    head,
                    holdout_cases,
                    pad_id,
                    device,
                    amp_dtype,
                    header_attends_options=header_attends_options,
                    max_seq_pad=win_max_total,
                )
                row_g = {
                    "beta_oof": b_cand,
                    "alpha_wise": a_cand,
                    "holdout_acc": round(h_acc, 6),
                    "holdout_kl": round(h_kl, 6),
                    "holdout_brier": round(h_br, 6),
                    "holdout_ece": round(h_ece, 6),
                }
                holdout_sweep_log.append(row_g)
                print(
                    f"  [Holdout Grid] beta={b_cand:.2f} alpha={a_cand:.2f} -> acc={h_acc:.4f} | KL={h_kl:.4f} | Brier={h_br:.4f}",
                    flush=True,
                )

        best_grid = select_best_holdout_grid(holdout_sweep_log)
        beta_oof = float(best_grid["beta_oof"])
        alpha_wise = float(best_grid["alpha_wise"])
        print(f"  >>> Selected on real 1/5 holdout: beta_oof={beta_oof} | alpha_wise={alpha_wise} <<<", flush=True)

        if len(student_seeds) >= 2:
            pool_cases_win = []
            for orig_i in teacher_pool_idx:
                pc = td_full_cases[orig_i]
                for q_i, qm in enumerate(pc["questions"]):
                    gp = qm["gold_probs"]
                    tp = F.softmax(inner_oof_logits[orig_i][q_i], dim=-1).tolist()
                    blended = [(1.0 - beta_oof) * g_v + beta_oof * t_v for g_v, t_v in zip(gp, tp)]
                    sb = sum(blended)
                    qm["train_probs"] = [v / sb for v in blended]
                pool_cases_win.append(pc)

            seed_bb_sds, seed_hd_sds, seed_reports = [], [], []
            for s_val in student_seeds:
                rng_sv = random.Random(s_val)
                rep_sv = list(gen_tr_final)
                rng_sv.shuffle(rep_sv)
                tr_sv = pool_cases_win + synth_final_cases + rep_sv[: int(cfg.get("replay_cases", 320))]
                bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in win_info["gen_bb_sd"].items()})
                head.load_state_dict({k: v.to(device=device) for k, v in win_info["gen_hd_sd"].items()})
                aux_sv = LatentFactorAuxHead(int(bb_model.config.hidden_size), factor_vocabs).to(device)
                s_bb, s_hd, _ = run_training_loop(
                    backbone=bb_model,
                    head=head,
                    aux_head=aux_sv,
                    train_cases=tr_sv,
                    val_cases=gen_va_final,
                    pad_id=pad_id,
                    device=device,
                    amp_dtype=amp_dtype,
                    epochs=int(cfg.get("student_epochs", 8)),
                    backbone_lr=2.8e-5,
                    head_lr=2.2e-4,
                    batch_cases=4,
                    grad_accum=4,
                    seed=s_val,
                    tag=f"holdout_soup_seed_{s_val}",
                    swa_start_epoch=6,
                    header_attends_options=header_attends_options,
                    max_seq_pad=win_max_total,
                )
                del aux_sv
                m_bb_s = {k: (1.0 - alpha_wise) * s_bb[k].float() + alpha_wise * win_info["gen_bb_sd"][k].float() for k in s_bb}
                m_hd_s = {k: (1.0 - alpha_wise) * s_hd[k].float() + alpha_wise * win_info["gen_hd_sd"][k].float() for k in s_hd}
                seed_bb_sds.append(m_bb_s)
                seed_hd_sds.append(m_hd_s)
                bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in m_bb_s.items()})
                head.load_state_dict({k: v.to(device=device) for k, v in m_hd_s.items()})
                sa, sk, sb_v, _, _ = eval_packed_cases(
                    bb_model,
                    head,
                    holdout_cases,
                    pad_id,
                    device,
                    amp_dtype,
                    header_attends_options=header_attends_options,
                    max_seq_pad=win_max_total,
                )
                seed_reports.append({"seed": s_val, "oof_accuracy": sa, "oof_kl": sk, "oof_brier": sb_v})

            soup_bb_h = average_state_dicts(seed_bb_sds)
            soup_hd_h = average_state_dicts(seed_hd_sds)
            bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in soup_bb_h.items()})
            head.load_state_dict({k: v.to(device=device) for k, v in soup_hd_h.items()})
            spa, spk, spb, _, _ = eval_packed_cases(
                bb_model,
                head,
                holdout_cases,
                pad_id,
                device,
                amp_dtype,
                header_attends_options=header_attends_options,
                max_seq_pad=win_max_total,
            )
            soup_report = {"oof_accuracy": spa, "oof_kl": spk, "oof_brier": spb}
            soup_selection_meta = {
                **select_soup_or_best_seed(seed_reports, soup_report),
                "seed_reports": seed_reports,
                "soup_report": soup_report,
            }
            print(f"  >>> 3-Seed Soup Gate on Holdout: {soup_selection_meta['selected']} ({soup_selection_meta}) <<<", flush=True)

    for i, pc in enumerate(td_full_cases):
        for qm in pc["questions"]:
            qid = qm["qid"]
            gp = qm["gold_probs"]
            tp = oof_teacher_probs[i][qid]
            blended = [(1.0 - beta_oof) * g_v + beta_oof * t_v for g_v, t_v in zip(gp, tp)]
            s_b = sum(blended)
            qm["train_probs"] = [v / s_b for v in blended]

    if soup_selection_meta.get("selected") == "soup" and len(student_seeds) >= 2:
        final_seed_list = student_seeds
    elif soup_selection_meta.get("selected") == "best_seed" and len(student_seeds) >= 2:
        final_seed_list = [student_seeds[int(soup_selection_meta["best_seed_idx"])]]
    else:
        final_seed_list = student_seeds

    skip_final_student = bool(cfg.get("skip_final_student", False))
    merged_bb_sd: dict[str, torch.Tensor] | None = None
    merged_hd_sd: dict[str, torch.Tensor] | None = None
    if not skip_final_student:
        final_bb_sds: list[dict[str, torch.Tensor]] = []
        final_hd_sds: list[dict[str, torch.Tensor]] = []
        for s_val in final_seed_list:
            rng_s = random.Random(s_val)
            replay_subset = list(gen_tr_final)
            rng_s.shuffle(replay_subset)
            joint_final_tr = td_full_cases + synth_final_cases + replay_subset[: int(cfg.get("replay_cases", 320))]

            bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in win_info["gen_bb_sd"].items()})
            head.load_state_dict({k: v.to(device=device) for k, v in win_info["gen_hd_sd"].items()})
            aux_head = LatentFactorAuxHead(int(bb_model.config.hidden_size), factor_vocabs).to(device)

            swa_bb_sd, swa_hd_sd, _ = run_training_loop(
                backbone=bb_model,
                head=head,
                aux_head=aux_head,
                train_cases=joint_final_tr,
                val_cases=gen_va_final,
                pad_id=pad_id,
                device=device,
                amp_dtype=amp_dtype,
                epochs=int(cfg.get("student_epochs", 8)),
                backbone_lr=2.8e-5,
                head_lr=2.2e-4,
                batch_cases=4,
                grad_accum=4,
                seed=s_val,
                tag=f"student_swa_full_train_s{s_val}",
                swa_start_epoch=6,
                header_attends_options=header_attends_options,
                max_seq_pad=win_max_total,
            )
            del aux_head
            final_bb_sds.append(
                {k: (1.0 - alpha_wise) * swa_bb_sd[k].float() + alpha_wise * win_info["gen_bb_sd"][k].float() for k in swa_bb_sd.keys()}
            )
            final_hd_sds.append(
                {k: (1.0 - alpha_wise) * swa_hd_sd[k].float() + alpha_wise * win_info["gen_hd_sd"][k].float() for k in swa_hd_sd.keys()}
            )

        merged_bb_sd = average_state_dicts(final_bb_sds)
        merged_hd_sd = average_state_dicts(final_hd_sds)

        bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in merged_bb_sd.items()})
        head.load_state_dict({k: v.to(device=device) for k, v in merged_hd_sd.items()})

    # Task 6 Step 2 (+ Task 10 Step 5 optional JevBench dev hard KL): Fit per-type temperature by L-BFGS
    win_oof = win_info["oof_logits_by_case"]
    if bool(cfg.get("eval_student_oof", False)):
        print(
            f"\n=== Evaluating Leak-Free 5-Fold Student OOF (seeds={final_seed_list}, alpha_wise={alpha_wise}, SWA=True) ===",
            flush=True,
        )
        for pc_rst in td_full_cases:
            for qm_rst in pc_rst["questions"]:
                qm_rst["train_probs"] = list(qm_rst["gold_probs"])

        student_oof_logits: dict[int, list[torch.Tensor]] = {}
        fold_case_splits_st = build_fold_splits_with_synth(td_full_cases, folds, synth_final_cases)
        for f_idx, ((_, va_idx), (fold_tr_st, fold_va_st)) in enumerate(zip(folds, fold_case_splits_st)):
            fold_seed_bb: list[dict[str, torch.Tensor]] = []
            fold_seed_hd: list[dict[str, torch.Tensor]] = []
            for s_val in final_seed_list:
                rng_fs = random.Random(s_val + 31 * f_idx)
                rep_fs = list(gen_tr_final)
                rng_fs.shuffle(rep_fs)
                tr_fs = fold_tr_st + rep_fs[: int(cfg.get("replay_cases", 320))]
                bb_model.load_state_dict(
                    {k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in win_info["gen_bb_sd"].items()}
                )
                head.load_state_dict({k: v.to(device=device) for k, v in win_info["gen_hd_sd"].items()})
                aux_fs = LatentFactorAuxHead(int(bb_model.config.hidden_size), factor_vocabs).to(device)
                s_bb_f, s_hd_f, _ = run_training_loop(
                    backbone=bb_model,
                    head=head,
                    aux_head=aux_fs,
                    train_cases=tr_fs,
                    val_cases=gen_va_final,
                    pad_id=pad_id,
                    device=device,
                    amp_dtype=amp_dtype,
                    epochs=int(cfg.get("student_epochs", 8)),
                    backbone_lr=2.8e-5,
                    head_lr=2.2e-4,
                    batch_cases=4,
                    grad_accum=4,
                    seed=s_val + 31 * f_idx,
                    tag=f"student_oof:fold{f_idx+1}/5:s{s_val}",
                    swa_start_epoch=int(cfg.get("swa_start_epoch", 6)),
                    header_attends_options=header_attends_options,
                    max_seq_pad=win_max_total,
                )
                del aux_fs
                fold_seed_bb.append(s_bb_f)
                fold_seed_hd.append(s_hd_f)

            fold_merged_bb = merge_wise_and_seed_soup(fold_seed_bb, win_info["gen_bb_sd"], alpha_wise=alpha_wise)
            fold_merged_hd = merge_wise_and_seed_soup(fold_seed_hd, win_info["gen_hd_sd"], alpha_wise=alpha_wise)
            bb_model.load_state_dict(
                {k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in fold_merged_bb.items()}
            )
            head.load_state_dict({k: v.to(device=device) for k, v in fold_merged_hd.items()})
            _, _, _, _, f_st_lgs = eval_packed_cases(
                bb_model,
                head,
                fold_va_st,
                pad_id,
                device,
                amp_dtype,
                header_attends_options=header_attends_options,
                max_seq_pad=win_max_total,
            )
            for loc_i, orig_i in enumerate(va_idx):
                student_oof_logits[orig_i] = f_st_lgs[loc_i]

        # Restore final full-train student weights on `bb_model` and `head` before saving!
        if merged_bb_sd is not None and merged_hd_sd is not None:
            bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in merged_bb_sd.items()})
            head.load_state_dict({k: v.to(device=device) for k, v in merged_hd_sd.items()})
        win_oof = student_oof_logits

    oof_flat_pairs: list[tuple[torch.Tensor, dict[str, Any]]] = []
    for i, pc in enumerate(td_full_cases):
        for q_i, qm in enumerate(pc["questions"]):
            oof_flat_pairs.append((win_oof[i][q_i], qm))

    jev_dev_hard_jsonl = cfg.get("jev_dev_hard_jsonl")
    if jev_dev_hard_jsonl and os.path.exists(jev_dev_hard_jsonl):
        jev_hard_cases = load_jev_dev_hard_cases(
            jev_dev_hard_jsonl,
            tok,
            max_state_tokens=max_state_infer,
            max_total_tokens=win_max_total,
        )
        if jev_hard_cases:
            _, _, _, _, jh_lgs = eval_packed_cases(
                bb_model,
                head,
                jev_hard_cases,
                pad_id,
                device,
                amp_dtype,
                header_attends_options=header_attends_options,
                max_seq_pad=win_max_total,
            )
            for pc_j, c_lgs_j in zip(jev_hard_cases, jh_lgs):
                for q_i, qm_j in enumerate(pc_j["questions"]):
                    oof_flat_pairs.append((c_lgs_j[q_i], qm_j))

    fitted_temps = fit_per_type_temperatures_lbfgs(oof_flat_pairs)
    nested_temps_by_case = fit_nested_fold_temperatures_lbfgs(win_oof, td_full_cases, folds)
    fitted_smooth = [0.0, 0.0, 0.0]

    with torch.no_grad():
        for qt_id in range(3):
            head.log_temps[qt_id] = math.log(fitted_temps[qt_id])
            head.prob_smooth[qt_id] = 0.0

    # Build calibrated 6,000-decision OOF report for `winner_key` using nested leave-one-fold-out temperatures
    per_decision: list[dict[str, Any]] = []
    for i, pc in enumerate(td_full_cases):
        ss, se = pc["state_span"]
        state_tok_len = max(0, int(se) - int(ss))
        case_temps = nested_temps_by_case[i]
        for q_i, qm in enumerate(pc["questions"]):
            qt_id = int(qm["qtype_id"])
            lv = win_oof[i][q_i]
            K = len(qm["option_keys"])
            ps = F.softmax(lv / case_temps[qt_id], dim=-1).tolist()
            eps = fitted_smooth[qt_id]
            if eps > 0.0:
                u = 1.0 / max(1, K)
                ps = [(1.0 - eps) * p + eps * u for p in ps]
            probs_map = {k: round(float(ps[idx]), 6) for idx, k in enumerate(qm["option_keys"])}
            gold_map = {k: round(float(qm["gold_probs"][idx]), 6) for idx, k in enumerate(qm["option_keys"])}
            ok, kl, br, cf = score_typed_decision_one(
                {"type": qm["qtype"], "probabilities": probs_map},
                qm["gold_raw"],
            )
            per_decision.append(
                {
                    "case_idx": i,
                    "workflow": pc["workflow"],
                    "qid": qm["qid"],
                    "qtype": qm["qtype"],
                    "state_tokens": state_tok_len,
                    "probs": probs_map,
                    "gold_probs": gold_map,
                    "correct": bool(ok),
                    "kl": round(float(kl), 6),
                    "brier": round(float(br), 6),
                    "confidence": round(float(cf), 6),
                }
            )

    oof_summary = {
        fk: {
            "backbone_id": fv["backbone_id"],
            "oof_accuracy": round(fv["oof_acc"], 6),
            "oof_kl": round(fv["oof_kl"], 6),
            "oof_brier": round(fv["oof_br"], 6),
            "oof_ece": round(fv["oof_ece"], 6),
        }
        for fk, fv in family_results.items()
    }
    oof_report = build_oof_report(
        per_decision,
        extra_meta={
            "winner_family": winner_key,
            "backbone_id": win_bb_id,
            "header_attends_options": header_attends_options,
            "beta_oof": beta_oof,
            "alpha_wise": alpha_wise,
            "student_holdout_sweep": holdout_sweep_log,
            "soup_selection": soup_selection_meta,
            "temperatures": {"noul": fitted_temps[0], "choice": fitted_temps[1], "score": fitted_temps[2]},
            "prob_smooth": {"noul": fitted_smooth[0], "choice": fitted_smooth[1], "score": fitted_smooth[2]},
            "per_family_raw_oof": oof_summary,
            "config": cfg,
        },
    )

    os.makedirs(model_dir, exist_ok=True)
    if not skip_final_student:
        s1_dir = os.path.join(model_dir, "systemone")
        if os.path.exists(s1_dir):
            shutil.rmtree(s1_dir)
        os.makedirs(s1_dir, exist_ok=True)
        tok.save_pretrained(s1_dir)
        bb_save = bb_model.to(torch.bfloat16) if device == "cuda" else bb_model
        bb_save.save_pretrained(s1_dir)

        ckpt_out = os.path.join(model_dir, "systemone_decision.pt")
        torch.save(
            {
                "arch_version": 3,
                "backbone_id": win_bb_id,
                "comp_dim": 512,
                "zs_dim": int(getattr(bb_model.config, "num_labels", 2)),
                "max_state_tokens": max_state_infer,
                "max_total_tokens": win_max_total,
                "header_attends_options": header_attends_options,
                "temperatures": {"noul": fitted_temps[0], "choice": fitted_temps[1], "score": fitted_temps[2]},
                "prob_smooth": {"noul": fitted_smooth[0], "choice": fitted_smooth[1], "score": fitted_smooth[2]},
                "schema_calib": {},
                "oof_summary": oof_summary,
                "head_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            },
            ckpt_out,
        )

    oof_out = os.path.join(model_dir, "oof_report.json")
    with open(oof_out, "w", encoding="utf-8") as ofh:
        json.dump(oof_report, ofh, indent=2)
    print(
        f"Saved OOF report -> {oof_out} "
        f"(acc={oof_report['oof_accuracy']:.4f}, KL={oof_report['oof_kl']:.4f}, Brier={oof_report['oof_brier']:.4f})",
        flush=True,
    )
    return oof_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(ROOT, "artifacts", "nullspan"), help="Output artifact directory")
    parser.add_argument("--config", default=None, help="Optional JSON training configuration file")
    parser.add_argument("--synth-td", default=None, help="Optional teacher-labelled synthetic TD JSONL")
    parser.add_argument("--jev-train", default=None, help="Optional JevBench-style Stage 1 JSONL")
    parser.add_argument("--jev-dev-hard", default=None, help="Optional JevBench dev hard JSONL for L-BFGS calibration")
    parser.add_argument("--header-attends-options", action="store_true", help="Enable tree-isolated header->option attention")
    parser.add_argument("--use-phase1-specs", action="store_true", help="Run all 6 Phase 1 candidate backbone specs")
    parser.add_argument("--sweep-student-holdout", action="store_true", help="Run Task 6 carved 1/5 holdout sweep & 3-seed soup gate")
    parser.add_argument("--eval-student-oof", action="store_true", help="Evaluate leak-free 5-fold student OOF (SWA + WiSE-FT + soup)")
    parser.add_argument("--family-cache-dir", default=None, help="Optional directory to cache Stage 1 + 5-fold OOF per backbone")
    parser.add_argument("--skip-final-student", action="store_true", help="Skip Stage 2B full-train student when only OOF report is needed")
    args = parser.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    if args.synth_td:
        cfg["synth_td_jsonl"] = args.synth_td
    if args.jev_train:
        cfg.setdefault("extra_gen_jsonl", []).append(args.jev_train)
    if args.jev_dev_hard:
        cfg["jev_dev_hard_jsonl"] = args.jev_dev_hard
    if args.header_attends_options:
        cfg["header_attends_options"] = True
    if args.use_phase1_specs:
        cfg["use_phase1_specs"] = True
    if args.sweep_student_holdout:
        cfg["sweep_student_holdout"] = True
        cfg.setdefault("student_seeds", [303, 404, 505])
    if args.eval_student_oof:
        cfg["eval_student_oof"] = True
    if args.family_cache_dir:
        cfg["family_cache_dir"] = args.family_cache_dir
    if args.skip_final_student:
        cfg["skip_final_student"] = True
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    train_systemone(args.out, device=device, config=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
