#!/usr/bin/env python3
"""System One Training & 5-Fold Out-of-Fold Cross-Validation Pipeline (Zero Test Access).

Trains `NullspanSystemOneEngine` strictly on public `train` splits and `LocalLLaMA/typed-decisions` `train`,
writing `systemone/`, `systemone_decision.pt`, and `oof_report.json`.
Zero held-out test split access.
"""

from __future__ import annotations

import argparse
import collections
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
MAX_STATE_TRAIN = 2560
MAX_STATE_INFER = 3072


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
) -> dict[str, Any]:
    pc = encode_case_parallel(
        tokenizer,
        state,
        questions,
        max_state_tokens=max_state_tokens,
        case_id=case_id,
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
            source="td",
            workflow=str(row["workflow"]),
            factor_targets=ftargs,
            max_state_tokens=max_state_tokens,
        )
        pc["orig_train_idx"] = r_idx
        cases.append(pc)
    return cases


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
                    raw_tuples.append((item["state"], item.get("qid", "decision"), item["question"], item["probabilities"]))

    rng.shuffle(raw_tuples)
    n_val = min(600, max(200, len(raw_tuples) // 10))
    val_tups = raw_tuples[:n_val]
    tr_tups = raw_tuples[n_val:]

    tr_cases = [
        _pack_single_case(tokenizer, st, {qid: q}, {qid: {"probabilities": gp}}, case_id=500000 + idx, source="gen", max_state_tokens=max_state_tokens)
        for idx, (st, qid, q, gp) in enumerate(tr_tups)
    ]
    va_cases = [
        _pack_single_case(tokenizer, st, {qid: q}, {qid: {"probabilities": gp}}, case_id=900000 + idx, source="gen", max_state_tokens=max_state_tokens)
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


@torch.no_grad()
def eval_packed_cases(
    backbone: Any,
    head: ParallelSpanDecisionHead,
    cases: list[dict[str, Any]],
    pad_id: int,
    device: str,
    amp_dtype: torch.dtype,
    batch_cases: int = 8,
) -> tuple[float, float, float, float, list[list[torch.Tensor]]]:
    backbone.eval()
    head.eval()
    accs, kls, briers, confs = [], [], [], []
    per_case_logits: list[list[torch.Tensor]] = []

    for s_i in range(0, len(cases), batch_cases):
        sub_c = cases[s_i : s_i + batch_cases]
        seq_f, zs_fn = forward_parallel_backbone(backbone, sub_c, pad_id=pad_id, device=device, amp_dtype=amp_dtype)
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
            seq_f, zs_fn = forward_parallel_backbone(backbone, sub_c, pad_id=pad_id, device=device, amp_dtype=amp_dtype)
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

            if (b_i + 1) % grad_accum == 0 or (b_i + 1) == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        v_acc, v_kl, v_br, v_ece, v_lgs = eval_packed_cases(backbone, head, val_cases, pad_id, device, amp_dtype)
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
        return (
            {k: v * inv_c for k, v in swa_bb_sum.items()},
            {k: v * inv_c for k, v in swa_hd_sum.items()},
            best_val_tuple,
        )
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


def instantiate_family(backbone_id: str, device: str, amp_dtype: torch.dtype) -> tuple[Any, Any, ParallelSpanDecisionHead]:
    from transformers import AutoConfig

    orig_cfg = AutoConfig.from_pretrained(backbone_id)
    orig_i2l = {int(k): str(v).lower() for k, v in (getattr(orig_cfg, "id2label", None) or {}).items()}
    has_nli_head = any("entail" in v for v in orig_i2l.values()) and len(orig_i2l) in (2, 3)

    orig_cfg.reference_compile = False
    orig_cfg._attn_implementation = "sdpa"
    if getattr(orig_cfg, "model_type", "") == "modernbert":
        orig_cfg.local_attention = 16384
        orig_cfg.sliding_window = 16384
        orig_cfg.global_attn_every_n_layers = 1
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
    backbones = [tuple(x) for x in cfg.get("candidate_backbones", CANDIDATE_BACKBONES)]
    max_state_train = int(cfg.get("max_state_train", MAX_STATE_TRAIN))
    max_state_infer = int(cfg.get("max_state_infer", MAX_STATE_INFER))
    beta_oof = float(cfg.get("beta_oof", 0.32))
    alpha_wise = float(cfg.get("alpha_wise", 0.04))
    synth_td_jsonl = cfg.get("synth_td_jsonl")
    extra_gen_jsonl = list(cfg.get("extra_gen_jsonl") or [])

    print("=== Training System One (5-Fold Stratified OOF, Zero Test Access) ===", flush=True)
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32)
    print(f"Device={device} | Precision={amp_dtype} | max_state_train={max_state_train}", flush=True)

    ds_train_full = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    factor_vocabs = extract_factor_vocabs(ds_train_full)
    folds = build_stratified_5folds(ds_train_full, seed=int(cfg.get("fold_seed", 42)))

    synth_rows: list[dict[str, Any]] = []
    if synth_td_jsonl and os.path.exists(synth_td_jsonl):
        with open(synth_td_jsonl, "r", encoding="utf-8") as sfh:
            synth_rows = [json.loads(line) for line in sfh if line.strip()]
        print(f"Loaded {len(synth_rows):,} synthetic teacher-labelled Typed Decisions cases from {synth_td_jsonl}", flush=True)

    family_results: dict[str, dict[str, Any]] = {}

    for fam_idx, (fam_key, bb_id) in enumerate(backbones):
        fam_seed_base = int(cfg.get("seed_base", 42)) + fam_idx * 17
        print(f"\n--- Candidate Family `{fam_key}` ({bb_id}) ---", flush=True)
        tok, bb_model, head = instantiate_family(bb_id, device, amp_dtype)
        pad_id = tok.pad_token_id or 0
        rng_fam = random.Random(fam_seed_base)

        gen_tr, gen_va = build_general_systemone_pool(
            tok, rng_fam, max_state_tokens=max_state_train, extra_jsonl_paths=extra_gen_jsonl
        )
        td_all_cases = prepare_td_cases(
            ds_train_full, tok, factor_vocabs, base_case_id=100000, max_state_tokens=max_state_train
        )
        synth_cases = (
            prepare_td_cases(synth_rows, tok, factor_vocabs, base_case_id=300000, max_state_tokens=max_state_train)
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
            batch_cases=4,
            grad_accum=4,
            seed=100 + fam_idx,
            tag=f"{fam_key}:stage1_gen",
        )
        print(f"  [Stage 1 · {fam_key} Best General] gen_val_acc={g_acc:.4f} | gen_val_kl={g_kl:.4f} | gen_val_ece={g_ece:.4f}", flush=True)

        oof_logits_by_case: dict[int, list[torch.Tensor]] = {}
        for f_idx, (tr_idx, va_idx) in enumerate(folds):
            bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in gen_bb_sd.items()})
            head.load_state_dict({k: v.to(device=device) for k, v in gen_hd_sd.items()})
            aux_head = LatentFactorAuxHead(int(bb_model.config.hidden_size), factor_vocabs).to(device)

            # Synthetic cases enter training side ONLY; OOF validation stays strictly on real train cases!
            fold_tr = [td_all_cases[i] for i in tr_idx] + synth_cases
            fold_va = [td_all_cases[i] for i in va_idx]
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
                seed=200 + fam_idx * 20 + f_idx,
                tag=f"{fam_key}:fold{f_idx+1}/5",
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
    print(f"Selected winning single-student backbone `{winner_key}` ({win_bb_id}) for Stage 2B!", flush=True)

    tok, bb_model, head = instantiate_family(win_bb_id, device, amp_dtype)
    pad_id = tok.pad_token_id or 0
    rng_final = random.Random(777)
    gen_tr_final, gen_va_final = build_general_systemone_pool(
        tok, rng_final, max_state_tokens=max_state_train, extra_jsonl_paths=extra_gen_jsonl
    )
    td_full_cases = prepare_td_cases(
        ds_train_full, tok, factor_vocabs, base_case_id=100000, max_state_tokens=max_state_train
    )
    synth_final_cases = (
        prepare_td_cases(synth_rows, tok, factor_vocabs, base_case_id=300000, max_state_tokens=max_state_train)
        if synth_rows
        else []
    )

    for i, pc in enumerate(td_full_cases):
        for qm in pc["questions"]:
            qid = qm["qid"]
            gp = qm["gold_probs"]
            tp = oof_teacher_probs[i][qid]
            blended = [(1.0 - beta_oof) * g_v + beta_oof * t_v for g_v, t_v in zip(gp, tp)]
            s_b = sum(blended)
            qm["train_probs"] = [v / s_b for v in blended]

    rng_s = random.Random(303)
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
        seed=303,
        tag="student_swa_full_train",
        swa_start_epoch=6,
    )
    del aux_head

    merged_bb_sd = {
        k: (1.0 - alpha_wise) * swa_bb_sd[k].float() + alpha_wise * win_info["gen_bb_sd"][k].float()
        for k in swa_bb_sd.keys()
    }
    merged_hd_sd = {
        k: (1.0 - alpha_wise) * swa_hd_sd[k].float() + alpha_wise * win_info["gen_hd_sd"][k].float()
        for k in swa_hd_sd.keys()
    }

    bb_model.load_state_dict({k: v.to(device=device, dtype=next(bb_model.parameters()).dtype) for k, v in merged_bb_sd.items()})
    head.load_state_dict({k: v.to(device=device) for k, v in merged_hd_sd.items()})

    # Fit per-type temperature & smoothing strictly on winning family's 6,000 OOF logits
    win_oof = win_info["oof_logits_by_case"]
    oof_flat_pairs: list[tuple[torch.Tensor, dict[str, Any]]] = []
    for i, pc in enumerate(td_full_cases):
        for q_i, qm in enumerate(pc["questions"]):
            oof_flat_pairs.append((win_oof[i][q_i], qm))

    fitted_temps = [1.0, 1.0, 1.0]
    fitted_smooth = [0.0, 0.0, 0.0]
    for qt_id in (0, 1, 2):
        sub_vl = [(lv, qm) for lv, qm in oof_flat_pairs if int(qm["qtype_id"]) == qt_id]
        if not sub_vl:
            continue
        best_t, best_eps = 1.0, 0.0
        best_cal_obj = float("inf")
        for cand_t in [0.60 + 0.05 * j for j in range(32)]:
            for cand_eps in (0.0, 0.005, 0.01):
                kls_t, brs_t = [], []
                for lv, qm in sub_vl:
                    K = len(qm["option_keys"])
                    ps = F.softmax(lv / cand_t, dim=-1).tolist()
                    if cand_eps > 0.0:
                        u = 1.0 / max(1, K)
                        ps = [(1.0 - cand_eps) * p + cand_eps * u for p in ps]
                    pred_obj = {"type": qm["qtype"], "probabilities": {k: ps[idx] for idx, k in enumerate(qm["option_keys"])}}
                    _, kl, br, _ = score_typed_decision_one(pred_obj, qm["gold_raw"])
                    kls_t.append(kl)
                    brs_t.append(br)
                cal_obj = (sum(kls_t) / len(kls_t)) + 0.45 * (sum(brs_t) / len(brs_t))
                if cal_obj < best_cal_obj:
                    best_cal_obj = cal_obj
                    best_t = round(cand_t, 3)
                    best_eps = cand_eps
        fitted_temps[qt_id] = best_t
        fitted_smooth[qt_id] = best_eps

    with torch.no_grad():
        for qt_id in range(3):
            head.log_temps[qt_id] = math.log(fitted_temps[qt_id])
            head.prob_smooth[qt_id] = fitted_smooth[qt_id]

    # Build calibrated 6,000-decision OOF report for `winner_key`
    per_decision: list[dict[str, Any]] = []
    for i, pc in enumerate(td_full_cases):
        ss, se = pc["state_span"]
        state_tok_len = max(0, int(se) - int(ss))
        for q_i, qm in enumerate(pc["questions"]):
            qt_id = int(qm["qtype_id"])
            lv = win_oof[i][q_i]
            K = len(qm["option_keys"])
            ps = F.softmax(lv / fitted_temps[qt_id], dim=-1).tolist()
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
            "temperatures": {"noul": fitted_temps[0], "choice": fitted_temps[1], "score": fitted_temps[2]},
            "prob_smooth": {"noul": fitted_smooth[0], "choice": fitted_smooth[1], "score": fitted_smooth[2]},
            "per_family_raw_oof": oof_summary,
            "config": cfg,
        },
    )

    os.makedirs(model_dir, exist_ok=True)
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
        f"Saved checkpoint -> {ckpt_out} and OOF report -> {oof_out} "
        f"(acc={oof_report['oof_accuracy']:.4f}, KL={oof_report['oof_kl']:.4f}, Brier={oof_report['oof_brier']:.4f})",
        flush=True,
    )
    return oof_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(ROOT, "artifacts", "nullspan"), help="Output artifact directory")
    parser.add_argument("--config", default=None, help="Optional JSON training configuration file")
    args = parser.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    train_systemone(args.out, device=device, config=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
