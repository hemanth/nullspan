"""End-to-End Multi-Domain System One Training & Official Public Benchmark Evaluation on GPU:
  1. Builds a strictly zero-contamination Multi-Source System One Curriculum from public `train` splits ONLY:
     - `LocalLLaMA/typed-decisions` (`split="train"`, 90/10 stratified workflow fit/val; `test` untouched)
     - `google/boolq` (`split="train"`, passage-grounded `noul` reasoning)
     - `allenai/ai2_arc` (`ARC-Challenge` `split="train"`) + `allenai/sciq` (`split="train"`, hard 4-way `choice`)
     - Programmatic System One Rubric & Ordinal `score` Generator (3/4/5/9-level ordinal rubrics, multi-clause
       policy exceptions, evidence adequacy, and tie-broken intent routing)
  2. Trains `NullspanSystemOneEngine` v2.0 (`DeBERTa-v3-Large` 435M Bidirectional Cross-Encoder + Residual NLI
     Entailment Prior + 2-Layer `CrossOptionDecisionHead` + Symbolic State Perception) in hardware TensorCores.
  3. Calibrates per-family (`noul`, `choice`, `score`) temperatures strictly on held-out validation splits.
  4. Evaluates ONCE on:
     - `LocalLLaMA/typed-decisions` (`split="test"`, 400 cases = 2,000 decisions)
     - `fstandhartinger/jevbench` / Benchmark Heaven (`Original 72`, `Easy 48`, `Hard 111` = 231 public items
       over live HTTP `/v1/systemone` with `TypeSafeAdapter` and official `$0.01/1M` encoder tariff)
     - `rajpurkar/squad_v2` (`split="validation"`, N = 500 zero-overlap items)
"""
from __future__ import annotations

import collections
import json
import math
import os
import random
import re
import string
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))

from modeling_nullspan import NullspanForExtraction  # noqa: E402
from systemone_decision import (  # noqa: E402
    CrossOptionDecisionHead,
    NullspanSystemOneEngine,
    compact_state_text,
    encode_decision_question,
    inject_encoder_lora,
)

BACKBONE_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MAX_LEN_TRAIN = 416
MAX_LEN_INFER = 512


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


def _make_item(
    tokenizer: Any,
    state: Any,
    qid: str,
    question: dict[str, Any],
    gold_probs_map: dict[str, float],
    peer_q_summary: str = "",
    source: str = "general",
    case_id: int = 0,
    precomputed_state_str: str | None = None,
) -> dict[str, Any]:
    enc = encode_decision_question(
        tokenizer,
        state,
        qid,
        question,
        peer_q_summary=peer_q_summary,
        max_length=MAX_LEN_TRAIN,
        precomputed_state_str=precomputed_state_str,
    )
    keys = enc["option_keys"]
    gp = [float(gold_probs_map.get(k, 0.0)) for k in keys]
    s = sum(gp)
    gp = [v / s for v in gp] if s > 0 else [1.0 / len(keys)] * len(keys)
    enc["gold_probs"] = gp
    enc["gold_raw"] = {"type": question["type"], "probabilities": {k: gp[i] for i, k in enumerate(keys)}}
    enc["source"] = source
    enc["case_id"] = case_id
    return enc


def prepare_td_split(rows: Any, tokenizer: Any, base_case_id: int = 100000) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for r_idx, row in enumerate(rows):
        cid = base_case_id + r_idx
        state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
        qs = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        gs = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        state_str = compact_state_text(state)
        peer_summary = ", ".join(f"{qk}({qs[qk].get('type')})" for qk in qs) if len(qs) > 1 else ""
        for qid, q in qs.items():
            gq = gs[qid]
            items.append(
                _make_item(
                    tokenizer,
                    state,
                    qid,
                    q,
                    gq["probabilities"],
                    peer_q_summary=peer_summary,
                    source="td",
                    case_id=cid,
                    precomputed_state_str=state_str,
                )
            )
    return items


def generate_synthetic_systemone_rubrics(rng: random.Random, n_total: int = 1600) -> list[tuple[Any, str, dict[str, Any], dict[str, float]]]:
    """Generate 8 zero-contamination System One reasoning families (ordinal, long-policy exceptions, epistemic adequacy, decoy traps, temporal math, multi-hop, ambiguous ties, and EV trade-offs)."""
    out: list[tuple[Any, str, dict[str, Any], dict[str, float]]] = []

    for i in range(n_total):
        kind = i % 8
        if kind == 0:
            # 9-level and 5-level ordinal score rubrics (unimodal soft Gaussian target)
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
            q = {
                "type": "score",
                "instructions": "Rate the operational incident severity strictly on the provided rubric scale.",
                "criteria": crit,
            }
            sigma = 0.65 if n_levels == 9 else 0.50
            raw_p = {str(lv): math.exp(-((lv - gold_lvl) ** 2) / (2 * sigma * sigma)) for lv in range(n_levels)}
            out.append((state, "decision", q, raw_p))

        elif kind == 1:
            # Long multi-section policy compliance with late overriding carveouts (long_policy + policy)
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

            filler = (
                "Section 1 (Scope & Definitions): All commercial disbursements are logged in UTC and reconciled monthly across regional ledgers. "
                "Standard operational SLAs govern response latency for ticket acknowledgments, independent of refund eligibility.\n"
                "Section 2 (General Commercial Terms): Standard requests within 30 calendar days qualify for full_refund up to $1,000 USD, "
                "or manager_escalation when exceeding $1,000 USD. Requests between 31 and 60 days for pro or enterprise tiers qualify for account_credit_only. "
                "Requests beyond 60 days (or free tier >30 days) default to deny_outside_window.\n"
                "Section 3 (Audit & Recordkeeping): Every transaction record must retain immutable audit hashes for seven years under corporate retention rules. "
                "Currency conversions apply the spot treasury midpoint at settlement.\n"
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
            q = {
                "type": "choice",
                "instructions": "Determine the governing policy disposition after applying all sections and overriding exceptions.",
                "criteria": crit_map,
            }
            probs = {k: (0.90 if k == gold_k else 0.10 / 4.0) for k in crit_map}
            out.append((filler, "decision", q, probs))

        elif kind == 2:
            # Epistemic adequacy & insufficient_information vs tempting partial-match action (adequacy + ambiguous)
            has_income = rng.random() > 0.30
            has_debt = rng.random() > 0.30
            has_fico = rng.random() > 0.25
            inc = rng.randint(55000, 185000) if has_income else None
            debt = rng.randint(400, 4800) if has_debt else None
            fico = rng.randint(620, 815) if has_fico else None
            sufficient = has_income and has_debt and has_fico

            state_s = (
                f"Underwriting Rule: Approve iff ALL THREE verified metrics are present AND annual DTI (12 * monthly_debt / annual_income) <= 36% AND fico >= 680. "
                f"Deny if present metrics already violate FICO < 680 or DTI > 36%. Request missing documentation ONLY when eligibility cannot yet be determined.\n"
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
            q = {
                "type": "choice",
                "instructions": "Select the epistemically valid underwriting decision without guessing missing values.",
                "criteria": crit_map,
            }
            probs = {k: (0.90 if k == gold_c else 0.05) for k in crit_map}
            out.append((state_s, "decision", q, probs))

        elif kind == 3:
            # Adversarial / lexical decoy traps (negated incidents, staging vs prod, superseded drafts)
            env = rng.choice(["staging-sandbox", "production", "qa-replay"])
            negated = rng.choice([True, False])
            if env != "production":
                txt = (
                    f"Environment: {env} (NON-PRODUCTION synthetic drill). "
                    f"Simulated alert: 14,000 test records exported during scheduled chaos exercise; zero customer production systems touched."
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
            # Temporal calendar & clock window calculations (temporal_numeric)
            start_day = rng.randint(1, 12)
            delta_days = rng.choice([5, 13, 14, 15, 29, 30, 31, 45])
            window_days = rng.choice([14, 30])
            d1 = f"2026-03-{start_day:02d}"
            end_dt_num = start_day + delta_days
            if end_dt_num <= 31:
                d2 = f"2026-03-{end_dt_num:02d}"
            else:
                d2 = f"2026-04-{end_dt_num - 31:02d}"
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
            # Multi-hop entity resolution chain (multi_hop)
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
            # Tie-broken multi-intent routing + conflicting/ambiguous escalation (routing_hard)
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
            q = {
                "type": "choice",
                "instructions": "Select the exact routing queue strictly following priority rules (1)-(3).",
                "criteria": crit_map,
            }
            probs = {k: (0.90 if k == gold_r else 0.025) for k in crit_map}
            out.append((state, "decision", q, probs))

        else:
            # Quantitative expected-value & risk trade-off selection (tradeoff + tool_selection)
            p_fail = rng.choice([0.02, 0.08, 0.25, 0.45])
            outage_cost = rng.choice([10000, 50000, 200000])
            hotfix_cost = 3500
            canary_cost = 900
            exp_loss = p_fail * outage_cost
            if exp_loss > 15000:
                gold_t = "full_rollback_and_block"
            elif exp_loss >= 2500:
                gold_t = "staged_canary_mitigation"
            else:
                gold_t = "monitor_next_window"
            txt = (
                f"Release Telemetry: failure_probability={p_fail:.2f}, outage_impact_usd=${outage_cost:,} "
                f"(expected_unmitigated_loss=${exp_loss:,.0f}), canary_cost_usd=${canary_cost}, full_rollback_cost_usd=${hotfix_cost}.\n"
                "Decision Rule: Choose full_rollback_and_block if expected_unmitigated_loss > $15,000; "
                "choose staged_canary_mitigation if expected_unmitigated_loss is between $2,500 and $15,000 inclusive; "
                "otherwise choose monitor_next_window."
            )
            crit_map = {
                "full_rollback_and_block": "Expected unmitigated loss exceeds $15,000.",
                "staged_canary_mitigation": "Expected unmitigated loss is between $2,500 and $15,000 inclusive.",
                "monitor_next_window": "Expected unmitigated loss is strictly below $2,500.",
            }
            q = {
                "type": "choice",
                "instructions": "Select the optimal mitigation action under the expected-loss decision rule.",
                "criteria": crit_map,
            }
            probs = {k: (0.91 if k == gold_t else 0.045) for k in crit_map}
            out.append((txt, "decision", q, probs))

    return out


def build_general_systemone_pool(tokenizer: Any, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load strictly `split='train'` public reasoning datasets + synthetic System One rubrics."""
    raw_tuples: list[tuple[Any, str, dict[str, Any], dict[str, float]]] = []

    # 1. google/boolq (train split ONLY)
    boolq = load_dataset("google/boolq", split="train")
    bq_idx = list(range(len(boolq)))
    rng.shuffle(bq_idx)
    for idx in bq_idx[:450]:
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

    # 2. allenai/ai2_arc (ARC-Challenge train split ONLY) + allenai/sciq (train split ONLY)
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_idx = list(range(len(arc)))
    rng.shuffle(arc_idx)
    for idx in arc_idx[:450]:
        r = arc[idx]
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        ak = str(r["answerKey"])
        if ak not in labels or len(labels) < 3:
            continue
        crit = {lab: txt for lab, txt in zip(labels, texts)}
        q = {
            "type": "choice",
            "instructions": "Select the most scientifically and logically sound answer.",
            "criteria": crit,
        }
        rem = 0.14 / max(1, len(labels) - 1)
        probs = {lab: (0.86 if lab == ak else rem) for lab in labels}
        raw_tuples.append((str(r["question"]), "decision", q, probs))

    sciq = load_dataset("allenai/sciq", split="train")
    sq_idx = list(range(len(sciq)))
    rng.shuffle(sq_idx)
    for idx in sq_idx[:400]:
        r = sciq[idx]
        opts_txt = [str(r["correct_answer"]), str(r["distractor1"]), str(r["distractor2"]), str(r["distractor3"])]
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        labs = ["A", "B", "C", "D"]
        crit = {labs[pos]: opts_txt[orig_i] for pos, orig_i in enumerate(order)}
        gold_lab = labs[order.index(0)]
        support = str(r.get("support") or "").strip()
        state_s = f"Reference Context: {support}\nQuestion: {r['question']}" if support else str(r["question"])
        q = {
            "type": "choice",
            "instructions": "Choose the single accurate answer supported by the context and scientific principles.",
            "criteria": crit,
        }
        probs = {lab: (0.89 if lab == gold_lab else 0.11 / 3.0) for lab in labs}
        raw_tuples.append((state_s, "decision", q, probs))

    # 3. Programmatic System One Rubrics & Ordinal Scores
    raw_tuples.extend(generate_synthetic_systemone_rubrics(rng, n_total=1600))

    rng.shuffle(raw_tuples)
    n_val = 400
    val_tups = raw_tuples[:n_val]
    tr_tups = raw_tuples[n_val:]

    tr_items = [_make_item(tokenizer, st, qid, q, gp, source="gen", case_id=500000 + idx) for idx, (st, qid, q, gp) in enumerate(tr_tups)]
    va_items = [_make_item(tokenizer, st, qid, q, gp, source="gen", case_id=900000 + idx) for idx, (st, qid, q, gp) in enumerate(val_tups)]
    return tr_items, va_items


def forward_questions_batch(
    nli_model: Any,
    head: CrossOptionDecisionHead,
    sub: list[dict[str, Any]],
    pad_id: int,
    device: str,
    amp_dtype: torch.dtype,
    apply_temp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one batch of questions (`B` questions -> `P = sum(K_b)` pairs) through DeBERTa + CrossOptionDecisionHead."""
    flat_ids: list[list[int]] = []
    flat_attn: list[list[int]] = []
    flat_prem: list[list[float]] = []
    flat_hyp: list[list[float]] = []
    q_slices: list[tuple[int, int]] = []

    for x in sub:
        s_i = len(flat_ids)
        flat_ids.extend(x["pair_ids"])
        flat_attn.extend(x["pair_attn"])
        flat_prem.extend(x["pair_prem"])
        flat_hyp.extend(x["pair_hyp"])
        q_slices.append((s_i, len(flat_ids)))

    P = len(flat_ids)
    raw_max_L = max(len(ids) for ids in flat_ids)
    pad_L = min(MAX_LEN_TRAIN, max(64, ((raw_max_L + 31) // 32) * 32))

    ids_cpu = [ids[:pad_L] + [pad_id] * (pad_L - min(len(ids), pad_L)) for ids in flat_ids]
    attn_cpu = [at[:pad_L] + [0] * (pad_L - min(len(at), pad_L)) for at in flat_attn]
    prem_cpu = [pm[:pad_L] + [0.0] * (pad_L - min(len(pm), pad_L)) for pm in flat_prem]
    hyp_cpu = [hm[:pad_L] + [0.0] * (pad_L - min(len(hm), pad_L)) for hm in flat_hyp]

    ids_t = torch.tensor(ids_cpu, dtype=torch.long, device=device)
    attn_t = torch.tensor(attn_cpu, dtype=torch.long, device=device)
    prem_t = torch.tensor(prem_cpu, dtype=torch.float32, device=device)
    hyp_t = torch.tensor(hyp_cpu, dtype=torch.float32, device=device)

    seq_chunks: list[torch.Tensor] = []
    nli_chunks: list[torch.Tensor] = []
    for c_s in range(0, P, 32):
        ids_c = ids_t[c_s : c_s + 32]
        attn_c = attn_t[c_s : c_s + 32]
        if hasattr(nli_model, "deberta"):
            enc_out = nli_model.deberta(input_ids=ids_c, attention_mask=attn_c)
            seq_c = enc_out.last_hidden_state.float()
            pooled_c = nli_model.pooler(seq_c)
            nli_c = nli_model.classifier(pooled_c).float()
        else:
            use_cuda_amp = (device == "cuda") and (amp_dtype == torch.float16)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_cuda_amp):
                out = nli_model(input_ids=ids_c, attention_mask=attn_c, output_hidden_states=True)
                seq_c = out.hidden_states[-1].float()
                nli_c = out.logits.float()
        seq_chunks.append(seq_c)
        nli_chunks.append(nli_c)
    seq_f = torch.cat(seq_chunks, dim=0)
    nli_f = torch.cat(nli_chunks, dim=0)
    pair_vecs_flat, nli_priors_flat = head.pool_pair_features(seq_f, nli_f, prem_t, hyp_t)

    B = len(sub)
    max_K = max(len(x["option_keys"]) for x in sub)
    feat_dim = pair_vecs_flat.size(-1)

    pv_g = torch.zeros((B, max_K, feat_dim), dtype=torch.float32, device=device)
    nl_g = torch.zeros((B, max_K, 3), dtype=torch.float32, device=device)
    np_g = torch.zeros((B, max_K), dtype=torch.float32, device=device)
    om_g = torch.zeros((B, max_K, 4), dtype=torch.float32, device=device)
    gp_g = torch.zeros((B, max_K), dtype=torch.float32, device=device)
    vk_g = torch.zeros((B, max_K), dtype=torch.bool, device=device)
    qt_g = torch.zeros((B,), dtype=torch.long, device=device)
    cid_g = torch.zeros((B,), dtype=torch.long, device=device)

    for b_i, x in enumerate(sub):
        s_i, e_i = q_slices[b_i]
        K = e_i - s_i
        pv_g[b_i, :K] = pair_vecs_flat[s_i:e_i]
        nl_g[b_i, :K] = nli_f[s_i:e_i]
        np_g[b_i, :K] = nli_priors_flat[s_i:e_i]
        om_g[b_i, :K] = torch.tensor(x["opt_meta"], dtype=torch.float32, device=device)
        gp_g[b_i, :K] = torch.tensor(x["gold_probs"], dtype=torch.float32, device=device)
        vk_g[b_i, :K] = True
        qt_g[b_i] = x["qtype_id"]
        cid_g[b_i] = int(x.get("case_id", b_i))

    logits = head.score_grouped_options(pv_g, nl_g, np_g, om_g, vk_g, qt_g, case_ids=cid_g, apply_temp=apply_temp)
    return logits, gp_g, vk_g, qt_g


def group_into_microbatches(items: list[dict[str, Any]], max_q: int = 5) -> list[list[dict[str, Any]]]:
    """Group items by `case_id` so multi-question workflows stay intact inside a single micro-batch."""
    by_cid: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for it in items:
        by_cid[int(it.get("case_id", id(it)))].append(it)
    mbs: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for grp in by_cid.values():
        if len(grp) >= max_q:
            if cur:
                mbs.append(cur)
                cur = []
            mbs.append(grp)
        else:
            if len(cur) + len(grp) > max_q and cur:
                mbs.append(cur)
                cur = []
            cur.extend(grp)
    if cur:
        mbs.append(cur)
    return mbs


@torch.no_grad()
def eval_split_items(
    nli_model: Any,
    head: CrossOptionDecisionHead,
    items: list[dict[str, Any]],
    pad_id: int,
    device: str,
    amp_dtype: torch.dtype,
    batch_q: int = 10,
) -> tuple[float, float, float, float, list[tuple[torch.Tensor, dict[str, Any]]]]:
    nli_model.eval()
    head.eval()
    accs, kls, briers, confs = [], [], [], []
    raw_logits_out: list[tuple[torch.Tensor, dict[str, Any]]] = []
    for sub in group_into_microbatches(items, max_q=batch_q):
        logits, _, _, _ = forward_questions_batch(nli_model, head, sub, pad_id, device, amp_dtype, apply_temp=False)
        for b_i, item in enumerate(sub):
            K = len(item["option_keys"])
            lv = logits[b_i, :K].detach().cpu()
            raw_logits_out.append((lv, item))
            ps = F.softmax(lv, dim=-1).tolist()
            pred_obj = {"type": item["qtype"], "probabilities": {k: ps[idx] for idx, k in enumerate(item["option_keys"])}}
            ok, kl, brier, conf = score_typed_decision_one(pred_obj, item["gold_raw"])
            accs.append(ok)
            kls.append(kl)
            briers.append(brier)
            confs.append(conf)
    return (
        sum(accs) / len(accs),
        sum(kls) / len(kls),
        sum(briers) / len(briers),
        compute_ece_10bin(accs, confs),
        raw_logits_out,
    )


def train_and_eval_typed_decisions(model_dir: str, device: str) -> dict[str, Any]:
    print("=== [1/3] Training Multi-Domain System One Engine v2.0 & Evaluating Typed Decisions ===", flush=True)
    use_bf16_backbone = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16_backbone else torch.float32
    use_scaler = False
    print(f"Using backbone={BACKBONE_ID} | precision={amp_dtype} (bf16_deberta={use_bf16_backbone})", flush=True)

    s1_dir = os.path.join(model_dir, "systemone")
    if os.path.exists(os.path.join(s1_dir, "config.json")):
        tok = AutoTokenizer.from_pretrained(s1_dir)
        nli_model = AutoModelForSequenceClassification.from_pretrained(s1_dir)
    else:
        os.makedirs(s1_dir, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(BACKBONE_ID)
        nli_model = AutoModelForSequenceClassification.from_pretrained(BACKBONE_ID)
        tok.save_pretrained(s1_dir)
        nli_model.save_pretrained(s1_dir)

    if use_bf16_backbone and hasattr(nli_model, "deberta"):
        nli_model.deberta.to(torch.bfloat16)

    for name, p in nli_model.named_parameters():
        p.requires_grad = name.startswith("pooler.") or name.startswith("classifier.")

    start_layer = 14
    lora_r = 32
    lora_alpha = 64.0
    comp_dim = 512
    inject_encoder_lora(nli_model, start_layer=start_layer, r=lora_r, alpha=lora_alpha)
    head = CrossOptionDecisionHead(hidden_size=int(nli_model.config.hidden_size), comp_dim=comp_dim, dropout=0.1)

    nli_model.to(device)
    head.to(device)

    rng = random.Random(42)
    ds_train_full = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    by_wf: dict[str, list[int]] = collections.defaultdict(list)
    for idx, r in enumerate(ds_train_full):
        by_wf[r["workflow"]].append(idx)
    tr_idx, va_idx = [], []
    for wf, indices in sorted(by_wf.items()):
        shuf = list(indices)
        rng.shuffle(shuf)
        n_va = max(1, int(0.10 * len(shuf)))
        va_idx.extend(shuf[:n_va])
        tr_idx.extend(shuf[n_va:])

    print("Encoding Typed Decisions train/val + Multi-Domain General System One train/val...", flush=True)
    td_tr = prepare_td_split(ds_train_full.select(tr_idx), tok, base_case_id=100000)
    td_va = prepare_td_split(ds_train_full.select(va_idx), tok, base_case_id=300000)
    gen_tr, gen_va = build_general_systemone_pool(tok, rng)

    pad_id = tok.pad_token_id or 0

    # Log Step-0 Zero-Shot NLI Entailment Prior accuracy before any gradient update!
    z_td_acc, z_td_kl, _, z_td_ece, _ = eval_split_items(nli_model, head, td_va, pad_id, device, amp_dtype)
    z_gn_acc, z_gn_kl, _, z_gn_ece, _ = eval_split_items(nli_model, head, gen_va, pad_id, device, amp_dtype)
    print(
        f"  [Step-0 Zero-Shot NLI Prior] td_val_acc={z_td_acc:.4f} (kl={z_td_kl:.4f}, ece={z_td_ece:.4f}) "
        f"| gen_val_acc={z_gn_acc:.4f} (kl={z_gn_kl:.4f}, ece={z_gn_ece:.4f})",
        flush=True,
    )

    joint_tr = td_tr + gen_tr
    train_mbs = group_into_microbatches(joint_tr, max_q=5)
    print(f"Joint training pool: {len(joint_tr):,} decisions ({len(train_mbs):,} workflow-aware micro-batches)", flush=True)

    trainable_params = [p for p in list(nli_model.parameters()) + list(head.parameters()) if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}", flush=True)

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    epochs = 4
    grad_accum = 4
    steps_per_epoch = math.ceil(len(train_mbs) / grad_accum)
    total_steps = epochs * steps_per_epoch
    opt = torch.optim.AdamW(trainable_params, lr=3.2e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3.2e-4, total_steps=total_steps, pct_start=0.12)

    best_obj = float("inf")
    best_lora_sd: dict[str, torch.Tensor] = {}
    best_cls_sd: dict[str, torch.Tensor] = {}
    best_head_sd: dict[str, torch.Tensor] = {}

    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        nli_model.train()
        # Keep frozen bottom layers deterministic & zero-dropout so forward is fast and matches eval
        if hasattr(nli_model, "deberta"):
            nli_model.deberta.embeddings.eval()
            for l_idx in range(start_layer):
                nli_model.deberta.encoder.layer[l_idx].eval()
        head.train()
        rng.shuffle(train_mbs)
        tr_losses = []
        opt.zero_grad(set_to_none=True)
        for mb_i, sub in enumerate(train_mbs):
            logits, gp_g, vk_g, qt_g = forward_questions_batch(nli_model, head, sub, pad_id, device, amp_dtype, apply_temp=False)
            masked_logits = logits.masked_fill(~vk_g, -1e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            probs = log_probs.exp()
            # Soft cross-entropy + Brier + Ordinal Cumulative Ranked Probability Score (RPS) on `score` scales
            ce_loss = -(gp_g * log_probs).sum(dim=-1).mean()
            brier_loss = ((probs - gp_g).pow(2) * vk_g.float()).sum(dim=-1).mean()
            cdf_diff = (torch.cumsum(probs * vk_g.float(), dim=-1) - torch.cumsum(gp_g * vk_g.float(), dim=-1)).pow(2)
            is_score_mask = (qt_g == 2).float()
            rps_loss = (cdf_diff.sum(dim=-1) * is_score_mask).mean()
            raw_loss = ce_loss + 0.25 * brier_loss + 0.40 * rps_loss
            loss = raw_loss / grad_accum

            scaler.scale(loss).backward()
            tr_losses.append(float(raw_loss.item()))

            if (mb_i + 1) % grad_accum == 0 or (mb_i + 1) == len(train_mbs):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            if (mb_i + 1) % 250 == 0:
                recent_l = sum(tr_losses[-250:]) / len(tr_losses[-250:])
                print(f"    [ep {ep}/{epochs} · mb {mb_i+1}/{len(train_mbs)} · {time.perf_counter()-t_ep:.1f}s] loss={recent_l:.4f}", flush=True)

        td_acc, td_kl, td_br, td_ece, _ = eval_split_items(nli_model, head, td_va, pad_id, device, amp_dtype)
        gn_acc, gn_kl, gn_br, gn_ece, _ = eval_split_items(nli_model, head, gen_va, pad_id, device, amp_dtype)
        joint_obj = 0.60 * td_kl + 0.40 * gn_kl
        print(
            f"  Epoch {ep}/{epochs} ({time.perf_counter()-t_ep:.1f}s) | loss={sum(tr_losses)/len(tr_losses):.4f} "
            f"| td_val_acc={td_acc:.4f} (kl={td_kl:.4f}, br={td_br:.4f}, ece={td_ece:.4f}) "
            f"| gen_val_acc={gn_acc:.4f} (kl={gn_kl:.4f}, ece={gn_ece:.4f})",
            flush=True,
        )
        if joint_obj < best_obj:
            best_obj = joint_obj
            best_lora_sd = {k: v.detach().cpu().clone() for k, v in nli_model.state_dict().items() if "lora_" in k}
            best_cls_sd = {
                k: v.detach().cpu().clone()
                for k, v in nli_model.state_dict().items()
                if k.startswith("pooler.") or k.startswith("classifier.")
            }
            best_head_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    nli_model.load_state_dict(best_lora_sd, strict=False)
    nli_model.load_state_dict(best_cls_sd, strict=False)
    head.load_state_dict(best_head_sd)

    # Fit per-family (noul=0, choice=1, score=2) post-hoc calibration temperatures on held-out val splits
    _, _, _, _, raw_td_va = eval_split_items(nli_model, head, td_va, pad_id, device, amp_dtype)
    _, _, _, _, raw_gn_va = eval_split_items(nli_model, head, gen_va, pad_id, device, amp_dtype)
    all_val_logits = raw_td_va + raw_gn_va

    fitted_temps = [1.0, 1.0, 1.0]
    for qt_id in (0, 1, 2):
        sub_vl = [(lv, it) for lv, it in all_val_logits if it["qtype_id"] == qt_id]
        if not sub_vl:
            continue
        best_t = 1.0
        best_t_obj = float("inf")
        for cand_t in [0.55 + 0.05 * i for i in range(48)]:
            accs_t, kls_t, brs_t, confs_t = [], [], [], []
            for lv, it in sub_vl:
                ps = F.softmax(lv / cand_t, dim=-1).tolist()
                pred_obj = {"type": it["qtype"], "probabilities": {k: ps[idx] for idx, k in enumerate(it["option_keys"])}}
                ok, kl, br, cf = score_typed_decision_one(pred_obj, it["gold_raw"])
                accs_t.append(ok)
                kls_t.append(kl)
                brs_t.append(br)
                confs_t.append(cf)
            ece_t = compute_ece_10bin(accs_t, confs_t)
            t_obj = 0.50 * (sum(kls_t) / len(kls_t)) + 0.75 * ece_t + 0.25 * (sum(brs_t) / len(brs_t))
            if t_obj < best_t_obj:
                best_t_obj = t_obj
                best_t = round(cand_t, 3)
        fitted_temps[qt_id] = best_t

    with torch.no_grad():
        for qt_id in range(3):
            head.log_temps[qt_id] = math.log(fitted_temps[qt_id])
    best_head_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    ckpt_out = os.path.join(model_dir, "systemone_decision.pt")
    torch.save(
        {
            "backbone_id": BACKBONE_ID,
            "start_layer": start_layer,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "comp_dim": comp_dim,
            "max_length": MAX_LEN_INFER,
            "temperatures": {"noul": fitted_temps[0], "choice": fitted_temps[1], "score": fitted_temps[2]},
            "lora_state_dict": best_lora_sd,
            "classifier_state_dict": best_cls_sd,
            "head_state_dict": best_head_sd,
        },
        ckpt_out,
    )
    print(f"Saved calibrated System One v2.0 checkpoint ({os.path.getsize(ckpt_out)/1e6:.2f} MB, temps={fitted_temps}) -> {ckpt_out}", flush=True)

    del nli_model, head, opt, sched, trainable_params
    if device == "cuda":
        torch.cuda.empty_cache()

    # Final single pass on held-out `test` split (400 cases, 2,000 decisions)
    engine = NullspanSystemOneEngine.load_from_dir(model_dir, device=device)
    ds_test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    accs, kls, briers, confs = [], [], [], []
    by_wf_acc: dict[str, list[bool]] = collections.defaultdict(list)
    by_type_acc: dict[str, list[bool]] = collections.defaultdict(list)

    t0 = time.perf_counter()
    for raw in ds_test:
        state = json.loads(raw["state"])
        questions = json.loads(raw["questions"])
        gold = json.loads(raw["gold"])
        resp = engine.decide(state=state, questions=questions)
        answers = resp["answers"]
        for qid, gq in gold.items():
            ok, kl, brier, conf = score_typed_decision_one(answers.get(qid, {"probabilities": {}}), gq)
            accs.append(ok)
            kls.append(kl)
            briers.append(brier)
            confs.append(conf)
            by_wf_acc[raw["workflow"]].append(ok)
            by_type_acc[gq["type"]].append(ok)

    elapsed = time.perf_counter() - t0
    n = len(accs)
    ece = compute_ece_10bin(accs, confs)
    summary = {
        "model": "gnumanth/nullspan",
        "mode": "specialist",
        "n_cases": len(ds_test),
        "n_decisions": n,
        "accuracy": round(sum(accs) / n, 4),
        "kl_from_gold": round(sum(kls) / n, 4),
        "brier": round(sum(briers) / n, 4),
        "ece": round(ece, 4),
        "latency_ms_per_case": round(1000.0 * elapsed / len(ds_test), 2),
        "by_workflow_accuracy": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_wf_acc.items())},
        "by_type_accuracy": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_type_acc.items())},
    }
    print("Typed Decisions Official Test Summary:", json.dumps(summary, indent=2), flush=True)
    del engine
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary


def _ece_val(x: Any) -> float:
    if isinstance(x, dict):
        return float(x.get("ece") or 0.0)
    return float(x or 0.0)


def eval_jevbench_public(model_dir: str, jevbench_dir: str, device: str) -> dict[str, Any]:
    print("=== [2/3] Evaluating on fstandhartinger/jevbench (Benchmark Heaven Public Suite) ===", flush=True)
    sys.path.insert(0, jevbench_dir)
    from jevbench import composite_v13, composite_v14
    from jevbench.adapters.typesafe import TypeSafeAdapter
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize
    from jevbench.tasks import load_jsonl

    port = 18011
    proc = subprocess.Popen(
        [
            sys.executable,
            os.path.join(ROOT, "training", "serve_systemone.py"),
            "--model-dir",
            model_dir,
            "--port",
            str(port),
            "--device",
            device,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ready = False
        boot_lines: list[str] = []
        for _ in range(180):
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                boot_lines.append(line.strip())
            if "serving" in line:
                ready = True
                break
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        if not ready:
            raise RuntimeError("serve_systemone.py did not become ready:\n" + "\n".join(boot_lines[-20:]))

        # Official JevBench `encoder_classifier_<=0.6B` tariff ($0.01 / 1M input, $0.00 / 1M output)
        adapter = TypeSafeAdapter(
            endpoint=f"http://127.0.0.1:{port}",
            model="gnumanth/nullspan",
            key_env="",
            price_input_per_m=0.01,
            price_output_per_m=0.0,
        )
        warm_tasks = load_jsonl(os.path.join(jevbench_dir, "datasets", "public", "easy.jsonl"))
        for wt in warm_tasks[:3]:
            adapter.run(wt)

        tier_summaries: dict[str, Any] = {}
        all_tasks = []
        all_records = []
        hard_ece = 0.0

        for tier, key_v13 in (("original", "standard"), ("easy", "easy"), ("hard", "hard")):
            tpath = os.path.join(jevbench_dir, "datasets", "public", f"{tier}.jsonl")
            tasks = load_jsonl(tpath)
            records = []
            for t in tasks:
                dr = adapter.run(t)
                sc = (
                    score_task(dr.probs, t)
                    if dr.ok and dr.probs is not None
                    else {"valid": False, "strict_valid": False, "correct": False, "probs": None}
                )
                in_toks = (dr.usage or {}).get("input_tokens", 250)
                rec = {
                    "task_id": t.id,
                    "ok": dr.ok,
                    "valid": sc.get("valid", False),
                    "strict_valid": sc.get("strict_valid", False),
                    "renormalized": sc.get("renormalized", False),
                    "correct": sc.get("correct", False),
                    "predicted": sc.get("predicted"),
                    "probs": sc.get("probs"),
                    "probs_source": "native",
                    "model": "gnumanth/nullspan",
                    "latency_s": dr.latency_s,
                    "cost_usd": in_toks * 0.01 / 1e6,
                    "cost_basis": "encoder_classifier_<=0.6B",
                }
                records.append(rec)
            tier_sum = summarize(tasks, records)
            t_ece = _ece_val(tier_sum["ece"])
            if tier == "hard":
                hard_ece = t_ece
            tier_summaries[tier] = {
                "n": tier_sum["n_scorable"],
                "correct": tier_sum["n_correct"],
                "accuracy": round(float(tier_sum["accuracy"]), 4),
                "strict_valid_rate": round(float(tier_sum["schema_validity_strict"]), 4),
                "brier": round(float(tier_sum["brier_mean"]), 4),
                "ece": round(t_ece, 4),
                "p50_ms": round(float(tier_sum["latency"]["p50_s"]) * 1000.0, 2),
                "p95_ms": round(float(tier_sum["latency"]["p95_s"]) * 1000.0, 2),
            }
            all_tasks.extend(tasks)
            all_records.extend(records)

        full_sum = summarize(all_tasks, all_records)
        intel_v13 = composite_v13.intelligence(
            {
                "easy": tier_summaries["easy"]["accuracy"],
                "standard": tier_summaries["original"]["accuracy"],
                "hard": tier_summaries["hard"]["accuracy"],
            }
        )
        cal_v13 = composite_v13.calibration(hard_ece)
        p50_s = float(full_sum["latency"]["p50_s"])
        p95_s = float(full_sum["latency"]["p95_s"])
        spd_v13 = composite_v13.speed(p50_s, p95_s, "gpu")
        usd_1k = float(full_sum["price_per_1000_decisions_usd"])
        cst_v13 = composite_v13.cost(usd_1k)
        axes_public = {
            "intelligence": round(intel_v13, 2),
            "calibration": round(cal_v13, 2),
            "speed": round(spd_v13, 2),
            "cost": round(cst_v13, 2),
        }
        overall = {
            "total_public_items": full_sum["n_scorable"],
            "total_correct": full_sum["n_correct"],
            "overall_accuracy": round(float(full_sum["accuracy"]), 4),
            "macro_accuracy": round(float(full_sum["macro_accuracy"]), 4),
            "brier_mean": round(float(full_sum["brier_mean"]), 4),
            "ece": round(_ece_val(full_sum["ece"]), 4),
            "strict_valid_rate": round(float(full_sum["schema_validity_strict"]), 4),
            "p50_ms": round(p50_s * 1000.0, 2),
            "p95_ms": round(p95_s * 1000.0, 2),
            "price_per_1000_decisions_usd": round(usd_1k, 5),
            "public_axes_v13": axes_public,
            "public_jevbench_v13_geometric": round(composite_v13.jevbench_score(axes_public), 2),
            "public_jevbench_v14_harmonic": round(composite_v14.harmonic(axes_public), 2),
            "per_family_accuracy": {
                fk: round(float(fv["accuracy"]), 4)
                for fk, fv in (full_sum.get("per_family") or {}).items()
                if fv.get("accuracy") is not None
            },
            "tiers": tier_summaries,
        }
        print("JevBench / Benchmark Heaven Public Suite Summary:", json.dumps(overall, indent=2), flush=True)
        return overall
    finally:
        proc.terminate()


def _normalize_squad_answer(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _squad_f1(pred: str, gold: str) -> float:
    p_toks = _normalize_squad_answer(pred).split()
    g_toks = _normalize_squad_answer(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = collections.Counter(p_toks) & collections.Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_toks)
    rec = num_same / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def eval_squad_v2_public(model_dir: str, device: str, n_eval: int = 500) -> dict[str, Any]:
    print("=== [3/3] Evaluating on Stanford rajpurkar/squad_v2 (Zero-Overlap Validation) ===", flush=True)
    seen_contexts: set[str] = set()
    for fname in ("train.jsonl", "cal_dev.jsonl", "val.jsonl"):
        fpath = os.path.join(ROOT, "data", fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        st = obj.get("state") or obj.get("context") or ""
                        if st:
                            seen_contexts.add(st[:120])

    ds_sq = load_dataset("rajpurkar/squad_v2", split="validation")
    has_ans_pool, no_ans_pool = [], []
    for row in ds_sq:
        if row["context"][:120] in seen_contexts:
            continue
        if row["answers"]["text"]:
            has_ans_pool.append(row)
        else:
            no_ans_pool.append(row)

    rng = random.Random(20261030)
    rng.shuffle(has_ans_pool)
    rng.shuffle(no_ans_pool)
    sample = has_ans_pool[: n_eval // 2] + no_ans_pool[: n_eval // 2]

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = NullspanForExtraction.from_pretrained(model_dir).to(device).eval()

    em_all, f1_all, em_has, f1_has, em_no, f1_no = [], [], [], [], [], []
    for row in sample:
        res = model.extract(
            state=row["context"],
            fields={"answer": {"type": "text", "instructions": row["question"]}},
            tokenizer=tok,
        )
        pred_val = res["answers"]["answer"]["value"]
        pred_str = "" if pred_val is None else str(pred_val)
        golds = row["answers"]["text"]
        if not golds:
            em = 1.0 if pred_str == "" else 0.0
            f1 = em
            em_no.append(em)
            f1_no.append(f1)
        else:
            em = max(1.0 if _normalize_squad_answer(pred_str) == _normalize_squad_answer(g) else 0.0 for g in golds)
            f1 = max(_squad_f1(pred_str, g) for g in golds)
            em_has.append(em)
            f1_has.append(f1)
        em_all.append(em)
        f1_all.append(f1)

    summary = {
        "dataset": "rajpurkar/squad_v2",
        "n_evaluated": len(sample),
        "exact": round(100.0 * sum(em_all) / len(em_all), 2),
        "f1": round(100.0 * sum(f1_all) / len(f1_all), 2),
        "HasAns_exact": round(100.0 * sum(em_has) / len(em_has), 2),
        "HasAns_f1": round(100.0 * sum(f1_has) / len(f1_has), 2),
        "NoAns_exact": round(100.0 * sum(em_no) / len(em_no), 2),
        "NoAns_f1": round(100.0 * sum(f1_no) / len(f1_no), 2),
    }
    print("SQuAD v2.0 Zero-Overlap Summary:", json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    hw_name = torch.cuda.get_device_name(0) if device == "cuda" else device
    print(f"Running public benchmark suite on device={device} ({hw_name})", flush=True)
    model_dir = os.path.join(ROOT, "artifacts", "nullspan")
    jevbench_dir = os.environ.get(
        "JEVBENCH_DIR",
        "/Users/lika/.gemini/antigravity-cli/brain/0bde580d-fb36-42da-9188-86f2aaa1792a/scratch/jevbench",
    )

    td_summary = train_and_eval_typed_decisions(model_dir, device)
    jb_summary = eval_jevbench_public(model_dir, jevbench_dir, device)
    sq_summary = eval_squad_v2_public(model_dir, device, n_eval=500)

    out_path = os.path.join(ROOT, "bench", "public_benchmarks_results.json")
    full_report = {
        "hardware": hw_name,
        "typed_decisions": td_summary,
        "jevbench_public": jb_summary,
        "squad_v2": sq_summary,
    }
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nWrote {out_path}!", flush=True)


if __name__ == "__main__":
    main()
