"""Build the Sealed Held-Out Multi-Numeric Contract Benchmark (`bench/contracts_sealed_golden.json`).

Contains 12 realistic, long-context (300-490 token) legal & commercial agreements across 12 distinct
industries, with 8 typed fields per contract = 96 held-out observations in the Sealed Pool (P).
Zero template, party, or text overlap with `data/train.jsonl`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "bench" / "contracts_sealed_golden.json"

OP_SPAN_TEXT = 0
OP_SPAN_NUMBER = 1
OP_ARITH_SUM = 2
OP_ARITH_DIFF = 3
OP_ARITH_MUL = 4
OP_ISO_DATETIME = 5
OP_DATE_DURATION = 6
OP_DIST_ESTIMATE_FORECAST = 7
OP_BOOL_JUDGE = 8
OP_ENUM_CLASSIFY = 9


def span_of(state: str, needle: str, after: str | None = None) -> list[int]:
    start_search = 0
    if after is not None:
        a_idx = state.find(after)
        if a_idx < 0:
            raise ValueError(f"Anchor {after!r} not found in contract")
        start_search = a_idx + len(after)
    idx = state.find(needle, start_search)
    if idx < 0:
        raise ValueError(f"Substring {needle!r} not found after {after!r}")
    return [idx, idx + len(needle)]


def build_contracts() -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []

    # 1. Enterprise Cloud SaaS MSA
    c1 = (
        "ENTERPRISE CLOUD SUBSCRIPTION AGREEMENT (Contract ID: ECSA-2026-9041)\n"
        "Effective as of January 15, 2026 between Northstar Telemetry Inc. ('Provider') and Meridian BioLabs ('Customer') "
        "covering 450 named analysts across 6 regional workspaces.\n\n"
        "1. SUBSCRIPTION TERM & RENEWAL\n"
        "The initial subscription term shall run for 36 months from the Effective Date and shall automatically renew for "
        "successive 12-month renewal terms unless written non-renewal notice is delivered at least 60 days prior to expiration.\n\n"
        "2. PLATFORM AVAILABILITY & DISASTER RECOVERY\n"
        "Provider commits to a monthly uptime SLA of 99.98% across primary ingestion endpoints measured over each 30-day period. "
        "In the event of a regional outage, Provider guarantees a disaster recovery RTO of 4 hours and RPO of 15 minutes.\n\n"
        "3. FEES, BILLING & LATE CHARGES\n"
        "Customer shall pay an annual platform subscription fee of $185,000 USD billed net 30 days from invoice date. "
        "Past-due balances remaining unpaid after a 14-day cure period shall accrue late payment interest at 1.25% per month "
        "(15% per annum ceiling).\n\n"
        "4. LIMITATION OF LIABILITY\n"
        "Except for indemnification obligations, neither party's aggregate cumulative liability shall exceed the liability cap "
        "of $550,000 USD, with data-migration secondary claims sub-limited to $90,000 USD."
    )
    contracts.append({
        "id": "sealed_contract_saas_enterprise_msa",
        "domain": "contract_extraction",
        "archetype": "Enterprise SaaS Subscription MSA",
        "state": c1,
        "fields": {
            "initial_term_months": {
                "spec": {"type": "number", "integer": True, "unit": "months", "instructions": "Initial subscription term duration in months"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c1, "36", "run for "), "value": 36},
            },
            "annual_platform_fee": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Annual platform subscription fee in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c1, "$185,000"), "value": 185000},
            },
            "uptime_sla_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Monthly uptime SLA percentage commitment"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c1, "99.98%"), "value": 99.98},
            },
            "disaster_recovery_rto_hours": {
                "spec": {"type": "number", "integer": True, "unit": "hours", "instructions": "Disaster recovery RTO target in hours"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c1, "4", "RTO of "), "value": 4},
            },
            "late_interest_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Monthly late payment interest rate percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c1, "1.25%"), "value": 1.25},
            },
            "cure_period_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Past-due payment cure period in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c1, "14", "after a "), "value": 14},
            },
            "liability_cap": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Aggregate cumulative liability cap in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c1, "$550,000"), "value": 550000},
            },
            "royalty_rate_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Gross revenue royalty rate percentage if specified"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c1, "1.25%"), "value": None, "guess": 1.25, "abstention_reason": "missing_field"},
            },
        },
    })

    # 2. Healthcare HIPAA BAA & Data Processing Addendum
    c2 = (
        "HIPAA BUSINESS ASSOCIATE & DATA PROCESSING ADDENDUM (Ref: BAA-77210)\n"
        "Entered into on February 3, 2026 between Cascade Health Network ('Covered Entity') and VaultMed Analytics ('Business Associate') "
        "governing Protected Health Information across 18 hospital clinics.\n\n"
        "1. BREACH NOTIFICATION & INCIDENT ESCALATION\n"
        "Business Associate shall notify Covered Entity of any confirmed PHI security breach within 24 hours of discovery, "
        "followed by a written root-cause forensic report within 10 days.\n\n"
        "2. ENCRYPTION, KEY ROTATION & AUDIT RETENTION\n"
        "All PHI databases are encrypted under FIPS 140-3 modules with mandatory cryptographic key rotation every 90 days. "
        "Tamper-evident HIPAA access audit logs must be retained for 2190 days in immutable cold storage.\n\n"
        "3. SUBPROCESSORS & COMPLIANCE ATTESTATION\n"
        "Business Associate shall provide 45 days advance written notice before engaging any new subprocessor. "
        "HIPAA HITECH compliance status is active and verified under annual HITRUST r2 certification.\n\n"
        "4. INDEMNIFICATION & CYBER SUB-LIMITS\n"
        "Aggregate regulatory and breach indemnification under this Addendum is capped at $1,200,000 USD, whereas third-party "
        "credit-monitoring reimbursement is sub-limited to $350,000 USD per incident."
    )
    contracts.append({
        "id": "sealed_contract_hipaa_baa_dpa",
        "domain": "contract_extraction",
        "archetype": "Healthcare HIPAA BAA & DPA",
        "state": c2,
        "fields": {
            "breach_notification_hours": {
                "spec": {"type": "number", "integer": True, "unit": "hours", "instructions": "Confirmed PHI security breach notification window in hours"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c2, "24", "within "), "value": 24},
            },
            "audit_retention_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "HIPAA access audit log retention period in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c2, "2190"), "value": 2190},
            },
            "key_rotation_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Mandatory cryptographic key rotation interval in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c2, "90", "every "), "value": 90},
            },
            "subprocessor_notice_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Advance written notice in days before engaging a new subprocessor"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c2, "45", "provide "), "value": 45},
            },
            "indemnity_cap": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Aggregate regulatory and breach indemnification cap in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c2, "$1,200,000"), "value": 1200000},
            },
            "credit_monitoring_sublimit": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Third-party credit-monitoring reimbursement sub-limit in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c2, "$350,000"), "value": 350000},
            },
            "hipaa_compliant": {
                "spec": {"type": "boolean", "instructions": "Whether HIPAA HITECH compliance status is active"},
                "target": {"is_null": 0, "op": OP_BOOL_JUDGE, "span": span_of(c2, "active", "status is "), "value": True},
            },
            "liquidated_damages": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Daily liquidated damages fee in USD if present"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c2, "$350,000"), "value": None, "guess": 350000, "abstention_reason": "missing_field"},
            },
        },
    })

    # 3. AI Data Center Colocation & Power Lease
    c3 = (
        "AI HIGH-DENSITY COLOCATION LEASE (Lease No. COLO-2026-318)\n"
        "Dated April 1, 2026 between Borealis Data Parks LLC ('Landlord') and FoundryAI Compute Corp ('Tenant') "
        "for Hall C Cage 12 housing 64 liquid-cooled racks.\n\n"
        "1. LEASE DURATION & POWER ALLOCATION\n"
        "The primary lease term is 60 months with a committed critical IT power allocation of 2400 kW delivered via "
        "dual 480V busways. Landlord guarantees a seasonal Power Usage Effectiveness (PUE) ceiling of 1.22 across Hall C.\n\n"
        "2. RECURRING RENT, DEPOSIT & FIT-OUT MILESTONES\n"
        "Tenant shall pay a base monthly colocation rent of $215,000 USD on the 1st of each month and post a refundable "
        "security deposit of $430,000 USD at signing. Custom liquid-manifold buildout payments are scheduled in 3 tranches: "
        "$60,000 at design freeze, $95,000 at cooling loop pressure test, and $45,000 at beneficial occupancy.\n\n"
        "3. EARLY TERMINATION BUYOUT\n"
        "Termination prior to month 36 requires 120 days notice and payment of an early exit buyout fee of $645,000 USD."
    )
    contracts.append({
        "id": "sealed_contract_gpu_colocation_lease",
        "domain": "contract_extraction",
        "archetype": "AI Data Center Colocation Lease",
        "state": c3,
        "fields": {
            "lease_term_months": {
                "spec": {"type": "number", "integer": True, "unit": "months", "instructions": "Primary colocation lease term in months"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c3, "60", "term is "), "value": 60},
            },
            "monthly_colocation_rent": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Base monthly colocation rent in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c3, "$215,000"), "value": 215000},
            },
            "security_deposit": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Refundable security deposit amount in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c3, "$430,000"), "value": 430000},
            },
            "pue_ceiling": {
                "spec": {"type": "number", "decimals": 2, "instructions": "Guaranteed Power Usage Effectiveness (PUE) ceiling"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c3, "1.22"), "value": 1.22},
            },
            "early_exit_fee": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Early exit buyout fee in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c3, "$645,000"), "value": 645000},
            },
            "termination_notice_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Early termination notice period in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c3, "120", "requires "), "value": 120},
            },
            "buildout_milestones": {
                "spec": {"type": "list", "items": "number", "instructions": "Tranche dollar amounts for liquid-manifold buildout payments"},
                "target": {
                    "is_null": 0,
                    "op": OP_SPAN_NUMBER,
                    "spans": [span_of(c3, "$60,000"), span_of(c3, "$95,000"), span_of(c3, "$45,000")],
                    "value": [60000, 95000, 45000],
                },
            },
            "carbon_offset_surcharge": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Monthly carbon offset surcharge in USD if present"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c3, "$215,000"), "value": None, "guess": 215000, "abstention_reason": "missing_field"},
            },
        },
    })

    # 4. FinTech Payment Clearing & Rolling Reserve Agreement
    c4 = (
        "MERCHANT ACQUIRING & CLEARING SCHEDULE (Schedule ID: MAC-2026-512)\n"
        "Effective May 10, 2026 between HarborPay Bank N.A. ('Acquirer') and NomadCart Commerce ('Merchant') "
        "under pricing tier enterprise-plus.\n\n"
        "1. SETTLEMENT & ROLLING RESERVE\n"
        "Net card proceeds settle within 2 business days of batch capture. Acquirer shall withhold a rolling reserve of "
        "8.5% of gross daily volume held in escrow for 120 days to cover reversal exposure.\n\n"
        "2. CHARGEBACK MONITORING & DISPUTE FEES\n"
        "Merchant must maintain a monthly chargeback ratio below the threshold of 0.85% of total transactions. Each cardholder "
        "dispute incurs a non-refundable per-dispute fee of $22.50 USD, subject to a monthly minimum processing fee of $3,500 USD."
    )
    contracts.append({
        "id": "sealed_contract_fintech_payment_clearing",
        "domain": "contract_extraction",
        "archetype": "FinTech Merchant Clearing Agreement",
        "state": c4,
        "fields": {
            "settlement_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Net card proceeds settlement window in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c4, "2", "within "), "value": 2},
            },
            "rolling_reserve_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Rolling reserve percentage of gross daily volume"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c4, "8.5%"), "value": 8.5},
            },
            "reserve_hold_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Escrow hold duration for rolling reserve in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c4, "120", "escrow for "), "value": 120},
            },
            "chargeback_threshold_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Monthly chargeback ratio threshold percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c4, "0.85%"), "value": 0.85},
            },
            "dispute_fee": {
                "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Per-dispute chargeback fee in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c4, "$22.50"), "value": 22.5},
            },
            "monthly_minimum_fee": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Monthly minimum processing fee in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c4, "$3,500"), "value": 3500},
            },
            "pricing_tier": {
                "spec": {"type": "enum", "options": ["starter", "growth", "enterprise-plus"], "instructions": "Assigned merchant pricing tier"},
                "target": {"is_null": 0, "op": OP_ENUM_CLASSIFY, "span": span_of(c4, "enterprise-plus"), "value": "enterprise-plus"},
            },
            "exclusivity_buyout": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Exclusivity buyout fee in USD if present"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c4, "$3,500"), "value": None, "guess": 3500, "abstention_reason": "missing_field"},
            },
        },
    })

    # 5. Biotech Clinical Trial CRO Agreement
    c5 = (
        "CLINICAL TRIAL MASTER SERVICES ORDER (Protocol: ONC-304-CRO)\n"
        "Executed March 22, 2026 between HelixVax Therapeutics ('Sponsor') and Praxion Clinical Research ('CRO') "
        "spanning 28 oncology trial sites across 24 months.\n\n"
        "1. ENROLLMENT TARGET & SITE COMPENSATION\n"
        "CRO shall recruit an enrolled patient target of 360 evaluable subjects. Sponsor shall pay a non-refundable site "
        "startup retainer of $85,000 USD upon IRB approval plus a per-patient site fee of $4,200 USD per randomized subject.\n\n"
        "2. PHARMACOVIGILANCE & PROTOCOL AMENDMENTS\n"
        "Serious Adverse Events (SAEs) must be reported to Sponsor safety desk within 24 hours of site awareness. "
        "Material protocol amendments require 30 days written review notice. Pass-through biomarker lab tranches are billed "
        "at 3 milestones: $50,000 at first-patient-in, $75,000 at 50% interim lock, and $65,000 at database lock.\n\n"
        "3. LIABILITY LIMITATION\n"
        "CRO's total contractual liability cap under Protocol ONC-304-CRO shall not exceed $950,000 USD."
    )
    contracts.append({
        "id": "sealed_contract_biotech_clinical_cro",
        "domain": "contract_extraction",
        "archetype": "Biotech Clinical Trial CRO Order",
        "state": c5,
        "fields": {
            "patient_target": {
                "spec": {"type": "number", "integer": True, "instructions": "Enrolled patient target count of evaluable subjects"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c5, "360"), "value": 360},
            },
            "startup_retainer": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Non-refundable site startup retainer in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c5, "$85,000"), "value": 85000},
            },
            "per_patient_fee": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Per-patient site fee in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c5, "$4,200"), "value": 4200},
            },
            "sae_reporting_hours": {
                "spec": {"type": "number", "integer": True, "unit": "hours", "instructions": "Serious Adverse Event (SAE) reporting window in hours"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c5, "24", "within "), "value": 24},
            },
            "amendment_notice_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Protocol amendment written review notice in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c5, "30", "require "), "value": 30},
            },
            "liability_cap": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Total contractual liability cap in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c5, "$950,000"), "value": 950000},
            },
            "biomarker_milestones": {
                "spec": {"type": "list", "items": "number", "instructions": "Milestone dollar amounts for biomarker lab tranches"},
                "target": {
                    "is_null": 0,
                    "op": OP_SPAN_NUMBER,
                    "spans": [span_of(c5, "$50,000"), span_of(c5, "$75,000"), span_of(c5, "$65,000")],
                    "value": [50000, 75000, 65000],
                },
            },
            "patent_royalty_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Commercial patent royalty percentage if specified"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c5, "50%"), "value": None, "guess": 50, "abstention_reason": "missing_field"},
            },
        },
    })

    # 6. Global 3PL Cold-Chain Logistics SLA
    c6 = (
        "COLD-CHAIN WAREHOUSING & 3PL FULFILLMENT AGREEMENT (Ref: 3PL-2026-619)\n"
        "Dated June 1, 2026 between PolarFreight Logistics ('Carrier') and Verdant Biologics ('Shipper') "
        "across 4 temperature-controlled hubs.\n\n"
        "1. CONTRACT DURATION & MONTHLY RETAINER\n"
        "This 3PL Agreement remains in force for a contract duration of 48 months. Shipper shall pay a monthly warehouse "
        "retainer of $32,000 USD covering 1,500 pallet positions, billed net 15 days.\n\n"
        "2. SHRINKAGE, DETENTION & COLD-CHAIN TELEMETRY\n"
        "Annual inventory shrinkage allowance shall not exceed 0.35% of audited unit value. Loading dock detention charges "
        "apply only after a 3-hour detention grace period. Continuous active cold-chain GPS telemetry is enabled on all reefers, "
        "and payload temperature excursions must not exceed 8 degrees Celsius.\n\n"
        "3. CARGO LIABILITY CAP\n"
        "Carrier's maximum cargo loss liability cap is $400,000 USD per occurrence."
    )
    contracts.append({
        "id": "sealed_contract_supply_chain_3pl",
        "domain": "contract_extraction",
        "archetype": "Cold-Chain 3PL Fulfillment SLA",
        "state": c6,
        "fields": {
            "contract_duration_months": {
                "spec": {"type": "number", "integer": True, "unit": "months", "instructions": "3PL contract duration in months"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c6, "48", "duration of "), "value": 48},
            },
            "warehouse_retainer": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Monthly warehouse retainer fee in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c6, "$32,000"), "value": 32000},
            },
            "shrinkage_allowance_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Annual inventory shrinkage allowance percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c6, "0.35%"), "value": 0.35},
            },
            "detention_grace_hours": {
                "spec": {"type": "number", "integer": True, "unit": "hours", "instructions": "Loading dock detention grace period in hours"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c6, "3", "after a "), "value": 3},
            },
            "cargo_liability_cap": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Maximum cargo loss liability cap in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c6, "$400,000"), "value": 400000},
            },
            "net_payment_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Invoice net payment window in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c6, "15", "net "), "value": 15},
            },
            "cold_chain_telemetry": {
                "spec": {"type": "boolean", "instructions": "Whether continuous active cold-chain GPS telemetry is enabled"},
                "target": {"is_null": 0, "op": OP_BOOL_JUDGE, "span": span_of(c6, "enabled", "telemetry is "), "value": True},
            },
            "fuel_surcharge_cap_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Diesel fuel surcharge cap percentage if present"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c6, "0.35%"), "value": None, "guess": 0.35, "abstention_reason": "missing_field"},
            },
        },
    })

    # 7. Enterprise Foundation Model License Agreement
    c7 = (
        "FOUNDATION MODEL ENTERPRISE LICENSE (Agreement: FMEL-2026-109)\n"
        "Effective July 1, 2026 between Lumina Frontier Labs ('Licensor') and Stratagem Financial ('Licensee') "
        "under deployment mode vpc-dedicated.\n\n"
        "1. COMMITTED SPEND, THROUGHPUT & OVERAGE DISCOUNT\n"
        "Licensee commits to an annual committed spend of $320,000 USD across a 24-month term for a guaranteed throughput "
        "quota of 18000 requests per minute. Token usage beyond committed quota receives an overage volume discount of 22.5% "
        "off list API rates.\n\n"
        "2. CHECKPOINT RETENTION, TERMINATION & IP INDEMNITY\n"
        "Fine-tuned LoRA adapter checkpoints are purged within 30 days after contract expiration. Either party may terminate "
        "for material breach on 45 days written notice. Licensor's copyright and IP indemnity cap is $1,500,000 USD."
    )
    contracts.append({
        "id": "sealed_contract_ai_foundation_license",
        "domain": "contract_extraction",
        "archetype": "Enterprise AI Model License",
        "state": c7,
        "fields": {
            "annual_committed_spend": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Annual committed spend in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c7, "$320,000"), "value": 320000},
            },
            "overage_discount_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Overage volume discount percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c7, "22.5%"), "value": 22.5},
            },
            "checkpoint_retention_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Fine-tuned checkpoint purge window in days after expiration"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c7, "30", "within "), "value": 30},
            },
            "termination_notice_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Material breach termination written notice in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c7, "45", "breach on "), "value": 45},
            },
            "ip_indemnity_cap": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Copyright and IP indemnity cap in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c7, "$1,500,000"), "value": 1500000},
            },
            "requests_per_minute": {
                "spec": {"type": "number", "integer": True, "instructions": "Guaranteed throughput quota in requests per minute"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c7, "18000"), "value": 18000},
            },
            "deployment_mode": {
                "spec": {"type": "enum", "options": ["multi-tenant", "vpc-dedicated", "air-gapped"], "instructions": "Licensed model deployment mode"},
                "target": {"is_null": 0, "op": OP_ENUM_CLASSIFY, "span": span_of(c7, "vpc-dedicated"), "value": "vpc-dedicated"},
            },
            "revenue_share_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Downstream product revenue share percentage if present"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c7, "22.5%"), "value": None, "guess": 22.5, "abstention_reason": "missing_field"},
            },
        },
    })

    # 8. Commercial Solar Power Purchase Agreement (PPA)
    c8 = (
        "RENEWABLE SOLAR POWER PURCHASE AGREEMENT (Contract: PPA-2026-440)\n"
        "Dated May 18, 2026 between SunMesa Generation LLC ('Seller') and Copperline Smelting Corp ('Buyer') "
        "for a 75 MW photovoltaic facility.\n\n"
        "1. PPA DELIVERY TERM, STRIKE PRICE & DEGRADATION\n"
        "The delivery PPA term is 15 years from Commercial Operation Date at a fixed strike price of $38.50 USD per MWh. "
        "Annual panel output degradation cap shall not exceed 0.50% per year, with a plant availability guarantee of 98.5%.\n\n"
        "2. CURTAILMENT, CURE WINDOW & INTERCONNECTION DEPOSITS\n"
        "Buyer-directed economic curtailment beyond 50 hours per year is compensated at a curtailment credit of $29.00 USD per MWh. "
        "Default notices carry a 20-day cure period. Seller's grid interconnection security deposits are funded in 3 tranches: "
        "$120,000 at system impact study, $180,000 at facilities study, and $200,000 at IA execution."
    )
    contracts.append({
        "id": "sealed_contract_renewable_ppa",
        "domain": "contract_extraction",
        "archetype": "Commercial Solar PPA",
        "state": c8,
        "fields": {
            "ppa_term_years": {
                "spec": {"type": "number", "integer": True, "unit": "years", "instructions": "Delivery PPA term duration in years"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c8, "15", "term is "), "value": 15},
            },
            "strike_price": {
                "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Fixed PPA strike price in USD per MWh"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c8, "$38.50"), "value": 38.5},
            },
            "degradation_cap_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Annual panel output degradation cap percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c8, "0.50%"), "value": 0.5},
            },
            "availability_guarantee_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Plant availability guarantee percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c8, "98.5%"), "value": 98.5},
            },
            "curtailment_credit": {
                "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Curtailment credit in USD per MWh"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c8, "$29.00"), "value": 29.0},
            },
            "cure_period_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Default notice cure period in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c8, "20", "carry a "), "value": 20},
            },
            "interconnection_tranches": {
                "spec": {"type": "list", "items": "number", "instructions": "Tranche dollar amounts for grid interconnection security deposits"},
                "target": {
                    "is_null": 0,
                    "op": OP_SPAN_NUMBER,
                    "spans": [span_of(c8, "$120,000"), span_of(c8, "$180,000"), span_of(c8, "$200,000")],
                    "value": [120000, 180000, 200000],
                },
            },
            "tax_credit_rebate": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Federal investment tax credit rebate in USD if specified"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c8, "$29.00"), "value": None, "guess": 29.0, "abstention_reason": "missing_field"},
            },
        },
    })

    # 9. Cybersecurity Incident Response Retainer (with cross-field algebraic constraint!)
    c9 = (
        "CYBER INCIDENT RESPONSE RETAINER AGREEMENT (Order: CIR-2026-812)\n"
        "Effective August 1, 2026 between Blackwatch DFIR Labs ('Responder') and Crestline Bancorp ('Client') "
        "covering 12-month emergency forensics readiness.\n\n"
        "1. PREPAID HOURS, RATE & RETAINER INVARIANT\n"
        "Client purchases 160 prepaid incident response hours at a blended hourly rate of $325.00 USD per hour, "
        "yielding an annual retainer subtotal of $52,000.00 USD due within 30 days of execution. Up to 25% of unused "
        "prepaid hours may roll over into the following renewal year.\n\n"
        "2. RESPONSE SLA, COVERAGE & LIABILITY\n"
        "Responder guarantees remote triage mobilization within a critical response SLA of 30 minutes. Continuous "
        "weekend SOC escalation coverage is true. Responder's aggregate professional liability cap is $500,000 USD."
    )
    contracts.append({
        "id": "sealed_contract_cyber_ir_retainer",
        "domain": "contract_extraction",
        "archetype": "Cybersecurity DFIR Retainer",
        "state": c9,
        "fields": {
            "prepaid_hours": {
                "spec": {"type": "number", "integer": True, "instructions": "Purchased prepaid incident response hours"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c9, "160"), "value": 160},
            },
            "hourly_rate": {
                "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Blended hourly rate in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c9, "$325.00"), "value": 325.0},
            },
            "annual_retainer": {
                "spec": {
                    "type": "number",
                    "decimals": 2,
                    "unit": "USD",
                    "constraint": "annual_retainer == prepaid_hours * hourly_rate",
                    "instructions": "Annual retainer subtotal in USD",
                },
                "target": {"is_null": 0, "op": OP_ARITH_MUL, "span": span_of(c9, "$52,000.00"), "value": 52000.0, "verified": True},
            },
            "rollover_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Maximum unused prepaid hours rollover percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c9, "25%"), "value": 25},
            },
            "response_sla_minutes": {
                "spec": {"type": "number", "integer": True, "unit": "minutes", "instructions": "Critical response SLA mobilization window in minutes"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c9, "30", "SLA of "), "value": 30},
            },
            "liability_cap": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Aggregate professional liability cap in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c9, "$500,000"), "value": 500000},
            },
            "weekend_coverage": {
                "spec": {"type": "boolean", "instructions": "Whether continuous weekend SOC escalation coverage is active"},
                "target": {"is_null": 0, "op": OP_BOOL_JUDGE, "span": span_of(c9, "true", "coverage is "), "value": True},
            },
            "ransom_negotiation_fee": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Separate ransom negotiation surcharge in USD if present"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c9, "$500,000"), "value": None, "guess": 500000, "abstention_reason": "missing_field"},
            },
        },
    })

    # 10. Commercial Wet-Lab Sublease Agreement (with false boolean!)
    c10 = (
        "COMMERCIAL WET-LAB SUBLEASE AGREEMENT (Sublease ID: WLS-2026-205)\n"
        "Entered July 15, 2026 between GeneCraft Corp ('Sublandlord') and Lattice Enzymes Inc. ('Subtenant') "
        "for 18,500 rentable sq ft on Floor 4.\n\n"
        "1. SUBLEASE TERM, BASE RENT & ESCALATION\n"
        "The sublease term is 42 months commencing August 1, 2026, with an initial rent abatement period of 2 months. "
        "Subtenant shall pay base monthly rent of $68,500 USD, subject to an annual rent escalation of 3.25% on each "
        "anniversary date, plus a security deposit of $137,000 USD.\n\n"
        "2. HOLDOVER & FURTHER SUBLETTING\n"
        "Unauthorized holdover occupancy incurs a holdover rent multiplier of 150% of prior monthly rent. Further "
        "assignment or sub-subletting by Subtenant is false and strictly prohibited. Late rent grace period is 5 days."
    )
    contracts.append({
        "id": "sealed_contract_commercial_sublease",
        "domain": "contract_extraction",
        "archetype": "Commercial Wet-Lab Sublease",
        "state": c10,
        "fields": {
            "sublease_term_months": {
                "spec": {"type": "number", "integer": True, "unit": "months", "instructions": "Sublease term duration in months"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c10, "42", "term is "), "value": 42},
            },
            "base_monthly_rent": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Base monthly rent in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c10, "$68,500"), "value": 68500},
            },
            "rent_escalation_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Annual rent escalation percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c10, "3.25%"), "value": 3.25},
            },
            "security_deposit": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Sublease security deposit in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c10, "$137,000"), "value": 137000},
            },
            "holdover_multiplier_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Holdover rent multiplier percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c10, "150%"), "value": 150},
            },
            "abatement_months": {
                "spec": {"type": "number", "integer": True, "unit": "months", "instructions": "Initial rent abatement period in months"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c10, "2", "period of "), "value": 2},
            },
            "further_subletting_allowed": {
                "spec": {"type": "boolean", "instructions": "Whether further sub-subletting by Subtenant is permitted"},
                "target": {"is_null": 0, "op": OP_BOOL_JUDGE, "span": span_of(c10, "false", "Subtenant is "), "value": False},
            },
            "tenant_improvement_allowance": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Landlord tenant improvement allowance in USD if present"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c10, "$137,000"), "value": None, "guess": 137000, "abstention_reason": "missing_field"},
            },
        },
    })

    # 11. Metro Dark Fiber IRU & Maintenance Agreement
    c11 = (
        "DARK FIBER INDEFEASIBLE RIGHT OF USE AGREEMENT (IRU-2026-774)\n"
        "Dated September 1, 2026 between MetroWave Fiber Corp ('Grantor') and QuantEdge Trading LLC ('Grantee') "
        "for 12 dedicated SMF-28 strands along route ring-east-diverse.\n\n"
        "1. IRU TERM, UPFRONT GRANT FEE & O&M\n"
        "Grantor conveys an IRU term of 20 years in exchange for a one-time upfront IRU fee of $480,000 USD due within "
        "30 days of acceptance testing, plus an annual O&M maintenance fee of $19,500 USD per year.\n\n"
        "2. OPTICAL ATTENUATION, MTTR & CHRONIC OUTAGE CREDIT\n"
        "End-to-end optical splice loss ceiling shall not exceed 0.25 dB per splice. Grantor commits to a mean time to "
        "repair (MTTR) of 4 hours for fiber cuts and a chronic outage service credit of 15% of annual O&M fees."
    )
    contracts.append({
        "id": "sealed_contract_telecom_dark_fiber",
        "domain": "contract_extraction",
        "archetype": "Metro Dark Fiber IRU Agreement",
        "state": c11,
        "fields": {
            "iru_term_years": {
                "spec": {"type": "number", "integer": True, "unit": "years", "instructions": "Dark fiber IRU term duration in years"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c11, "20", "term of "), "value": 20},
            },
            "upfront_iru_fee": {
                "spec": {"type": "number", "unit": "USD", "instructions": "One-time upfront IRU grant fee in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c11, "$480,000"), "value": 480000},
            },
            "annual_om_fee": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Annual O&M maintenance fee in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c11, "$19,500"), "value": 19500},
            },
            "mttr_hours": {
                "spec": {"type": "number", "integer": True, "unit": "hours", "instructions": "Mean time to repair (MTTR) target in hours"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c11, "4", "(MTTR) of "), "value": 4},
            },
            "splice_loss_db": {
                "spec": {"type": "number", "decimals": 2, "instructions": "Optical splice loss ceiling in dB"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c11, "0.25"), "value": 0.25},
            },
            "chronic_outage_credit_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Chronic outage service credit percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c11, "15%"), "value": 15},
            },
            "fiber_route": {
                "spec": {"type": "enum", "options": ["ring-west-primary", "ring-east-diverse", "trans-hudson-direct"], "instructions": "Designated dark fiber route topology"},
                "target": {"is_null": 0, "op": OP_ENUM_CLASSIFY, "span": span_of(c11, "ring-east-diverse"), "value": "ring-east-diverse"},
            },
            "pole_attachment_fee": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Municipal pole attachment fee in USD if present"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c11, "$19,500"), "value": None, "guess": 19500, "abstention_reason": "missing_field"},
            },
        },
    })

    # 12. Programmatic AdTech DSP Insertion Order
    c12 = (
        "PROGRAMMATIC DSP INSERTION ORDER & BRAND SAFETY RIDER (IO: DSP-2026-930)\n"
        "Executed October 1, 2026 between PulseBid Exchange ('Platform') and Vela Consumer Brands ('Advertiser') "
        "across CTV and premium mobile inventory.\n\n"
        "1. CAMPAIGN FLIGHT, MEDIA BUDGET & PLATFORM FEE\n"
        "The Q4 campaign flight spans 75 days with a committed total media budget of $275,000 USD billed on net 45 days "
        "payment terms. Platform deducts a transparent DSP technology fee of 11.5% of gross media spend.\n\n"
        "2. VIEWABILITY FLOOR, IVT CLAWBACK & BRAND SAFETY\n"
        "Impressions must meet an MRC video viewability floor of 78% completion. Audited invalid traffic (IVT) exceeding "
        "1.5% triggers an automatic IVT clawback credit of 100% of affected spend. Pre-bid IAS brand safety filtering is enabled."
    )
    contracts.append({
        "id": "sealed_contract_adtech_dsp_io",
        "domain": "contract_extraction",
        "archetype": "Programmatic DSP Insertion Order",
        "state": c12,
        "fields": {
            "campaign_flight_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Campaign flight duration in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c12, "75", "spans "), "value": 75},
            },
            "total_media_budget": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Committed total media budget in USD"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c12, "$275,000"), "value": 275000},
            },
            "payment_net_days": {
                "spec": {"type": "number", "integer": True, "unit": "days", "instructions": "Invoice net payment terms in days"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c12, "45", "net "), "value": 45},
            },
            "dsp_tech_fee_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "DSP technology fee percentage of gross media spend"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c12, "11.5%"), "value": 11.5},
            },
            "viewability_floor_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "MRC video viewability floor percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c12, "78%"), "value": 78},
            },
            "ivt_threshold_pct": {
                "spec": {"type": "number", "decimals": 2, "unit": "%", "instructions": "Audited invalid traffic (IVT) threshold percentage"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": span_of(c12, "1.5%"), "value": 1.5},
            },
            "brand_safety_enabled": {
                "spec": {"type": "boolean", "instructions": "Whether pre-bid IAS brand safety filtering is enabled"},
                "target": {"is_null": 0, "op": OP_BOOL_JUDGE, "span": span_of(c12, "enabled", "filtering is "), "value": True},
            },
            "cancellation_penalty": {
                "spec": {"type": "number", "unit": "USD", "instructions": "Early IO cancellation penalty fee in USD if present"},
                "target": {"is_null": 1, "op": OP_SPAN_NUMBER, "span": span_of(c12, "$275,000"), "value": None, "guess": 275000, "abstention_reason": "missing_field"},
            },
        },
    })

    return contracts


def main() -> None:
    contracts = build_contracts()
    total_fields = sum(len(c["fields"]) for c in contracts)
    null_fields = sum(
        1 for c in contracts for f in c["fields"].values() if f["target"].get("is_null") == 1
    )
    payload = {
        "source": "NullspanBench Held-Out Multi-Numeric Commercial & Legal Contract Suite",
        "split": "sealed_test",
        "total_contracts": len(contracts),
        "total_fields": total_fields,
        "null_fields": null_fields,
        "contracts": contracts,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(contracts)} sealed contracts ({total_fields} fields, {null_fields} null) to {OUT_PATH}")


if __name__ == "__main__":
    main()
