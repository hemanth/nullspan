"""Nullspan System One Probabilistic Typed-Decision Engine v3.0 (`/v1/systemone`).

Single-Pass Parallel-Span 8K Bidirectional Encoder Architecture (Zero Output Tokens):
  1. Schema-Agnostic Symbolic & Reconciliation Perception (`compact_state_text` v3.0):
     Deterministically surfaces aggregate line-item sums vs scalar delivery/order quantities,
     partial-delivery exact matches vs overbilling, unpermitted freight surcharges, missing PO
     identifiers, currency mismatches, exact percentage variances, trace error/step ratios,
     calendar/clock deltas, set membership, and null flags.
  2. Single-Pass Case-Packed Parallel-Span 8K Encoding (`encode_case_parallel`):
     Packs the entire workflow State + ALL questions + ALL candidate option rubrics into ONE
     single `ModernBERT-large` (`395M`, `28` layers, `8,192`-token RoPE context) forward pass!
     Every option of Question `q` shares identical starting rotary positions (`opt_base_pos`)
     with full 28-layer bidirectional attention (`full_attention == sliding_attention`),
     guaranteeing 100% mathematical option-order permutation equivariance (`< 1e-7` max diff).
  3. Residual Zero-Shot NLI Prior + `ParallelSpanDecisionHead`:
     Combines pretrained zero-shot NLI entailment log-odds with 2-layer Cross-Option Competition,
     Cross-Question Workflow Propagation, Ordinal Neighbor Smoothing on `score` scales, and
     held-out OOF temperature calibration.
"""
from __future__ import annotations

import datetime as dt
import inspect
import json
import math
import os
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_leaf_key(k: str) -> str:
    kl = k.lower()
    if kl in ("received_qty", "ordered_qty", "billed_qty", "shipped_qty", "expected_qty", "quantity", "qty"):
        return "qty"
    if kl in ("total_usd", "po_total", "invoice_total", "order_total", "grand_total", "amount_usd", "amount", "total"):
        return "total_amt"
    if kl in ("subtotal_usd", "subtotal", "net_usd", "net_total"):
        return "subtotal_amt"
    if kl in ("tax_usd", "tax", "vat_usd"):
        return "tax_amt"
    if kl in ("unit_usd", "unit_price", "unit_cost", "rate_usd", "price_usd"):
        return "unit_amt"
    return kl


def _plain_text_temporal_notes(text: str) -> str:
    """Surface deterministic calendar-day and clock-minute deltas on plain-text states."""
    notes: list[str] = []
    seen_d: list[tuple[str, dt.date]] = []
    for m in re.finditer(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", text):
        raw_d = m.group(0)
        if any(raw_d == r for r, _ in seen_d):
            continue
        try:
            seen_d.append((raw_d, dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))))
        except ValueError:
            pass
    if 2 <= len(seen_d) <= 6:
        d_parts = []
        for i in range(len(seen_d) - 1):
            r1, d1 = seen_d[i]
            r2, d2 = seen_d[i + 1]
            d_parts.append(f"{r1}->{r2} ({(d2 - d1).days:+d}d)")
        if len(seen_d) >= 3:
            r0, d0 = seen_d[0]
            rn, dn = seen_d[-1]
            d_parts.append(f"{r0}->{rn} ({(dn - d0).days:+d}d)")
        notes.append("_dates: " + "; ".join(d_parts[:5]))

    seen_t: list[tuple[str, int]] = []
    for m in re.finditer(r"(?<![\d:])([01]\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?(?![\d:])", text):
        raw_t = f"{m.group(1)}:{m.group(2)}"
        mins = int(m.group(1)) * 60 + int(m.group(2))
        if not seen_t or seen_t[-1][0] != raw_t:
            seen_t.append((raw_t, mins))
    if 2 <= len(seen_t) <= 5:
        t_parts = []
        for i in range(len(seen_t) - 1):
            r1, m1 = seen_t[i]
            r2, m2 = seen_t[i + 1]
            dm = m2 - m1
            t_parts.append(f"{r1}->{r2} ({dm:+d}m = {dm/60.0:+.2f}h)")
        notes.append("_times: " + "; ".join(t_parts[:4]))

    if not notes:
        return text
    return "[Temporal Checks] " + " | ".join(notes) + "\n" + text


def _is_surcharge_line(item: dict[str, Any]) -> bool:
    for k in ("sku", "code", "item", "type", "description", "name"):
        v = item.get(k)
        if isinstance(v, str) and any(tok in v.lower() for tok in ("freight", "shipping", "surcharge", "handling", "delivery_fee")):
            return True
    return False


def compact_state_text(state: Any) -> str:
    """Flatten structured JSON/dict state and prepend schema-agnostic symbolic, reconciliation & ratio checks."""
    if isinstance(state, str):
        s_strip = state.strip()
        if s_strip.startswith("{") or s_strip.startswith("["):
            try:
                state = json.loads(s_strip)
            except Exception:
                return _plain_text_temporal_notes(state)
        else:
            return _plain_text_temporal_notes(state)

    if not isinstance(state, (dict, list)):
        return str(state)

    lines: list[str] = []
    num_leaves: list[tuple[str, str, str, float]] = []  # (top_group, path, norm_key, val)
    str_leaves: list[tuple[str, str, str, str]] = []    # (top_group, path, raw_key, val)
    str_lists: list[tuple[str, list[str]]] = []
    null_paths: list[str] = []
    arith_notes: list[str] = []
    group_goods_qty: dict[str, float] = {}
    group_unit_prices: dict[str, list[float]] = {}
    surcharge_lines: list[tuple[str, str, float]] = []
    freight_terms_vals: list[tuple[str, str]] = []
    has_usd_key = False

    def _check_dict_math(top_group: str, prefix: str, d: dict[str, Any]) -> None:
        nonlocal has_usd_key
        for k in d.keys():
            if k.lower().endswith("_usd"):
                has_usd_key = True

        # 1. Operational / trace telemetry ratios (steps, errors, violations, duration)
        steps_v = d.get("steps")
        if isinstance(steps_v, (int, float)) and not isinstance(steps_v, bool) and steps_v > 0:
            ratio_parts = [f"steps={steps_v:g}"]
            errs_v = d.get("tool_errors")
            if isinstance(errs_v, (int, float)) and not isinstance(errs_v, bool):
                ratio_parts.append(f"tool_errors={errs_v:g} ({100.0 * errs_v / steps_v:.1f}%/step)")
            viol_v = d.get("constraint_violations")
            if isinstance(viol_v, (int, float)) and not isinstance(viol_v, bool):
                ratio_parts.append(f"constraint_violations={viol_v:g} (VIOLATED={viol_v > 0})")
            irrev_v = d.get("irreversible_actions")
            if isinstance(irrev_v, (int, float)) and not isinstance(irrev_v, bool):
                ratio_parts.append(f"irreversible_actions={irrev_v:g} (IRREVERSIBLE={irrev_v > 0})")
            dur_v = d.get("duration_s")
            if isinstance(dur_v, (int, float)) and not isinstance(dur_v, bool):
                ratio_parts.append(f"sec_per_step={dur_v / steps_v:.1f}s")
            if len(ratio_parts) >= 2:
                arith_notes.append(f"_telemetry.{prefix or 'root'}: " + ", ".join(ratio_parts))

        # 2. Check qty * unit == total inside any dict
        q_val, u_val, t_val = None, None, None
        for k, v in d.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            nk = _norm_leaf_key(k)
            if nk == "qty" and q_val is None:
                q_val = float(v)
            elif nk == "unit_amt" and u_val is None:
                u_val = float(v)
            elif nk == "total_amt" and t_val is None:
                t_val = float(v)
        if q_val is not None and u_val is not None and t_val is not None:
            expected = round(q_val * u_val, 4)
            diff = round(t_val - expected, 4)
            arith_notes.append(
                f"_math.{prefix or 'root'}.qty_x_unit: {q_val:g} * {u_val:g} = {expected:g} vs stated {t_val:g} (diff={diff:+.4g}, exact={abs(diff) < 1e-2})"
            )

        # 3. Check sum of child line totals & goods quantities vs parent total
        for k, v in d.items():
            if isinstance(v, str) and "freight" in k.lower():
                freight_terms_vals.append((f"{prefix}.{k}" if prefix else k, v))
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                child_totals = []
                goods_qtys = []
                u_prices = []
                status_counts: dict[str, int] = {}
                for x in v:
                    st_val = x.get("status")
                    if isinstance(st_val, str):
                        status_counts[st_val] = status_counts.get(st_val, 0) + 1
                    is_sur = _is_surcharge_line(x)
                    c_tot = None
                    c_qty = None
                    c_unit = None
                    for ck, cv in x.items():
                        if not isinstance(cv, bool) and isinstance(cv, (int, float)):
                            cnk = _norm_leaf_key(ck)
                            if cnk == "total_amt" and c_tot is None:
                                c_tot = float(cv)
                            elif cnk == "qty" and c_qty is None:
                                c_qty = float(cv)
                            elif cnk == "unit_amt" and c_unit is None:
                                c_unit = float(cv)
                    if c_tot is not None:
                        child_totals.append(c_tot)
                    if is_sur:
                        surcharge_lines.append((f"{prefix}.{k}", str(x.get("sku") or x.get("type") or "SURCHARGE"), c_tot or c_unit or 0.0))
                    else:
                        if c_qty is not None:
                            goods_qtys.append(c_qty)
                        if c_unit is not None:
                            u_prices.append(c_unit)
                tg_key = top_group or prefix or k
                if goods_qtys and tg_key not in group_goods_qty:
                    group_goods_qty[tg_key] = round(sum(goods_qtys), 4)
                if u_prices and tg_key not in group_unit_prices:
                    group_unit_prices[tg_key] = u_prices
                if status_counts and len(v) >= 2:
                    st_summary = ", ".join(f"{sk}:{sv}" for sk, sv in sorted(status_counts.items()))
                    arith_notes.append(f"_history.{prefix}.{k}_statuses (total={len(v)}): {st_summary}")
                if t_val is not None and child_totals:
                    s_tot = round(sum(child_totals), 4)
                    s_diff = round(t_val - s_tot, 4)
                    pct_s = 100.0 * s_diff / max(abs(s_tot), 1e-9)
                    tag = "EXACT_LINES_SUM" if abs(s_diff) < 1e-2 else f"TAX_OR_HEADER_SURCHARGE_DIFF={s_diff:+.4g} ({pct_s:+.2f}%)"
                    arith_notes.append(
                        f"_math.{prefix or 'root'}.lines_sum: sum={s_tot:g} vs parent={t_val:g} ({tag})"
                    )

    def _walk(top_group: str, prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            _check_dict_math(top_group, prefix, obj)
            for k, v in obj.items():
                tg = top_group or str(k)
                p = f"{prefix}.{k}" if prefix else str(k)
                _walk(tg, p, v)
        elif isinstance(obj, list):
            if not obj:
                lines.append(f"{prefix}: [] (empty_list, count=0)")
                return
            if all(isinstance(x, dict) and "role" in x and "text" in x for x in obj):
                cust_turns = sum(1 for x in obj if str(x.get("role")).lower() in ("customer", "user"))
                lines.append(
                    f"{prefix}.turn_count: {len(obj)} (customer_turns={cust_turns}, last_speaker={obj[-1].get('role')})"
                )
                for idx, turn in enumerate(obj):
                    lines.append(f"{prefix}[{idx} · {turn.get('role')}]: {turn.get('text')}")
                return
            if all(not isinstance(x, (dict, list)) for x in obj):
                joined = "; ".join("null" if x is None else str(x) for x in obj)
                lines.append(f"{prefix} (count={len(obj)}): {joined}")
                if all(isinstance(x, str) for x in obj):
                    str_lists.append((prefix, [str(x) for x in obj]))
            else:
                lines.append(f"{prefix}.count: {len(obj)}")
                for idx, item in enumerate(obj):
                    p_i = f"{prefix}[{idx}]"
                    if isinstance(item, dict):
                        _check_dict_math(top_group or prefix, p_i, item)
                        flat_parts = []
                        nested = {}
                        for ik, iv in item.items():
                            sub_p = f"{p_i}.{ik}"
                            if isinstance(iv, (dict, list)):
                                nested[ik] = iv
                            else:
                                flat_parts.append(f"{ik}={iv}")
                                if iv is None:
                                    null_paths.append(sub_p)
                                elif isinstance(iv, (int, float)) and not isinstance(iv, bool):
                                    num_leaves.append((top_group or prefix, sub_p, _norm_leaf_key(ik), float(iv)))
                                elif isinstance(iv, str):
                                    str_leaves.append((top_group or prefix, sub_p, ik.lower(), iv))
                        if flat_parts:
                            lines.append(f"{p_i}: " + ", ".join(flat_parts))
                        for ik, iv in nested.items():
                            _walk(top_group or prefix, f"{p_i}.{ik}", iv)
                    else:
                        _walk(top_group or prefix, p_i, item)
        else:
            leaf_name = prefix.split(".")[-1].split("[")[0]
            if obj is None:
                null_paths.append(prefix)
                lines.append(f"{prefix}: null")
            elif isinstance(obj, bool):
                lines.append(f"{prefix}: {str(obj).lower()}")
            elif isinstance(obj, (int, float)):
                tag = " [NEGATIVE]" if obj < 0 else ""
                lines.append(f"{prefix}: {obj}{tag}")
                nk = _norm_leaf_key(leaf_name)
                tg = top_group or prefix
                num_leaves.append((tg, prefix, nk, float(obj)))
                if nk == "qty" and "[" not in prefix and (
                    tg not in group_goods_qty
                    or any(w in leaf_name.lower() for w in ("received", "delivered", "accepted"))
                ):
                    group_goods_qty[tg] = float(obj)
            else:
                s_val = str(obj)
                lines.append(f"{prefix}: {s_val}")
                if 1 <= len(s_val) <= 100:
                    str_leaves.append((top_group or prefix, prefix, leaf_name.lower(), s_val))

    _walk("", "", state)

    derived: list[str] = []

    # High-signal cross-entity quantity, delivery, freight, PO & currency reconciliation
    if len(group_goods_qty) >= 2:
        q_summary = ", ".join(f"{gk}={gv:g}" for gk, gv in sorted(group_goods_qty.items()))
        rec_tags: list[str] = []
        deliv_q = group_goods_qty.get("delivery")
        inv_q = group_goods_qty.get("invoice")
        po_q = group_goods_qty.get("purchase_order")
        u_match = (
            "invoice" in group_unit_prices
            and "purchase_order" in group_unit_prices
            and group_unit_prices["invoice"] == group_unit_prices["purchase_order"]
        )
        if deliv_q is not None and deliv_q == 0 and (inv_q or 0) > 0:
            rec_tags.append("NOT_YET_DELIVERED_EARLY_INVOICE=True")
        elif deliv_q is not None and inv_q is not None:
            if inv_q > deliv_q:
                rec_tags.append(f"OVERBILLED_VS_DELIVERY=True (diff={inv_q - deliv_q:+g})")
            elif po_q is not None and inv_q == deliv_q and inv_q < po_q:
                rec_tags.append(f"PARTIAL_DELIVERY_BILLED_FOR_RECEIVED=True (unit_price_match={u_match})")
            elif po_q is not None and inv_q == deliv_q == po_q:
                rec_tags.append(f"ALL_QUANTITIES_EXACT_MATCH=True (unit_price_match={u_match})")
        tag_str = (" -> " + "; ".join(rec_tags)) if rec_tags else ""
        derived.append(f"_reconcile.goods_qty: {q_summary}{tag_str}")

    if surcharge_lines:
        for s_path, s_sku, s_amt in surcharge_lines:
            unpermitted = any(
                "not separately billable" in ft.lower() or "prepaid" in ft.lower() or "included" in ft.lower()
                for _, ft in freight_terms_vals
            )
            derived.append(
                f"_reconcile.surcharge_line: {s_path} sku={s_sku!r} amt={s_amt:g} (UNPERMITTED_BY_FREIGHT_TERMS={unpermitted})"
            )

    for _, s_path, s_key, s_val in str_leaves:
        if s_key == "currency" and has_usd_key and s_val.upper() != "USD":
            derived.append(f"_reconcile.currency_mismatch: {s_path}={s_val!r} vs USD-denominated fields (MISMATCH=True)")

    for np_path in null_paths:
        if np_path.endswith(".id") or np_path in ("purchase_order", "po_id", "purchase_order.id"):
            derived.append(f"_reconcile.missing_reference: {np_path}=null (MISSING_AUTHORIZATION_OR_PO=True)")

    derived.extend(arith_notes)

    # Cross-subtree numeric comparisons on matching normalized leaf keys & matching list indices
    seen_pairs: set[tuple[str, str]] = set()
    for i in range(len(num_leaves)):
        g1, p1, k1, v1 = num_leaves[i]
        idx1 = re.findall(r"\[\d+\]", p1)
        for j in range(i + 1, len(num_leaves)):
            g2, p2, k2, v2 = num_leaves[j]
            if g1 != g2 and k1 == k2 and idx1 == re.findall(r"\[\d+\]", p2) and (p1, p2) not in seen_pairs:
                seen_pairs.add((p1, p2))
                diff = v1 - v2
                denom_ref = abs(v2) if abs(v2) > 1e-9 else max(abs(v1), 1e-9)
                pct = 100.0 * diff / denom_ref
                eq = abs(diff) < 1e-6
                derived.append(
                    f"_compare.{p1}_vs_{p2}: exact_equal={eq}, diff={diff:+.4g}, pct_vs_ref={pct:+.2f}%"
                )

    # Cross-subtree short string identifier comparisons
    for i in range(len(str_leaves)):
        g1, p1, k1, s1 = str_leaves[i]
        if k1 not in ("sku", "currency", "vendor", "account", "id", "po_id", "tier", "model"):
            continue
        idx1 = re.findall(r"\[\d+\]", p1)
        for j in range(i + 1, len(str_leaves)):
            g2, p2, k2, s2 = str_leaves[j]
            if g1 != g2 and k1 == k2 and idx1 == re.findall(r"\[\d+\]", p2):
                eq_s = s1.strip().lower() == s2.strip().lower()
                derived.append(f"_compare.{p1}_vs_{p2}: exact_match={eq_s} ({s1!r} vs {s2!r})")

    # Scalar-in-list membership checks (duplicate IDs, allowlists, queues)
    for _, s_path, s_key, s_val in str_leaves:
        if len(s_val) < 2 or (" " in s_val and len(s_val) > 40):
            continue
        for arr_path, arr_vals in str_lists:
            in_arr = s_val in arr_vals or any(s_val.lower() == a.lower() for a in arr_vals)
            key_toks = set(re.findall(r"[a-z]+", s_path.lower()))
            arr_toks = set(re.findall(r"[a-z]+", arr_path.lower()))
            key_stems = {t.rstrip("s") for t in key_toks if len(t) >= 2}
            arr_stems = {t.rstrip("s") for t in arr_toks if len(t) >= 2}
            if in_arr or (key_stems & arr_stems):
                dup_tag = " [DUPLICATE_MATCH]" if in_arr else ""
                derived.append(f"_membership.{s_path}_in_{arr_path}: {in_arr}{dup_tag}")

    if null_paths:
        derived.append("_null_fields: " + ", ".join(null_paths[:12]))

    body_txt = "\n".join(lines)
    t_notes_raw = _plain_text_temporal_notes(body_txt)
    if t_notes_raw.startswith("[Temporal Checks] "):
        derived.insert(0, t_notes_raw.split("\n", 1)[0])

    if derived:
        return "\n".join(["=== Derived Structural Checks ===", *derived[:20], "--- State ---", *lines])

    return body_txt


QTYPE_TO_ID = {"noul": 0, "choice": 1, "score": 2}


def canonical_options(question: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract ordered (option_key, rubric_description) pairs for noul, choice, or score."""
    qtype = question["type"]
    crit = question.get("criteria")
    instr = str(question.get("instructions") or "").strip().rstrip(".")
    if qtype == "score":
        if isinstance(crit, list) and crit:
            return [(str(i), str(desc)) for i, desc in enumerate(crit)]
        mn = int(question.get("min", 0))
        mx = int(question.get("max", 3))
        return [(str(lv), f"Level {lv} on scale {mn}..{mx}") for lv in range(mn, mx + 1)]
    if qtype == "noul":
        cdict = crit if isinstance(crit, dict) else {}
        false_desc = str(
            cdict.get("false")
            or cdict.get("no")
            or (f"False / No — the evidence does NOT support that: {instr}." if instr else "False / No: statement does not hold.")
        )
        true_desc = str(
            cdict.get("true")
            or cdict.get("yes")
            or (f"True / Yes — the evidence confirms that: {instr}." if instr else "True / Yes: statement holds.")
        )
        return [("false", false_desc), ("true", true_desc)]
    if isinstance(crit, dict):
        return [(str(k), str(v) if v else str(k)) for k, v in crit.items()]
    opts = question.get("options") or []
    return [(str(o), str(o)) for o in opts]


def question_signature(qid: str, question: dict[str, Any]) -> str:
    """Order-invariant canonical signature `(qid, qtype, sorted option rubrics)` for per-schema OOF calibration."""
    qtype = str(question.get("type") or "")
    opts = canonical_options(question)
    parts = sorted(f"{k.strip()}={v.strip()[:64]}" for k, v in opts)
    return f"{qid}::{qtype}::" + "||".join(parts)


class CrossOptionBlock(nn.Module):
    """Permutation-equivariant Pre-LN Self-Attention block over the K candidate options."""

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, K, D], valid_mask: [B, K] bool"""
        B, K, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, K, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # [B, nh, K, hd]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        bias = torch.zeros((B, 1, 1, K), dtype=x.dtype, device=x.device).masked_fill(~valid_mask[:, None, None, :], -1e4)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale + bias, dim=-1)
        ctx = torch.matmul(self.drop(attn), v).transpose(1, 2).contiguous().view(B, K, D)
        x = x + self.drop(self.proj(ctx))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class ParallelSpanDecisionHead(nn.Module):
    """v3.0 Single-Pass Parallel-Span Cross-Option + Cross-Question + Ordinal Neighbor Scorer."""

    def __init__(self, hidden_size: int = 1024, comp_dim: int = 512, zs_dim: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.comp_dim = comp_dim
        self.zs_dim = zs_dim

        # Option-to-State fine-grained cross-attention pooling
        self.state_q = nn.Linear(hidden_size * 2, hidden_size)
        self.state_k = nn.Linear(hidden_size, hidden_size)
        self.scale_h = 1.0 / math.sqrt(hidden_size)

        # Fuse [opt_mean, opt_max, q_hdr_mean, state_att, cls_v, zs_logits (2), opt_meta (6)]
        in_dim = hidden_size * 5 + zs_dim + 6
        self.opt_proj = nn.Sequential(
            nn.Linear(in_dim, comp_dim),
            nn.LayerNorm(comp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            CrossOptionBlock(dim=comp_dim, num_heads=8, dropout=dropout),
            CrossOptionBlock(dim=comp_dim, num_heads=8, dropout=dropout),
        ])

        # Cross-question workflow propagation over sibling questions on the same case
        self.wf_norm = nn.LayerNorm(comp_dim)
        self.wf_qkv = nn.Linear(comp_dim, comp_dim * 3)
        self.wf_out = nn.Linear(comp_dim, comp_dim)
        nn.init.zeros_(self.wf_out.weight)
        nn.init.zeros_(self.wf_out.bias)

        # Ordinal neighbor context projection (active ONLY when is_score == 1.0; zero on choice/noul!)
        self.ord_neighbor = nn.Linear(comp_dim * 2, comp_dim)
        nn.init.zeros_(self.ord_neighbor.weight)
        nn.init.zeros_(self.ord_neighbor.bias)

        # Final rival-margin MLP over [x_k, x_k - rival_max, x_k - mean_x]
        self.out_mlp = nn.Sequential(
            nn.Linear(comp_dim * 3, comp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(comp_dim // 2, 1),
        )
        nn.init.zeros_(self.out_mlp[-1].weight)
        nn.init.zeros_(self.out_mlp[-1].bias)

        # Learnable zero-shot NLI prior weights ([entailment, not_entailment] -> +0.55, -0.55)
        init_w = [0.55, -0.55] if zs_dim == 2 else ([0.75, -0.10, -0.65] if zs_dim == 3 else [0.0] * zs_dim)
        self.zs_weights = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
        # Per-question-type log temperatures: [noul (0), choice (1), score (2)]
        self.log_temps = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        # Per-question-type logit uniform-smoothing alpha (in logit space, default 0)
        self.prob_smooth = nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=False)

    def forward_packed_cases(
        self,
        seq_out: torch.Tensor,
        packed_cases: list[dict[str, Any]],
        zs_head_fn: Any | None = None,
        apply_temp: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """
        seq_out: [B_cases, L, H] float32
        packed_cases: list of length B_cases from `encode_case_parallel`
        Returns (logits [Q_total, max_K], flat_questions_meta)
        """
        device = seq_out.device
        H = self.hidden_size
        flat_q_meta: list[dict[str, Any]] = []
        for c_idx, pc in enumerate(packed_cases):
            for qm in pc["questions"]:
                flat_q_meta.append({**qm, "_c_idx": c_idx, "_state_span": pc["state_span"], "_case_id": pc.get("case_id", c_idx)})

        Q_tot = len(flat_q_meta)
        max_K = max(len(qm["option_keys"]) for qm in flat_q_meta)

        opt_mean_t = torch.zeros((Q_tot, max_K, H), dtype=torch.float32, device=device)
        opt_max_t = torch.zeros((Q_tot, max_K, H), dtype=torch.float32, device=device)
        q_hdr_t = torch.zeros((Q_tot, max_K, H), dtype=torch.float32, device=device)
        st_att_t = torch.zeros((Q_tot, max_K, H), dtype=torch.float32, device=device)
        cls_t = torch.zeros((Q_tot, max_K, H), dtype=torch.float32, device=device)
        om_t = torch.zeros((Q_tot, max_K, 6), dtype=torch.float32, device=device)
        vk_t = torch.zeros((Q_tot, max_K), dtype=torch.bool, device=device)
        qt_t = torch.zeros((Q_tot,), dtype=torch.long, device=device)
        cid_t = torch.zeros((Q_tot,), dtype=torch.long, device=device)

        # Precompute state_k per case once for speed
        case_state_h: list[torch.Tensor] = []
        case_state_k: list[torch.Tensor] = []
        for c_idx, pc in enumerate(packed_cases):
            ss, se = pc["state_span"]
            sh = seq_out[c_idx, ss : max(ss + 1, se)]  # [Ls, H]
            case_state_h.append(sh)
            case_state_k.append(self.state_k(sh))      # [Ls, H]

        for q_i, qm in enumerate(flat_q_meta):
            c_idx = qm["_c_idx"]
            cls_v = seq_out[c_idx, 0]
            hs, he = qm["hdr_span"]
            q_hdr_v = seq_out[c_idx, hs : max(hs + 1, he)].mean(dim=0)
            K = len(qm["option_keys"])
            sh = case_state_h[c_idx]
            sk = case_state_k[c_idx]

            om_spans = []
            ox_spans = []
            for k_i, (os_i, oe_i) in enumerate(qm["opt_spans"]):
                sl = seq_out[c_idx, os_i : max(os_i + 1, oe_i)]
                om_spans.append(sl.mean(dim=0))
                ox_spans.append(sl.max(dim=0).values)
            om_stack = torch.stack(om_spans, dim=0)  # [K, H]
            ox_stack = torch.stack(ox_spans, dim=0)  # [K, H]
            q_hdr_exp = q_hdr_v.unsqueeze(0).expand(K, H)

            # Query-conditioned state attention per option: [K, H] @ [H, Ls] -> [K, Ls]
            sq = self.state_q(torch.cat([om_stack, q_hdr_exp], dim=-1))  # [K, H]
            att_w = F.softmax(torch.matmul(sq, sk.t()) * self.scale_h, dim=-1)
            s_att = torch.matmul(att_w, sh)  # [K, H]

            opt_mean_t[q_i, :K] = om_stack
            opt_max_t[q_i, :K] = ox_stack
            q_hdr_t[q_i, :K] = q_hdr_exp
            st_att_t[q_i, :K] = s_att
            cls_t[q_i, :K] = cls_v.unsqueeze(0).expand(K, H)
            om_t[q_i, :K] = torch.tensor(qm["opt_meta"], dtype=torch.float32, device=device)
            vk_t[q_i, :K] = True
            qt_t[q_i] = int(qm["qtype_id"])
            cid_t[q_i] = int(qm["_case_id"])

        if zs_head_fn is not None and self.zs_dim > 0:
            blend_h = 0.5 * opt_mean_t + 0.5 * st_att_t
            zs_logits = zs_head_fn(blend_h).float()  # [Q_tot, max_K, zs_dim]
            zs_priors = (zs_logits * self.zs_weights.view(1, 1, -1)).sum(dim=-1)
        else:
            zs_logits = torch.zeros((Q_tot, max_K, self.zs_dim), dtype=torch.float32, device=device)
            zs_priors = torch.zeros((Q_tot, max_K), dtype=torch.float32, device=device)

        feats = torch.cat([opt_mean_t, opt_max_t, q_hdr_t, st_att_t, cls_t, zs_logits, om_t], dim=-1)
        x = self.opt_proj(feats)  # [Q_tot, max_K, comp_dim]
        x = self.blocks[0](x, vk_t)

        # Ordinal neighbor message passing strictly for `score` scales (is_score == 1)
        is_score_q = (qt_t == 2).to(x.dtype).view(Q_tot, 1, 1)
        if max_K > 1:
            left_nb = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
            right_nb = torch.cat([x[:, 1:, :], x[:, -1:, :]], dim=1)
            # Clamp right neighbor at K_q - 1 boundary
            k_counts = vk_t.sum(dim=1).clamp(min=1)  # [Q_tot]
            for q_i in range(Q_tot):
                kq = int(k_counts[q_i].item())
                if kq < max_K:
                    right_nb[q_i, kq - 1] = x[q_i, kq - 1]
            x = x + is_score_q * self.ord_neighbor(torch.cat([x - left_nb, right_nb - x], dim=-1))

        # Cross-question workflow propagation across sibling questions sharing `_case_id`
        prior_w = F.softmax(zs_priors.masked_fill(~vk_t, -1e4), dim=-1).unsqueeze(-1)
        q_post = (x * prior_w).sum(dim=1)  # [Q_tot, comp_dim]
        qh = self.wf_norm(q_post)
        D = qh.size(-1)
        nh = 8
        hd = D // nh
        qkv = self.wf_qkv(qh).view(Q_tot, 3, nh, hd)
        qw = qkv[:, 0].transpose(0, 1)
        kw = qkv[:, 1].transpose(0, 1)
        vw = qkv[:, 2].transpose(0, 1)
        same_case = cid_t[:, None] == cid_t[None, :]
        wf_bias = torch.zeros((1, Q_tot, Q_tot), dtype=x.dtype, device=device).masked_fill(~same_case.unsqueeze(0), -1e4)
        wf_attn = F.softmax(torch.matmul(qw, kw.transpose(-2, -1)) / math.sqrt(hd) + wf_bias, dim=-1)
        wf_ctx = torch.matmul(wf_attn, vw).transpose(0, 1).contiguous().view(Q_tot, D)
        x = x + self.wf_out(wf_ctx).unsqueeze(1)

        x = self.blocks[1](x, vk_t)

        vk_f = vk_t.to(x.dtype).unsqueeze(-1)
        mean_x = (x * vk_f).sum(dim=1, keepdim=True) / vk_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        eye = torch.eye(max_K, dtype=torch.bool, device=device).unsqueeze(0)
        other_valid = vk_t.unsqueeze(1) & (~eye)
        x_exp = x.unsqueeze(1).expand(Q_tot, max_K, max_K, D)
        masked_other = x_exp.masked_fill(~other_valid.unsqueeze(-1), -1e4)
        rival_max = masked_other.max(dim=2).values
        has_other = other_valid.any(dim=2, keepdim=True)
        rival_max = torch.where(has_other, rival_max, mean_x.expand_as(rival_max))

        comp_feats = torch.cat([x, x - rival_max, x - mean_x], dim=-1)
        res_logits = self.out_mlp(comp_feats).squeeze(-1)
        logits = zs_priors + res_logits

        if apply_temp:
            temps = torch.exp(self.log_temps[qt_t]).clamp(0.15, 5.0).unsqueeze(-1)
            logits = logits / temps
        return logits, flat_q_meta


def encode_case_parallel(
    tokenizer: Any,
    state: Any,
    questions: dict[str, dict[str, Any]],
    max_state_tokens: int = 3072,
    case_id: int = 0,
    precomputed_state_str: str | None = None,
) -> dict[str, Any]:
    """Pack State + ALL questions + ALL candidate options into ONE parallel-positioned token sequence."""
    state_str = precomputed_state_str if precomputed_state_str is not None else compact_state_text(state)
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else (tokenizer.bos_token_id or 0)
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else (tokenizer.eos_token_id or 0)

    state_toks = tokenizer.encode("State:\n" + state_str, add_special_tokens=False)
    if len(state_toks) > max_state_tokens:
        # Preserve head (derived checks + intro) + tail (late overriding exceptions / final turns)
        half_h = int(max_state_tokens * 0.55)
        half_t = max_state_tokens - half_h
        state_toks = state_toks[:half_h] + state_toks[-half_t:]

    input_ids: list[int] = [cls_id, *state_toks, sep_id]
    pos_ids: list[int] = list(range(len(input_ids)))
    q_owner_ids: list[int] = [0] * len(input_ids)
    opt_branch_ids: list[int] = [0] * len(input_ids)
    state_span = (1, 1 + len(state_toks))
    cur_pos = len(input_ids)

    q_records: list[dict[str, Any]] = []
    for q_idx, (qid, question) in enumerate(questions.items(), start=1):
        qtype = question["type"]
        qtype_id = QTYPE_TO_ID.get(qtype, 1)
        opts = canonical_options(question)
        K = len(opts)
        instr = str(question.get("instructions") or "").strip()

        hdr_txt = f"\nQuestion [{qid} | {qtype}]: {instr}\nOptions:"
        hdr_toks = tokenizer.encode(hdr_txt, add_special_tokens=False)[:192]
        if not hdr_toks:
            hdr_toks = [sep_id]
        hdr_start = len(input_ids)
        input_ids.extend(hdr_toks)
        pos_ids.extend(range(cur_pos, cur_pos + len(hdr_toks)))
        q_owner_ids.extend([q_idx] * len(hdr_toks))
        opt_branch_ids.extend([0] * len(hdr_toks))
        hdr_end = len(input_ids)
        cur_pos += len(hdr_toks)

        # CRITICAL: All options of Question `qid` start at the exact same `opt_base_pos`
        # AND are isolated in a parallel tree mask (`opt_branch_ids == k_i + 1`) so sibling options
        # never bleed across identical RoPE coordinates while remaining 100% option-order invariant!
        opt_base_pos = cur_pos
        opt_spans: list[tuple[int, int]] = []
        opt_meta: list[list[float]] = []
        max_o_len = 1

        is_noul = 1.0 if qtype == "noul" else 0.0
        is_choice = 1.0 if qtype == "choice" else 0.0
        is_score = 1.0 if qtype == "score" else 0.0
        inv_k = 1.0 / max(1, K)

        for k_i, (key, rubric) in enumerate(opts):
            if qtype == "noul":
                is_yes = key.lower() in ("true", "yes")
                verdict = "YES / TRUE" if is_yes else "NO / FALSE"
                opt_txt = f" [{verdict}: {rubric}]"
                ord_pos = 1.0 if is_yes else 0.0
            elif qtype == "score":
                opt_txt = f" [SCORE {key} of 0..{max(0, K - 1)}: {rubric}]"
                ord_pos = k_i / max(1, K - 1) if K > 1 else 0.5
            else:
                # Strip single-character positional labels (A/B/C/D/0/1) so option relabeling & permutation are invariant
                clean_k = key.strip()
                if len(clean_k) > 1 and not clean_k.isdigit() and clean_k.lower() != rubric.strip().lower():
                    opt_txt = f" [OPTION {clean_k}: {rubric}]"
                else:
                    opt_txt = f" [OPTION: {rubric}]"
                ord_pos = 0.5  # Exact symmetry across all unordered `choice` options!

            o_toks = tokenizer.encode(opt_txt, add_special_tokens=False)[:159] + [sep_id]
            o_start = len(input_ids)
            input_ids.extend(o_toks)
            pos_ids.extend(range(opt_base_pos, opt_base_pos + len(o_toks)))
            q_owner_ids.extend([q_idx] * len(o_toks))
            opt_branch_ids.extend([k_i + 1] * len(o_toks))
            o_end = len(input_ids)
            opt_spans.append((o_start, o_end))
            max_o_len = max(max_o_len, len(o_toks))
            opt_meta.append([ord_pos, ord_pos * ord_pos, is_noul, is_choice, is_score, inv_k])

        q_records.append({
            "qid": qid,
            "qsig": question_signature(qid, question),
            "qtype": qtype,
            "qtype_id": qtype_id,
            "option_keys": [k for k, _ in opts],
            "hdr_span": (hdr_start, hdr_end),
            "opt_spans": opt_spans,
            "opt_meta": opt_meta,
        })

    return {
        "case_id": case_id,
        "input_ids": input_ids,
        "position_ids": pos_ids,
        "q_owner_ids": q_owner_ids,
        "opt_branch_ids": opt_branch_ids,
        "state_span": state_span,
        "questions": q_records,
    }


def configure_modernbert_full_bidir(backbone: Any, tokenizer: Any | None = None) -> None:
    """Configure ModernBERT / Ettin for 8,192-token full bidirectional SDPA with custom parallel `position_ids` across HF 4.x & 5.x."""
    if tokenizer is not None:
        tokenizer.model_max_length = 8192
    cfg = getattr(backbone, "config", None)
    if cfg is not None:
        cfg.local_attention = 16384
        cfg.sliding_window = 16384
        cfg.global_attn_every_n_layers = 1
        cfg._attn_implementation = "sdpa"
        cfg.reference_compile = False
        if hasattr(cfg, "layer_types") and isinstance(cfg.layer_types, list):
            cfg.layer_types = ["full_attention"] * len(cfg.layer_types)
    enc_model = getattr(backbone, "model", None)
    for layer in getattr(enc_model, "layers", []):
        if hasattr(layer, "attention_type"):
            layer.attention_type = "full_attention"
        attn_mod = getattr(layer, "attn", None)
        if attn_mod is not None and hasattr(attn_mod, "local_attention"):
            attn_mod.local_attention = (-1, -1)


def build_parallel_tree_mask(
    q_t: torch.Tensor,
    o_t: torch.Tensor,
    dtype: torch.dtype,
    return_bool: bool = False,
) -> torch.Tensor:
    """Build exact `[B, 1, L, L]` parallel-option tree mask:
      - State & Question Headers (`o_t == 0`) attend freely to State & Question Headers (`o_t == 0`), never options.
      - Each Option `(q, k)` (`o_t > 0`) attends to State (`q_t == 0`), its own Question Header `q` (`q_t == q, o_t == 0`),
        and ITSELF (`q_t == q, o_t == k`), strictly eliminating cross-option RoPE collision!
    """
    valid = q_t >= 0
    is_opt_q = o_t[:, :, None] > 0
    allow_non_opt = o_t[:, None, :] == 0
    same_q = q_t[:, None, :] == q_t[:, :, None]
    allow_opt = (q_t[:, None, :] == 0) | (same_q & ((o_t[:, None, :] == 0) | (o_t[:, None, :] == o_t[:, :, None])))
    allow = valid[:, :, None] & valid[:, None, :] & torch.where(is_opt_q, allow_opt, allow_non_opt)
    if return_bool:
        return allow.unsqueeze(1)
    neg_val = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9
    mask_4d = torch.zeros(allow.shape, dtype=dtype, device=q_t.device).masked_fill(~allow, neg_val)
    return mask_4d.unsqueeze(1)


def forward_parallel_backbone(
    backbone: Any,
    packed_cases: list[dict[str, Any]],
    pad_id: int,
    device: str,
    amp_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, Any]:
    """Run ONE batched forward pass of ModernBERT across `B` packed multi-question cases with exact parallel tree isolation."""
    raw_max_L = max(len(pc["input_ids"]) for pc in packed_cases)
    pad_L = max(32, ((raw_max_L + 31) // 32) * 32)

    ids_cpu = [pc["input_ids"] + [pad_id] * (pad_L - len(pc["input_ids"])) for pc in packed_cases]
    pos_cpu = [pc["position_ids"] + [0] * (pad_L - len(pc["position_ids"])) for pc in packed_cases]
    qow_cpu = [pc["q_owner_ids"] + [-1] * (pad_L - len(pc["q_owner_ids"])) for pc in packed_cases]
    obr_cpu = [pc["opt_branch_ids"] + [-1] * (pad_L - len(pc["opt_branch_ids"])) for pc in packed_cases]

    ids_t = torch.tensor(ids_cpu, dtype=torch.long, device=device)
    pos_t = torch.tensor(pos_cpu, dtype=torch.long, device=device)
    qow_t = torch.tensor(qow_cpu, dtype=torch.long, device=device)
    obr_t = torch.tensor(obr_cpu, dtype=torch.long, device=device)

    use_amp = (device == "cuda") and (amp_dtype in (torch.bfloat16, torch.float16))
    mask_dt = amp_dtype if use_amp else next(backbone.parameters()).dtype
    if hasattr(backbone, "model"):
        m = backbone.model
        is_hf4 = "sliding_window_mask" in inspect.signature(m.forward).parameters
        tree_mask = build_parallel_tree_mask(qow_t, obr_t, dtype=mask_dt, return_bool=not is_hf4)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            if is_hf4:
                hidden_states = m.embeddings(input_ids=ids_t)
                for encoder_layer in m.layers:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask=tree_mask,
                        sliding_window_mask=tree_mask,
                        position_ids=pos_t,
                    )
                    hidden_states = layer_outputs[0]
                seq_f = m.final_norm(hidden_states).float()
            else:
                enc_out = m(
                    input_ids=ids_t,
                    attention_mask={"full_attention": tree_mask, "sliding_attention": tree_mask},
                    position_ids=pos_t,
                )
                seq_f = enc_out.last_hidden_state.float()

        def zs_fn(h_tensor: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                in_dt = next(backbone.head.parameters()).dtype
                return backbone.classifier(backbone.head(h_tensor.to(in_dt))).float()

        return seq_f, zs_fn

    # Generic fallback for non-ModernBERT encoders
    att_t = (qow_t >= 0).long()
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        out = backbone(input_ids=ids_t, attention_mask=att_t, position_ids=pos_t, output_hidden_states=True)
        seq_f = out.hidden_states[-1].float()
    return seq_f, None


# --- Legacy v2.0 LoRA & Pairwise Helpers Retained for Backward Compatibility ---


class LoRALinear(nn.Module):
    """Rank-r LoRA wrapper preserving frozen base nn.Linear weights."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        out = self.base(x.to(self.base.weight.dtype))
        lora_delta = (self.drop(x.to(self.lora_A.dtype)) @ self.lora_A.t() @ self.lora_B.t()) * self.scale
        return out.to(orig_dtype) + lora_delta.to(orig_dtype)


def inject_encoder_lora(backbone: nn.Module, start_layer: int = 8, r: int = 16, alpha: float = 32.0) -> list[str]:
    """Inject LoRA adapters into DeBERTa-v3 or ELECTRA/BERT encoder layers from `start_layer` up."""
    wrapped: list[str] = []
    if hasattr(backbone, "deberta"):
        layers = backbone.deberta.encoder.layer
    elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        layers = backbone.encoder.layer
    else:
        raise ValueError(f"Unsupported encoder architecture: {type(backbone)}")

    for idx in range(start_layer, len(layers)):
        blk = layers[idx]
        attn_self = blk.attention.self
        for attr in ("query_proj", "key_proj", "value_proj", "query", "key", "value"):
            if hasattr(attn_self, attr):
                orig = getattr(attn_self, attr)
                if isinstance(orig, nn.Linear):
                    setattr(attn_self, attr, LoRALinear(orig, r=r, alpha=alpha))
                    wrapped.append(f"layer.{idx}.attention.self.{attr}")
        if hasattr(blk.attention, "output") and isinstance(blk.attention.output.dense, nn.Linear):
            blk.attention.output.dense = LoRALinear(blk.attention.output.dense, r=r, alpha=alpha)
            wrapped.append(f"layer.{idx}.attention.output.dense")
        if hasattr(blk, "intermediate") and isinstance(blk.intermediate.dense, nn.Linear):
            blk.intermediate.dense = LoRALinear(blk.intermediate.dense, r=r, alpha=alpha)
            wrapped.append(f"layer.{idx}.intermediate.dense")
        if hasattr(blk, "output") and isinstance(blk.output.dense, nn.Linear):
            blk.output.dense = LoRALinear(blk.output.dense, r=r, alpha=alpha)
            wrapped.append(f"layer.{idx}.output.dense")
    return wrapped


class CrossOptionDecisionHead(nn.Module):
    """v2.0 Pairwise Residual NLI-Prior + 2-Layer Cross-Option Competition Scorer (legacy support)."""

    def __init__(self, hidden_size: int = 1024, comp_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_size * 3 + 7, comp_dim),
            nn.LayerNorm(comp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prem_q = nn.Linear(hidden_size, hidden_size)
        self.prem_k = nn.Linear(hidden_size, hidden_size)
        self.scale_h = 1.0 / math.sqrt(hidden_size)
        self.blocks = nn.ModuleList([
            CrossOptionBlock(dim=comp_dim, num_heads=8, dropout=dropout),
            CrossOptionBlock(dim=comp_dim, num_heads=8, dropout=dropout),
        ])
        self.wf_norm = nn.LayerNorm(comp_dim)
        self.wf_qkv = nn.Linear(comp_dim, comp_dim * 3)
        self.wf_out = nn.Linear(comp_dim, comp_dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(comp_dim * 3, comp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(comp_dim // 2, 1),
        )
        self.nli_weights = nn.Parameter(torch.tensor([0.75, -0.10, -0.65], dtype=torch.float32))
        self.log_temps = nn.Parameter(torch.zeros(3, dtype=torch.float32))


class NullspanSystemOneEngine:
    """Unified System One v3.0 inference engine supporting `/v1/systemone` for Typed Decisions & JevBench."""

    def __init__(
        self,
        nli_backbone: Any,
        tokenizer: Any,
        head: nn.Module,
        device: str = "cpu",
        max_length: int = 3072,
        arch_version: int = 3,
    ) -> None:
        self.nli_backbone = nli_backbone
        self.tokenizer = tokenizer
        self.head = head
        self.device = device
        self.max_length = max_length
        self.arch_version = arch_version
        self.schema_calib: dict[str, dict[str, Any]] = {}
        if arch_version >= 3:
            configure_modernbert_full_bidir(self.nli_backbone, self.tokenizer)
        self.amp_dtype = (
            torch.bfloat16
            if (device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8)
            else (torch.float16 if device == "cuda" else torch.float32)
        )
        self.nli_backbone.to(device).eval()
        if device == "cuda" and arch_version == 3:
            self.nli_backbone.to(self.amp_dtype)
        self.head.to(device).eval()

    @classmethod
    def load_from_dir(
        cls,
        model_dir: str,
        device: str | None = None,
        revision: str | None = None,
    ) -> "NullspanSystemOneEngine":
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if not os.path.isdir(model_dir):
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                repo_id=model_dir,
                revision=revision,
                allow_patterns=["systemone/*", "systemone_decision.pt"],
            )

        s1_dir = os.path.join(model_dir, "systemone")
        ckpt_path = os.path.join(model_dir, "systemone_decision.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        arch_version = int(ckpt.get("arch_version", 2))
        backbone_src = s1_dir if os.path.isdir(s1_dir) else ckpt.get(
            "backbone_id", "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
        )

        cfg_json_path = os.path.join(backbone_src, "config.json") if os.path.isdir(backbone_src) else ""
        if cfg_json_path and os.path.isfile(cfg_json_path):
            with open(cfg_json_path, "r", encoding="utf-8") as cf:
                raw_cfg = json.load(cf)
            if raw_cfg.get("model_type") == "modernbert" and raw_cfg.get("global_attn_every_n_layers") == 1:
                raw_cfg["global_attn_every_n_layers"] = 3
                raw_cfg["local_rope_theta"] = float(raw_cfg.get("global_rope_theta", 160000.0))
                raw_cfg.pop("sliding_window", None)
                from transformers import ModernBertConfig
                bb_cfg = ModernBertConfig(**raw_cfg)
            else:
                bb_cfg = AutoConfig.from_pretrained(backbone_src)
        else:
            bb_cfg = AutoConfig.from_pretrained(backbone_src)
        tok = AutoTokenizer.from_pretrained(backbone_src, config=bb_cfg)
        if arch_version >= 3:
            bb_cfg.reference_compile = False
            bb_cfg._attn_implementation = "sdpa"
            bb_cfg.local_attention = 16384
            bb_cfg.global_attn_every_n_layers = 1
            if hasattr(bb_cfg, "layer_types"):
                bb_cfg.layer_types = ["full_attention"] * int(bb_cfg.num_hidden_layers)
        nli_model = AutoModelForSequenceClassification.from_pretrained(backbone_src, config=bb_cfg)
        hidden_size = int(nli_model.config.hidden_size)

        if arch_version >= 3:
            zs_dim = int(ckpt.get("zs_dim", int(getattr(nli_model.config, "num_labels", 2))))
            head: nn.Module = ParallelSpanDecisionHead(
                hidden_size=hidden_size,
                comp_dim=int(ckpt.get("comp_dim", 512)),
                zs_dim=zs_dim,
                dropout=0.0,
            )
            head.load_state_dict(ckpt["head_state_dict"], strict=False)
            max_len = int(ckpt.get("max_state_tokens", 3072))
        else:
            if ckpt.get("lora_r", 0) > 0:
                inject_encoder_lora(
                    nli_model,
                    start_layer=int(ckpt.get("start_layer", 6)),
                    r=int(ckpt["lora_r"]),
                    alpha=float(ckpt.get("lora_alpha", 32.0)),
                )
                nli_model.load_state_dict(ckpt["lora_state_dict"], strict=False)
            if "classifier_state_dict" in ckpt:
                nli_model.load_state_dict(ckpt["classifier_state_dict"], strict=False)
            head = CrossOptionDecisionHead(
                hidden_size=hidden_size,
                comp_dim=int(ckpt.get("comp_dim", 512)),
                dropout=0.0,
            )
            head.load_state_dict(ckpt["head_state_dict"])
            max_len = min(512, int(ckpt.get("max_length", 512)))

        eng = cls(
            nli_backbone=nli_model,
            tokenizer=tok,
            head=head,
            device=device,
            max_length=max_len,
            arch_version=arch_version,
        )
        eng.schema_calib = dict(ckpt.get("schema_calib") or {})
        return eng

    @torch.no_grad()
    def decide(self, state: Any, questions: dict[str, dict[str, Any]], max_length: int | None = None) -> dict[str, Any]:
        max_state_toks = max_length or self.max_length
        state_str = compact_state_text(state)
        pc = encode_case_parallel(
            self.tokenizer,
            state,
            questions,
            max_state_tokens=max_state_toks,
            case_id=0,
            precomputed_state_str=state_str,
        )
        pad_id = self.tokenizer.pad_token_id or 0
        seq_f, zs_fn = forward_parallel_backbone(
            self.nli_backbone,
            [pc],
            pad_id=pad_id,
            device=self.device,
            amp_dtype=self.amp_dtype,
        )
        assert isinstance(self.head, ParallelSpanDecisionHead)
        raw_logits, flat_q_meta = self.head.forward_packed_cases(seq_f, [pc], zs_head_fn=zs_fn, apply_temp=False)

        total_in_toks = len(pc["input_ids"])
        answers: dict[str, Any] = {}

        for b_i, qm in enumerate(flat_q_meta):
            qid = qm["qid"]
            qsig = qm.get("qsig", "")
            keys = qm["option_keys"]
            K = len(keys)
            qtype = qm["qtype"]
            qt_id = int(qm["qtype_id"])
            lv = raw_logits[b_i, :K].float().cpu()
            sc = self.schema_calib.get(qsig)
            if sc is not None:
                t_q = max(0.25, float(sc.get("temp", 1.0)))
                b_map = sc.get("bias") or {}
                b_vec = torch.tensor([float(b_map.get(k, 0.0)) for k in keys], dtype=torch.float32)
                lv = lv / t_q + b_vec
            else:
                t_g = float(torch.exp(self.head.log_temps[qt_id]).clamp(0.15, 5.0).item())
                lv = lv / t_g
            probs_raw = F.softmax(lv, dim=-1).tolist()
            smooth_eps = float(sc.get("smooth", self.head.prob_smooth[qt_id].item())) if sc is not None else float(self.head.prob_smooth[qt_id].item())
            if smooth_eps > 0.0:
                u = 1.0 / max(1, K)
                probs_raw = [(1.0 - smooth_eps) * p + smooth_eps * u for p in probs_raw]
            s_p = sum(probs_raw)
            probs = [p / s_p for p in probs_raw]
            best_idx = max(range(K), key=lambda idx: probs[idx])

            if qtype == "noul":
                p_true = float(probs[keys.index("true")]) if "true" in keys else float(probs[-1])
                p_true_r = round(min(1.0, max(0.0, p_true)), 6)
                answers[qid] = {
                    "type": "noul",
                    "noul": p_true_r,
                    "probabilities": {"yes": p_true_r, "no": round(1.0 - p_true_r, 6)},
                }
            elif qtype == "score":
                prob_map = {k: round(float(p), 6) for k, p in zip(keys, probs)}
                diff = round(1.0 - sum(prob_map.values()), 6)
                prob_map[keys[best_idx]] = round(min(1.0, max(0.0, prob_map[keys[best_idx]] + diff)), 6)
                ev = sum(float(k) * prob_map[k] for k in keys)
                answers[qid] = {
                    "type": "score",
                    "score": round(ev, 6),
                    "probabilities": prob_map,
                }
            else:
                prob_map = {k: round(float(p), 6) for k, p in zip(keys, probs)}
                diff = round(1.0 - sum(prob_map.values()), 6)
                prob_map[keys[best_idx]] = round(min(1.0, max(0.0, prob_map[keys[best_idx]] + diff)), 6)
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[best_idx],
                    "probabilities": prob_map,
                }

        return {
            "model": "gnumanth/nullspan",
            "answers": answers,
            "usage": {"input_tokens": total_in_toks, "output_tokens": 0},
        }
