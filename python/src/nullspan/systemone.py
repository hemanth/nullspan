"""Nullspan System One Probabilistic Typed-Decision Engine v2.0 (`/v1/systemone`).

Pure Bidirectional Cross-Encoder Architecture (Zero Autoregressive Output Tokens):
  1. Schema-Agnostic Symbolic State Perception (`compact_state_text`):
     Deterministically surfaces cross-subtree numeric deltas/percentages, line-item
     product/sum checks, string-in-array set membership, thread structure, and null flags.
  2. Pairwise Premise-Hypothesis Cross-Encoding (`max_length = 640`):
     Scores every candidate option `(State, Question + Option_k Rubric)` in ONE single
     batched GPU forward pass (`output_tokens = 0`) with full bidirectional attention.
  3. Residual NLI Entailment Prior + 2-Layer Cross-Option Competition Transformer:
     Combines pretrained 3-way NLI (`entailment - contradiction`) log-odds with a
     permutation-equivariant Cross-Option Self-Attention & Rival-Margin head (`h_k - max_{j!=k} h_j`)
     and per-family (`noul`, `choice`, `score`) calibrated temperature scaling.
"""
from __future__ import annotations

import datetime as dt
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


def compact_state_text(state: Any) -> str:
    """Flatten structured JSON/dict state and prepend schema-agnostic symbolic & temporal checks."""
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

    def _check_dict_math(prefix: str, d: dict[str, Any]) -> None:
        # Check qty * unit == total inside any dict
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
                f"_math.{prefix or 'root'}.qty_x_unit: {q_val:g} * {u_val:g} = {expected:g} vs stated {t_val:g} (diff={diff:+.4g})"
            )

        # Check sum of child line totals vs parent total
        if t_val is not None:
            for k, v in d.items():
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    child_totals = []
                    for x in v:
                        for ck, cv in x.items():
                            if not isinstance(cv, bool) and isinstance(cv, (int, float)) and _norm_leaf_key(ck) == "total_amt":
                                child_totals.append(float(cv))
                                break
                    if child_totals:
                        s_tot = round(sum(child_totals), 4)
                        s_diff = round(t_val - s_tot, 4)
                        arith_notes.append(
                            f"_math.{prefix or 'root'}.lines_sum: sum={s_tot:g} vs parent={t_val:g} (diff={s_diff:+.4g}, exact={abs(s_diff) < 1e-2})"
                        )

    def _walk(top_group: str, prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            _check_dict_math(prefix, obj)
            for k, v in obj.items():
                tg = top_group or str(k)
                p = f"{prefix}.{k}" if prefix else str(k)
                _walk(tg, p, v)
        elif isinstance(obj, list):
            if not obj:
                lines.append(f"{prefix}: [] (empty_list, count=0)")
                return
            if all(isinstance(x, dict) and "role" in x and "text" in x for x in obj):
                lines.append(f"{prefix}.turn_count: {len(obj)} (last_speaker={obj[-1].get('role')})")
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
                        _check_dict_math(p_i, item)
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
                num_leaves.append((top_group or prefix, prefix, _norm_leaf_key(leaf_name), float(obj)))
            else:
                s_val = str(obj)
                lines.append(f"{prefix}: {s_val}")
                if 1 <= len(s_val) <= 80:
                    str_leaves.append((top_group or prefix, prefix, leaf_name.lower(), s_val))

    _walk("", "", state)

    # Cross-subtree numeric comparisons on matching normalized leaf keys & matching list indices
    derived: list[str] = list(arith_notes)
    seen_pairs: set[tuple[str, str]] = set()
    for i in range(len(num_leaves)):
        g1, p1, k1, v1 = num_leaves[i]
        idx1 = re.findall(r"\[\d+\]", p1)
        for j in range(i + 1, len(num_leaves)):
            g2, p2, k2, v2 = num_leaves[j]
            if g1 != g2 and k1 == k2 and idx1 == re.findall(r"\[\d+\]", p2) and (p1, p2) not in seen_pairs:
                seen_pairs.add((p1, p2))
                diff = v1 - v2
                denom = max(abs(v2), abs(v1), 1e-9)
                pct = 100.0 * diff / denom
                eq = abs(diff) < 1e-6
                derived.append(
                    f"_compare.{p1}_vs_{p2}: exact_equal={eq}, diff={diff:+.4g}, pct_diff={pct:+.2f}%"
                )

    # Cross-subtree short string identifier comparisons (sku, currency, vendor, tier, status)
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
        if len(s_val) < 2 or " " in s_val and len(s_val) > 40:
            continue
        for arr_path, arr_vals in str_lists:
            in_arr = s_val in arr_vals or any(s_val.lower() == a.lower() for a in arr_vals)
            key_toks = set(re.findall(r"[a-z]+", s_path.lower()))
            arr_toks = set(re.findall(r"[a-z]+", arr_path.lower()))
            # Strip trailing plural 's' for token overlap (e.g. invoice.id vs prior_invoice_ids)
            key_stems = {t.rstrip("s") for t in key_toks if len(t) >= 2}
            arr_stems = {t.rstrip("s") for t in arr_toks if len(t) >= 2}
            if in_arr or (key_stems & arr_stems):
                derived.append(f"_membership.{s_path}_in_{arr_path}: {in_arr}")

    if null_paths:
        derived.append("_null_fields: " + ", ".join(null_paths[:12]))

    body_txt = "\n".join(lines)
    t_notes_raw = _plain_text_temporal_notes(body_txt)
    if t_notes_raw.startswith("[Temporal Checks] "):
        derived.insert(0, t_notes_raw.split("\n", 1)[0])

    if derived:
        return "\n".join(["=== Derived Structural Checks ===", *derived[:16], "--- State ---", *lines])

    return body_txt


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


class CrossOptionDecisionHead(nn.Module):
    """Residual NLI-Prior + 2-Layer Cross-Option Competition & Rival-Margin Scorer."""

    def __init__(self, hidden_size: int = 1024, comp_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        # Fuse [CLS], Hypothesis Mean, Premise Attended Evidence, and 3 NLI logits + 4 ordinal/type features
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
        # Cross-question workflow propagation over posterior-weighted question states (same_case_mask)
        self.wf_norm = nn.LayerNorm(comp_dim)
        self.wf_qkv = nn.Linear(comp_dim, comp_dim * 3)
        self.wf_out = nn.Linear(comp_dim, comp_dim)
        nn.init.zeros_(self.wf_out.weight)
        nn.init.zeros_(self.wf_out.bias)

        # Score [h_k, h_k - rival_max, h_k - mean_h]
        self.out_mlp = nn.Sequential(
            nn.Linear(comp_dim * 3, comp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(comp_dim // 2, 1),
        )
        # Zero-init final residual layer so step-0 starts cleanly at the 885k-pair NLI entailment prior
        nn.init.zeros_(self.out_mlp[-1].weight)
        nn.init.zeros_(self.out_mlp[-1].bias)

        # Learnable NLI prior weights [entailment, neutral, contradiction]
        self.nli_weights = nn.Parameter(torch.tensor([0.75, -0.10, -0.65], dtype=torch.float32))
        # Per-question-type log temperatures: [noul (0), choice (1), score (2)]
        self.log_temps = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def pool_pair_features(
        self,
        seq_out: torch.Tensor,
        nli_logits: torch.Tensor,
        prem_mask: torch.Tensor,
        hyp_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        seq_out: [P, L, H]
        nli_logits: [P, 3]
        prem_mask, hyp_mask: [P, L] float/bool
        Returns (pair_repr [P, 3*H], nli_prior [P])
        """
        cls_v = seq_out[:, 0, :]  # [P, H]
        hm = hyp_mask.to(seq_out.dtype)
        hyp_v = (seq_out * hm.unsqueeze(-1)).sum(dim=1) / hm.sum(dim=1, keepdim=True).clamp(min=1.0)

        pm = prem_mask.to(seq_out.dtype).clone()
        pm[:, 0] = 1.0
        q = self.prem_q(hyp_v).unsqueeze(1)  # [P, 1, H]
        k = self.prem_k(seq_out)             # [P, L, H]
        att_sc = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * self.scale_h + (1.0 - pm) * (-1e4)
        att_w = F.softmax(att_sc, dim=-1)
        prem_v = torch.bmm(att_w.unsqueeze(1), seq_out).squeeze(1)  # [P, H]

        nli_prior = (nli_logits * self.nli_weights.unsqueeze(0)).sum(dim=-1)  # [P]
        return torch.cat([cls_v, hyp_v, prem_v], dim=-1), nli_prior

    def score_grouped_options(
        self,
        pair_vecs: torch.Tensor,
        nli_logits: torch.Tensor,
        nli_priors: torch.Tensor,
        opt_meta: torch.Tensor,
        valid_k: torch.Tensor,
        qtype_ids: torch.Tensor,
        case_ids: torch.Tensor | None = None,
        apply_temp: bool = False,
    ) -> torch.Tensor:
        """
        pair_vecs: [B, K, 3*H]
        nli_logits: [B, K, 3]
        nli_priors: [B, K]
        opt_meta: [B, K, 4]  (norm_pos, norm_pos^2, is_score, inv_K)
        valid_k: [B, K] bool
        qtype_ids: [B] long (0=noul, 1=choice, 2=score)
        case_ids: [B] long grouping questions belonging to the same workflow state
        Returns logits: [B, K]
        """
        B, K, _ = pair_vecs.shape
        x = self.pair_proj(torch.cat([pair_vecs, nli_logits, opt_meta], dim=-1))  # [B, K, comp_dim]
        x = self.blocks[0](x, valid_k)

        # Propagate posterior-weighted latent conclusions across sibling questions on the same state
        prior_w = F.softmax(nli_priors.masked_fill(~valid_k, -1e4), dim=-1).unsqueeze(-1)  # [B, K, 1]
        q_post = (x * prior_w).sum(dim=1)  # [B, comp_dim]
        qh = self.wf_norm(q_post)
        D = qh.size(-1)
        nh = 8
        hd = D // nh
        qkv = self.wf_qkv(qh).view(B, 3, nh, hd)
        qw = qkv[:, 0].transpose(0, 1)  # [nh, B, hd]
        kw = qkv[:, 1].transpose(0, 1)
        vw = qkv[:, 2].transpose(0, 1)
        if case_ids is None:
            same_case = torch.ones((B, B), dtype=torch.bool, device=x.device)
        else:
            same_case = case_ids[:, None] == case_ids[None, :]
        wf_bias = torch.zeros((1, B, B), dtype=x.dtype, device=x.device).masked_fill(~same_case.unsqueeze(0), -1e4)
        wf_attn = F.softmax(torch.matmul(qw, kw.transpose(-2, -1)) / math.sqrt(hd) + wf_bias, dim=-1)
        wf_ctx = torch.matmul(wf_attn, vw).transpose(0, 1).contiguous().view(B, D)
        x = x + self.wf_out(wf_ctx).unsqueeze(1)

        x = self.blocks[1](x, valid_k)

        # Compute mean and strongest-rival ("max other") representations across valid options
        vk_f = valid_k.to(x.dtype).unsqueeze(-1)  # [B, K, 1]
        mean_x = (x * vk_f).sum(dim=1, keepdim=True) / vk_f.sum(dim=1, keepdim=True).clamp(min=1.0)

        # Rival max: for each k, max over j != k among valid options
        eye = torch.eye(K, dtype=torch.bool, device=x.device).unsqueeze(0)  # [1, K, K]
        other_valid = valid_k.unsqueeze(1) & (~eye)                         # [B, K, K]
        x_exp = x.unsqueeze(1).expand(B, K, K, x.size(-1))                  # [B, K, K, D]
        masked_other = x_exp.masked_fill(~other_valid.unsqueeze(-1), -1e4)
        rival_max = masked_other.max(dim=2).values
        # Fallback if K == 1
        has_other = other_valid.any(dim=2, keepdim=True)
        rival_max = torch.where(has_other, rival_max, mean_x.expand_as(rival_max))

        comp_feats = torch.cat([x, x - rival_max, x - mean_x], dim=-1)
        res_logits = self.out_mlp(comp_feats).squeeze(-1)  # [B, K]
        logits = nli_priors + res_logits

        if apply_temp:
            temps = torch.exp(self.log_temps[qtype_ids]).clamp(0.15, 5.0).unsqueeze(-1)
            logits = logits / temps
        return logits


QTYPE_TO_ID = {"noul": 0, "choice": 1, "score": 2}


def canonical_options(question: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract ordered (option_key, rubric_description) pairs for noul, choice, or score."""
    qtype = question["type"]
    crit = question.get("criteria")
    if qtype == "score":
        return [(str(i), str(desc)) for i, desc in enumerate(crit or [])]
    if qtype == "noul":
        cdict = crit if isinstance(crit, dict) else {}
        return [
            ("false", str(cdict.get("false") or cdict.get("no") or "False / No: the statement does not hold.")),
            ("true", str(cdict.get("true") or cdict.get("yes") or "True / Yes: the statement holds based on the state.")),
        ]
    if isinstance(crit, dict):
        return [(str(k), str(v) if v else str(k)) for k, v in crit.items()]
    opts = question.get("options") or []
    return [(str(o), str(o)) for o in opts]


def build_hypothesis_text(
    qid: str,
    question: dict[str, Any],
    opt_idx: int,
    opts: list[tuple[str, str]],
    peer_q_summary: str = "",
) -> str:
    qtype = question["type"]
    instr = str(question.get("instructions") or "").strip()
    key, rubric = opts[opt_idx]
    K = len(opts)
    others = [f"{ok}" for idx, (ok, _) in enumerate(opts) if idx != opt_idx]
    others_s = ", ".join(others[:8])
    if qtype == "noul":
        verdict = "YES / TRUE" if key in ("true", "yes") else "NO / FALSE"
        base = f"Decision [{qid} · yes/no]: {instr}\nHypothesis ({verdict}): {rubric}"
    elif qtype == "score":
        base = (
            f"Ordinal Rating [{qid} · level {key} of 0..{K-1}]: {instr}\n"
            f"Hypothesis (Level {key}): {rubric}"
        )
    else:
        base = (
            f"Choice [{qid} · 1 of {K} ({key} vs {others_s})]: {instr}\n"
            f"Hypothesis ({key}): {rubric}"
        )
    if peer_q_summary:
        base = f"{base}\nWorkflow context: {peer_q_summary}"
    return base


_FOCUS_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has", "have",
    "had", "not", "but", "what", "which", "who", "whom", "when", "where", "why", "how", "does",
    "did", "can", "could", "should", "would", "will", "into", "onto", "upon", "over", "under",
    "based", "state", "given", "select", "choose", "rate", "level", "option", "true", "false",
    "yes", "statement", "holds", "whether", "strictly", "provided", "rubric", "criteria",
}


def focus_state_for_question(
    state_str: str,
    qid: str,
    question: dict[str, Any],
    opts: list[tuple[str, str]],
    max_chars: int = 1480,
) -> str:
    """Preserve Head (derived checks + intro) + Query-Matched Middle Clauses + Tail (late turns/exceptions)."""
    if len(state_str) <= max_chars:
        return state_str

    head_budget = 460
    tail_budget = 440
    mid_budget = max(240, max_chars - head_budget - tail_budget - 30)

    head_part = state_str[:head_budget]
    nl_h = head_part.rfind("\n")
    if nl_h >= 260:
        head_part = head_part[:nl_h]

    tail_part = state_str[-tail_budget:]
    nl_t = tail_part.find("\n")
    if 0 <= nl_t <= 160:
        tail_part = tail_part[nl_t + 1 :]

    mid_region = state_str[len(head_part) : max(len(head_part), len(state_str) - len(tail_part))].strip()
    if not mid_region:
        return f"{head_part}\n...\n{tail_part}"

    q_blob = f"{qid} {question.get('instructions', '')} " + " ".join(f"{k} {r}" for k, r in opts)
    q_stems = {
        w[:5] if len(w) >= 6 else w.rstrip("s")
        for w in re.findall(r"[a-z0-9]{3,}", q_blob.lower())
        if w not in _FOCUS_STOP
    }
    chunks = [c.strip() for c in re.split(r"(?<=[.!?\n])\s+", mid_region) if c.strip()]
    if not chunks:
        return f"{head_part}\n...\n{tail_part}"

    scored: list[tuple[float, int, str]] = []
    for idx, ch in enumerate(chunks):
        cl = ch.lower()
        hits = sum(1.0 for st in q_stems if st and st in cl)
        # Boost explicit exception/override/negation/threshold clauses in long policies & traces
        if any(kw in cl for kw in ("except", "unless", "override", "carveout", "however", "critical", "error", "fail", "breach", "null", "missing")):
            hits += 0.85
        scored.append((hits, idx, ch))

    scored.sort(key=lambda x: (x[0], -abs(x[1] - len(chunks) // 2)), reverse=True)
    picked_idx: list[int] = []
    used = 0
    for sc, idx, ch in scored:
        if used + len(ch) + 2 > mid_budget and picked_idx:
            continue
        picked_idx.append(idx)
        used += len(ch) + 2
        if used >= mid_budget:
            break

    picked_idx.sort()
    mid_selected = "\n".join(chunks[i] for i in picked_idx)
    return f"{head_part}\n[...]\n{mid_selected}\n[...]\n{tail_part}"


def encode_decision_question(
    tokenizer: Any,
    state: Any,
    qid: str,
    question: dict[str, Any],
    peer_q_summary: str = "",
    max_length: int = 512,
    precomputed_state_str: str | None = None,
) -> dict[str, Any]:
    """Encode all K candidate options for one question as Premise=FocusedState, Hypothesis=Option_k."""
    opts = canonical_options(question)
    qtype = question["type"]
    K = len(opts)
    raw_state_str = precomputed_state_str if precomputed_state_str is not None else compact_state_text(state)
    state_str = focus_state_for_question(raw_state_str, qid, question, opts, max_chars=max(950, int(max_length * 2.9)))

    pair_ids: list[list[int]] = []
    pair_attn: list[list[int]] = []
    pair_prem: list[list[float]] = []
    pair_hyp: list[list[float]] = []
    opt_meta: list[list[float]] = []

    is_score = 1.0 if qtype == "score" else 0.0
    inv_k = 1.0 / max(1, K)

    for k_i in range(K):
        hyp = build_hypothesis_text(qid, question, k_i, opts, peer_q_summary=peer_q_summary)
        enc = tokenizer(
            state_str,
            hyp,
            truncation="only_first",
            max_length=max_length,
        )
        ids = enc["input_ids"]
        seq_ids = enc.sequence_ids()
        L = len(ids)
        pm = [1.0 if s == 0 else 0.0 for s in seq_ids]
        hm = [1.0 if s == 1 else 0.0 for s in seq_ids]
        if sum(hm) == 0.0:
            hm[0] = 1.0
        pair_ids.append(ids)
        pair_attn.append(enc["attention_mask"])
        pair_prem.append(pm)
        pair_hyp.append(hm)
        npos = k_i / max(1, K - 1) if K > 1 else 0.5
        opt_meta.append([npos, npos * npos, is_score, inv_k])

    return {
        "qid": qid,
        "qtype": qtype,
        "qtype_id": QTYPE_TO_ID.get(qtype, 1),
        "option_keys": [k for k, _ in opts],
        "pair_ids": pair_ids,
        "pair_attn": pair_attn,
        "pair_prem": pair_prem,
        "pair_hyp": pair_hyp,
        "opt_meta": opt_meta,
    }


class NullspanSystemOneEngine:
    """Unified System One inference engine supporting `/v1/systemone` for Typed Decisions & JevBench."""

    def __init__(
        self,
        nli_backbone: Any,
        tokenizer: Any,
        head: CrossOptionDecisionHead,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self.nli_backbone = nli_backbone
        self.tokenizer = tokenizer
        self.head = head
        self.device = device
        self.max_length = max_length
        self.nli_backbone.to(device).eval()
        if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8 and hasattr(self.nli_backbone, "deberta"):
            self.nli_backbone.deberta.to(torch.bfloat16)
        self.head.to(device).eval()

    @classmethod
    def load_from_dir(
        cls,
        model_dir: str,
        device: str | None = None,
        revision: str | None = None,
    ) -> "NullspanSystemOneEngine":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        backbone_src = s1_dir if os.path.isdir(s1_dir) else ckpt.get(
            "backbone_id", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        )

        tok = AutoTokenizer.from_pretrained(backbone_src)
        nli_model = AutoModelForSequenceClassification.from_pretrained(backbone_src)
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

        hidden_size = int(nli_model.config.hidden_size)
        head = CrossOptionDecisionHead(
            hidden_size=hidden_size,
            comp_dim=int(ckpt.get("comp_dim", 512)),
            dropout=0.0,
        )
        head.load_state_dict(ckpt["head_state_dict"])
        return cls(
            nli_backbone=nli_model,
            tokenizer=tok,
            head=head,
            device=device,
            max_length=min(512, int(ckpt.get("max_length", 512))),
        )

    @torch.no_grad()
    def decide(self, state: Any, questions: dict[str, dict[str, Any]], max_length: int | None = None) -> dict[str, Any]:
        max_len = max_length or self.max_length
        qids = list(questions.keys())
        state_str = compact_state_text(state)
        peer_summary = ", ".join(f"{qk}({questions[qk].get('type')})" for qk in qids) if len(qids) > 1 else ""

        encoded_qs = [
            encode_decision_question(
                self.tokenizer,
                state,
                qid,
                questions[qid],
                peer_q_summary=peer_summary,
                max_length=max_len,
                precomputed_state_str=state_str,
            )
            for qid in qids
        ]

        # Flatten all pairs across all questions into ONE single parallel GPU batch
        flat_ids: list[list[int]] = []
        flat_attn: list[list[int]] = []
        flat_prem: list[list[float]] = []
        flat_hyp: list[list[float]] = []
        q_slices: list[tuple[int, int]] = []

        for eq in encoded_qs:
            s_i = len(flat_ids)
            flat_ids.extend(eq["pair_ids"])
            flat_attn.extend(eq["pair_attn"])
            flat_prem.extend(eq["pair_prem"])
            flat_hyp.extend(eq["pair_hyp"])
            q_slices.append((s_i, len(flat_ids)))

        P = len(flat_ids)
        raw_max_L = max(len(x) for x in flat_ids)
        pad_L = min(max_len, max(64, ((raw_max_L + 31) // 32) * 32))
        pad_id = self.tokenizer.pad_token_id or 0

        ids_cpu = [ids[:pad_L] + [pad_id] * (pad_L - min(len(ids), pad_L)) for ids in flat_ids]
        attn_cpu = [at[:pad_L] + [0] * (pad_L - min(len(at), pad_L)) for at in flat_attn]
        prem_cpu = [pm[:pad_L] + [0.0] * (pad_L - min(len(pm), pad_L)) for pm in flat_prem]
        hyp_cpu = [hm[:pad_L] + [0.0] * (pad_L - min(len(hm), pad_L)) for hm in flat_hyp]

        ids_t = torch.tensor(ids_cpu, dtype=torch.long, device=self.device)
        attn_t = torch.tensor(attn_cpu, dtype=torch.long, device=self.device)
        prem_t = torch.tensor(prem_cpu, dtype=torch.float32, device=self.device)
        hyp_t = torch.tensor(hyp_cpu, dtype=torch.float32, device=self.device)

        seq_chunks: list[torch.Tensor] = []
        nli_chunks: list[torch.Tensor] = []
        for c_s in range(0, P, 24):
            ids_c = ids_t[c_s : c_s + 24]
            attn_c = attn_t[c_s : c_s + 24]
            if hasattr(self.nli_backbone, "deberta"):
                enc_out = self.nli_backbone.deberta(input_ids=ids_c, attention_mask=attn_c)
                seq_c = enc_out.last_hidden_state.float()
                pooled_c = self.nli_backbone.pooler(seq_c)
                nli_c = self.nli_backbone.classifier(pooled_c).float()
            else:
                out = self.nli_backbone(input_ids=ids_c, attention_mask=attn_c, output_hidden_states=True)
                seq_c = out.hidden_states[-1].float()
                nli_c = out.logits.float()
            seq_chunks.append(seq_c)
            nli_chunks.append(nli_c)
        seq_f = torch.cat(seq_chunks, dim=0)
        nli_f = torch.cat(nli_chunks, dim=0)
        pair_vecs_flat, nli_priors_flat = self.head.pool_pair_features(seq_f, nli_f, prem_t, hyp_t)

        B = len(encoded_qs)
        max_K = max(len(eq["option_keys"]) for eq in encoded_qs)
        feat_dim = pair_vecs_flat.size(-1)

        pv_g = torch.zeros((B, max_K, feat_dim), dtype=torch.float32, device=self.device)
        nl_g = torch.zeros((B, max_K, 3), dtype=torch.float32, device=self.device)
        np_g = torch.zeros((B, max_K), dtype=torch.float32, device=self.device)
        om_g = torch.zeros((B, max_K, 4), dtype=torch.float32, device=self.device)
        vk_g = torch.zeros((B, max_K), dtype=torch.bool, device=self.device)
        qt_g = torch.zeros((B,), dtype=torch.long, device=self.device)

        for b_i, eq in enumerate(encoded_qs):
            s_i, e_i = q_slices[b_i]
            K = e_i - s_i
            pv_g[b_i, :K] = pair_vecs_flat[s_i:e_i]
            nl_g[b_i, :K] = nli_f[s_i:e_i]
            np_g[b_i, :K] = nli_priors_flat[s_i:e_i]
            om_g[b_i, :K] = torch.tensor(eq["opt_meta"], dtype=torch.float32, device=self.device)
            vk_g[b_i, :K] = True
            qt_g[b_i] = eq["qtype_id"]

        logits = self.head.score_grouped_options(pv_g, nl_g, np_g, om_g, vk_g, qt_g, apply_temp=True)

        # Estimate realistic input tokens (state + questions once, not Kx duplicated)
        approx_state_toks = len(self.tokenizer.encode(state_str, add_special_tokens=False))
        approx_q_toks = sum(45 * len(eq["option_keys"]) for eq in encoded_qs)
        total_in_toks = approx_state_toks + approx_q_toks

        answers: dict[str, Any] = {}
        for b_i, qid in enumerate(qids):
            eq = encoded_qs[b_i]
            keys = eq["option_keys"]
            K = len(keys)
            probs_t = F.softmax(logits[b_i, :K], dim=-1).cpu().tolist()
            s_p = sum(probs_t)
            probs = [p / s_p for p in probs_t]
            best_idx = max(range(K), key=lambda idx: probs[idx])
            qtype = eq["qtype"]
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
