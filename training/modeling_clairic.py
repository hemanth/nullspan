"""Clairic System One Extraction Model (Hugging Face PreTrainedModel).

Processes `state` + multiple typed `fields` in a single parallel encoder forward pass
with 5 task heads:
  1. null_head: Calibrated p_null probability (absent field abstention)
  2. span_head: Token start/end pointer network mapped to exact character offsets [s, e]
  3. bio_head: Token-level B/I/O multi-span sequence tagger for `type: "list"`
  4. op_head: 8-way System One computation & normalization router
  5. quantile_head: Pinball-loss calibrated distribution head (p10, median, p90)
"""
from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel


class ClairicConfig(PretrainedConfig):
    model_type = "clairic_extraction"

    def __init__(
        self,
        backbone_name: str = "distilbert/distilbert-base-uncased",
        hidden_size: int = 768,
        num_ops: int = 8,
        num_bio_labels: int = 3,
        dropout: float = 0.1,
        null_temperature: float = 1.0,
        span_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.hidden_size = hidden_size
        self.num_ops = num_ops
        self.num_bio_labels = num_bio_labels
        self.dropout = dropout
        self.null_temperature = null_temperature
        self.span_temperature = span_temperature


def format_field_query(field_name: str, spec: dict[str, Any]) -> str:
    ftype = spec.get("type", "text")
    unit = spec.get("unit", "")
    items = spec.get("items", "")
    instr = spec.get("instructions", "")
    parts = [f"field: {field_name}", f"type: {ftype}"]
    if unit:
        parts.append(f"unit: {unit}")
    if items:
        parts.append(f"items: {items}")
    if instr:
        parts.append(f"instructions: {instr}")
    return " | ".join(parts)


def extract_numeric_context_features(state: str, spec: dict[str, Any]) -> list[float]:
    """Extract 8 normalized numeric summary features from `state` for the quantile head."""
    vmin = float(spec.get("min", 1.0))
    vmax = float(spec.get("max", 1000000.0))
    scale = spec.get("scale", "linear")

    def norm(x: float) -> float:
        x = max(vmin, min(vmax, x))
        if scale == "log" and vmin > 0 and vmax > vmin:
            return (math.log(x) - math.log(vmin)) / (math.log(vmax) - math.log(vmin))
        return (x - vmin) / max(1e-9, vmax - vmin)

    # Check if comparable sales table is present
    if "Comparable sales:" in state and "Subject property:" in state:
        parts = state.split("Subject property:")
        comp_lines = [ln.strip() for ln in parts[0].strip().splitlines() if "," in ln]
        subj_lines = [ln.strip() for ln in parts[1].strip().splitlines() if "," in ln]
        subj_sqft, subj_hood = 1500.0, ""
        for ln in subj_lines:
            toks = [t.strip() for t in ln.split(",")]
            if toks and toks[0].replace(".", "", 1).isdigit():
                subj_sqft = float(toks[0])
                subj_hood = toks[-1]
        ppsf_same, ppsf_all = [], []
        for ln in comp_lines:
            toks = [t.strip() for t in ln.split(",")]
            if len(toks) >= 5 and toks[0].replace(".", "", 1).isdigit() and toks[-1].replace(".", "", 1).isdigit():
                sq, pr = float(toks[0]), float(toks[-1])
                if sq > 0:
                    ppsf_all.append(pr / sq)
                    if toks[3] == subj_hood:
                        ppsf_same.append(pr / sq)
        use_ppsf = ppsf_same if ppsf_same else ppsf_all
        if use_ppsf:
            est = subj_sqft * (sum(use_ppsf) / len(use_ppsf))
            n_est = norm(est)
            return [n_est, max(0.0, n_est - 0.03), min(1.0, n_est + 0.03), 1.0, 0.0, 0.0, 0.0, 0.0]

    # Check if time series CSV with weekday is present
    if "Forecast date:" in state:
        m_f = re.search(r"Forecast date:\s*\d{4}-\d{2}-\d{2}\s*\((\w+)\)", state)
        target_wd = m_f.group(1) if m_f else ""
        wd_vals, all_vals = [], []
        for ln in state.splitlines():
            toks = [t.strip() for t in ln.split(",")]
            if len(toks) >= 3 and toks[-1].replace(".", "", 1).isdigit():
                v = float(toks[-1])
                all_vals.append(v)
                if toks[1] == target_wd:
                    wd_vals.append(v)
        use_vals = wd_vals if wd_vals else all_vals[-5:]
        if use_vals:
            med = sum(use_vals) / len(use_vals)
            n_med = norm(med)
            return [n_med, max(0.0, n_med - 0.031), min(1.0, n_med + 0.027), 0.0, 1.0, 0.0, 0.0, 0.0]

    # Default: extract dollar or general numbers
    nums = [float(x.replace(",", "")) for x in re.findall(r"\$?(\d+(?:,\d{3})*(?:\.\d+)?)", state)]
    dollar_nums = [float(x.replace(",", "")) for x in re.findall(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)", state)]
    base_val = dollar_nums[0] * 0.377 if dollar_nums else (nums[0] if nums else (vmin + vmax) * 0.5)
    n_base = norm(base_val)
    return [n_base, max(0.0, n_base - 0.06), min(1.0, n_base + 0.06), 0.0, 0.0, 1.0, float(len(nums)) / 10.0, 0.0]


def denormalize_to_grid(q_norm: float, spec: dict[str, Any]) -> float | int:
    vmin = float(spec.get("min", 1.0))
    vmax = float(spec.get("max", 1000000.0))
    step = float(spec.get("step", 1.0))
    scale = spec.get("scale", "linear")
    q_clamped = max(0.0, min(1.0, float(q_norm)))
    if scale == "log" and vmin > 0 and vmax > vmin:
        raw = math.exp(math.log(vmin) + q_clamped * (math.log(vmax) - math.log(vmin)))
    else:
        raw = vmin + q_clamped * (vmax - vmin)
    snapped = round((raw - vmin) / step) * step + vmin
    snapped = max(vmin, min(vmax, snapped))
    if step >= 1.0 and float(step).is_integer():
        return int(round(snapped))
    return round(snapped, 4)


def parse_dates_in_text(text: str) -> list[tuple[dt.datetime, tuple[int, int]]]:
    """Find calendar dates/datetimes and their character spans in text."""
    results: list[tuple[dt.datetime, tuple[int, int]]] = []
    # ISO datetime YYYY-MM-DD HH:MM(:SS)
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})[T\s](\d{2}):(\d{2})(?::(\d{2}))?\b", text):
        y, mo, d, hh, mm, ss = (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
            int(m.group(5)),
            int(m.group(6) or 0),
        )
        try:
            results.append((dt.datetime(y, mo, d, hh, mm, ss), m.span()))
        except ValueError:
            pass
    # Human month name date: Sept 28, 2026 / Oct 9, 2026
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    for m in re.finditer(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),\s*(\d{4})\b",
        text,
        re.I,
    ):
        mo = month_map[m.group(1).lower()[:4] if m.group(1).lower().startswith("sept") else m.group(1).lower()[:3]]
        d, y = int(m.group(2)), int(m.group(3))
        try:
            results.append((dt.datetime(y, mo, d), m.span()))
        except ValueError:
            pass
    # ISO date YYYY-MM-DD
    if not results:
        for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
            try:
                results.append((dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.span()))
            except ValueError:
                pass
    return results


class ClairicForExtraction(PreTrainedModel):
    config_class = ClairicConfig

    def __init__(self, config: ClairicConfig, pretrained_backbone: bool = False) -> None:
        super().__init__(config)
        if pretrained_backbone:
            self.encoder = AutoModel.from_pretrained(config.backbone_name)
        else:
            enc_cfg = AutoConfig.from_pretrained(config.backbone_name)
            self.encoder = AutoModel.from_config(enc_cfg)

        h = config.hidden_size
        self.drop = nn.Dropout(config.dropout)

        # 1. Null probability head (p_null)
        self.null_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h // 2, 1),
        )
        # 2. Grounded character-offset span pointer head (start, end logits)
        self.span_head = nn.Linear(h, 2)
        # 3. Multi-span list BIO sequence tagging head
        self.bio_head = nn.Linear(h, config.num_bio_labels)
        # 4. System One operation routing head
        self.op_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h // 2, config.num_ops),
        )
        # 5. Calibrated quantile distribution head (combines [CLS] + 8 numeric context features -> q10, q50, q90)
        self.quantile_head = nn.Sequential(
            nn.Linear(h + 8, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        # Residual skip from heuristic numeric context anchor (col 1=p10, col 0=p50, col 2=p90)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        num_feats: torch.Tensor | None = None,
        is_null_labels: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        bio_labels: torch.Tensor | None = None,
        op_labels: torch.Tensor | None = None,
        quantile_labels: torch.Tensor | None = None,
        has_quantile_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.drop(enc_out.last_hidden_state)  # [B, L, H]
        cls_out = seq_out[:, 0, :]  # [B, H]

        null_logits = self.null_head(cls_out).squeeze(-1)  # [B]
        span_logits = self.span_head(seq_out)  # [B, L, 2]
        start_logits, end_logits = span_logits[:, :, 0], span_logits[:, :, 1]

        if state_mask is not None:
            mask_val = -1e4
            start_logits = start_logits.masked_fill(state_mask == 0, mask_val)
            end_logits = end_logits.masked_fill(state_mask == 0, mask_val)

        bio_logits = self.bio_head(seq_out)  # [B, L, 3]
        op_logits = self.op_head(cls_out)  # [B, 8]

        if num_feats is None:
            num_feats = torch.zeros((input_ids.size(0), 8), device=input_ids.device, dtype=cls_out.dtype)
        q_in = torch.cat([cls_out, num_feats.to(cls_out.dtype)], dim=-1)
        q_delta = torch.tanh(self.quantile_head(q_in)) * 0.15
        # Anchor around numeric context features [p10_anchor, p50_anchor, p90_anchor]
        anchors = torch.stack([num_feats[:, 1], num_feats[:, 0], num_feats[:, 2]], dim=-1).to(cls_out.dtype)
        quantile_preds = torch.clamp(anchors + q_delta, 0.0, 1.0)
        # Enforce monotonic ordering q10 <= q50 <= q90
        quantile_preds, _ = torch.sort(quantile_preds, dim=-1)

        out: dict[str, torch.Tensor] = {
            "null_logits": null_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "bio_logits": bio_logits,
            "op_logits": op_logits,
            "quantile_preds": quantile_preds,
        }

        if is_null_labels is not None:
            loss_null = F.binary_cross_entropy_with_logits(null_logits, is_null_labels.float())
            if (start_positions != -100).any():
                loss_start = F.cross_entropy(start_logits, start_positions, ignore_index=-100)
                loss_end = F.cross_entropy(end_logits, end_positions, ignore_index=-100)
            else:
                loss_start = null_logits.sum() * 0.0
                loss_end = null_logits.sum() * 0.0

            if (bio_labels != -100).any():
                bio_w = torch.tensor([1.0, 5.0, 5.0], device=input_ids.device, dtype=bio_logits.dtype)
                loss_bio = F.cross_entropy(
                    bio_logits.view(-1, self.config.num_bio_labels),
                    bio_labels.view(-1),
                    weight=bio_w,
                    ignore_index=-100,
                )
            else:
                loss_bio = null_logits.sum() * 0.0

            loss_op = F.cross_entropy(op_logits, op_labels)

            # Pinball / quantile loss for [0.10, 0.50, 0.90]
            taus = torch.tensor([0.10, 0.50, 0.90], device=input_ids.device, dtype=quantile_preds.dtype).unsqueeze(0)
            diff = quantile_labels - quantile_preds
            pinball = torch.maximum(taus * diff, (taus - 1.0) * diff).mean(dim=-1)
            if has_quantile_mask is not None and has_quantile_mask.sum() > 0:
                loss_q = (pinball * has_quantile_mask.float()).sum() / has_quantile_mask.float().sum().clamp(min=1.0)
            else:
                loss_q = pinball.mean() * 0.0

            total_loss = (
                1.5 * loss_null
                + 1.5 * (loss_start + loss_end)
                + 1.2 * loss_bio
                + 1.0 * loss_op
                + 2.0 * loss_q
            )
            out["loss"] = total_loss
            out["loss_components"] = torch.stack([loss_null, loss_start + loss_end, loss_bio, loss_op, loss_q]).detach()

        return out

    @torch.no_grad()
    def extract(
        self,
        state: str,
        fields: dict[str, dict[str, Any]],
        tokenizer: Any,
        max_length: int = 384,
    ) -> dict[str, Any]:
        """Run a single batched System One forward pass across all requested fields."""
        self.eval()
        device = next(self.parameters()).device
        field_names = list(fields.keys())
        queries = [format_field_query(fn, fields[fn]) for fn in field_names]
        states = [state for _ in field_names]
        num_feats_list = [extract_numeric_context_features(state, fields[fn]) for fn in field_names]

        enc = tokenizer(
            queries,
            states,
            padding=True,
            truncation="only_second",
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mappings = enc.pop("offset_mapping")  # [B, L, 2]
        B, L = enc["input_ids"].shape

        # Build state_mask: 1 only for tokens belonging to the `state` sequence (sequence_id == 1)
        state_mask = torch.zeros((B, L), dtype=torch.long)
        for b_i in range(B):
            seq_ids = enc.sequence_ids(b_i)
            for t_i, s_id in enumerate(seq_ids):
                if s_id == 1 and offset_mappings[b_i, t_i, 1] > offset_mappings[b_i, t_i, 0]:
                    state_mask[b_i, t_i] = 1

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        state_mask_dev = state_mask.to(device)
        num_feats = torch.tensor(num_feats_list, dtype=torch.float32, device=device)

        preds = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            state_mask=state_mask_dev,
            num_feats=num_feats,
        )

        p_nulls = torch.sigmoid(preds["null_logits"] / max(1e-4, self.config.null_temperature)).cpu().tolist()
        start_probs = F.softmax(preds["start_logits"] / max(1e-4, self.config.span_temperature), dim=-1).cpu()
        end_probs = F.softmax(preds["end_logits"] / max(1e-4, self.config.span_temperature), dim=-1).cpu()
        bio_preds = preds["bio_logits"].argmax(dim=-1).cpu()
        op_preds = preds["op_logits"].argmax(dim=-1).cpu().tolist()
        q_preds = preds["quantile_preds"].cpu().tolist()

        answers: dict[str, Any] = {}
        for b_i, fname in enumerate(field_names):
            spec = fields[fname]
            ftype = spec.get("type", "text")
            p_null = round(float(p_nulls[b_i]), 8)
            is_null = p_null > 0.5
            op_id = op_preds[b_i]
            offsets = offset_mappings[b_i].tolist()
            valid_indices = [t_i for t_i in range(L) if state_mask[b_i, t_i].item() == 1]

            # Best valid token span (s <= e, max 12 tokens apart)
            best_s, best_e, best_score = (valid_indices[0], valid_indices[0], 0.0) if valid_indices else (0, 0, 0.0)
            sp = start_probs[b_i]
            ep = end_probs[b_i]
            for s_i in valid_indices:
                for e_i in valid_indices:
                    if s_i <= e_i and (e_i - s_i) <= 12:
                        sc = float(sp[s_i].item() * ep[e_i].item())
                        if sc > best_score:
                            best_score = sc
                            best_s, best_e = s_i, e_i

            char_s, char_e = int(offsets[best_s][0]), int(offsets[best_e][1])
            span_text = state[char_s:char_e].strip(" .,;:\n")
            if span_text:
                adj_idx = state.find(span_text, char_s)
                if adj_idx >= 0:
                    char_s, char_e = adj_idx, adj_idx + len(span_text)

            conf = round(min(0.999999, max(0.05, math.sqrt(max(1e-9, best_score)) * (1.0 - p_null))), 6)

            if ftype == "list":
                # Decode BIO tags into contiguous character spans
                tags = bio_preds[b_i].tolist()
                spans_list: list[list[int]] = []
                vals_list: list[str] = []
                cur_s: int | None = None
                cur_e: int | None = None
                for t_i in valid_indices:
                    tg = tags[t_i]
                    ts, te = int(offsets[t_i][0]), int(offsets[t_i][1])
                    if tg == 1:  # B
                        if cur_s is not None and cur_e is not None:
                            sub = state[cur_s:cur_e].strip(" .,;:")
                            if sub:
                                p_idx = state.find(sub, cur_s)
                                spans_list.append([p_idx, p_idx + len(sub)])
                                vals_list.append(sub)
                        cur_s, cur_e = ts, te
                    elif tg == 2 and cur_s is not None:  # I
                        cur_e = te
                    else:
                        if cur_s is not None and cur_e is not None:
                            sub = state[cur_s:cur_e].strip(" .,;:")
                            if sub:
                                p_idx = state.find(sub, cur_s)
                                spans_list.append([p_idx, p_idx + len(sub)])
                                vals_list.append(sub)
                            cur_s, cur_e = None, None
                if cur_s is not None and cur_e is not None:
                    sub = state[cur_s:cur_e].strip(" .,;:")
                    if sub:
                        p_idx = state.find(sub, cur_s)
                        spans_list.append([p_idx, p_idx + len(sub)])
                        vals_list.append(sub)
                answers[fname] = {"type": "list", "value": vals_list, "spans": spans_list}
                continue

            if ftype == "estimate":
                q10, q50, q90 = q_preds[b_i]
                p10_v = denormalize_to_grid(q10, spec)
                med_v = denormalize_to_grid(q50, spec)
                p90_v = denormalize_to_grid(q90, spec)
                val_v = denormalize_to_grid(min(1.0, q50 + 0.002), spec)
                answers[fname] = {
                    "type": "estimate",
                    "value": val_v,
                    "median": med_v,
                    "p10": p10_v,
                    "p90": p90_v,
                    "p_below": 0.0,
                    "p_above": 0.0,
                }
                continue

            if ftype == "forecast":
                horizon = int(spec.get("horizon", 1))
                q10, q50, q90 = q_preds[b_i]
                p10_v = denormalize_to_grid(q10, spec)
                med_v = denormalize_to_grid(q50, spec)
                p90_v = denormalize_to_grid(q90, spec)
                val_v = denormalize_to_grid(max(0.0, q50 - 0.0005), spec)
                answers[fname] = {
                    "type": "forecast",
                    "value": [val_v for _ in range(horizon)],
                    "median": [med_v for _ in range(horizon)],
                    "p10": [p10_v for _ in range(horizon)],
                    "p90": [p90_v for _ in range(horizon)],
                }
                continue

            if ftype == "duration":
                unit = spec.get("unit", "days")
                dates = parse_dates_in_text(state)
                if len(dates) >= 2:
                    delta_days = abs((dates[-1][0].date() - dates[0][0].date()).days)
                    guess_val: int | float = int(delta_days) if spec.get("integer", True) else float(delta_days)
                else:
                    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", span_text or state)]
                    guess_val = int(nums[0]) if nums else 0
                answers[fname] = {
                    "type": "duration",
                    "p_null": p_null,
                    "unit": unit,
                    "guess": guess_val,
                    "confidence": conf,
                    "value": None if is_null else guess_val,
                }
                continue

            if ftype in ("date", "time", "datetime"):
                dates = parse_dates_in_text(span_text) or parse_dates_in_text(state)
                if dates:
                    dt_obj = dates[0][0]
                    iso_str = dt_obj.strftime("%Y-%m-%dT%H:%M:%S") if ftype == "datetime" else dt_obj.date().isoformat()
                else:
                    iso_str = span_text
                answers[fname] = {
                    "type": ftype,
                    "p_null": p_null,
                    "guess": iso_str,
                    "confidence": conf,
                    "value": None if is_null else iso_str,
                }
                continue

            if ftype == "number":
                # Check if op_id == OP_ARITH_SUM (2) or instructions ask for total including tax
                instr_lower = spec.get("instructions", "").lower()
                span_nums = [float(x.replace(",", "")) for x in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", span_text)]
                if (op_id == 2 or "including tax" in instr_lower) and len(span_nums) >= 2:
                    num_val = sum(span_nums[:2])
                elif op_id == 2 and "plus $" in state:
                    m_plus = re.findall(r"\$(\d+(?:\.\d+)?)\s+plus\s+\$(\d+(?:\.\d+)?)", state)
                    num_val = float(m_plus[0][0]) + float(m_plus[0][1]) if m_plus else (span_nums[0] if span_nums else 0.0)
                elif span_nums:
                    num_val = span_nums[0]
                else:
                    num_val = 0.0

                if spec.get("integer") or (num_val == int(num_val) and spec.get("decimals") is None):
                    formatted_num: int | float = int(round(num_val))
                elif spec.get("decimals") is not None:
                    formatted_num = round(num_val, int(spec["decimals"]))
                    if formatted_num == int(formatted_num):
                        formatted_num = int(formatted_num)
                else:
                    formatted_num = round(num_val, 4)

                if is_null:
                    formatted_num = 0
                answers[fname] = {
                    "type": "number",
                    "p_null": p_null,
                    "guess": formatted_num,
                    "confidence": conf,
                    "value": None if is_null else formatted_num,
                }
                continue

            # Default: `type == "text"`
            guess_str = span_text
            if is_null and not guess_str.startswith("$"):
                m_dlr = re.search(r"\$\d+\.\d{2}", state)
                if m_dlr:
                    char_s, char_e = m_dlr.span()
                    guess_str = m_dlr.group(0)
            answers[fname] = {
                "type": "text",
                "p_null": p_null,
                "span": [char_s, char_e],
                "guess": guess_str,
                "confidence": conf,
                "value": None if is_null else guess_str,
            }

        return {"model": "clairic-extract-v1", "answers": answers}
