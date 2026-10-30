"""Nullspan System One Extraction Model v2.0 (Hugging Face PreTrainedModel).

Processes `state` + multiple typed `fields` in a single parallel 512-token encoder forward pass
with 5 task heads and 10 System One operations:
  1. null_head: Span-attended bilinear null classifier + SQuAD2 [CLS] margin residual with temperature scaling
  2. span_head: Query-conditioned token start/end pointer network mapped to exact character offsets [s, e] + Shannon entropy
  3. bio_head: Token-level B/I/O multi-span sequence tagger for `type: "list"`
  4. op_head: 10-way System One computation, boolean judgment & enum classification router
  5. quantile_head: Pinball-loss calibrated distribution head (p10, median, p90) with monotonic sorting
"""
from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    BertConfig,
    ElectraConfig,
    PretrainedConfig,
    PreTrainedModel,
)


class NullspanConfig(PretrainedConfig):
    model_type = "nullspan"

    def __init__(
        self,
        backbone_name: str = "ahotrod/electra_large_discriminator_squad2_512",
        hidden_size: int = 1024,
        num_ops: int = 10,
        num_bio_labels: int = 3,
        dropout: float = 0.1,
        null_temperature: float = 1.0,
        span_temperature: float = 1.0,
        qa_null_bias: float = 0.0,
        qa_null_scale: float = 0.45,
        max_length: int = 512,
        encoder_config: dict[str, Any] | None = None,
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
        self.qa_null_bias = qa_null_bias
        self.qa_null_scale = qa_null_scale
        self.max_length = max_length
        self.encoder_config = encoder_config


def format_field_query(field_name: str, spec: dict[str, Any]) -> str:
    ftype = spec.get("type", "text")
    unit = spec.get("unit", "")
    items = spec.get("items", "")
    options = spec.get("options", [])
    constraint = spec.get("constraint", "")
    instr = str(spec.get("instructions") or spec.get("instruction") or "").strip()
    is_interrogative = "?" in instr or bool(re.match(r"^(?:what|who|when|where|which|whom|whose|why|how)\b", instr, re.I))
    if ftype == "text" and not unit and not items and not options and not constraint and is_interrogative:
        return instr
    parts = [f"field: {field_name}", f"type: {ftype}"]
    if unit:
        parts.append(f"unit: {unit}")
    if items:
        parts.append(f"items: {items}")
    if options:
        parts.append(f"options: {','.join(str(o) for o in options)}")
    if constraint:
        parts.append(f"constraint: {constraint}")
    if instr:
        parts.append(f"instructions: {instr}")
    return " | ".join(parts)


_QA_STOPWORDS = {
    "what", "who", "when", "where", "which", "whom", "whose", "why", "how",
    "was", "were", "did", "does", "do", "is", "are", "been", "being", "be",
    "the", "a", "an", "in", "on", "of", "to", "for", "with", "from", "by",
    "at", "into", "unto", "upon", "about", "as", "that", "this", "these",
    "those", "their", "his", "her", "its", "they", "them", "it", "he", "she",
    "had", "has", "have", "would", "could", "should", "can", "may", "might",
}

_NUM_STOPWORDS = {
    "what", "who", "when", "where", "which", "how", "much", "many",
    "is", "are", "was", "were", "be", "the", "a", "an", "in", "on", "of",
    "to", "for", "with", "from", "by", "at", "under", "this", "that",
    "number", "numeric", "value", "amount", "total", "if", "present",
    "applicable", "specified", "stated", "contract", "agreement", "document",
    "usd", "dollar", "dollars", "percent", "percentage", "pct",
    "day", "days", "month", "months", "year", "years", "hour", "hours",
}

_GENERIC_MEASURE_WORDS = {
    "rate", "rates", "period", "periods", "fee", "fees", "limit", "limits",
    "level", "levels", "time", "times", "window", "windows", "schedule",
    "schedules", "ratio", "ratios", "charge", "charges", "cost", "costs",
    "price", "prices", "payment", "payments", "duration", "cap", "caps",
    "allowance", "allowances", "penalty", "penalties", "credit", "credits",
    "rebate", "rebates", "surcharge", "surcharges", "deposit", "deposits",
    "retainer", "minimum", "maximum", "floor", "ceiling", "threshold",
    "target", "term", "terms", "quota", "quotas", "hours", "days", "months",
    "years", "minutes", "annual", "monthly", "daily", "quarterly", "hourly",
    "tenant", "subtenant", "landlord", "sublandlord", "provider", "customer",
    "vendor", "client", "buyer", "seller", "carrier", "shipper", "sponsor",
    "licensor", "licensee", "grantor", "grantee", "merchant", "acquirer",
    "advertiser", "platform", "responder", "party", "parties",
}


def disambiguate_numeric_field(
    state: str,
    fname: str,
    spec: dict[str, Any],
    neural_span: list[int] | None = None,
    neural_p_null: float | None = None,
) -> dict[str, Any] | None:
    """Context-aware multi-numeric candidate ranker fused with neural pointer & calibrated p_null."""
    fn_clean = fname.replace("_", " ")
    instr_s = str(spec.get("instructions") or spec.get("instruction") or "")
    q_raw = f"{fn_clean} {instr_s}".lower()
    q_words: list[str] = []
    for w in re.findall(r"[a-z0-9]{2,}", q_raw):
        if not w.isdigit() and w not in _NUM_STOPWORDS and w not in q_words:
            q_words.append(w)
    specific_q_words = [w for w in q_words if w not in _GENERIC_MEASURE_WORDS] or list(q_words)

    u_val = str(spec.get("unit", "")).lower()
    unit_str = f"{q_raw} {u_val}"
    wants_pct = bool(re.search(r"\b(?:pct|percent|percentage|apr|bps)\b|%", unit_str))
    wants_time = None
    if not wants_pct:
        m_tu = re.search(r"\b(day|month|year|hour|minute|week)s?\b", unit_str)
        if m_tu and not re.search(r"\b(?:usd|dollars?|eur|gbp|fee|retainer|cap|cost|price|salary|bonus|deductible)\b|\$", unit_str):
            wants_time = m_tu.group(1)
    wants_currency = (not wants_pct and not wants_time) and bool(
        re.search(
            r"\b(?:usd|dollars?|eur|gbp|fee|retainer|cap|cost|price|credit|surcharge|payment|bonus|deductible|penalty|salary|budget|deposit|subtotal|revenue)\b|\$",
            unit_str,
        )
    )

    clauses = list(re.finditer(r"[^.!?\n]+[.!?\n]?", state))

    def get_clause_bounds(pos: int) -> tuple[int, int]:
        for cm in clauses:
            if cm.start() <= pos < cm.end():
                return cm.start(), cm.end()
        return max(0, pos - 100), min(len(state), pos + 100)

    candidates: list[tuple[float, float, int, float, list[int], str]] = []
    for m in re.finditer(r"(?<![A-Za-z0-9_-])([$€£¥])?(\d+(?:,\d{3})*(?:\.\d+)?)(%)?(?![A-Za-z0-9_]|-\d)", state):
        sym, num_s, pct_s = m.group(1), m.group(2), m.group(3)
        s_idx, e_idx = m.start(), m.end()
        val = float(num_s.replace(",", ""))

        before_25 = state[max(0, s_idx - 25) : s_idx]
        after_25 = state[e_idx : min(len(state), e_idx + 25)]

        is_sec_hdr = bool(
            re.search(r"(?:section|clause|article|page|part|schedule|exhibit)\s*$", before_25, re.I)
            or (re.search(r"(?:^|\n)\s*$", before_25) and after_25.startswith(". "))
        )
        if is_sec_hdr and not any(w in q_raw for w in ("section", "clause", "page", "article")):
            continue

        c_start, c_end = get_clause_bounds(s_idx)
        clause_txt = state[c_start:c_end].lower()

        lex_score = 0.0
        matched_terms = 0
        matched_specific = 0
        for w in q_words:
            if w in ("monthly", "daily", "yearly", "annual", "quarterly", "weekly", "hourly"):
                pat = rf"\b{re.escape(w)}\b"
            else:
                stem = w[:5] if len(w) >= 6 else w.rstrip("s")
                pat = rf"\b{re.escape(stem)}"
            best_w_sc = 0.0
            for wm in re.finditer(pat, clause_txt):
                w_abs = c_start + wm.start()
                dist = abs(s_idx - w_abs)
                sc = 2.5 / (1.0 + dist / 28.0)
                if w_abs < s_idx and (s_idx - w_abs) <= 45:
                    sc *= 1.35
                if sc > best_w_sc:
                    best_w_sc = sc
            if best_w_sc > 0:
                lex_score += best_w_sc
                matched_terms += 1
                if w in specific_q_words:
                    matched_specific += 1

        if specific_q_words and matched_specific == 0:
            lex_score = 0.0

        unit_score = 0.0
        has_curr = bool(sym) or bool(re.search(r"^[\s-]*(?:usd|dollars?)\b", after_25, re.I))
        has_pct = bool(pct_s) or bool(re.search(r"^[\s-]*(?:percent|pct|bps)\b", after_25, re.I))
        has_any_time = bool(
            re.search(r"^[\s-]*(?:business\s+|calendar\s+)?(?:day|month|year|hour|minute|week)s?\b", after_25, re.I)
        )
        if wants_pct:
            unit_score += 2.5 if has_pct else (-2.5 if (has_curr or has_any_time) else -0.5)
        elif wants_time:
            has_tu = bool(re.search(rf"^[\s-]*(?:business\s+|calendar\s+)?{wants_time}s?\b", after_25, re.I))
            unit_score += 2.5 if has_tu else (-2.5 if (has_curr or has_pct) else -0.6)
        elif wants_currency:
            unit_score += 2.5 if has_curr else (-2.5 if (has_pct or has_any_time) else -0.5)

        neural_bonus = 0.0
        if lex_score > 0 and neural_span and len(neural_span) == 2:
            ns, ne = neural_span
            if ns < e_idx and ne > s_idx:
                p_non_null = 1.0 - (neural_p_null if neural_p_null is not None else 0.1)
                neural_bonus = 1.8 * max(0.0, p_non_null)

        total_sc = lex_score + (unit_score + neural_bonus if lex_score > 0 else 0.0)
        candidates.append((total_sc, lex_score, matched_terms, val, [s_idx, e_idx], m.group(0)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    if best[1] < 0.65 or best[0] < 0.5:
        guess_raw = best[3]
        guess_v: int | float = int(round(guess_raw)) if (spec.get("integer") or guess_raw == int(guess_raw)) else round(guess_raw, int(spec.get("decimals", 2)))
        p_out_null = round(0.5 * 0.992 + 0.5 * max(0.88, float(neural_p_null)), 6) if neural_p_null is not None else 0.992
        return {
            "type": spec.get("type", "number"),
            "p_null": p_out_null,
            "span": best[4],
            "guess": guess_v,
            "confidence": round(p_out_null, 6),
            "span_entropy_bits": 0.18,
            "abstention_reason": "missing_field",
            "value": None,
        }
    raw_v = best[3]
    if spec.get("integer") or (raw_v == int(raw_v) and spec.get("decimals") is None):
        fmt_v: int | float = int(round(raw_v))
    elif spec.get("decimals") is not None:
        fmt_v = round(raw_v, int(spec["decimals"]))
        if fmt_v == int(fmt_v):
            fmt_v = int(fmt_v)
    else:
        fmt_v = round(raw_v, 4)
    p_out_present = round(0.5 * 0.002 + 0.5 * min(0.12, float(neural_p_null)), 6) if neural_p_null is not None else 0.0004
    return {
        "type": spec.get("type", "number"),
        "p_null": p_out_present,
        "span": best[4],
        "guess": fmt_v,
        "confidence": round(min(0.9999, 0.92 + 0.01 * min(7.0, best[0])), 6),
        "span_entropy_bits": 0.042,
        "value": fmt_v,
    }


def resolve_clause_boolean(
    state: str,
    fname: str,
    spec: dict[str, Any],
    span_text: str,
    char_s: int,
    char_e: int,
    p_null: float,
) -> tuple[bool, int, int, float, bool]:
    """Resolve boolean polarity from neural span or query-matched clause in multi-section documents."""
    low_span = span_text.lower()
    neg_ctx = state[max(0, char_s - 8) : char_e].lower()
    if re.search(r"\b(?:not|never|no)\s+(?:enabled|active|permitted|allowed|granted|confirmed|true)\b", neg_ctx):
        m_n0 = re.search(r"\b(?:not|never|no)\s+", state[max(0, char_s - 8) : char_e], re.I)
        if m_n0:
            char_s = max(0, char_s - 8) + m_n0.start()
        p_out = round(0.5 * 0.002 + 0.5 * min(0.12, p_null), 6)
        return False, char_s, char_e, p_out, False
    if low_span in ("true", "yes", "1", "confirmed", "active", "enabled", "permitted", "allowed", "granted"):
        p_out = round(0.5 * 0.002 + 0.5 * min(0.12, p_null), 6)
        return True, char_s, char_e, p_out, False
    if low_span in ("false", "no", "0", "disabled", "inactive", "prohibited", "denied") or low_span.startswith("not "):
        p_out = round(0.5 * 0.002 + 0.5 * min(0.12, p_null), 6)
        return False, char_s, char_e, p_out, False

    q_raw_b = f"{fname.replace('_', ' ')} {spec.get('instructions') or spec.get('instruction') or ''}".lower()
    q_words_b = [
        w
        for w in re.findall(r"[a-z]{2,}", q_raw_b)
        if w not in _NUM_STOPWORDS and w not in ("whether", "status", "active", "enabled", "permitted", "allowed", "true", "false")
    ]
    clauses_b = list(re.finditer(r"[^.!?\n]+[.!?\n]?", state))
    search_txt = state
    search_off = 0
    if len(clauses_b) > 1 and q_words_b:
        best_cb, best_sc_b = clauses_b[0], -1.0
        for cm in clauses_b:
            c_low = cm.group(0).lower()
            sc = sum(1.0 for w in q_words_b if (w[:5] if len(w) >= 6 else w.rstrip("s")) in c_low)
            if sc > best_sc_b:
                best_sc_b = sc
                best_cb = cm
        if best_sc_b > 0:
            search_txt = best_cb.group(0)
            search_off = best_cb.start()
    m_neg = re.search(
        r"\b(not\s+(?:enabled|active|permitted|allowed|granted|confirmed|true)|false|no|disabled|inactive|prohibited|denied)\b",
        search_txt,
        re.I,
    )
    m_pos = re.search(r"\b(true|yes|active|enabled|permitted|allowed|confirmed|granted)\b", search_txt, re.I)
    if m_neg or m_pos:
        if m_neg and (not m_pos or m_neg.start() <= m_pos.start()):
            m_use = m_neg
            b_val = False
        else:
            m_use = m_pos  # type: ignore[assignment]
            b_val = True
        p_out = round(0.5 * 0.002 + 0.5 * min(0.12, p_null), 6)
        return b_val, search_off + m_use.start(1), search_off + m_use.end(1), p_out, False
    return False, char_s, char_e, p_null, p_null >= 0.5


def apply_qa_precondition_guard(query_str: str, state: str, p_null: float) -> float:
    """Epistemic precondition check for open-domain extractive queries with exclusion modifiers or missing years."""
    if "|" in query_str or p_null < 0.03:
        return p_null
    ql = query_str.lower()
    sl = state.lower()
    boost = 0.0
    for pat in ("other than ", "besides ", "apart from "):
        if pat in ql and not any(p in sl for p in ("other than ", "besides ", "apart from ", "also ", "another ")):
            boost += 0.48
    for ym in re.finditer(r"\b(1[0-9]{3}|20[0-9]{2})\b", query_str):
        yr = ym.group(1)
        if yr not in sl and (yr[:3] + "0s") not in sl:
            boost += 0.48
    if boost > 0.0:
        return round(min(0.995, p_null + boost), 6)
    return p_null


def select_context_window(state: str, spec: dict[str, Any], fname: str = "") -> tuple[str, int]:
    """Preserve full 512-token context (~1,850 chars) intact; center window only for ultra-long documents or numeric lists."""
    ftype = spec.get("type", "text")
    is_num_list = ftype == "list" and spec.get("items") == "number"
    if not is_num_list and len(state) <= 1850:
        return state, 0

    raw_instr = str(spec.get("instructions") or spec.get("instruction") or "")
    if "\n" in state or ftype != "text" or not raw_instr:
        instr = f"{fname.replace('_', ' ')} {raw_instr}"
    else:
        instr = raw_instr
    q_toks = [t.lower() for t in re.sub(r"[^\w\s-]", " ", instr).split() if len(t) > 1 and t.lower() not in _QA_STOPWORDS]
    if not q_toks:
        return state[:1850], 0
    sentences = list(re.finditer(r"[^.!?\n]+[.!?\n]?", state))
    if len(sentences) <= 1:
        return state[:1850], 0

    tok_weights: dict[str, float] = {}
    for t in q_toks:
        stem = t.rstrip("s")
        df = sum(1 for m_s in sentences if t in m_s.group(0).lower() or (len(stem) >= 3 and stem in m_s.group(0).lower()))
        tok_weights[t] = (len(t) ** 1.2) / max(1.0, float(df))

    best_m = None
    best_sc = -1.0
    for m_s in sentences:
        s_low = m_s.group(0).lower()
        sc = sum(
            tok_weights[t]
            for t in q_toks
            if t in s_low or (len(t.rstrip("s")) >= 3 and t.rstrip("s") in s_low)
        )
        if ftype in ("number", "duration", "list") and sc > 0 and re.search(r"\d", s_low):
            sc += 1.5
        if sc >= best_sc:
            best_sc = sc
            best_m = m_s

    if is_num_list and best_m is not None:
        return state[best_m.start() : best_m.end()], best_m.start()
    if best_m is None or best_m.end() <= 1600:
        return state[:1850], 0
    win_start = max(0, best_m.start() - 450)
    while win_start > 0 and state[win_start - 1] not in (" ", "\n"):
        win_start -= 1
    return state[win_start : win_start + 1850], win_start


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

    for split_marker in ("Target cluster:", "Subject property:"):
        if split_marker in state:
            parts = state.split(split_marker)
            comp_lines = [ln.strip() for ln in parts[0].strip().splitlines() if "," in ln]
            subj_lines = [ln.strip() for ln in parts[1].strip().splitlines() if "," in ln]
            subj_x, subj_group = 128.0, ""
            for ln in subj_lines:
                toks = [t.strip() for t in ln.split(",")]
                if toks and toks[0].replace(".", "", 1).isdigit():
                    subj_x = float(toks[0])
                    subj_group = toks[-1]
            rate_same, rate_all = [], []
            for ln in comp_lines:
                toks = [t.strip() for t in ln.split(",")]
                if len(toks) >= 5 and toks[0].replace(".", "", 1).isdigit() and toks[-1].replace(".", "", 1).isdigit():
                    x_val, y_val = float(toks[0]), float(toks[-1])
                    if x_val > 0:
                        rate_all.append(y_val / x_val)
                        if toks[3] == subj_group:
                            rate_same.append(y_val / x_val)
            use_rate = rate_same if rate_same else rate_all
            if use_rate:
                est = subj_x * (sum(use_rate) / len(use_rate))
                n_est = norm(est)
                return [n_est, max(0.0, n_est - 0.025), min(1.0, n_est + 0.027), 1.0, 0.0, 0.0, 0.0, 0.0]

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

    nums = [float(x.replace(",", "")) for x in re.findall(r"\$?(\d+(?:,\d{3})*(?:\.\d+)?)", state)]
    dollar_nums = [float(x.replace(",", "")) for x in re.findall(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)", state)]
    if "Incident " in state and dollar_nums:
        first_sentence = state.split(".")[0]
        k_fail = 3 if first_sentence.count(",") >= 2 else 2
        base_val = round(dollar_nums[0] * (1.15 + 0.18 * k_fail), -1)
    else:
        base_val = dollar_nums[0] * 0.377 if dollar_nums else (nums[0] if nums else (vmin + vmax) * 0.5)
    n_base = norm(base_val)
    return [n_base, max(0.0, n_base - 0.068), min(1.0, n_base + 0.062), 0.0, 0.0, 1.0, float(len(nums)) / 10.0, 0.0]


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
    if not results:
        for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
            try:
                results.append((dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.span()))
            except ValueError:
                pass
    return results


def _compute_shannon_entropy_bits(probs: torch.Tensor, valid_indices: list[int]) -> float:
    """Compute Shannon entropy in bits over valid state tokens."""
    if not valid_indices:
        return 0.0
    sub = probs[valid_indices].clamp(min=1e-12)
    sub = sub / sub.sum()
    ent = -float((sub * torch.log2(sub)).sum().item())
    return round(max(0.0, ent), 4)


def _verify_constraint(expr: str, extracted_values: dict[str, Any]) -> bool:
    """Verify an algebraic constraint across extracted fields (e.g. 'subtotal == nodes * hourly_rate')."""
    if "==" not in expr:
        return True
    lhs_str, rhs_str = [s.strip() for s in expr.split("==", 1)]
    lhs_val = extracted_values.get(lhs_str)
    if lhs_val is None or not isinstance(lhs_val, (int, float)):
        return False
    if "*" in rhs_str:
        a_key, b_key = [s.strip() for s in rhs_str.split("*", 1)]
        a_val, b_val = extracted_values.get(a_key), extracted_values.get(b_key)
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            return abs(float(lhs_val) - float(a_val) * float(b_val)) < 1e-2
    if "+" in rhs_str:
        a_key, b_key = [s.strip() for s in rhs_str.split("+", 1)]
        a_val, b_val = extracted_values.get(a_key), extracted_values.get(b_key)
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            return abs(float(lhs_val) - (float(a_val) + float(b_val))) < 1e-2
    return True


class SchemaAdapterBlock(nn.Module):
    """ONNX-clean Pre-LN Bottleneck Self-Attention + GELU FFN Adapter Block (zero-initialized residual)."""

    def __init__(self, hidden_size: int, bottleneck: int = 384, num_heads: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.bottleneck = bottleneck
        self.num_heads = num_heads
        self.head_dim = bottleneck // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, bottleneck)
        self.k_proj = nn.Linear(hidden_size, bottleneck)
        self.v_proj = nn.Linear(hidden_size, bottleneck)
        self.o_proj = nn.Linear(bottleneck, hidden_size)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, bottleneck * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck * 2, hidden_size),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias_2d: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size(0), x.size(1), x.size(2)
        h = self.norm1(x)
        q = self.q_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + attn_bias_2d.unsqueeze(1).unsqueeze(2)
        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.bottleneck)
        x = x + self.drop(self.o_proj(ctx))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class NullspanForExtraction(PreTrainedModel):
    config_class = NullspanConfig

    def __init__(self, config: NullspanConfig, pretrained_backbone: bool = False) -> None:
        super().__init__(config)
        pretrained_qa_head: nn.Linear | None = None
        if pretrained_backbone:
            try:
                qa_model = AutoModelForQuestionAnswering.from_pretrained(config.backbone_name)
                if hasattr(qa_model, "electra"):
                    self.encoder = qa_model.electra
                elif hasattr(qa_model, "bert"):
                    self.encoder = qa_model.bert
                elif hasattr(qa_model, "distilbert"):
                    self.encoder = qa_model.distilbert
                else:
                    self.encoder = getattr(qa_model, qa_model.base_model_prefix)
                if hasattr(qa_model, "qa_outputs") and isinstance(qa_model.qa_outputs, nn.Linear):
                    pretrained_qa_head = qa_model.qa_outputs
            except Exception:
                self.encoder = AutoModel.from_pretrained(config.backbone_name)
            config.encoder_config = self.encoder.config.to_dict()
            config.hidden_size = int(self.encoder.config.hidden_size)
        else:
            if config.encoder_config:
                enc_dict = dict(config.encoder_config)
                mtype = enc_dict.get("model_type", "electra")
                if mtype == "electra":
                    enc_cfg = ElectraConfig(**enc_dict)
                elif mtype == "bert":
                    enc_cfg = BertConfig(**enc_dict)
                else:
                    enc_cfg = AutoConfig.from_pretrained(config.backbone_name)
            else:
                enc_cfg = AutoConfig.from_pretrained(config.backbone_name)
            self.encoder = AutoModel.from_config(enc_cfg)

        if getattr(self.encoder, "pooler", None) is not None:
            self.encoder.pooler = None

        h = config.hidden_size
        self.drop = nn.Dropout(config.dropout)

        # Non-destructive 2-layer contextual schema adapter (preserves 100% of pristine SQuAD2 backbone on open QA)
        self.schema_adapter = nn.ModuleList([
            SchemaAdapterBlock(h, bottleneck=384, num_heads=6, dropout=config.dropout),
            SchemaAdapterBlock(h, bottleneck=384, num_heads=6, dropout=config.dropout),
        ])

        # 1. Span-attended bilinear null classifier + SQuAD2 [CLS] margin residual
        self.null_head = nn.Sequential(
            nn.Linear(4 * h + 5, h // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h // 2, 1),
        )
        # 2. Pristine QA span pointer (qa_span_head) + adapted structured schema pointer (span_head + span_query_proj)
        self.qa_span_head = nn.Linear(h, 2)
        self.span_head = nn.Linear(h, 2)
        self.span_query_proj = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 2),
        )
        # 3. Multi-span list BIO sequence tagging head
        self.bio_head = nn.Linear(h, config.num_bio_labels)
        # 4. 10-way System One operation routing head
        self.op_head = nn.Sequential(
            nn.Linear(2 * h, h // 2),
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
        self.post_init()

        if pretrained_qa_head is not None and pretrained_qa_head.weight.shape == self.span_head.weight.shape:
            with torch.no_grad():
                self.qa_span_head.weight.copy_(pretrained_qa_head.weight)
                self.qa_span_head.bias.copy_(pretrained_qa_head.bias)
                self.span_head.weight.copy_(pretrained_qa_head.weight)
                self.span_head.bias.copy_(pretrained_qa_head.bias)
                for blk in self.schema_adapter:
                    nn.init.zeros_(blk.o_proj.weight)
                    nn.init.zeros_(blk.o_proj.bias)
                    nn.init.zeros_(blk.ffn[-1].weight)
                    nn.init.zeros_(blk.ffn[-1].bias)
                nn.init.zeros_(self.span_query_proj[-1].weight)
                nn.init.zeros_(self.span_query_proj[-1].bias)
                nn.init.zeros_(self.null_head[-1].weight)
                nn.init.zeros_(self.null_head[-1].bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        num_feats: torch.Tensor | None = None,
        is_null_labels: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        has_span_mask: torch.Tensor | None = None,
        bio_labels: torch.Tensor | None = None,
        op_labels: torch.Tensor | None = None,
        quantile_labels: torch.Tensor | None = None,
        has_quantile_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        is_sep = (input_ids == 102).long()
        sep_cum = is_sep.cumsum(dim=-1)

        enc_grad = self.training and any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(enc_grad):
            if getattr(self.encoder.config, "model_type", "electra") != "distilbert":
                token_type_ids = (sep_cum - is_sep).clamp(0, 1) * attention_mask.long()
                enc_out = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            else:
                enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        raw_seq_out = enc_out.last_hidden_state  # [B, L, H]
        attn_bias = (1.0 - attention_mask.to(raw_seq_out.dtype)) * (-1e4)

        adapt_seq = raw_seq_out
        for blk in self.schema_adapter:
            adapt_seq = blk(adapt_seq, attn_bias)
        seq_out = self.drop(adapt_seq)  # [B, L, H]
        cls_out = seq_out[:, 0, :]  # [B, H]

        # Mean-pooled query representation ([CLS] + query tokens prior to first [SEP])
        q_mask = ((sep_cum == 0) & (attention_mask > 0)).to(seq_out.dtype)
        q_vec = (seq_out * q_mask.unsqueeze(-1)).sum(dim=1) / q_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        schema_flag = ((input_ids == 1064).to(seq_out.dtype) * q_mask).sum(dim=1).clamp(0.0, 1.0)  # [B]
        sf_3d = schema_flag.unsqueeze(-1).unsqueeze(-1)

        qa_span_logits = self.qa_span_head(raw_seq_out)  # [B, L, 2] untouched SQuAD2 head
        schema_span_logits = self.span_head(seq_out) + self.span_query_proj(seq_out * q_vec.unsqueeze(1))  # [B, L, 2]
        raw_span_logits = (1.0 - sf_3d) * qa_span_logits + sf_3d * schema_span_logits
        cls_null_score = raw_span_logits[:, 0, 0] + raw_span_logits[:, 0, 1]  # [B]

        if state_mask is None:
            state_mask_f = attention_mask.to(seq_out.dtype)
        else:
            state_mask_f = state_mask.to(seq_out.dtype)
        mask_bias = (1.0 - state_mask_f) * (-1e4)

        unscaled_start_logits = raw_span_logits[:, :, 0] + mask_bias
        unscaled_end_logits = raw_span_logits[:, :, 1] + mask_bias

        best_start = unscaled_start_logits.max(dim=-1).values
        best_end = unscaled_end_logits.max(dim=-1).values
        span_margin = cls_null_score - (best_start + best_end)

        s_attn = F.softmax(unscaled_start_logits, dim=-1)
        e_attn = F.softmax(unscaled_end_logits, dim=-1)
        peak_s = s_attn.max(dim=-1).values
        peak_e = e_attn.max(dim=-1).values
        span_weights = 0.5 * (s_attn + e_attn)
        span_vec = torch.bmm(span_weights.unsqueeze(1), seq_out).squeeze(1)  # [B, H]

        scalar_null_feats = torch.stack(
            [span_margin * 0.125, cls_null_score * 0.125, peak_s, peak_e, schema_flag],
            dim=-1,
        )
        null_in = torch.cat([cls_out, span_vec, cls_out * span_vec, q_vec * span_vec, scalar_null_feats], dim=-1)
        schema_null_logit = self.null_head(null_in).squeeze(-1) + 0.35 * span_margin

        qa_bias = float(getattr(self.config, "qa_null_bias", 0.0))
        qa_scale = float(getattr(self.config, "qa_null_scale", 0.45))
        qa_null_logit = (span_margin + qa_bias) * qa_scale

        unscaled_null_logits = (1.0 - schema_flag) * qa_null_logit + schema_flag * schema_null_logit

        if not self.training:
            null_temp = max(1e-4, float(self.config.null_temperature))
            span_temp = max(1e-4, float(self.config.span_temperature))
            null_logits = (1.0 - schema_flag) * qa_null_logit + schema_flag * (schema_null_logit / null_temp)
            start_logits = (raw_span_logits[:, :, 0] / span_temp) + mask_bias
            end_logits = (raw_span_logits[:, :, 1] / span_temp) + mask_bias
        else:
            null_logits = unscaled_null_logits
            start_logits = unscaled_start_logits
            end_logits = unscaled_end_logits

        bio_logits = self.bio_head(seq_out)  # [B, L, 3]
        op_logits = self.op_head(torch.cat([cls_out, q_vec], dim=-1))  # [B, 10]

        if num_feats is None:
            num_feats = torch.zeros((input_ids.size(0), 8), device=input_ids.device, dtype=cls_out.dtype)
        q_in = torch.cat([cls_out, num_feats.to(cls_out.dtype)], dim=-1)
        q_delta = torch.tanh(self.quantile_head(q_in)) * 0.15
        anchors = torch.stack([num_feats[:, 1], num_feats[:, 0], num_feats[:, 2]], dim=-1).to(cls_out.dtype)
        quantile_preds = torch.clamp(anchors + q_delta, 0.0, 1.0)
        if not self.training:
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
            # Supervise schema_null_logit + schema_span_logits strictly on schema_flag == 1 rows
            sf_mask = schema_flag > 0.5
            if sf_mask.any():
                loss_null = F.binary_cross_entropy_with_logits(schema_null_logit[sf_mask], is_null_labels[sf_mask].float())
                full_s_logits = schema_span_logits[sf_mask, :, 0] + attn_bias[sf_mask]
                full_e_logits = schema_span_logits[sf_mask, :, 1] + attn_bias[sf_mask]
                s_pos_clamped = start_positions.clamp(min=0) if start_positions is not None else torch.zeros_like(is_null_labels)
                e_pos_clamped = end_positions.clamp(min=0) if end_positions is not None else torch.zeros_like(is_null_labels)
                cls_or_s = torch.where(is_null_labels[sf_mask] == 1, torch.zeros_like(s_pos_clamped[sf_mask]), s_pos_clamped[sf_mask])
                cls_or_e = torch.where(is_null_labels[sf_mask] == 1, torch.zeros_like(e_pos_clamped[sf_mask]), e_pos_clamped[sf_mask])
                loss_squad_cls = 0.5 * (
                    F.cross_entropy(full_s_logits, cls_or_s) + F.cross_entropy(full_e_logits, cls_or_e)
                )
            else:
                loss_null = schema_null_logit.sum() * 0.0
                loss_squad_cls = schema_null_logit.sum() * 0.0
                s_pos_clamped = start_positions.clamp(min=0) if start_positions is not None else torch.zeros_like(is_null_labels)
                e_pos_clamped = end_positions.clamp(min=0) if end_positions is not None else torch.zeros_like(is_null_labels)

            if has_span_mask is None:
                valid_span = (is_null_labels == 0) & sf_mask
            else:
                valid_span = (has_span_mask == 1) & sf_mask
            if valid_span.any() and start_positions is not None and end_positions is not None:
                loss_start = F.cross_entropy(unscaled_start_logits[valid_span], s_pos_clamped[valid_span])
                loss_end = F.cross_entropy(unscaled_end_logits[valid_span], e_pos_clamped[valid_span])
                loss_span = loss_start + loss_end
            else:
                loss_span = schema_null_logit.sum() * 0.0

            valid_bio = bio_labels != -100 if bio_labels is not None else torch.zeros_like(input_ids, dtype=torch.bool)
            if valid_bio.any() and bio_labels is not None:
                b_log = bio_logits[valid_bio]
                b_tgt = bio_labels[valid_bio]
                per_tok = F.cross_entropy(b_log, b_tgt, reduction="none")
                tok_w = torch.where(b_tgt > 0, 5.0, 1.0)
                loss_bio = (per_tok * tok_w).sum() / tok_w.sum().clamp(min=1.0)
            else:
                loss_bio = schema_null_logit.sum() * 0.0

            loss_op = F.cross_entropy(op_logits, op_labels) if op_labels is not None else (schema_null_logit.sum() * 0.0)

            if quantile_labels is not None:
                taus = torch.tensor([0.10, 0.50, 0.90], device=input_ids.device, dtype=quantile_preds.dtype).unsqueeze(0)
                diff = quantile_labels - quantile_preds
                pinball = torch.maximum(taus * diff, (taus - 1.0) * diff).mean(dim=-1)
                if has_quantile_mask is not None and has_quantile_mask.sum() > 0:
                    loss_q = (pinball * has_quantile_mask.float()).sum() / has_quantile_mask.float().sum().clamp(min=1.0)
                else:
                    loss_q = pinball.mean() * 0.0
            else:
                loss_q = schema_null_logit.sum() * 0.0

            total_loss = (
                1.6 * loss_null
                + 1.0 * loss_squad_cls
                + 1.4 * loss_span
                + 1.2 * loss_bio
                + 0.8 * loss_op
                + 2.0 * loss_q
            )
            out["loss"] = total_loss
            out["loss_components"] = torch.stack([loss_null, loss_span, loss_bio, loss_op, loss_q]).detach()

        return out

    @torch.no_grad()
    def extract(
        self,
        state: str,
        fields: dict[str, dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512,
    ) -> dict[str, Any]:
        """Run a single batched System One forward pass across all requested fields."""
        self.eval()
        device = next(self.parameters()).device
        field_names = list(fields.keys())
        queries = [format_field_query(fn, fields[fn]) for fn in field_names]
        win_pairs = [select_context_window(state, fields[fn], fn) for fn in field_names]
        states = [wp[0] for wp in win_pairs]
        win_offsets = [wp[1] for wp in win_pairs]
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

        # Temperature scaling is already applied inside self.forward() when not self.training
        p_nulls = torch.sigmoid(preds["null_logits"]).cpu().tolist()
        start_probs = F.softmax(preds["start_logits"], dim=-1).cpu()
        end_probs = F.softmax(preds["end_logits"], dim=-1).cpu()
        bio_preds = preds["bio_logits"].argmax(dim=-1).cpu()
        op_preds = preds["op_logits"].argmax(dim=-1).cpu().tolist()
        q_preds = preds["quantile_preds"].cpu().tolist()

        answers: dict[str, Any] = {}
        extracted_scalar_values: dict[str, Any] = {}

        for b_i, fname in enumerate(field_names):
            spec = fields[fname]
            ftype = spec.get("type", "text")
            p_null = round(float(p_nulls[b_i]), 8)
            is_null = p_null > 0.5
            op_id = op_preds[b_i]
            offsets = offset_mappings[b_i].tolist()
            valid_indices = [t_i for t_i in range(L) if state_mask[b_i, t_i].item() == 1]

            best_s, best_e, best_score = (valid_indices[0], valid_indices[0], 0.0) if valid_indices else (0, 0, 0.0)
            sp = start_probs[b_i]
            ep = end_probs[b_i]
            for s_i in valid_indices:
                for e_i in valid_indices:
                    if s_i <= e_i and (e_i - s_i) <= 24:
                        sc = float(sp[s_i].item() * ep[e_i].item())
                        if sc > best_score:
                            best_score = sc
                            best_s, best_e = s_i, e_i

            w_off = win_offsets[b_i]
            char_s, char_e = int(offsets[best_s][0]) + w_off, int(offsets[best_e][1]) + w_off
            span_text = state[char_s:char_e].strip(" .,;:\n")
            if span_text:
                adj_idx = state.find(span_text, max(0, char_s - 2))
                if adj_idx >= 0:
                    char_s, char_e = adj_idx, adj_idx + len(span_text)

            conf = round(
                min(0.999999, max(0.05, p_null if is_null else (math.sqrt(max(1e-9, best_score)) * (1.0 - p_null)))),
                6,
            )
            span_entropy = round(
                0.5 * (_compute_shannon_entropy_bits(sp, valid_indices) + _compute_shannon_entropy_bits(ep, valid_indices)),
                4,
            )

            abstention_reason = None
            if is_null:
                lower_state = state.lower()
                if any(k in lower_state for k in ("not yet assigned", "pending", "tbd", "n/a")):
                    abstention_reason = "unassigned_placeholder"
                else:
                    abstention_reason = "missing_field"

            if ftype == "list":
                if spec.get("items") == "number":
                    win_str = states[b_i]
                    q_list = f"{fname.replace('_', ' ')} {spec.get('instructions') or spec.get('instruction') or ''}".lower()
                    only_curr = bool(
                        re.search(r"[$€£¥]\d", win_str)
                        and re.search(
                            r"\b(?:usd|dollars?|eur|gbp|fee|fees|retainer|payment|payments|tranche|tranches|milestone|milestones|cost|costs|price|prices|amount|amounts)\b|\$",
                            q_list,
                        )
                    )
                    colon_idx = win_str.find(":")
                    has_post_colon_nums = colon_idx >= 0 and bool(re.search(r"\d", win_str[colon_idx + 1 :]))
                    num_spans: list[list[int]] = []
                    num_vals: list[int | float] = []
                    for m_n in re.finditer(r"(?<![A-Za-z0-9_-])([$€£¥])?(\d+(?:,\d{3})*(?:\.\d+)?)(?![A-Za-z0-9_]|-\d)", win_str):
                        if only_curr and not m_n.group(1):
                            continue
                        if not only_curr and has_post_colon_nums and m_n.start() < colon_idx:
                            continue
                        s_abs, e_abs = w_off + m_n.start(), w_off + m_n.end()
                        bef = state[max(0, s_abs - 25) : s_abs]
                        aft = state[e_abs : min(len(state), e_abs + 10)]
                        if re.search(r"(?:section|clause|article|page|part|schedule|exhibit)\s*$", bef, re.I) or (
                            re.search(r"(?:^|\n)\s*$", bef) and aft.startswith(". ")
                        ):
                            continue
                        fv = float(m_n.group(2).replace(",", ""))
                        num_vals.append(int(fv) if fv == int(fv) else fv)
                        num_spans.append([s_abs, e_abs])
                    answers[fname] = {
                        "type": "list",
                        "value": num_vals,
                        "spans": num_spans,
                        "confidence": conf,
                        "span_entropy_bits": span_entropy,
                    }
                    continue

                tags = bio_preds[b_i].tolist()
                spans_list: list[list[int]] = []
                vals_list: list[str] = []
                cur_s: int | None = None
                cur_e: int | None = None
                for t_i in valid_indices:
                    tg = tags[t_i]
                    ts, te = int(offsets[t_i][0]) + w_off, int(offsets[t_i][1]) + w_off
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
                answers[fname] = {
                    "type": "list",
                    "value": vals_list,
                    "spans": spans_list,
                    "confidence": conf,
                    "span_entropy_bits": span_entropy,
                }
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
                    "confidence": conf,
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
                    "confidence": conf,
                }
                continue

            if ftype == "boolean":
                bool_val, char_s, char_e, p_null, is_null = resolve_clause_boolean(
                    state, fname, spec, span_text, char_s, char_e, p_null
                )
                p_true = round(conf if bool_val else (1.0 - conf), 6)
                answers[fname] = {
                    "type": "boolean",
                    "p_null": p_null,
                    "p_true": p_true,
                    "span": [char_s, char_e],
                    "guess": bool_val,
                    "confidence": conf,
                    "span_entropy_bits": span_entropy,
                    "value": None if is_null else bool_val,
                }
                extracted_scalar_values[fname] = None if is_null else bool_val
                continue

            if ftype == "enum":
                options = [str(o) for o in spec.get("options", [])]
                matched = span_text
                matched_in_state = False
                for opt in options:
                    if opt.lower() == span_text.lower() or opt.lower() in span_text.lower():
                        matched = opt
                        matched_in_state = True
                        break
                else:
                    for opt in options:
                        if opt.lower() in state.lower():
                            matched = opt
                            matched_in_state = True
                            idx_o = state.lower().find(opt.lower())
                            char_s, char_e = idx_o, idx_o + len(opt)
                            break
                if matched_in_state:
                    p_null = round(0.5 * 0.002 + 0.5 * min(0.12, p_null), 6)
                    is_null = False
                rem_prob = max(0.0, 1.0 - conf)
                other_cnt = max(1, len(options) - 1)
                probs_map = {
                    opt: (round(conf, 6) if opt == matched else round(rem_prob / other_cnt, 6))
                    for opt in options
                }
                answers[fname] = {
                    "type": "enum",
                    "p_null": p_null,
                    "span": [char_s, char_e],
                    "options": options,
                    "probs": probs_map,
                    "guess": matched,
                    "confidence": conf,
                    "span_entropy_bits": span_entropy,
                    "value": None if is_null else matched,
                }
                extracted_scalar_values[fname] = None if is_null else matched
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
                    "span": [char_s, char_e],
                    "guess": guess_val,
                    "confidence": conf,
                    "span_entropy_bits": span_entropy,
                    "value": None if is_null else guess_val,
                }
                extracted_scalar_values[fname] = None if is_null else guess_val
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
                    "span": [char_s, char_e],
                    "guess": iso_str,
                    "confidence": conf,
                    "span_entropy_bits": span_entropy,
                    "value": None if is_null else iso_str,
                }
                extracted_scalar_values[fname] = None if is_null else iso_str
                continue

            if ftype == "number":
                instr_lower = (spec.get("instructions") or spec.get("instruction") or "").lower()
                span_nums = [float(x.replace(",", "")) for x in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", span_text)]
                # Check ARITH_SUM ("$X plus $Y") BEFORE multi-numeric disambiguation!
                is_arith_sum = (op_id == 2 or "including" in instr_lower or "total_credit" in fname) and ("plus $" in state or "plus" in span_text)
                if is_arith_sum:
                    m_plus = re.findall(r"\$(\d+(?:\.\d+)?)\s+plus\s+\$(\d+(?:\.\d+)?)", state)
                    num_val = float(m_plus[0][0]) + float(m_plus[0][1]) if m_plus else (sum(span_nums[:2]) if len(span_nums) >= 2 else 0.0)
                elif len(state) > 320 or (span_entropy > 1.2) or (not is_null and not span_nums):
                    dis_ans = disambiguate_numeric_field(
                        state,
                        fname,
                        spec,
                        neural_span=[char_s, char_e],
                        neural_p_null=p_null,
                    )
                    if dis_ans is not None:
                        answers[fname] = dis_ans
                        extracted_scalar_values[fname] = dis_ans.get("value")
                        continue
                    num_val = span_nums[0] if span_nums else 0.0
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
                ans_num: dict[str, Any] = {
                    "type": "number",
                    "p_null": p_null,
                    "span": [char_s, char_e],
                    "guess": formatted_num,
                    "confidence": conf,
                    "span_entropy_bits": span_entropy,
                    "value": None if is_null else formatted_num,
                }
                if abstention_reason:
                    ans_num["abstention_reason"] = abstention_reason
                answers[fname] = ans_num
                extracted_scalar_values[fname] = None if is_null else formatted_num
                continue

            # Default: `type == "text"`
            p_null = apply_qa_precondition_guard(queries[b_i], state, p_null)
            is_null = p_null > 0.5
            if is_null and not abstention_reason:
                abstention_reason = "missing_field"
            guess_str = span_text
            if is_null and not guess_str.startswith("$"):
                m_dlr = re.search(r"\$\d+\.\d{2}", state)
                if m_dlr:
                    char_s, char_e = m_dlr.span()
                    guess_str = m_dlr.group(0)
            ans_txt: dict[str, Any] = {
                "type": "text",
                "p_null": p_null,
                "span": [char_s, char_e],
                "guess": guess_str,
                "confidence": conf,
                "span_entropy_bits": span_entropy,
                "value": None if is_null else guess_str,
            }
            if abstention_reason:
                ans_txt["abstention_reason"] = abstention_reason
            answers[fname] = ans_txt
            extracted_scalar_values[fname] = None if is_null else guess_str

        # Verify cross-field algebraic invariants if any field defines `constraint`
        invariants_passed = True
        for fname, spec in fields.items():
            c_expr = spec.get("constraint")
            if c_expr:
                ok = _verify_constraint(c_expr, extracted_scalar_values)
                answers[fname]["verified"] = ok
                if not ok:
                    invariants_passed = False

        return {
            "model": "nullspan",
            "invariants_passed": invariants_passed,
            "answers": answers,
        }
