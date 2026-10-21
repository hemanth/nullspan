#!/usr/bin/env python3
"""
Render a 2x supersampled (2400x1520 -> 1200x760, 15fps, 60 frames) social media GIF
comparing Cloudflare `@cf/cloudflare/clairic + Clef` vs `nullspan` in the EXACT
`nullspan` studio visual theme (`docs/index.html` & `docs/nullspan-social-mobile.png`):
- Warm paper canvas (`#FBFAF7`) + subtle dot grid + 8px top `#111111` bar
- Solid black `nullspan` brand-mark + 3 semantic highlighter badges (`#FFE484`, `#F5D0CE`, `#C8F0D4`)
- Crisp white (`#FFFFFF`) cards with `#111111` black header strips, `#F3F0E8` muted sub-boxes,
  and hard `#111111` offset shadows (zero lavender/pastel card fills).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 1200 * S, 760 * S
OUT_W, OUT_H = 1200, 760

HN_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"
JB_BOLD = "/Users/lika/Library/Fonts/JetBrainsMonoNerdFontMono-Bold.ttf"
JB_REG = "/Users/lika/Library/Fonts/JetBrainsMonoNerdFontMono-Regular.ttf"
MENLO_PATH = "/System/Library/Fonts/Menlo.ttc"


def font_sans_bold(px: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(HN_PATH, int(px * S), index=1)


def font_sans_med(px: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(HN_PATH, int(px * S), index=10)


def font_mono_bold(px: float) -> ImageFont.FreeTypeFont:
    path = JB_BOLD if os.path.exists(JB_BOLD) else MENLO_PATH
    return ImageFont.truetype(path, int(px * S))


def font_mono_reg(px: float) -> ImageFont.FreeTypeFont:
    path = JB_REG if os.path.exists(JB_REG) else MENLO_PATH
    return ImageFont.truetype(path, int(px * S))


# Exact nullspan CSS tokens from docs/index.html & docs/social-poster.html
THEME = {
    "bg": (251, 250, 247),           # --bg: #FBFAF7
    "dot": (228, 224, 214),          # radial dot grid
    "surface": (255, 255, 255),      # --surface: #FFFFFF
    "muted_bg": (243, 240, 232),     # --muted-bg: #F3F0E8
    "border_subtle": (229, 225, 216),# --border-subtle: #E5E1D8
    "ink": (17, 17, 17),             # --ink: #111111
    "ink_muted": (82, 80, 74),       # --ink-muted: #52504A
    "span": (255, 228, 132),         # --accent-span: #FFE484
    "list": (200, 240, 212),         # --accent-list: #C8F0D4
    "null": (245, 208, 206),         # --accent-null: #F5D0CE
}


def draw_studio_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    header_text: str | None = None,
    header_right: str | None = None,
    active: bool = False,
    radius: float = 8,
) -> None:
    """Draw a crisp white nullspan studio card with optional solid #111111 top header strip."""
    x0, y0, x1, y1 = [int(v * S) for v in box]
    r = int(radius * S)
    so = int((4.5 if active else 3.0) * S)
    bw = int((2.4 if active else 1.8) * S)

    # Hard offset ink shadow
    draw.rounded_rectangle([x0 + so, y0 + so, x1 + so, y1 + so], radius=r, fill=THEME["ink"])
    # Crisp white card body
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=THEME["surface"], outline=THEME["ink"], width=bw)

    if header_text:
        hh = int(28 * S)
        # Top black bar matching SINGLE-PASS SYSTEM ONE X-RAY in social-poster.html
        draw.rounded_rectangle([x0, y0, x1, y0 + hh], radius=r, fill=THEME["ink"])
        draw.rectangle([x0, y0 + hh - r, x1, y0 + hh], fill=THEME["ink"])
        hf = font_mono_bold(9.5)
        draw.text((x0 + int(12 * S), y0 + int(7 * S)), header_text, font=hf, fill=THEME["bg"])
        if header_right:
            rb = draw.textbbox((0, 0), header_right, font=hf)
            rw = rb[2] - rb[0]
            draw.text((x1 - int(12 * S) - rw, y0 + int(7 * S)), header_right, font=hf, fill=THEME["span"])


def draw_pill(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    border_color: tuple[int, int, int] = THEME["ink"],
    font: ImageFont.FreeTypeFont | None = None,
    pad_x: float = 9,
    pad_y: float = 4,
    radius: float = 5,
    shadow: bool = False,
) -> float:
    if font is None:
        font = font_mono_bold(10.0)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px, py = int(pad_x * S), int(pad_y * S)
    rx0, ry0 = int(x * S), int(y * S)
    rx1, ry1 = rx0 + tw + px * 2, ry0 + th + py * 2
    if shadow:
        so = int(2.5 * S)
        draw.rounded_rectangle([rx0 + so, ry0 + so, rx1 + so, ry1 + so], radius=int(radius * S), fill=THEME["ink"])
    draw.rounded_rectangle(
        [rx0, ry0, rx1, ry1],
        radius=int(radius * S),
        fill=bg,
        outline=border_color,
        width=max(1, int(1.6 * S)),
    )
    draw.text((rx0 + px - bbox[0], ry0 + py - bbox[1]), text, font=font, fill=fg)
    return (rx1 - rx0) / S


def draw_arrow(draw: ImageDraw.ImageDraw, x0: float, x1: float, y: float, active: bool) -> None:
    col = THEME["ink"] if active else (175, 170, 160)
    w = int((2.6 if active else 1.6) * S)
    sx0, sx1, sy = int(x0 * S), int(x1 * S), int(y * S)
    draw.line([(sx0, sy), (sx1, sy)], fill=col, width=w)
    ah = int(4.5 * S)
    draw.polygon([(sx1, sy), (sx1 - ah, sy - ah), (sx1 - ah, sy + ah)], fill=col)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_gif = root / "docs" / "nullspan-vs-clairic.gif"
    space_gif = root / "space" / "nullspan-vs-clairic.gif"
    preview_png = root / "docs" / "nullspan-vs-clairic-preview.png"
    frames_dir = Path("/tmp/nullspan_vs_clairic_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 4 Stage Cards matching the 3 semantic badges (Span #FFE484, List/Verified #C8F0D4, Null #F5D0CE)
    stages = [
        {
            "tag": "1 · ARCHITECTURE",
            "badge": "1 PASS vs 2 SPLIT",
            "badge_bg": THEME["span"],
            "title": "Unified vs Split",
            "rows": [
                ("Clairic + Clef", "2 Split RPCs", THEME["null"]),
                ("nullspan 67M", "1 Encoder Pass", THEME["list"]),
                ("op_head Router", "10 System1 Ops", THEME["span"]),
                ("Output Tokens", "0 Generated", THEME["list"]),
            ],
        },
        {
            "tag": "2 · RUNTIME TIERS",
            "badge": "WASM + ZERO-DEP",
            "badge_bg": THEME["list"],
            "title": "Local vs Lock-In",
            "rows": [
                ("Workers AI", "Closed Cloud", THEME["null"]),
                ("Browser WASM", "64.3 MB Int8", THEME["span"]),
                ("PyTorch FP32", "Open MIT (HF)", THEME["list"]),
                ("Zero-Dep Fast", "0 MB · 5 µs", THEME["list"]),
            ],
        },
        {
            "tag": "3 · MATH & RANGES",
            "badge": "VERIFIED MATH",
            "badge_bg": THEME["list"],
            "title": "Audited Equations",
            "rows": [
                ("Clairic Math", "No Cross-Check", THEME["null"]),
                ("constraint", "nodes * rate", THEME["list"]),
                ("verified Flag", "true (100% EM)", THEME["list"]),
                ("quantile_head", "p10<=p50<=p90", THEME["span"]),
            ],
        },
        {
            "tag": "4 · ABSTENTION GATE",
            "badge": "P_NULL >= 0.5",
            "badge_bg": THEME["null"],
            "title": "Reason + Entropy",
            "rows": [
                ("Clairic Null", "Bare p_null", THEME["null"]),
                ("null_head", "AUROC 0.9998", THEME["list"]),
                ("Reason Tag", "unassigned_cve", THEME["null"]),
                ("Span Entropy", "H = 0.03 bits", THEME["span"]),
            ],
        },
    ]

    kpis = [
        ("98.7 vs 77.1", "Capability (50I:50C)", THEME["span"]),
        ("97.7 vs 71.5", "Composite Score", THEME["list"]),
        ("64.3 MB WASM", "Browser ONNX Int8", THEME["list"]),
        ("$0.0003 / 1k", "100x Lower Cost", THEME["null"]),
    ]

    code_lines = [
        "import { extract } from 'nullspan'; // 1 pass",
        "const out = extract(doc, { // Clairic + Clef:",
        "  subtotal: { type:'number', constraint:'...' },",
        "  severity: { type:'enum', options:['sev-1'] }",
        "}); // => verified: true, 0 output tokens",
    ]

    total_frames = 60
    for idx in range(total_frames):
        t = idx / float(total_frames)
        active_stage = int((t * 4.0) % 4.0)
        scene_idx = idx // 20  # 0, 1, 2

        img = Image.new("RGB", (W, H), THEME["bg"])
        draw = ImageDraw.Draw(img)

        # Subtle dot grid matching social-poster.html
        step = int(20 * S)
        r_dot = int(1.0 * S)
        for gx in range(step, W, step):
            for gy in range(step, H, step):
                draw.ellipse([gx - r_dot, gy - r_dot, gx + r_dot, gy + r_dot], fill=THEME["dot"])

        # 8px solid #111111 top studio border
        draw.rectangle([0, 0, W, int(6 * S)], fill=THEME["ink"])

        # Top Studio Header Bar (y = 16..54) matching social-poster.html & docs/index.html
        cur_x = 32.0
        w_brand = draw_pill(
            draw,
            cur_x,
            16,
            "nullspan",
            THEME["ink"],
            THEME["bg"],
            THEME["ink"],
            font_mono_bold(13.0),
            pad_x=12,
            pad_y=4.5,
            radius=5,
        )
        cur_x += w_brand + 10
        for b_txt, b_col in [
            ("[START, END] POINTER", THEME["span"]),
            ("CALIBRATED P_NULL", THEME["null"]),
            ("VERIFIED INVARIANTS", THEME["list"]),
        ]:
            wb = draw_pill(draw, cur_x, 18, b_txt, b_col, THEME["ink"], THEME["ink"], font_mono_bold(9.5), pad_x=8, pad_y=3.5, radius=4)
            cur_x += wb + 8

        draw_pill(
            draw,
            838,
            18,
            "npm i nullspan  ·  pip install nullspan",
            THEME["surface"],
            THEME["ink"],
            THEME["ink"],
            font_mono_bold(9.8),
            pad_x=10,
            pad_y=3.5,
            radius=4,
        )

        # Divider line under studio bar
        draw.line([(int(32 * S), int(52 * S)), (int(1168 * S), int(52 * S))], fill=THEME["ink"], width=int(1.8 * S))

        # Hero Title with Inline Boxed Highlighter Tokens (y = 62..118)
        draw.text((int(32 * S), int(64 * S)), "Unify", font=font_sans_bold(31), fill=THEME["ink"])
        w1 = draw_pill(draw, 122, 61, "[Clairic Extraction]", THEME["span"], THEME["ink"], THEME["ink"], font_sans_bold(23), pad_x=10, pad_y=3, radius=6, shadow=True)
        px_plus = 122 + w1 + 10
        draw.text((int(px_plus * S), int(64 * S)), "+", font=font_sans_bold(31), fill=THEME["ink"])
        w2 = draw_pill(draw, px_plus + 26, 61, "[Clef Decisions]", THEME["null"], THEME["ink"], THEME["ink"], font_sans_bold(23), pad_x=10, pad_y=3, radius=6, shadow=True)
        px_in = px_plus + 26 + w2 + 12
        draw.text((int(px_in * S), int(64 * S)), "in", font=font_sans_bold(31), fill=THEME["ink"])
        draw_pill(draw, px_in + 36, 61, "1 Open WASM Pass.", THEME["list"], THEME["ink"], THEME["ink"], font_sans_bold(23), pad_x=10, pad_y=3, radius=6, shadow=True)

        draw.text(
            (int(32 * S), int(103 * S)),
            "Cloudflare splits System One across 2 closed Workers AI models. nullspan runs all 10 ops + math verification in 1 local pass:",
            font=font_sans_med(13.5),
            fill=THEME["ink_muted"],
        )

        # 4-Stage Pipeline Row (y = 132..424)
        card_y0, card_y1 = 132, 424
        xs = [(32, 290), (324, 582), (616, 874), (908, 1168)]
        arrows = [(293, 321), (585, 613), (877, 905)]
        for ai, (ax0, ax1) in enumerate(arrows):
            draw_arrow(draw, ax0, ax1, 278, active=(ai == active_stage))

        for si, (cx0, cx1) in enumerate(xs):
            st = stages[si]
            is_act = si == active_stage
            draw_studio_card(
                draw,
                (cx0, card_y0, cx1, card_y1),
                header_text=st["tag"],
                header_right="● ACTIVE" if is_act else None,
                active=is_act,
            )

            # Subheader title + semantic highlighter badge matching social-poster.html
            draw.text((int((cx0 + 14) * S), int((card_y0 + 38) * S)), st["title"], font=font_sans_bold(16.0), fill=THEME["ink"])
            bf = font_mono_bold(8.5)
            bb = draw.textbbox((0, 0), st["badge"], font=bf)
            bw_px = (bb[2] - bb[0]) / S + 14
            draw_pill(draw, cx1 - 12 - bw_px, card_y0 + 37, st["badge"], st["badge_bg"], THEME["ink"], THEME["ink"], bf, pad_x=6, pad_y=2.5, radius=4)

            for ri, (rleft, rright, pill_col) in enumerate(st["rows"]):
                ry = card_y0 + 74 + ri * 52
                is_cf_row = ri == 0
                row_hi = is_act and (ri == 1 + int((t * 12) % 3))
                row_bg = THEME["muted_bg"] if is_cf_row else (THEME["span"] if row_hi else THEME["surface"])
                draw.rounded_rectangle(
                    [int((cx0 + 12) * S), int(ry * S), int((cx1 - 12) * S), int((ry + 40) * S)],
                    radius=int(6 * S),
                    fill=row_bg,
                    outline=THEME["ink"],
                    width=int((2.0 if row_hi else 1.3) * S),
                )
                draw.text(
                    (int((cx0 + 20) * S), int((ry + 12) * S)),
                    rleft,
                    font=font_mono_bold(9.6),
                    fill=THEME["ink_muted"] if is_cf_row else THEME["ink"],
                )
                # Right-aligned semantic pill inside the row
                rf = font_mono_bold(8.8)
                rb = draw.textbbox((0, 0), rright, font=rf)
                rw_px = (rb[2] - rb[0]) / S + 12
                draw_pill(
                    draw,
                    cx1 - 18 - rw_px,
                    ry + 8,
                    rright,
                    THEME["surface"] if row_hi else pill_col,
                    THEME["ink"],
                    THEME["ink"],
                    rf,
                    pad_x=6,
                    pad_y=2.5,
                    radius=4,
                )

        # Bottom Left Card (x = 32..756, y = 444..734): Single-Pass System One X-Ray vs Clairic + Clef
        bot_y0, bot_y1 = 444, 734
        draw_studio_card(
            draw,
            (32, bot_y0, 756, bot_y1),
            header_text="HEAD-TO-HEAD X-RAY  ·  @cf/cloudflare/clairic + Clef  vs  nullspan",
            header_right="0 OUTPUT TOKENS",
        )

        # Scene selector pills on muted strip
        scene_tabs = [
            ("1 · UNIFIED PASS (INC-40928)", THEME["span"]),
            ("2 · INVARIANT MATH (#88310)", THEME["list"]),
            ("3 · JEVBENCH 4-AXIS SCORES", THEME["null"]),
        ]
        cur_tx = 46.0
        for tab_i, (tab_lbl, tab_col) in enumerate(scene_tabs):
            is_tab = tab_i == scene_idx
            pw = draw_pill(
                draw,
                cur_tx,
                bot_y0 + 36,
                tab_lbl,
                tab_col if is_tab else THEME["muted_bg"],
                THEME["ink"],
                THEME["ink"],
                font_mono_bold(9.0),
                pad_x=8,
                pad_y=3.0,
                radius=4,
                shadow=is_tab,
            )
            cur_tx += pw + 10

        paper = (46, bot_y0 + 66, 742, bot_y1 - 14)
        draw.rounded_rectangle(
            [int(v * S) for v in paper],
            radius=int(7 * S),
            fill=THEME["muted_bg"],
            outline=THEME["ink"],
            width=int(1.6 * S),
        )

        if scene_idx == 0:
            # Highlighted document strip + 2-call vs 1-pass comparison
            doc_box = (58, bot_y0 + 76, 730, bot_y0 + 114)
            draw.rounded_rectangle([int(v * S) for v in doc_box], radius=int(5 * S), fill=THEME["surface"], outline=THEME["ink"], width=int(1.3 * S))
            draw.text((int(68 * S), int((bot_y0 + 88) * S)), "Incident", font=font_mono_reg(9.8), fill=THEME["ink"])
            draw_pill(draw, 130, bot_y0 + 83, "INC-40928 [9,18]", THEME["span"], THEME["ink"], THEME["ink"], font_mono_bold(9.0), pad_x=6, pad_y=2)
            draw.text((int(258 * S), int((bot_y0 + 88) * S)), "credit", font=font_mono_reg(9.8), fill=THEME["ink"])
            draw_pill(draw, 304, bot_y0 + 83, "$1240.50 + $310.25", THEME["span"], THEME["ink"], THEME["ink"], font_mono_bold(9.0), pad_x=6, pad_y=2)
            draw_pill(draw, 456, bot_y0 + 83, "sev-1 [80,85]", THEME["list"], THEME["ink"], THEME["ink"], font_mono_bold(9.0), pad_x=6, pad_y=2)
            draw_pill(draw, 566, bot_y0 + 83, "CVE unassigned (p_null:0.99)", THEME["null"], THEME["ink"], THEME["ink"], font_mono_bold(8.8), pad_x=6, pad_y=2)

            # Row 1: Cloudflare Clairic + Clef
            cf_row = (58, bot_y0 + 122, 730, bot_y0 + 178)
            draw.rounded_rectangle([int(v * S) for v in cf_row], radius=int(5 * S), fill=THEME["surface"], outline=THEME["ink"], width=int(1.3 * S))
            draw_pill(draw, 68, bot_y0 + 130, "CLOUDFLARE: Clairic + Clef", THEME["null"], THEME["ink"], THEME["ink"], font_mono_bold(8.8), pad_x=6, pad_y=2)
            draw.text((int(268 * S), int((bot_y0 + 133) * S)), "Requires 2 Separate Closed Workers AI Calls · No Browser Mode", font=font_mono_bold(9.2), fill=THEME["ink_muted"])
            draw.text((int(68 * S), int((bot_y0 + 155) * S)), "RPC 1 (Clairic): spans + bare p_null   |   RPC 2 (Clef): severity='sev-1', data_loss=false", font=font_mono_reg(9.4), fill=THEME["ink_muted"])

            # Row 2: nullspan unified pass
            ns_row = (58, bot_y0 + 186, 730, bot_y1 - 22)
            draw.rounded_rectangle([int(v * S) for v in ns_row], radius=int(5 * S), fill=THEME["surface"], outline=THEME["ink"], width=int(2.0 * S))
            draw_pill(draw, 68, bot_y0 + 194, "NULLSPAN: 1 Unified Pass", THEME["list"], THEME["ink"], THEME["ink"], font_mono_bold(8.8), pad_x=6, pad_y=2)
            draw.text((int(258 * S), int((bot_y0 + 197) * S)), "5 Parallel Heads · 10 Routed Ops · Runs in Browser WASM & Python", font=font_mono_bold(9.2), fill=THEME["ink"])
            draw.text((int(68 * S), int((bot_y0 + 220) * S)), "incident_id: 'INC-40928' (H=0.03b) · total_credit: $1550.75 (ARITH_SUM) · severity: 'sev-1' (0.99)", font=font_mono_bold(9.3), fill=THEME["ink"])
            draw.text((int(68 * S), int((bot_y0 + 242) * S)), "cve_id: null (p_null: 0.994 · abstention_reason: 'unassigned_placeholder' · 0% FPHR)", font=font_mono_bold(9.3), fill=THEME["ink"])

        elif scene_idx == 1:
            # Scene 1: Cross-Field Algebraic Invariant Verification
            cf_box = (58, bot_y0 + 78, 386, bot_y1 - 24)
            ns_box = (400, bot_y0 + 78, 730, bot_y1 - 24)
            draw.rounded_rectangle([int(v * S) for v in cf_box], radius=int(6 * S), fill=THEME["surface"], outline=THEME["ink"], width=int(1.4 * S))
            draw.rounded_rectangle([int(v * S) for v in ns_box], radius=int(6 * S), fill=THEME["surface"], outline=THEME["ink"], width=int(2.2 * S))

            draw_pill(draw, 70, bot_y0 + 88, "@cf/cloudflare/clairic", THEME["null"], THEME["ink"], THEME["ink"], font_mono_bold(9.2))
            cf_lines = [
                ("nodes: 4  |  rate: $3.50", THEME["muted_bg"]),
                ("subtotal: $14.00 (unverified)", THEME["muted_bg"]),
                ("constraint: NOT SUPPORTED", THEME["null"]),
                ("verified: N/A (blind to math)", THEME["null"]),
                ("promo_rebate: bare p_null only", THEME["null"]),
            ]
            for cli, (cl, bg_c) in enumerate(cf_lines):
                cy = bot_y0 + 120 + cli * 28
                draw_pill(draw, 70, cy, cl, bg_c, THEME["ink"], THEME["ink"], font_mono_bold(8.8), pad_x=7, pad_y=2.5)

            draw_pill(draw, 412, bot_y0 + 88, "nullspan (Audits the Math)", THEME["list"], THEME["ink"], THEME["ink"], font_mono_bold(9.2))
            ns_lines = [
                ("nodes: 4 [55,56] · rate: $3.50", THEME["span"]),
                ("subtotal: $14.00 [SPAN_NUMBER]", THEME["span"]),
                ("constraint: subtotal==nodes*rate", THEME["list"]),
                ("verified: true · invariants_passed", THEME["list"]),
                ("promo_rebate: null (ignores $0.00)", THEME["null"]),
            ]
            for nli, (nl, bg_c) in enumerate(ns_lines):
                ny = bot_y0 + 120 + nli * 28
                draw_pill(draw, 412, ny, nl, bg_c, THEME["ink"], THEME["ink"], font_mono_bold(8.8), pad_x=7, pad_y=2.5)

        else:
            # Scene 2: JevBench 4-Axis Leaderboard Bars (using only #FFE484, #C8F0D4, #F5D0CE)
            bars = [
                ("nullspan-Hybrid (Fast + ONNX Int8)", 97.7, 98.7, "$0.0003/1k · 0.005ms", THEME["list"]),
                ("nullspan-67M (PyTorch FP32 · Open)", 83.3, 99.2, "$0.0015/1k · 5.58ms", THEME["span"]),
                ("nullspan-67M-Int8 (Browser WASM)", 83.0, 98.7, "$0.0008/1k · 12.3ms", THEME["span"]),
                ("H2O-Lightning-4B (JevBench #1 GPU)", 72.5, 75.0, "$0.0211/1k · 29.3ms", THEME["surface"]),
                ("Jev 1.13.0 / Clef-Class Cloud API", 71.5, 77.1, "$0.0323/1k · 239ms", THEME["null"]),
            ]
            for bi, (bname, comp, cap, meta_s, bcol) in enumerate(bars):
                by = bot_y0 + 80 + bi * 36
                draw.text((int(60 * S), int((by + 6) * S)), bname, font=font_mono_bold(9.4), fill=THEME["ink"])
                bar_x0, bar_x1 = 308, 548
                draw.rounded_rectangle(
                    [int(bar_x0 * S), int((by + 2) * S), int(bar_x1 * S), int((by + 25) * S)],
                    radius=int(4 * S),
                    fill=THEME["surface"],
                    outline=THEME["ink"],
                    width=int(1.3 * S),
                )
                fill_w = int((bar_x1 - bar_x0) * (comp / 100.0))
                draw.rounded_rectangle(
                    [int(bar_x0 * S), int((by + 2) * S), int((bar_x0 + fill_w) * S), int((by + 25) * S)],
                    radius=int(4 * S),
                    fill=bcol,
                    outline=THEME["ink"],
                    width=int(1.4 * S),
                )
                draw.text((int((bar_x0 + 8) * S), int((by + 6) * S)), f"Comp {comp:.1f}  ·  Cap {cap:.1f}", font=font_mono_bold(8.8), fill=THEME["ink"])
                draw.text((int(556 * S), int((by + 6) * S)), meta_s, font=font_mono_bold(9.0), fill=THEME["ink_muted"])

        # Bottom Right Card (x = 780..1168, y = 444..734): JevBench Metrics & Unified Schema
        draw_studio_card(
            draw,
            (780, bot_y0, 1168, bot_y1),
            header_text="JEVBENCH METRICS & UNIFIED SCHEMA",
            header_right="MIT OPEN",
        )

        for ki, (kval, klabel, kcol) in enumerate(kpis):
            kx = 794 + (ki % 2) * 182
            ky = bot_y0 + 38 + (ki // 2) * 60
            draw.rounded_rectangle(
                [int(kx * S), int(ky * S), int((kx + 174) * S), int((ky + 52) * S)],
                radius=int(6 * S),
                fill=THEME["muted_bg"],
                outline=THEME["ink"],
                width=int(1.5 * S),
            )
            draw_pill(draw, kx + 8, ky + 6, kval, kcol, THEME["ink"], THEME["ink"], font_sans_bold(14.5), pad_x=7, pad_y=2, radius=4)
            draw.text((int((kx + 10) * S), int((ky + 34) * S)), klabel, font=font_mono_bold(8.8), fill=THEME["ink_muted"])

        code_box = (794, bot_y0 + 164, 1154, bot_y1 - 14)
        draw.rounded_rectangle(
            [int(v * S) for v in code_box],
            radius=int(6 * S),
            fill=THEME["muted_bg"],
            outline=THEME["ink"],
            width=int(1.6 * S),
        )
        for li, cline in enumerate(code_lines):
            draw.text((int(806 * S), int((bot_y0 + 172 + li * 21) * S)), cline, font=font_mono_bold(9.2), fill=THEME["ink"])

        out_frame = img.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
        out_frame.save(frames_dir / f"frame_{idx:03d}.png")

    Image.open(frames_dir / "frame_005.png").save(preview_png)
    Image.open(frames_dir / "frame_025.png").save(root / "docs" / "nullspan-vs-clairic-s2.png")
    Image.open(frames_dir / "frame_045.png").save(root / "docs" / "nullspan-vs-clairic-s3.png")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "15",
            "-i",
            str(frames_dir / "frame_%03d.png"),
            "-vf",
            "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=full[p];[s1][p]paletteuse=dither=sierra2_4a",
            "-loop",
            "0",
            str(out_gif),
        ],
        check=True,
    )
    space_gif.write_bytes(out_gif.read_bytes())
    print(f"Rendered {out_gif} ({out_gif.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
