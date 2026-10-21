#!/usr/bin/env python3
"""
Render a Mobile-First Portrait (1080x1350, 4:5 aspect ratio, 2x supersampled at 2160x2700)
animated social media GIF comparing Cloudflare `@cf/cloudflare/clairic + Clef` vs `nullspan`.

Designed specifically for mobile phone feeds (X/Twitter, LinkedIn, Bluesky):
- Portrait 1080x1350 layout matching `docs/nullspan-social-mobile.png`
- Large 20px-52px typography (zero tiny text)
- Calm 12.0-second loop (120 frames @ 10 fps):
  - Each of the 4 cards in the 2x2 grid stays active for 3.0 full seconds
  - Each of the 3 Head-to-Head X-Ray scenes stays open for 4.0 full seconds with a progress bar
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

S = 2
OUT_W, OUT_H = 1080, 1350
W, H = OUT_W * S, OUT_H * S

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


THEME = {
    "bg": (251, 250, 247),           # --bg: #FBFAF7
    "dot": (226, 222, 212),          # subtle dot grid
    "surface": (255, 255, 255),      # --surface: #FFFFFF
    "muted_bg": (243, 240, 232),     # --muted-bg: #F3F0E8
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
    radius: float = 10,
    header_h: float = 36,
) -> None:
    x0, y0, x1, y1 = [int(v * S) for v in box]
    r = int(radius * S)
    so = int((6.0 if active else 3.5) * S)
    bw = int((3.0 if active else 2.0) * S)

    draw.rounded_rectangle([x0 + so, y0 + so, x1 + so, y1 + so], radius=r, fill=THEME["ink"])
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=THEME["surface"], outline=THEME["ink"], width=bw)

    if header_text:
        hh = int(header_h * S)
        draw.rounded_rectangle([x0, y0, x1, y0 + hh], radius=r, fill=THEME["ink"])
        draw.rectangle([x0, y0 + hh - r, x1, y0 + hh], fill=THEME["ink"])
        hf = font_mono_bold(13.0)
        draw.text((x0 + int(16 * S), y0 + int(9 * S)), header_text, font=hf, fill=THEME["bg"])
        if header_right:
            rb = draw.textbbox((0, 0), header_right, font=hf)
            rw = rb[2] - rb[0]
            draw.text((x1 - int(16 * S) - rw, y0 + int(9 * S)), header_right, font=hf, fill=THEME["span"])


def draw_pill(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int] = THEME["ink"],
    border_color: tuple[int, int, int] = THEME["ink"],
    font: ImageFont.FreeTypeFont | None = None,
    pad_x: float = 12,
    pad_y: float = 6,
    radius: float = 6,
    shadow: bool = False,
) -> float:
    if font is None:
        font = font_mono_bold(13.5)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px, py = int(pad_x * S), int(pad_y * S)
    rx0, ry0 = int(x * S), int(y * S)
    rx1, ry1 = rx0 + tw + px * 2, ry0 + th + py * 2
    if shadow:
        so = int(3.0 * S)
        draw.rounded_rectangle([rx0 + so, ry0 + so, rx1 + so, ry1 + so], radius=int(radius * S), fill=THEME["ink"])
    draw.rounded_rectangle(
        [rx0, ry0, rx1, ry1],
        radius=int(radius * S),
        fill=bg,
        outline=border_color,
        width=max(1, int(2.0 * S)),
    )
    draw.text((rx0 + px - bbox[0], ry0 + py - bbox[1]), text, font=font, fill=fg)
    return (rx1 - rx0) / S


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_gif = root / "docs" / "nullspan-vs-clairic.gif"
    space_gif = root / "space" / "nullspan-vs-clairic.gif"
    preview_png = root / "docs" / "nullspan-vs-clairic-preview.png"
    frames_dir = Path("/tmp/nullspan_vs_clairic_mobile_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_f in frames_dir.glob("frame_*.png"):
        old_f.unlink()

    stages = [
        {
            "tag": "1 · ARCHITECTURE",
            "badge": "1 PASS vs 2 RPCs",
            "badge_bg": THEME["span"],
            "rows": [
                ("Clairic + Clef", "2 Cloud Models", THEME["null"]),
                ("nullspan 67M", "1 Unified Pass", THEME["list"]),
                ("op_head Router", "10 System1 Ops", THEME["span"]),
            ],
        },
        {
            "tag": "2 · RUNTIME & OPENNESS",
            "badge": "WASM + ZERO-DEP",
            "badge_bg": THEME["list"],
            "rows": [
                ("Workers AI", "Closed Preview", THEME["null"]),
                ("Browser WASM", "64.3 MB Int8", THEME["span"]),
                ("Zero-Dep Fast", "0 MB · 5 µs", THEME["list"]),
            ],
        },
        {
            "tag": "3 · MATH & INVARIANTS",
            "badge": "AUDITED MATH",
            "badge_bg": THEME["list"],
            "rows": [
                ("Clairic Math", "No Cross-Check", THEME["null"]),
                ("constraint", "nodes * rate", THEME["list"]),
                ("verified Flag", "true (100% EM)", THEME["list"]),
            ],
        },
        {
            "tag": "4 · ABSTENTION GATE",
            "badge": "P_NULL >= 0.5",
            "badge_bg": THEME["null"],
            "rows": [
                ("Clairic Null", "Bare p_null", THEME["null"]),
                ("null_head", "AUROC 0.9998", THEME["list"]),
                ("Reason Tag", "unassigned_cve", THEME["span"]),
            ],
        },
    ]

    # 120 frames @ 10 fps = 12.0s calm loop (3.0s per card, 4.0s per scene)
    total_frames = 120
    fps = 10

    for idx in range(total_frames):
        active_stage = idx // 30       # 30 frames = 3.0 seconds per stage card
        scene_idx = idx // 40          # 40 frames = 4.0 seconds per bottom X-Ray scene
        scene_progress = (idx % 40) / 39.0

        img = Image.new("RGB", (W, H), THEME["bg"])
        draw = ImageDraw.Draw(img)

        # Dot grid
        step = int(20 * S)
        r_dot = int(1.1 * S)
        for gx in range(step, W, step):
            for gy in range(step, H, step):
                draw.ellipse([gx - r_dot, gy - r_dot, gx + r_dot, gy + r_dot], fill=THEME["dot"])

        # Top 8px solid #111111 border
        draw.rectangle([0, 0, W, int(8 * S)], fill=THEME["ink"])

        # Top Studio Bar (y = 24..72)
        cur_x = 40.0
        w_brand = draw_pill(
            draw,
            cur_x,
            24,
            "nullspan",
            THEME["ink"],
            THEME["bg"],
            THEME["ink"],
            font_mono_bold(19.0),
            pad_x=14,
            pad_y=6,
            radius=6,
        )
        cur_x += w_brand + 12
        for b_txt, b_col in [
            ("[START, END] SPAN", THEME["span"]),
            ("CALIBRATED P_NULL", THEME["null"]),
            ("VERIFIED MATH", THEME["list"]),
        ]:
            wb = draw_pill(draw, cur_x, 27, b_txt, b_col, THEME["ink"], THEME["ink"], font_mono_bold(13.0), pad_x=10, pad_y=5, radius=5)
            cur_x += wb + 10

        draw_pill(
            draw,
            cur_x + 4,
            27,
            "67M · WASM · 0-DEP",
            THEME["surface"],
            THEME["ink"],
            THEME["ink"],
            font_mono_bold(13.0),
            pad_x=10,
            pad_y=5,
            radius=5,
        )

        draw.line([(int(40 * S), int(76 * S)), (int(1040 * S), int(76 * S))], fill=THEME["ink"], width=int(2.0 * S))

        # Hero Headline (y = 92..210) — Huge 46px mobile-first text!
        draw.text((int(40 * S), int(96 * S)), "Unify", font=font_sans_bold(46), fill=THEME["ink"])
        w_c1 = draw_pill(draw, 172, 91, "[Clairic Extraction]", THEME["span"], THEME["ink"], THEME["ink"], font_sans_bold(36), pad_x=14, pad_y=4, radius=8, shadow=True)
        x_plus = 172 + w_c1 + 16
        draw.text((int(x_plus * S), int(96 * S)), "+", font=font_sans_bold(46), fill=THEME["ink"])
        draw_pill(draw, x_plus + 40, 91, "[Clef Decisions]", THEME["null"], THEME["ink"], THEME["ink"], font_sans_bold(36), pad_x=14, pad_y=4, radius=8, shadow=True)

        draw.text((int(40 * S), int(162 * S)), "in", font=font_sans_bold(46), fill=THEME["ink"])
        w_p1 = draw_pill(draw, 96, 157, "1 Open WASM Pass", THEME["list"], THEME["ink"], THEME["ink"], font_sans_bold(36), pad_x=14, pad_y=4, radius=8, shadow=True)
        draw.text((int((96 + w_p1 + 16) * S), int(162 * S)), "with 0 Output Tokens.", font=font_sans_bold(42), fill=THEME["ink"])

        draw.text(
            (int(40 * S), int(224 * S)),
            "Cloudflare splits System One across 2 closed Workers AI models. nullspan unifies both + audits math:",
            font=font_sans_med(19.5),
            fill=THEME["ink_muted"],
        )

        # 2x2 Comparison Grid (y = 264..744) — Each card is 486px wide x 226px tall!
        grid_coords = [
            (40, 264, 526, 492),
            (554, 264, 1040, 492),
            (40, 514, 526, 742),
            (554, 514, 1040, 742),
        ]
        for si, (cx0, cy0, cx1, cy1) in enumerate(grid_coords):
            st = stages[si]
            is_act = si == active_stage
            draw_studio_card(
                draw,
                (cx0, cy0, cx1, cy1),
                header_text=st["tag"],
                header_right="● ACTIVE" if is_act else st["badge"],
                active=is_act,
                header_h=36,
            )

            for ri, (rleft, rright, pill_col) in enumerate(st["rows"]):
                ry = cy0 + 48 + ri * 56
                is_cf_row = ri == 0
                row_hi = is_act and (ri > 0)
                row_bg = THEME["muted_bg"] if is_cf_row else (THEME["span"] if (is_act and ri == 1) else THEME["surface"])
                draw.rounded_rectangle(
                    [int((cx0 + 14) * S), int(ry * S), int((cx1 - 14) * S), int((ry + 46) * S)],
                    radius=int(7 * S),
                    fill=row_bg,
                    outline=THEME["ink"],
                    width=int((2.4 if row_hi else 1.5) * S),
                )
                draw.text(
                    (int((cx0 + 24) * S), int((ry + 12) * S)),
                    rleft,
                    font=font_mono_bold(15.0),
                    fill=THEME["ink_muted"] if is_cf_row else THEME["ink"],
                )
                rf = font_mono_bold(13.5)
                rb = draw.textbbox((0, 0), rright, font=rf)
                rw_px = (rb[2] - rb[0]) / S + 20
                draw_pill(
                    draw,
                    cx1 - 22 - rw_px,
                    ry + 7,
                    rright,
                    pill_col,
                    THEME["ink"],
                    THEME["ink"],
                    rf,
                    pad_x=9,
                    pad_y=4,
                    radius=5,
                )

        # Head-to-Head X-Ray Card (y = 768..1176) — Full 1000px width, 4.0s per scene!
        x_y0, x_y1 = 768, 1176
        draw_studio_card(
            draw,
            (40, x_y0, 1040, x_y1),
            header_text="HEAD-TO-HEAD X-RAY  ·  @cf/cloudflare/clairic + Clef  vs  nullspan",
            header_right=f"SCENE {scene_idx + 1}/3",
            active=True,
            header_h=38,
        )

        # 3 Large Scene Tabs with live progress bar inside the active tab
        scene_tabs = [
            ("1 · UNIFIED PASS (INC-40928)", THEME["span"]),
            ("2 · AUDITED MATH (#88310)", THEME["list"]),
            ("3 · JEVBENCH LEADERBOARD", THEME["null"]),
        ]
        tab_w = 312
        for tab_i, (tab_lbl, tab_col) in enumerate(scene_tabs):
            tx0 = 58 + tab_i * (tab_w + 14)
            ty0 = x_y0 + 50
            tx1 = tx0 + tab_w
            ty1 = ty0 + 38
            is_tab = tab_i == scene_idx
            draw.rounded_rectangle(
                [int(tx0 * S), int(ty0 * S), int(tx1 * S), int(ty1 * S)],
                radius=int(6 * S),
                fill=tab_col if is_tab else THEME["muted_bg"],
                outline=THEME["ink"],
                width=int((2.4 if is_tab else 1.5) * S),
            )
            if is_tab:
                # Progress bar along bottom of active tab
                prog_w = int((tab_w - 8) * scene_progress * S)
                draw.rectangle(
                    [int((tx0 + 4) * S), int((ty1 - 5) * S), int((tx0 + 4) * S) + prog_w, int((ty1 - 2) * S)],
                    fill=THEME["ink"],
                )
            tf = font_mono_bold(13.0)
            tb = draw.textbbox((0, 0), tab_lbl, font=tf)
            tw_t = tb[2] - tb[0]
            draw.text((int(tx0 * S) + (int(tab_w * S) - tw_t) // 2, int((ty0 + 8) * S)), tab_lbl, font=tf, fill=THEME["ink"])

        # Inner Stage Sheet (y = x_y0 + 100 .. x_y1 - 16)
        sheet = (58, x_y0 + 100, 1022, x_y1 - 16)
        draw.rounded_rectangle(
            [int(v * S) for v in sheet],
            radius=int(8 * S),
            fill=THEME["muted_bg"],
            outline=THEME["ink"],
            width=int(1.8 * S),
        )

        if scene_idx == 0:
            # Scene 1: 2 Cloud Calls vs 1 Local Pass (Large 16px-18px fonts)
            cf_box = (74, x_y0 + 114, 1006, x_y0 + 224)
            draw.rounded_rectangle([int(v * S) for v in cf_box], radius=int(7 * S), fill=THEME["surface"], outline=THEME["ink"], width=int(1.6 * S))
            draw_pill(draw, 88, x_y0 + 124, "CLOUDFLARE: @cf/cloudflare/clairic + Clef", THEME["null"], THEME["ink"], THEME["ink"], font_mono_bold(13.5), pad_x=10, pad_y=4)
            draw.text((int(520 * S), int((x_y0 + 129) * S)), "2 Separate Closed Cloud Calls · No Browser Mode", font=font_sans_bold(17.0), fill=THEME["ink_muted"])
            draw.text((int(88 * S), int((x_y0 + 166) * S)), "Call 1 (Clairic):  incident_id='INC-40928',  total_credit=1550.75,  cve_id=null (bare p_null)", font=font_mono_bold(14.5), fill=THEME["ink_muted"])
            draw.text((int(88 * S), int((x_y0 + 194) * S)), "Call 2 (Clef):     severity='sev-1' (0.98),  data_loss=false  [Requires 2nd Model RPC]", font=font_mono_bold(14.5), fill=THEME["ink_muted"])

            ns_box = (74, x_y0 + 236, 1006, x_y1 - 28)
            draw.rounded_rectangle([int(v * S) for v in ns_box], radius=int(7 * S), fill=THEME["surface"], outline=THEME["ink"], width=int(2.6 * S))
            draw_pill(draw, 88, x_y0 + 246, "NULLSPAN: 1 Unified Forward Pass", THEME["list"], THEME["ink"], THEME["ink"], font_mono_bold(13.5), pad_x=10, pad_y=4)
            draw.text((int(465 * S), int((x_y0 + 251) * S)), "All 10 Ops in 1 Pass · Runs in Browser WASM & Python", font=font_sans_bold(17.0), fill=THEME["ink"])

            # Highlighted badges row inside nullspan box
            draw_pill(draw, 88, x_y0 + 288, "SPAN: 'INC-40928' [9,18] (H=0.03b)", THEME["span"], THEME["ink"], THEME["ink"], font_mono_bold(13.5), pad_x=10, pad_y=4)
            draw_pill(draw, 435, x_y0 + 288, "ARITH_SUM: $1,550.75 ✓", THEME["list"], THEME["ink"], THEME["ink"], font_mono_bold(13.5), pad_x=10, pad_y=4)
            draw_pill(draw, 695, x_y0 + 288, "ENUM: 'sev-1' (0.99)", THEME["list"], THEME["ink"], THEME["ink"], font_mono_bold(13.5), pad_x=10, pad_y=4)

            draw_pill(draw, 88, x_y0 + 332, "NULL_HEAD: cve_id = null  (p_null: 0.994 · reason: 'unassigned_placeholder' · 0% FPHR)", THEME["null"], THEME["ink"], THEME["ink"], font_mono_bold(13.5), pad_x=10, pad_y=4)

        elif scene_idx == 1:
            # Scene 2: Cross-Field Algebraic Invariant Verification (Side by Side, Large 16px text)
            cf_box = (74, x_y0 + 114, 528, x_y1 - 28)
            ns_box = (552, x_y0 + 114, 1006, x_y1 - 28)
            draw.rounded_rectangle([int(v * S) for v in cf_box], radius=int(7 * S), fill=THEME["surface"], outline=THEME["ink"], width=int(1.6 * S))
            draw.rounded_rectangle([int(v * S) for v in ns_box], radius=int(7 * S), fill=THEME["surface"], outline=THEME["ink"], width=int(2.6 * S))

            draw_pill(draw, 90, x_y0 + 126, "@cf/cloudflare/clairic", THEME["null"], THEME["ink"], THEME["ink"], font_mono_bold(14.5), pad_x=10, pad_y=5)
            cf_rows = [
                ("nodes: 4  ·  rate: $3.50", THEME["muted_bg"]),
                ("subtotal: $14.00 (unverified)", THEME["muted_bg"]),
                ("✗ constraint: NOT SUPPORTED", THEME["null"]),
                ("✗ verified: N/A (blind to math)", THEME["null"]),
                ("✗ promo_rebate: bare p_null", THEME["null"]),
            ]
            for cri, (ctxt, cbg) in enumerate(cf_rows):
                draw_pill(draw, 90, x_y0 + 172 + cri * 40, ctxt, cbg, THEME["ink"], THEME["ink"], font_mono_bold(14.0), pad_x=10, pad_y=4)

            draw_pill(draw, 568, x_y0 + 126, "nullspan (Audits the Math)", THEME["list"], THEME["ink"], THEME["ink"], font_mono_bold(14.5), pad_x=10, pad_y=5)
            ns_rows = [
                ("nodes: 4 [55,56] · rate: $3.50", THEME["span"]),
                ("subtotal: $14.00 [SPAN_NUMBER]", THEME["span"]),
                ("✓ constraint: subtotal==nodes*rate", THEME["list"]),
                ("✓ verified: true (invariants_passed)", THEME["list"]),
                ("✓ promo_rebate: null (ignores $0.00)", THEME["null"]),
            ]
            for nri, (ntxt, nbg) in enumerate(ns_rows):
                draw_pill(draw, 568, x_y0 + 172 + nri * 40, ntxt, nbg, THEME["ink"], THEME["ink"], font_mono_bold(14.0), pad_x=10, pad_y=4)

        else:
            # Scene 3: JevBench 4-Axis Leaderboard Bars (Big, Mobile-Legible 16px-18px Bars)
            bars = [
                ("nullspan-Hybrid (Fast + WASM)", 97.7, 98.7, "$0.0003/1k · 5 µs", THEME["list"]),
                ("nullspan-67M (PyTorch FP32)", 83.3, 99.2, "$0.0015/1k · 5.6 ms", THEME["span"]),
                ("nullspan-67M-Int8 (Browser WASM)", 83.0, 98.7, "$0.0008/1k · 64 MB", THEME["span"]),
                ("H2O-Lightning-4B (JevBench #1)", 72.5, 75.0, "$0.0211/1k · 4B GPU", THEME["surface"]),
                ("Jev 1.13.0 / Clef-Class API", 71.5, 77.1, "$0.0323/1k · 239 ms", THEME["null"]),
            ]
            for bi, (bname, comp, cap, meta_s, bcol) in enumerate(bars):
                by = x_y0 + 118 + bi * 52
                draw.text((int(76 * S), int((by + 8) * S)), bname, font=font_mono_bold(15.0), fill=THEME["ink"])
                bar_x0, bar_x1 = 420, 770
                draw.rounded_rectangle(
                    [int(bar_x0 * S), int((by + 2) * S), int(bar_x1 * S), int((by + 38) * S)],
                    radius=int(6 * S),
                    fill=THEME["surface"],
                    outline=THEME["ink"],
                    width=int(1.8 * S),
                )
                fill_w = int((bar_x1 - bar_x0) * (comp / 100.0))
                draw.rounded_rectangle(
                    [int(bar_x0 * S), int((by + 2) * S), int((bar_x0 + fill_w) * S), int((by + 38) * S)],
                    radius=int(6 * S),
                    fill=bcol,
                    outline=THEME["ink"],
                    width=int(2.0 * S),
                )
                draw.text((int((bar_x0 + 14) * S), int((by + 9) * S)), f"Composite {comp:.1f}  ·  Cap {cap:.1f}", font=font_mono_bold(14.0), fill=THEME["ink"])
                draw.text((int(784 * S), int((by + 9) * S)), meta_s, font=font_mono_bold(14.0), fill=THEME["ink_muted"])

        # Bottom 3 KPI Cards (y = 1196..1278) matching docs/nullspan-social-mobile.png
        kpi_boxes = [
            (40, 1196, 360, 1276, "Capability Score", "50% Intel + 50% Calib", "98.7 vs 77.1", THEME["span"]),
            (380, 1196, 700, 1276, "In-Browser WASM", "Int8 ONNX SIMD", "64.3 MB", THEME["list"]),
            (720, 1196, 1040, 1276, "100x Lower Cost", "Zero Output Tokens", "$0.0003/1k", THEME["null"]),
        ]
        for kx0, ky0, kx1, ky1, ktitle, ksub, kval, kcol in kpi_boxes:
            draw_studio_card(draw, (kx0, ky0, kx1, ky1), active=False, radius=8)
            draw.text((int((kx0 + 16) * S), int((ky0 + 16) * S)), ktitle, font=font_sans_bold(18.0), fill=THEME["ink"])
            draw.text((int((kx0 + 16) * S), int((ky0 + 44) * S)), ksub, font=font_mono_reg(12.5), fill=THEME["ink_muted"])
            vf = font_mono_bold(16.5)
            vb = draw.textbbox((0, 0), kval, font=vf)
            vw_px = (vb[2] - vb[0]) / S + 20
            draw_pill(draw, kx1 - 14 - vw_px, ky0 + 20, kval, kcol, THEME["ink"], THEME["ink"], vf, pad_x=10, pad_y=6, radius=6)

        # Bottom Footer Strip (y = 1294..1334)
        draw_pill(
            draw,
            40,
            1294,
            "npm i nullspan  ·  pip install nullspan",
            THEME["ink"],
            THEME["bg"],
            THEME["ink"],
            font_mono_bold(14.5),
            pad_x=16,
            pad_y=6,
            radius=6,
        )
        draw.text(
            (int(490 * S), int(1301 * S)),
            "hemanth.github.io/nullspan/bench.html  ·  hf.co/gnumanth/nullspan",
            font=font_mono_bold(14.0),
            fill=THEME["ink"],
        )

        out_frame = img.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
        out_frame.save(frames_dir / f"frame_{idx:03d}.png")

    Image.open(frames_dir / "frame_010.png").save(preview_png)
    Image.open(frames_dir / "frame_050.png").save(root / "docs" / "nullspan-vs-clairic-s2.png")
    Image.open(frames_dir / "frame_090.png").save(root / "docs" / "nullspan-vs-clairic-s3.png")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
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
    print(f"Rendered {out_gif} ({out_gif.stat().st_size / 1024:.1f} KB, {total_frames / fps:.1f}s loop)")


if __name__ == "__main__":
    main()
