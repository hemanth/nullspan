#!/usr/bin/env python3
"""
Render a 2x supersampled (2400x1520 -> 1200x760, 15fps, 60 frames) social media GIF
comparing Cloudflare `@cf/cloudflare/clairic + Clef` vs `nullspan` using the
`project-social-gif` skill layout and `nullspan` warm-paper + highlighter/mint/rose theme.
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


# Exact nullspan CSS palette from docs/index.html
THEME = {
    "bg": (251, 250, 247),          # --bg: #FBFAF7
    "dot": (224, 219, 208),         # subtle warm dot grid
    "ink": (17, 17, 17),            # --ink: #111111
    "ink_muted": (82, 80, 74),      # --ink-muted: #52504A
    "surface": (255, 255, 255),     # --surface: #FFFFFF
    "muted_bg": (243, 240, 232),    # --muted-bg: #F3F0E8
    "border": (17, 17, 17),         # --border: #111111
    "span": (255, 228, 132),        # --accent-span: #FFE484 (Amber Highlighter)
    "list": (200, 240, 212),        # --accent-list: #C8F0D4 (Verified Mint)
    "null": (245, 208, 206),        # --accent-null: #F5D0CE (Abstention Rose)
    "quant": (233, 226, 255),       # Soft Lavender (Quantiles)
    "cards": [
        (255, 238, 170),  # Stage 1: Warm Span Amber
        (214, 245, 223),  # Stage 2: Verified Mint
        (236, 230, 255),  # Stage 3: Invariant Lavender
        (250, 220, 218),  # Stage 4: Calibrated Null Rose
    ],
    "accents": [
        (209, 119, 6),    # Amber accent
        (21, 128, 61),    # Emerald accent
        (91, 62, 214),    # Violet accent
        (190, 42, 42),    # Crimson accent
    ],
}


def draw_shadow_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    fill: tuple[int, int, int],
    border_color: tuple[int, int, int],
    shadow_color: tuple[int, int, int],
    radius: float = 12,
    active: bool = False,
) -> None:
    x0, y0, x1, y1 = [int(v * S) for v in box]
    r = int(radius * S)
    so = int((5.5 if active else 3.5) * S)
    bw = int((2.8 if active else 2.0) * S)
    draw.rounded_rectangle([x0 + so, y0 + so, x1 + so, y1 + so], radius=r, fill=shadow_color)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill, outline=border_color, width=bw)


def draw_pill(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    border_color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | None = None,
    pad_x: float = 10,
    pad_y: float = 4,
    radius: float = 6,
) -> float:
    if font is None:
        font = font_mono_bold(10.5)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px, py = int(pad_x * S), int(pad_y * S)
    rx0, ry0 = int(x * S), int(y * S)
    rx1, ry1 = rx0 + tw + px * 2, ry0 + th + py * 2
    draw.rounded_rectangle(
        [rx0, ry0, rx1, ry1],
        radius=int(radius * S),
        fill=bg,
        outline=border_color,
        width=max(1, int(1.5 * S)),
    )
    draw.text((rx0 + px - bbox[0], ry0 + py - bbox[1]), text, font=font, fill=fg)
    return (rx1 - rx0) / S


def draw_arrow(draw: ImageDraw.ImageDraw, x0: float, x1: float, y: float, active: bool) -> None:
    col = THEME["ink"] if active else (160, 155, 145)
    w = int((2.8 if active else 1.8) * S)
    sx0, sx1, sy = int(x0 * S), int(x1 * S), int(y * S)
    draw.line([(sx0, sy), (sx1, sy)], fill=col, width=w)
    ah = int(5 * S)
    draw.polygon([(sx1, sy), (sx1 - ah, sy - ah), (sx1 - ah, sy + ah)], fill=col)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_gif = root / "docs" / "nullspan-vs-clairic.gif"
    space_gif = root / "space" / "nullspan-vs-clairic.gif"
    preview_10 = root / "docs" / "nullspan-vs-clairic-preview.png"
    frames_dir = Path("/tmp/nullspan_vs_clairic_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        {
            "tag": "1 · ARCHITECTURE",
            "title": "1 Pass vs 2 Models",
            "rows": [
                ("Clairic + Clef", "2 Split RPCs"),
                ("nullspan 67M", "1 Unified Pass"),
                ("op_head Router", "10 System1 Ops"),
                ("Output Tokens", "0 Generated"),
            ],
        },
        {
            "tag": "2 · RUNTIME & WEIGHTS",
            "title": "Local WASM vs Cloud",
            "rows": [
                ("Workers AI", "Closed Preview"),
                ("Browser WASM", "64.3 MB Int8"),
                ("PyTorch FP32", "Open MIT (HF)"),
                ("Zero-Dep Fast", "0 MB · 5 µs"),
            ],
        },
        {
            "tag": "3 · MATH & INVARIANTS",
            "title": "Verified Equations",
            "rows": [
                ("Clairic Math", "No Cross-Check"),
                ("constraint", "nodes * rate"),
                ("verified Flag", "true (100% EM)"),
                ("quantile_head", "p10<=p50<=p90"),
            ],
        },
        {
            "tag": "4 · CALIBRATED NULL",
            "title": "Reason + Entropy",
            "rows": [
                ("Clairic Null", "Bare p_null"),
                ("null_head", "AUROC 0.9998"),
                ("Reason Tag", "unassigned_cve"),
                ("Span Entropy", "0.04 bits (H)"),
            ],
        },
    ]

    kpis = [
        ("98.7 vs 77.1", "Capability (50I:50C)"),
        ("97.7 vs 71.5", "Composite Score"),
        ("64.3 MB ONNX", "Browser WASM SIMD"),
        ("$0.0003 / 1k", "100x Cheaper than API"),
    ]

    code_lines = [
        "import { extract } from 'nullspan'; // 1 pass",
        "const out = extract(doc, { // Clairic + Clef:",
        "  subtotal: { type:'number', constraint:'...' },",
        "  severity: { type:'enum', options:['sev-1'] }",
        "}); // => invariants_passed: true, 0 tokens",
    ]

    total_frames = 60
    for idx in range(total_frames):
        t = idx / float(total_frames)
        active_stage = int((t * 4.0) % 4.0)
        scene_idx = idx // 20  # 0 (0..19), 1 (20..39), 2 (40..59)

        img = Image.new("RGB", (W, H), THEME["bg"])
        draw = ImageDraw.Draw(img)

        # Dot grid
        step = int(24 * S)
        r_dot = int(1.2 * S)
        for gx in range(step, W, step):
            for gy in range(step, H, step):
                draw.ellipse([gx - r_dot, gy - r_dot, gx + r_dot, gy + r_dot], fill=THEME["dot"])

        # Top Header (y = 20..120)
        draw_pill(
            draw,
            32,
            20,
            "SYSTEM ONE COMPARISON · @cf/cloudflare/clairic + Clef  vs  nullspan",
            THEME["span"],
            THEME["ink"],
            THEME["border"],
            font_mono_bold(10.5),
        )
        draw_pill(draw, 764, 20, "$ npm i nullspan", THEME["list"], THEME["ink"], THEME["border"], font_mono_bold(11.0))
        draw_pill(draw, 954, 20, "$ pip install nullspan", THEME["quant"], THEME["ink"], THEME["border"], font_mono_bold(11.0))

        draw.text((int(32 * S), int(54 * S)), "nullspan vs Clairic + Clef", font=font_sans_bold(36), fill=THEME["ink"])
        draw.text(
            (int(488 * S), int(62 * S)),
            "1 Unified Open Model (WASM ONNX + Zero-Dep) vs 2 Closed Cloud Primitives",
            font=font_sans_med(17.5),
            fill=THEME["ink_muted"],
        )

        stage_banners = [
            "ACTIVE STAGE 1/4 · UNIFYING CLAIRIC EXTRACTION + CLEF DECISIONS IN ONE FORWARD PASS",
            "ACTIVE STAGE 2/4 · 100% OPEN WEIGHTS (64.3 MB BROWSER WASM ONNX + 0 MB ZERO-DEP ENGINE)",
            "ACTIVE STAGE 3/4 · CROSS-FIELD ALGEBRAIC INVARIANTS (subtotal == nodes * hourly_rate)",
            "ACTIVE STAGE 4/4 · CALIBRATED p_null ABSTENTION REASON + SHANNON SPAN ENTROPY",
        ]
        draw_pill(
            draw,
            488,
            91,
            stage_banners[active_stage],
            THEME["surface"],
            THEME["ink"],
            THEME["border"],
            font_mono_bold(9.5),
            pad_x=8,
            pad_y=3,
        )

        # 4-Stage Pipeline Row (y = 132..422)
        card_y0, card_y1 = 132, 422
        xs = [(32, 290), (324, 582), (616, 874), (908, 1168)]
        arrows = [(293, 321), (585, 613), (877, 905)]
        for ai, (ax0, ax1) in enumerate(arrows):
            draw_arrow(draw, ax0, ax1, 277, active=(ai == active_stage))

        for si, (cx0, cx1) in enumerate(xs):
            st = stages[si]
            is_act = si == active_stage
            draw_shadow_card(
                draw,
                (cx0, card_y0, cx1, card_y1),
                fill=THEME["cards"][si],
                border_color=THEME["border"],
                shadow_color=THEME["ink"] if is_act else (120, 116, 108),
                active=is_act,
            )
            draw_pill(
                draw,
                cx0 + 14,
                card_y0 + 14,
                st["tag"],
                THEME["ink"] if is_act else THEME["surface"],
                THEME["bg"] if is_act else THEME["ink"],
                THEME["border"],
                font_mono_bold(9.8),
                pad_x=8,
                pad_y=3.5,
            )
            draw.text((int((cx0 + 14) * S), int((card_y0 + 46) * S)), st["title"], font=font_sans_bold(17.0), fill=THEME["ink"])

            for ri, (rleft, rright) in enumerate(st["rows"]):
                ry = card_y0 + 82 + ri * 48
                is_cf_row = ri == 0
                row_hi = is_act and (ri == 1 + int((t * 12) % 3))
                row_bg = (246, 243, 236) if is_cf_row else (THEME["span"] if row_hi else THEME["surface"])
                draw.rounded_rectangle(
                    [int((cx0 + 14) * S), int(ry * S), int((cx1 - 14) * S), int((ry + 38) * S)],
                    radius=int(7 * S),
                    fill=row_bg,
                    outline=THEME["border"],
                    width=int((2.2 if row_hi else 1.3) * S),
                )
                draw.text(
                    (int((cx0 + 22) * S), int((ry + 11) * S)),
                    rleft,
                    font=font_mono_bold(9.8),
                    fill=THEME["ink_muted"] if is_cf_row else THEME["ink"],
                )
                rf = font_mono_bold(9.5) if not is_cf_row else font_mono_reg(9.2)
                rb = draw.textbbox((0, 0), rright, font=rf)
                rw = rb[2] - rb[0]
                rcol = THEME["accents"][3] if is_cf_row else (THEME["ink"] if row_hi else THEME["accents"][1])
                draw.text((int((cx1 - 22) * S) - rw, int((ry + 11) * S)), rright, font=rf, fill=rcol)

        # Bottom Left Card (x = 32..756, y = 444..734): Live Head-to-Head Specimen
        bot_y0, bot_y1 = 444, 734
        draw_shadow_card(draw, (32, bot_y0, 756, bot_y1), THEME["surface"], THEME["border"], THEME["border"])
        draw_pill(
            draw,
            48,
            bot_y0 + 14,
            "HEAD-TO-HEAD SPECIMEN",
            THEME["ink"],
            THEME["bg"],
            THEME["border"],
            font_mono_bold(10.0),
        )

        scene_tabs = ["1 · UNIFIED PASS", "2 · MATH INVARIANT", "3 · JEVBENCH SCORES"]
        cur_tx = 264.0
        for tab_i, tab_lbl in enumerate(scene_tabs):
            is_tab = tab_i == scene_idx
            pw = draw_pill(
                draw,
                cur_tx,
                bot_y0 + 14,
                tab_lbl,
                THEME["span"] if is_tab else THEME["muted_bg"],
                THEME["ink"],
                THEME["border"],
                font_mono_bold(9.5),
                pad_x=9,
                pad_y=3.5,
            )
            cur_tx += pw + 8

        paper = (48, bot_y0 + 50, 740, bot_y1 - 16)
        draw.rounded_rectangle(
            [int(v * S) for v in paper],
            radius=int(10 * S),
            fill=THEME["bg"],
            outline=THEME["border"],
            width=int(1.6 * S),
        )

        # Split paper into Top (Cloudflare Clairic + Clef) vs Bottom (nullspan) or Bars in Scene 3
        if scene_idx == 0:
            # Scene 0: Unified Extraction + Decision on Incident INC-40928
            draw_pill(draw, 62, bot_y0 + 62, "@cf/cloudflare/clairic + Clef (Workers AI Cloud)", THEME["null"], THEME["ink"], THEME["border"], font_mono_bold(9.5))
            draw.text((int(445 * S), int((bot_y0 + 66) * S)), "2 Separate Models · Closed API Only", font=font_mono_reg(9.5), fill=THEME["accents"][3])
            draw.text((int(62 * S), int((bot_y0 + 94) * S)), "Call 1 (Clairic): incident_id='INC-40928', total_credit=1550.75, cve_id=null (bare p_null)", font=font_mono_reg(10.0), fill=THEME["ink_muted"])
            draw.text((int(62 * S), int((bot_y0 + 116) * S)), "Call 2 (Clef):    severity='sev-1' (0.98), data_loss=false  [Extra RPC + No Browser Mode]", font=font_mono_reg(10.0), fill=THEME["ink_muted"])

            draw.line([(int(62 * S), int((bot_y0 + 144) * S)), (int(726 * S), int((bot_y0 + 144) * S))], fill=THEME["border"], width=int(1.2 * S))

            draw_pill(draw, 62, bot_y0 + 154, "nullspan (1 Unified Pass · Browser WASM ONNX + Zero-Dep)", THEME["list"], THEME["ink"], THEME["border"], font_mono_bold(9.5))
            draw.text((int(490 * S), int((bot_y0 + 158) * S)), "10 Ops in 1 Pass · 0 Output Tokens", font=font_mono_bold(9.5), fill=THEME["accents"][1])

            ns_rows = [
                ("SPAN + ARITH_SUM", "incident_id: 'INC-40928' [9,18] (H=0.03b)  |  total_credit: $1550.75", THEME["span"]),
                ("BOOL + ENUM HEADS", "severity: 'sev-1' (probs:{sev-1:0.99})     |  data_loss: false (p_true:0.01)", THEME["list"]),
                ("CALIBRATED NULL", "cve_id: null (p_null: 0.994 · reason: 'unassigned_placeholder')", THEME["null"]),
            ]
            for nri, ( badge_t, line_t, bcol ) in enumerate(ns_rows):
                ny = bot_y0 + 186 + nri * 28
                pw = draw_pill(draw, 62, ny, badge_t, bcol, THEME["ink"], THEME["border"], font_mono_bold(8.8), pad_x=6, pad_y=2.5)
                draw.text((int((62 + pw + 10) * S), int((ny + 3) * S)), line_t, font=font_mono_bold(9.6), fill=THEME["ink"])

        elif scene_idx == 1:
            # Scene 1: Cross-Field Algebraic Invariant + Decoy Immunity on Invoice #88310
            draw_pill(draw, 62, bot_y0 + 62, "Input: Nodes: 4 @ $3.50 | Subtotal: $14.00 | Egress: $0.00 (no promo_rebate)", THEME["muted_bg"], THEME["ink"], THEME["border"], font_mono_bold(9.3))

            # Left box vs Right box inside paper
            cf_box = (62, bot_y0 + 96, 386, bot_y1 - 28)
            ns_box = (400, bot_y0 + 96, 726, bot_y1 - 28)
            draw.rounded_rectangle([int(v * S) for v in cf_box], radius=int(8 * S), fill=THEME["surface"], outline=THEME["border"], width=int(1.4 * S))
            draw.rounded_rectangle([int(v * S) for v in ns_box], radius=int(8 * S), fill=THEME["surface"], outline=THEME["border"], width=int(2.0 * S))

            draw_pill(draw, 74, bot_y0 + 106, "@cf/cloudflare/clairic", THEME["null"], THEME["ink"], THEME["border"], font_mono_bold(9.2))
            cf_lines = [
                "nodes: 4, hourly_rate: 3.50",
                "subtotal: 14.00",
                "✗ constraint: NOT SUPPORTED",
                "✗ verified: N/A (blind to math)",
                "✗ reason: bare p_null only",
            ]
            for cli, cl in enumerate(cf_lines):
                col = THEME["accents"][3] if cl.startswith("✗") else THEME["ink_muted"]
                draw.text((int(74 * S), int((bot_y0 + 138 + cli * 24) * S)), cl, font=font_mono_bold(9.5), fill=col)

            draw_pill(draw, 412, bot_y0 + 106, "nullspan (Constraint + Guarded)", THEME["list"], THEME["ink"], THEME["border"], font_mono_bold(9.2))
            ns_lines = [
                "nodes: 4, hourly_rate: 3.50",
                "subtotal: 14.00 [SPAN_NUMBER]",
                "✓ constraint: subtotal==nodes*rate",
                "✓ verified: true (invariants_passed)",
                "✓ promo_rebate: null (ignores $0.00)",
            ]
            for nli, nl in enumerate(ns_lines):
                col = THEME["accents"][1] if nl.startswith("✓") else THEME["ink"]
                draw.text((int(412 * S), int((bot_y0 + 138 + nli * 24) * S)), nl, font=font_mono_bold(9.5), fill=col)

        else:
            # Scene 2: JevBench 4-Axis Head-to-Head Bars (N=404 Scored Observations)
            draw_pill(draw, 62, bot_y0 + 62, "JEVBENCH 4-AXIS LEADERBOARD (Intelligence · Calibration · Speed · Cost)", THEME["span"], THEME["ink"], THEME["border"], font_mono_bold(9.5))
            bars = [
                ("nullspan-Hybrid (Fast + ONNX Int8)", 97.7, 98.7, "$0.0003/1k · 0.005ms", THEME["list"]),
                ("nullspan-67M (PyTorch FP32 · Open)", 83.3, 99.2, "$0.0015/1k · 5.58ms", THEME["span"]),
                ("nullspan-67M-Int8 (Browser WASM)", 83.0, 98.7, "$0.0008/1k · 12.3ms", THEME["span"]),
                ("H2O-Lightning-4B (JevBench #1 GPU)", 72.5, 75.0, "$0.0211/1k · 29.3ms", THEME["quant"]),
                ("Jev 1.13.0 / Clef-Class Cloud API", 71.5, 77.1, "$0.0323/1k · 239ms", THEME["null"]),
            ]
            for bi, (bname, comp, cap, meta_s, bcol) in enumerate(bars):
                by = bot_y0 + 96 + bi * 34
                draw.text((int(62 * S), int((by + 5) * S)), bname, font=font_mono_bold(9.5), fill=THEME["ink"])
                bar_x0, bar_x1 = 310, 545
                draw.rounded_rectangle(
                    [int(bar_x0 * S), int((by + 2) * S), int(bar_x1 * S), int((by + 24) * S)],
                    radius=int(5 * S),
                    fill=THEME["muted_bg"],
                    outline=THEME["border"],
                    width=int(1.2 * S),
                )
                fill_w = int((bar_x1 - bar_x0) * (comp / 100.0))
                draw.rounded_rectangle(
                    [int(bar_x0 * S), int((by + 2) * S), int((bar_x0 + fill_w) * S), int((by + 24) * S)],
                    radius=int(5 * S),
                    fill=bcol,
                    outline=THEME["border"],
                    width=int(1.2 * S),
                )
                draw.text((int((bar_x0 + 8) * S), int((by + 5) * S)), f"Comp {comp:.1f} | Cap {cap:.1f}", font=font_mono_bold(9.0), fill=THEME["ink"])
                draw.text((int(554 * S), int((by + 5) * S)), meta_s, font=font_mono_reg(9.2), fill=THEME["ink_muted"])

        # Bottom Right Card (x = 780..1168, y = 444..734): KPIs + Code Box
        draw_shadow_card(draw, (780, bot_y0, 1168, bot_y1), THEME["surface"], THEME["border"], THEME["border"])
        draw_pill(
            draw,
            796,
            bot_y0 + 14,
            "NULLSPAN ADVANTAGE & UNIFIED API",
            THEME["ink"],
            THEME["bg"],
            THEME["border"],
            font_mono_bold(10.0),
        )

        tile_colors = [THEME["span"], THEME["list"], THEME["quant"], THEME["null"]]
        for ki, (kval, klabel) in enumerate(kpis):
            kx = 796 + (ki % 2) * 182
            ky = bot_y0 + 50 + (ki // 2) * 58
            draw.rounded_rectangle(
                [int(kx * S), int(ky * S), int((kx + 174) * S), int((ky + 50) * S)],
                radius=int(8 * S),
                fill=tile_colors[ki],
                outline=THEME["border"],
                width=int(1.5 * S),
            )
            draw.text((int((kx + 10) * S), int((ky + 6) * S)), kval, font=font_sans_bold(17.5), fill=THEME["ink"])
            draw.text((int((kx + 10) * S), int((ky + 30) * S)), klabel, font=font_mono_bold(9.2), fill=THEME["ink_muted"])

        code_box = (796, bot_y0 + 172, 1152, bot_y1 - 16)
        draw.rounded_rectangle(
            [int(v * S) for v in code_box],
            radius=int(8 * S),
            fill=THEME["bg"],
            outline=THEME["border"],
            width=int(1.6 * S),
        )
        for li, cline in enumerate(code_lines):
            draw.text((int(808 * S), int((bot_y0 + 180 + li * 20) * S)), cline, font=font_mono_bold(9.3), fill=THEME["ink"])

        out_frame = img.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
        out_frame.save(frames_dir / f"frame_{idx:03d}.png")

    # Save static keyframes from each scene for visual inspection
    Image.open(frames_dir / "frame_005.png").save(preview_10)
    Image.open(frames_dir / "frame_025.png").save(root / "docs" / "nullspan-vs-clairic-scene2.png")
    Image.open(frames_dir / "frame_045.png").save(root / "docs" / "nullspan-vs-clairic-scene3.png")

    # Encode 256-color full-stats sierra2_4a GIF
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
