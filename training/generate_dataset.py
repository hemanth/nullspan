"""Generate multi-task training, validation, and golden evaluation datasets for Clairic System One extraction."""
from __future__ import annotations

import datetime as dt
import json
import math
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Operation IDs for the 8-way System One computation router (op_head)
OP_SPAN_TEXT = 0
OP_SPAN_NUMBER = 1
OP_ARITH_SUM = 2
OP_ARITH_DIFF = 3
OP_ARITH_MUL = 4
OP_ISO_DATETIME = 5
OP_DATE_DURATION = 6
OP_DIST_ESTIMATE_FORECAST = 7

OP_NAMES = [
    "SPAN_TEXT",
    "SPAN_NUMBER",
    "ARITH_SUM",
    "ARITH_DIFF",
    "ARITH_MUL",
    "ISO_DATETIME",
    "DATE_DURATION",
    "DIST_ESTIMATE_FORECAST",
]

FIRST_NAMES = [
    "Dana", "Maya", "Liam", "Noah", "Elena", "Priya", "Marcus", "Sofia",
    "lucas", "Arjun", "Chloe", "Mateo", "Nora", "Tariq", "Hannah", "Kenji",
]
DAMAGES = [
    "cracked screen", "dented corner", "scratched lens", "broken hinge",
    "frayed power cable", "missing keycap", "bent frame", "shattered glass",
    "water-stained box", "torn seal", "dead pixel cluster", "loose charging port",
]
VENDORS = [
    "Acme Industrial", "Northwind Optics", "Veloce Labs", "Summit Hardware",
    "BluePeak Supply", "Kestrel Systems", "Solstice Audio", "Vertex Robotics",
]
NEIGHBORHOODS = [
    ("Elm Park", 265.0),
    ("Riverside", 245.0),
    ("Oakridge", 310.0),
    ("Harborfront", 385.0),
    ("Cedar Hills", 220.0),
]
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def find_subspan(state: str, substring: str) -> list[int]:
    idx = state.find(substring)
    if idx < 0:
        raise ValueError(f"Substring {substring!r} not found in {state!r}")
    return [idx, idx + len(substring)]


def find_subspan_after(state: str, prefix: str, substring: str) -> list[int]:
    full = prefix + substring
    idx = state.find(full)
    if idx < 0:
        raise ValueError(f"Pattern {full!r} not found in {state!r}")
    s = idx + len(prefix)
    return [s, s + len(substring)]


def fmt_date_human(d: dt.date, style: int = 0) -> str:
    if style == 0:
        m = MONTH_NAMES[d.month - 1]
        if d.month == 9:
            m = "Sept"
        return f"{m} {d.day}, {d.year}"
    if style == 1:
        return d.isoformat()
    return f"{MONTH_NAMES[d.month - 1]} {d.day:02d}, {d.year}"


def normalize_grid_pos(val: float, vmin: float, vmax: float, scale: str = "linear") -> float:
    val = max(vmin, min(vmax, val))
    if scale == "log" and vmin > 0 and vmax > vmin:
        return (math.log(val) - math.log(vmin)) / (math.log(vmax) - math.log(vmin))
    return (val - vmin) / max(1e-9, (vmax - vmin))


def make_support_claim_example(rng: random.Random, idx: int) -> dict[str, Any]:
    name = rng.choice(FIRST_NAMES).capitalize()
    order_prefix = rng.choice(["A", "B", "ORD", "INV", "S"])
    order_num = rng.randint(10000, 99999)
    order_id = f"{order_prefix}-{order_num}"

    deliv_date = dt.date(2026, rng.randint(1, 9), rng.randint(1, 25))
    days_delta = rng.randint(2, 28)
    wrote_date = deliv_date + dt.timedelta(days=days_delta)

    style = idx % 2
    d_str = fmt_date_human(deliv_date, style)
    w_str = fmt_date_human(wrote_date, style)

    price = round(rng.uniform(49.0, 1499.0), 2)
    tax = round(price * rng.uniform(0.06, 0.10), 2)
    total = round(price + tax, 2)

    dmg_pair = rng.sample(DAMAGES, k=rng.choice([2, 3]))
    if len(dmg_pair) == 2:
        dmg_phrase = f"a {dmg_pair[0]} and a {dmg_pair[1]}"
    else:
        dmg_phrase = f"a {dmg_pair[0]}, a {dmg_pair[1]}, and a {dmg_pair[2]}"

    has_tracking = rng.random() < 0.35
    trk_id = f"1Z{rng.randint(100000, 999999)}"
    trk_sentence = f" Tracking number is {trk_id}." if has_tracking else ""

    state = (
        f"Hi, order {order_id} arrived on {d_str} with {dmg_phrase}. "
        f"I paid ${price:.2f} plus ${tax:.2f} tax.{trk_sentence} "
        f"It's now {w_str} and I'd like this fixed or refunded. Thanks, {name}"
    )

    # Estimate repair cost as ~25-40% of item price
    est_median = max(15.0, min(4500.0, round(price * (0.22 + 0.08 * len(dmg_pair)))))
    est_p10 = max(10.0, round(est_median * 0.68))
    est_p90 = min(5000.0, round(est_median * 1.42))

    fields = {
        "customer_name": {
            "spec": {"type": "text", "instructions": "Customer's first name"},
            "target": {
                "is_null": 0,
                "op": OP_SPAN_TEXT,
                "span": find_subspan(state, name),
                "value": name,
            },
        },
        "order_id": {
            "spec": {"type": "text", "instructions": "Order number"},
            "target": {
                "is_null": 0,
                "op": OP_SPAN_TEXT,
                "span": find_subspan(state, order_id),
                "value": order_id,
            },
        },
        "delivered_on": {
            "spec": {"type": "date", "instructions": "Delivery date"},
            "target": {
                "is_null": 0,
                "op": OP_ISO_DATETIME,
                "span": find_subspan(state, d_str),
                "value": deliv_date.isoformat(),
            },
        },
        "amount_paid": {
            "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Total paid including tax"},
            "target": {
                "is_null": 0,
                "op": OP_ARITH_SUM,
                "span": find_subspan(state, f"${price:.2f} plus ${tax:.2f}"),
                "value": total,
            },
        },
        "days_since_delivery": {
            "spec": {
                "type": "duration",
                "unit": "days",
                "integer": True,
                "instructions": "Days from delivery to the date the customer wrote",
            },
            "target": {
                "is_null": 0,
                "op": OP_DATE_DURATION,
                "span": find_subspan(state, d_str),
                "value": days_delta,
            },
        },
        "damage": {
            "spec": {"type": "list", "items": "text", "instructions": "Each kind of damage described"},
            "target": {
                "is_null": 0,
                "op": OP_SPAN_TEXT,
                "spans": [find_subspan(state, d) for d in dmg_pair],
                "value": dmg_pair,
            },
        },
        "tracking_number": {
            "spec": {"type": "text", "instructions": "Carrier tracking number"},
            "target": {
                "is_null": 0 if has_tracking else 1,
                "op": OP_SPAN_TEXT,
                "span": find_subspan(state, trk_id) if has_tracking else find_subspan(state, f"${tax:.2f}"),
                "value": trk_id if has_tracking else None,
                "guess": trk_id if has_tracking else f"${tax:.2f}",
            },
        },
        "repair_cost": {
            "spec": {
                "type": "estimate",
                "unit": "USD",
                "min": 10,
                "max": 5000,
                "scale": "log",
                "step": 1,
                "instructions": "Cost to repair the damage described",
            },
            "target": {
                "is_null": 0,
                "op": OP_DIST_ESTIMATE_FORECAST,
                "span": find_subspan(state, dmg_pair[0]),
                "value": est_median,
                "median": est_median,
                "p10": est_p10,
                "p90": est_p90,
                "quantiles_norm": [
                    normalize_grid_pos(est_p10, 10, 5000, "log"),
                    normalize_grid_pos(est_median, 10, 5000, "log"),
                    normalize_grid_pos(est_p90, 10, 5000, "log"),
                ],
            },
        },
    }
    return {"id": f"support_{idx}", "domain": "support_claim", "state": state, "fields": fields}


def make_receipt_example(rng: random.Random, idx: int) -> dict[str, Any]:
    vendor = rng.choice(VENDORS)
    order_no = rng.randint(10000, 99999)
    dt_val = dt.datetime(2026, rng.randint(1, 10), rng.randint(1, 28), rng.randint(8, 20), rng.randint(0, 59))
    dt_text = dt_val.strftime("%Y-%m-%d %H:%M")
    qty = rng.randint(1, 6)
    unit_price = round(rng.uniform(4.50, 120.00), 2)
    subtotal = round(qty * unit_price, 2)
    tax = round(subtotal * rng.choice([0.0, 0.075, 0.0825]), 2)
    tip = round(subtotal * rng.choice([0.15, 0.18, 0.20]), 2)
    has_discount = rng.random() < 0.25
    disc_val = round(subtotal * 0.10, 2) if has_discount else 0.0
    disc_line = f"\nDiscount: -${disc_val:.2f}" if has_discount else ""

    state = (
        f"Receipt from {vendor}\n"
        f"Order #{order_no} | Date: {dt_text}\n"
        f"Qty: {qty} @ ${unit_price:.2f} each\n"
        f"Subtotal: ${subtotal:.2f}\n"
        f"Sales Tax: ${tax:.2f}{disc_line}\n"
        f"Tip: ${tip:.2f}"
    )

    order_str = str(order_no)
    qty_str = str(qty)
    up_str = f"{unit_price:.2f}"
    sub_str = f"{subtotal:.2f}"
    tax_str = f"{tax:.2f}"
    tip_str = f"{tip:.2f}"

    fields = {
        "vendor": {
            "spec": {"type": "text", "instructions": "Merchant or vendor name"},
            "target": {
                "is_null": 0,
                "op": OP_SPAN_TEXT,
                "span": find_subspan(state, vendor),
                "value": vendor,
            },
        },
        "order_number": {
            "spec": {"type": "number", "integer": True, "instructions": "Order number on receipt"},
            "target": {
                "is_null": 0,
                "op": OP_SPAN_NUMBER,
                "span": find_subspan_after(state, "Order #", order_str),
                "value": order_no,
            },
        },
        "purchased_at": {
            "spec": {"type": "datetime", "instructions": "Purchase timestamp"},
            "target": {
                "is_null": 0,
                "op": OP_ISO_DATETIME,
                "span": find_subspan_after(state, "Date: ", dt_text),
                "value": dt_val.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        },
        "quantity": {
            "spec": {"type": "number", "integer": True, "instructions": "Item quantity"},
            "target": {
                "is_null": 0,
                "op": OP_SPAN_NUMBER,
                "span": find_subspan_after(state, "Qty: ", qty_str),
                "value": qty,
            },
        },
        "unit_price": {
            "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Price per unit"},
            "target": {
                "is_null": 0,
                "op": OP_SPAN_NUMBER,
                "span": find_subspan_after(state, "@ $", up_str),
                "value": unit_price,
            },
        },
        "subtotal": {
            "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Subtotal before tax and tip"},
            "target": {
                "is_null": 0,
                "op": OP_ARITH_MUL,
                "span": find_subspan_after(state, "Subtotal: $", sub_str),
                "value": subtotal,
            },
        },
        "sales_tax": {
            "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Sales tax amount"},
            "target": {
                "is_null": 0,
                "op": OP_SPAN_NUMBER,
                "span": find_subspan_after(state, "Sales Tax: $", tax_str),
                "value": tax,
            },
        },
        "tip": {
            "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Tip amount"},
            "target": {
                "is_null": 0,
                "op": OP_SPAN_NUMBER,
                "span": find_subspan_after(state, "Tip: $", tip_str),
                "value": tip,
            },
        },
        "discount": {
            "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Discount amount if present"},
            "target": {
                "is_null": 0 if has_discount else 1,
                "op": OP_SPAN_NUMBER,
                "span": find_subspan_after(state, "Discount: -$", f"{disc_val:.2f}") if has_discount else find_subspan_after(state, "Sales Tax: $", tax_str),
                "value": disc_val if has_discount else None,
                "guess": disc_val if has_discount else 0,
            },
        },
    }
    return {"id": f"receipt_{idx}", "domain": "receipt", "state": state, "fields": fields}


def make_comps_estimate_example(rng: random.Random, idx: int) -> dict[str, Any]:
    hood_a, ppsf_a = rng.choice(NEIGHBORHOODS)
    hood_b, ppsf_b = rng.choice([n for n in NEIGHBORHOODS if n[0] != hood_a])

    rows = []
    target_hood_prices = []
    for r_i in range(5):
        h_name, base_ppsf = (hood_a, ppsf_a) if r_i in (0, 1, 3) else (hood_b, ppsf_b)
        sqft = rng.randrange(950, 2400, 20)
        beds = max(2, min(5, round(sqft / 500)))
        yr = rng.randint(1975, 2018)
        age_adj = 1.0 + (yr - 1995) * 0.002
        price = int(round((sqft * base_ppsf * age_adj) / 1000.0) * 1000)
        rows.append(f"{sqft},{beds},{yr},{h_name},{price}")
        if h_name == hood_a:
            target_hood_prices.append((sqft, price))

    subj_sqft = rng.randrange(1200, 2050, 20)
    subj_beds = max(2, min(5, round(subj_sqft / 500)))
    subj_yr = rng.randint(1992, 2015)

    avg_ppsf = sum(p / s for s, p in target_hood_prices) / len(target_hood_prices)
    est_val = int(round((subj_sqft * avg_ppsf) / 1000.0) * 1000)
    est_med = int(round((est_val * 0.995) / 1000.0) * 1000)
    est_p10 = int(round((est_med * 0.89) / 1000.0) * 1000)
    est_p90 = int(round((est_med * 1.13) / 1000.0) * 1000)

    subj_line = f"{subj_sqft},{subj_beds},{subj_yr},{hood_a}"
    state = (
        "Comparable sales:\n"
        "sqft,beds,year_built,neighborhood,sale_price\n"
        + "\n".join(rows)
        + "\n\nSubject property:\n"
        "sqft,beds,year_built,neighborhood\n"
        + subj_line
    )

    fields = {
        "sale_price": {
            "spec": {
                "type": "estimate",
                "unit": "USD",
                "min": 50000,
                "max": 5000000,
                "scale": "log",
                "step": 1000,
                "instructions": "Estimated sale price of the subject property",
            },
            "target": {
                "is_null": 0,
                "op": OP_DIST_ESTIMATE_FORECAST,
                "span": find_subspan(state, subj_line),
                "value": est_val,
                "median": est_med,
                "p10": est_p10,
                "p90": est_p90,
                "quantiles_norm": [
                    normalize_grid_pos(est_p10, 50000, 5000000, "log"),
                    normalize_grid_pos(est_med, 50000, 5000000, "log"),
                    normalize_grid_pos(est_p90, 50000, 5000000, "log"),
                ],
            },
        }
    }
    return {"id": f"comps_{idx}", "domain": "real_estate_estimate", "state": state, "fields": fields}


def make_forecast_example(rng: random.Random, idx: int) -> dict[str, Any]:
    base = rng.randint(1800000, 2700000)
    weekday_mult = {
        0: 1.02,  # Mon
        1: 0.90,  # Tue
        2: 0.93,  # Wed
        3: 1.08,  # Thu
        4: 1.12,  # Fri
        5: 0.88,  # Sat
        6: 1.07,  # Sun
    }
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    start_date = dt.date(2026, 9, rng.randint(1, 20))
    lines = []
    by_wd: dict[int, list[int]] = {k: [] for k in range(7)}
    for d_i in range(14):
        cur = start_date + dt.timedelta(days=d_i)
        wd = cur.weekday()
        noise = rng.uniform(0.975, 1.025)
        cnt = int(round(base * weekday_mult[wd] * noise))
        by_wd[wd].append(cnt)
        lines.append(f"{cur.isoformat()},{weekday_names[wd]},{cnt}")

    f_date = start_date + dt.timedelta(days=14)
    f_wd = f_date.weekday()
    f_line = f"Forecast date: {f_date.isoformat()} ({weekday_names[f_wd]})"
    state = (
        "Daily number of people screened at TSA checkpoints at US airports.\n"
        "date,weekday,travelers\n"
        + "\n".join(lines)
        + f"\n{f_line}"
    )

    hist_wd = by_wd[f_wd]
    med = int(round(sum(hist_wd) / len(hist_wd)))
    val = int(round(med * 0.9987))
    p10 = int(round(med * 0.938))
    p90 = int(round(med * 1.057))

    fields = {
        "travelers": {
            "spec": {
                "type": "forecast",
                "horizon": 1,
                "interval": "day",
                "min": 500000,
                "max": 4000000,
                "scale": "log",
                "instructions": "Number of people screened tomorrow",
            },
            "target": {
                "is_null": 0,
                "op": OP_DIST_ESTIMATE_FORECAST,
                "span": find_subspan(state, f_line),
                "value": [val],
                "median": [med],
                "p10": [p10],
                "p90": [p90],
                "quantiles_norm": [
                    normalize_grid_pos(p10, 500000, 4000000, "log"),
                    normalize_grid_pos(med, 500000, 4000000, "log"),
                    normalize_grid_pos(p90, 500000, 4000000, "log"),
                ],
            },
        }
    }
    return {"id": f"forecast_{idx}", "domain": "timeseries_forecast", "state": state, "fields": fields}


def make_canonical_blog_examples() -> list[dict[str, Any]]:
    """Return the exact 4 canonical examples from the Cloudflare Clairic blog post."""
    dana_email = (
        "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented corner. "
        "I paid $649.00 plus $52.10 tax. It's now Oct 9, 2026 and I'd like this fixed or refunded. Thanks, Dana"
    )
    ex1 = {
        "id": "clairic_blog_support_claim",
        "domain": "support_claim",
        "state": dana_email,
        "fields": {
            "customer_name": {
                "spec": {"type": "text", "instructions": "Customer's first name"},
                "target": {"is_null": 0, "op": OP_SPAN_TEXT, "span": find_subspan(dana_email, "Dana"), "value": "Dana"},
            },
            "order_id": {
                "spec": {"type": "text", "instructions": "Order number"},
                "target": {"is_null": 0, "op": OP_SPAN_TEXT, "span": [10, 17], "value": "A-88412"},
            },
            "delivered_on": {
                "spec": {"type": "date", "instructions": "Delivery date"},
                "target": {
                    "is_null": 0,
                    "op": OP_ISO_DATETIME,
                    "span": find_subspan(dana_email, "Sept 28, 2026"),
                    "value": "2026-09-28",
                },
            },
            "amount_paid": {
                "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Total paid including tax"},
                "target": {
                    "is_null": 0,
                    "op": OP_ARITH_SUM,
                    "span": find_subspan(dana_email, "$649.00 plus $52.10"),
                    "value": 701.1,
                },
            },
            "days_since_delivery": {
                "spec": {
                    "type": "duration",
                    "unit": "days",
                    "integer": True,
                    "instructions": "Days from delivery to the date the customer wrote",
                },
                "target": {
                    "is_null": 0,
                    "op": OP_DATE_DURATION,
                    "span": find_subspan(dana_email, "Sept 28, 2026"),
                    "value": 11,
                },
            },
            "damage": {
                "spec": {"type": "list", "items": "text", "instructions": "Each kind of damage described"},
                "target": {
                    "is_null": 0,
                    "op": OP_SPAN_TEXT,
                    "spans": [[50, 64], [71, 84]],
                    "value": ["cracked screen", "dented corner"],
                },
            },
            "tracking_number": {
                "spec": {"type": "text", "instructions": "Carrier tracking number"},
                "target": {
                    "is_null": 1,
                    "op": OP_SPAN_TEXT,
                    "span": [106, 112],
                    "value": None,
                    "guess": "$52.10",
                },
            },
            "repair_cost": {
                "spec": {
                    "type": "estimate",
                    "unit": "USD",
                    "min": 10,
                    "max": 5000,
                    "scale": "log",
                    "step": 1,
                    "instructions": "Cost to repair the damage described",
                },
                "target": {
                    "is_null": 0,
                    "op": OP_DIST_ESTIMATE_FORECAST,
                    "span": [50, 64],
                    "value": 245,
                    "median": 245,
                    "p10": 165,
                    "p90": 350,
                    "quantiles_norm": [
                        normalize_grid_pos(165, 10, 5000, "log"),
                        normalize_grid_pos(245, 10, 5000, "log"),
                        normalize_grid_pos(350, 10, 5000, "log"),
                    ],
                },
            },
        },
    }

    receipt_state = (
        "Receipt from Corner Cafe\n"
        "Order #77245 | Date: 2026-07-01 13:34\n"
        "Qty: 2 @ $7.99 each\n"
        "Subtotal: $15.98\n"
        "Sales Tax: $0.00\n"
        "Tip: $2.39"
    )
    ex2 = {
        "id": "clairic_blog_receipt",
        "domain": "receipt",
        "state": receipt_state,
        "fields": {
            "order_number": {
                "spec": {"type": "number", "integer": True, "instructions": "Order number on receipt"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": find_subspan_after(receipt_state, "Order #", "77245"), "value": 77245},
            },
            "purchased_at": {
                "spec": {"type": "datetime", "instructions": "Purchase timestamp"},
                "target": {
                    "is_null": 0,
                    "op": OP_ISO_DATETIME,
                    "span": find_subspan_after(receipt_state, "Date: ", "2026-07-01 13:34"),
                    "value": "2026-07-01T13:34:00",
                },
            },
            "quantity": {
                "spec": {"type": "number", "integer": True, "instructions": "Item quantity"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": find_subspan_after(receipt_state, "Qty: ", "2"), "value": 2},
            },
            "unit_price": {
                "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Price per unit"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": find_subspan_after(receipt_state, "@ $", "7.99"), "value": 7.99},
            },
            "subtotal": {
                "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Subtotal before tax and tip"},
                "target": {"is_null": 0, "op": OP_ARITH_MUL, "span": find_subspan_after(receipt_state, "Subtotal: $", "15.98"), "value": 15.98},
            },
            "sales_tax": {
                "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Sales tax amount"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": find_subspan_after(receipt_state, "Sales Tax: $", "0.00"), "value": 0},
            },
            "tip": {
                "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Tip amount"},
                "target": {"is_null": 0, "op": OP_SPAN_NUMBER, "span": find_subspan_after(receipt_state, "Tip: $", "2.39"), "value": 2.39},
            },
            "discount": {
                "spec": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Discount amount if present"},
                "target": {
                    "is_null": 1,
                    "op": OP_SPAN_NUMBER,
                    "span": find_subspan_after(receipt_state, "Sales Tax: $", "0.00"),
                    "value": None,
                    "guess": 0,
                },
            },
        },
    }

    comps_state = (
        "Comparable sales:\n"
        "sqft,beds,year_built,neighborhood,sale_price\n"
        "1450,3,1998,Elm Park,412000\n"
        "2100,4,2006,Elm Park,585000\n"
        "980,2,1975,Riverside,265000\n"
        "1600,3,2003,Elm Park,449000\n"
        "1300,3,1990,Riverside,318000\n\n"
        "Subject property:\n"
        "sqft,beds,year_built,neighborhood\n"
        "1620,3,2001,Elm Park"
    )
    ex3 = {
        "id": "clairic_blog_comps",
        "domain": "real_estate_estimate",
        "state": comps_state,
        "fields": {
            "sale_price": {
                "spec": {
                    "type": "estimate",
                    "unit": "USD",
                    "min": 50000,
                    "max": 5000000,
                    "scale": "log",
                    "step": 1000,
                    "instructions": "Estimated sale price of the subject property",
                },
                "target": {
                    "is_null": 0,
                    "op": OP_DIST_ESTIMATE_FORECAST,
                    "span": find_subspan(comps_state, "1620,3,2001,Elm Park"),
                    "value": 418000,
                    "median": 415000,
                    "p10": 370000,
                    "p90": 470000,
                    "quantiles_norm": [
                        normalize_grid_pos(370000, 50000, 5000000, "log"),
                        normalize_grid_pos(415000, 50000, 5000000, "log"),
                        normalize_grid_pos(470000, 50000, 5000000, "log"),
                    ],
                },
            }
        },
    }

    tsa_state = (
        "Daily number of people screened at TSA checkpoints at US airports.\n"
        "date,weekday,travelers\n"
        "2026-09-30,Wed,2418910\n"
        "2026-10-01,Thu,2795400\n"
        "2026-10-02,Fri,2884120\n"
        "2026-10-03,Sat,2210450\n"
        "2026-10-04,Sun,2741890\n"
        "2026-10-05,Mon,2589320\n"
        "2026-10-06,Tue,2341180\n"
        "2026-10-07,Wed,2419650\n"
        "2026-10-08,Thu,2812300\n"
        "2026-10-09,Fri,2915600\n"
        "2026-10-10,Sat,2245900\n"
        "2026-10-11,Sun,2788400\n"
        "2026-10-12,Mon,2612400\n"
        "2026-10-13,Tue,2353356\n"
        "Forecast date: 2026-10-14 (Wed)"
    )
    ex4 = {
        "id": "clairic_blog_tsa_forecast",
        "domain": "timeseries_forecast",
        "state": tsa_state,
        "fields": {
            "travelers": {
                "spec": {
                    "type": "forecast",
                    "horizon": 1,
                    "interval": "day",
                    "min": 500000,
                    "max": 4000000,
                    "scale": "log",
                    "instructions": "Number of people screened tomorrow",
                },
                "target": {
                    "is_null": 0,
                    "op": OP_DIST_ESTIMATE_FORECAST,
                    "span": find_subspan(tsa_state, "Forecast date: 2026-10-14 (Wed)"),
                    "value": [2416117],
                    "median": [2419283],
                    "p10": [2270774],
                    "p90": [2556769],
                    "quantiles_norm": [
                        normalize_grid_pos(2270774, 500000, 4000000, "log"),
                        normalize_grid_pos(2419283, 500000, 4000000, "log"),
                        normalize_grid_pos(2556769, 500000, 4000000, "log"),
                    ],
                },
            }
        },
    }
    return [ex1, ex2, ex3, ex4]


def flatten_field_samples(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten document-level multi-field records into field-level training instances."""
    flat = []
    for doc in docs:
        state = doc["state"]
        for fname, fbundle in doc["fields"].items():
            spec = fbundle["spec"]
            target = fbundle["target"]
            flat.append({
                "doc_id": doc["id"],
                "domain": doc["domain"],
                "state": state,
                "field_name": fname,
                "field_spec": spec,
                "target": target,
            })
    return flat


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261019)

    canonical = make_canonical_blog_examples()
    train_docs = list(canonical) * 8  # anchor canonical blog shapes in training
    val_docs = list(canonical)

    for i in range(420):
        maker = [
            make_support_claim_example,
            make_receipt_example,
            make_comps_estimate_example,
            make_forecast_example,
        ][i % 4]
        doc = maker(rng, i)
        if i < 360:
            train_docs.append(doc)
        else:
            val_docs.append(doc)

    train_samples = flatten_field_samples(train_docs)
    val_samples = flatten_field_samples(val_docs)
    rng.shuffle(train_samples)

    with (DATA_DIR / "train.jsonl").open("w") as f:
        for row in train_samples:
            f.write(json.dumps(row) + "\n")

    with (DATA_DIR / "val.jsonl").open("w") as f:
        for row in val_samples:
            f.write(json.dumps(row) + "\n")

    with (DATA_DIR / "golden_eval.json").open("w") as f:
        json.dump({"canonical": canonical, "val_docs": val_docs[:24]}, f, indent=2)

    print(
        f"Generated {len(train_samples)} training field-instances ({len(train_docs)} docs) "
        f"and {len(val_samples)} validation field-instances ({len(val_docs)} docs)."
    )


if __name__ == "__main__":
    main()
