"""Unit tests for nullspan across all 7 System One extraction field types."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nullspan import extract


class TestNullspan(unittest.TestCase):
    def test_support_claim_extraction_and_computation(self) -> None:
        email = (
            "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented corner. "
            "I paid $649.00 plus $52.10 tax. It's now Oct 9, 2026 and I'd like this fixed or refunded. Thanks, Dana"
        )
        res = extract(email, {
            "customer_name": {"type": "text", "instructions": "Customer's first name"},
            "order_id": {"type": "text", "instructions": "Order number"},
            "delivered_on": {"type": "date", "instructions": "Delivery date"},
            "amount_paid": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Total paid including tax"},
            "days_since_delivery": {
                "type": "duration",
                "unit": "days",
                "integer": True,
                "instructions": "Days from delivery to the date the customer wrote",
            },
            "damage": {"type": "list", "items": "text", "instructions": "Each kind of damage described"},
            "tracking_number": {"type": "text", "instructions": "Carrier tracking number"},
            "repair_cost": {
                "type": "estimate",
                "unit": "USD",
                "min": 10,
                "max": 5000,
                "scale": "log",
                "step": 1,
                "instructions": "Cost to repair the damage described",
            },
        })
        ans = res["answers"]
        self.assertEqual(ans["customer_name"]["value"], "Dana")
        self.assertEqual(ans["order_id"]["value"], "A-88412")
        self.assertEqual(ans["order_id"]["span"], [10, 17])
        self.assertEqual(email[10:17], "A-88412")
        self.assertEqual(ans["delivered_on"]["value"], "2026-09-28")
        self.assertAlmostEqual(ans["amount_paid"]["value"], 701.10, places=2)
        self.assertEqual(ans["days_since_delivery"]["value"], 11)
        self.assertEqual(ans["damage"]["value"], ["cracked screen", "dented corner"])
        self.assertEqual(ans["damage"]["spans"], [[50, 64], [71, 84]])
        self.assertIsNone(ans["tracking_number"]["value"])
        self.assertGreater(ans["tracking_number"]["p_null"], 0.9)
        self.assertTrue(10 <= ans["repair_cost"]["p10"] <= ans["repair_cost"]["median"] <= ans["repair_cost"]["p90"] <= 5000)

    def test_receipt_extraction_and_null_discount(self) -> None:
        receipt = (
            "Receipt from Corner Cafe\n"
            "Order #77245 | Date: 2026-07-01 13:34\n"
            "Qty: 2 @ $7.99 each\n"
            "Subtotal: $15.98\n"
            "Sales Tax: $0.00\n"
            "Tip: $2.39"
        )
        res = extract(receipt, {
            "order_number": {"type": "number", "integer": True, "instructions": "Order number on receipt"},
            "purchased_at": {"type": "datetime", "instructions": "Purchase timestamp"},
            "quantity": {"type": "number", "integer": True, "instructions": "Item quantity"},
            "unit_price": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Price per unit"},
            "subtotal": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Subtotal before tax and tip"},
            "sales_tax": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Sales tax amount"},
            "tip": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Tip amount"},
            "discount": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Discount amount if present"},
        })
        ans = res["answers"]
        self.assertEqual(ans["order_number"]["value"], 77245)
        self.assertEqual(ans["purchased_at"]["value"], "2026-07-01T13:34:00")
        self.assertEqual(ans["quantity"]["value"], 2)
        self.assertAlmostEqual(ans["unit_price"]["value"], 7.99, places=2)
        self.assertAlmostEqual(ans["subtotal"]["value"], 15.98, places=2)
        self.assertEqual(ans["sales_tax"]["value"], 0)
        self.assertAlmostEqual(ans["tip"]["value"], 2.39, places=2)
        self.assertIsNone(ans["discount"]["value"])

    def test_comparable_sales_estimation_and_tsa_forecast(self) -> None:
        comps = (
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
        est = extract(comps, {
            "sale_price": {
                "type": "estimate",
                "unit": "USD",
                "min": 50000,
                "max": 5000000,
                "scale": "log",
                "step": 1000,
                "percentiles": True,
            },
        })["answers"]["sale_price"]
        self.assertTrue(350000 <= est["p10"] <= est["median"] <= est["p90"] <= 500000)
        self.assertEqual(est["median"] % 1000, 0)
        self.assertEqual(len(est["percentiles"]), 99)

        tsa = (
            "Daily number of people screened at TSA checkpoints at US airports.\n"
            "date,weekday,travelers\n"
            "2026-09-30,Wed,2418910\n"
            "2026-10-07,Wed,2419650\n"
            "2026-10-13,Tue,2353356\n"
            "Forecast date: 2026-10-14 (Wed)"
        )
        fc = extract(tsa, {
            "travelers": {
                "type": "forecast",
                "horizon": 1,
                "interval": "day",
                "min": 500000,
                "max": 4000000,
                "scale": "log",
            },
        })["answers"]["travelers"]
        self.assertEqual(len(fc["median"]), 1)
        self.assertTrue(2200000 <= fc["p10"][0] <= fc["median"][0] <= fc["p90"][0] <= 2600000)

    def test_reusable_extractor_object(self) -> None:
        schema = extract({"order_id": {"type": "text", "instructions": "Order number"}})
        batch_out = schema.batch(["Order B-10492 shipped.", "Order C-99812 delayed."])
        self.assertEqual(len(batch_out), 2)
        self.assertEqual(batch_out[0]["answers"]["order_id"]["value"], "B-10492")
        self.assertEqual(batch_out[1]["answers"]["order_id"]["value"], "C-99812")


if __name__ == "__main__":
    unittest.main()
