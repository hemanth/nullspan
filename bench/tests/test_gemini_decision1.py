import unittest


class TestGeminiDecision1Comparison(unittest.TestCase):
    def test_build_request_and_normalize_response_across_all_three_types(self) -> None:
        from compare_decision1 import (
            build_gemini_decision1_request,
            normalize_gemini_decision1_response,
        )

        questions = {
            "is_damaged": {
                "type": "noul",
                "instructions": "Did the item arrive damaged?",
            },
            "queue": {
                "type": "choice",
                "instructions": "Which queue handles this?",
                "criteria": {
                    "billing": "Billing inquiries",
                    "replacements": "Damaged items",
                },
            },
            "severity": {
                "type": "score",
                "instructions": "Rate severity 0..2.",
                "criteria": ["0 - Low", "1 - Medium", "2 - High"],
            },
        }
        req = build_gemini_decision1_request({"order_id": "A-9", "note": "Cracked mug"}, questions)
        self.assertEqual(req["model"], "gemini-decision-1")
        self.assertIn("A-9", req["input"])
        self.assertEqual(set(req["questions"].keys()), {"is_damaged", "queue", "severity"})
        # Score criteria list should map cleanly to ordered rubric criteria
        self.assertIn("severity", req["questions"])

        raw_api_resp = {
            "answers": {
                "is_damaged": {"type": "noul", "noul": 0.96, "confidence": 0.96},
                "queue": {
                    "type": "choice",
                    "choice": "replacements",
                    "probabilities": {"billing": 0.03, "replacements": 0.97},
                    "confidence": 0.97,
                },
                "severity": {
                    "type": "score",
                    "score": 1.85,
                    "probabilities": {"0": 0.05, "1": 0.05, "2": 0.90},
                    "confidence": 0.90,
                },
            },
            "usage": {
                "total_input_tokens": 142,
                "total_output_tokens": 0,
                "total_tokens": 142,
            },
        }
        norm = normalize_gemini_decision1_response(raw_api_resp, questions)
        self.assertEqual(norm["usage"]["input_tokens"], 142)
        self.assertEqual(norm["usage"]["output_tokens"], 0)
        self.assertAlmostEqual(norm["answers"]["is_damaged"]["probabilities"]["true"], 0.96, places=5)
        self.assertAlmostEqual(norm["answers"]["is_damaged"]["probabilities"]["false"], 0.04, places=5)
        self.assertEqual(norm["answers"]["queue"]["label"], "replacements")
        self.assertAlmostEqual(norm["answers"]["severity"]["probabilities"]["2"], 0.90, places=5)

    def test_render_decision1_comparison_md_distinguishes_measured_vs_published_reference(self) -> None:
        from compare_decision1 import render_decision1_comparison_md

        nullspan_metrics = {
            "version": "v1.4.0-rc",
            "params": "395M (ModernBERT-large)",
            "td_test_acc": 0.814,
            "td_test_kl": 0.0725,
            "td_test_brier": 0.0402,
            "jev_public_acc": 0.788,
            "jev_v16_capability": 68.4,
            "jev_v16_calibration": 89.2,
            "p50_ms_l4": 18.4,
            "usd_per_1m_in": 0.04,
            "nullspan_sealed_em": 0.6524,
        }
        md = render_decision1_comparison_md(nullspan_metrics, live_d1_metrics=None, snapshot_date="2026-11-11")
        self.assertIn("gemini-decision-1", md)
        self.assertIn("gnumanth/nullspan", md)
        self.assertIn("2026-11-11", md)
        self.assertIn("$0.10", md)
        self.assertIn("$0.04", md)
        self.assertIn("92.2%", md)
        self.assertIn("published", md.lower())


if __name__ == "__main__":
    unittest.main()
