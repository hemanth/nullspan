import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestAnalyzeTdErrors(unittest.TestCase):
    def test_four_row_fixture_gold_matching_ece_is_nonzero_and_metrics_exact(self):
        from bench.analyze_td_errors import analyze_oof_report

        fixture = {
            "oof_accuracy": 0.5,
            "oof_kl": 0.25,
            "oof_brier": 0.12,
            "oof_ece": 0.15,
            "per_decision": [
                {
                    "case_idx": 0,
                    "workflow": "wf_a",
                    "qid": "q1",
                    "qtype": "noul",
                    "state_tokens": 500,
                    "probs": {"true": 2.0 / 3.0, "false": 1.0 / 3.0},
                    "gold_probs": {"true": 2.0 / 3.0, "false": 1.0 / 3.0},
                    "correct": True,
                    "kl": 0.0,
                    "brier": 0.0,
                    "confidence": 2.0 / 3.0,
                },
                {
                    "case_idx": 1,
                    "workflow": "wf_a",
                    "qid": "q1",
                    "qtype": "noul",
                    "state_tokens": 1200,
                    "probs": {"true": 0.3, "false": 0.7},
                    "gold_probs": {"true": 2.0 / 3.0, "false": 1.0 / 3.0},
                    "correct": False,
                    "kl": 0.31,
                    "brier": 0.27,
                    "confidence": 0.7,
                },
                {
                    "case_idx": 2,
                    "workflow": "wf_b",
                    "qid": "q2",
                    "qtype": "choice",
                    "state_tokens": 2800,
                    "probs": {"x": 0.2, "y": 0.8},
                    "gold_probs": {"x": 1.0, "y": 0.0},
                    "correct": False,
                    "kl": 1.61,
                    "brier": 1.28,
                    "confidence": 0.8,
                },
                {
                    "case_idx": 3,
                    "workflow": "wf_b",
                    "qid": "q2",
                    "qtype": "choice",
                    "state_tokens": 600,
                    "probs": {"x": 0.75, "y": 0.25},
                    "gold_probs": {"x": 2.0 / 3.0, "false_opt": 0.0, "y": 1.0 / 3.0},
                    "correct": True,
                    "kl": 0.02,
                    "brier": 0.01,
                    "confidence": 0.75,
                },
            ],
        }

        report = analyze_oof_report(fixture)

        # (d) Gold-matching predictor outputs exact gold distribution; against gold argmax
        # 3 rows have max(gold)=2/3 (acc=1.0, conf=2/3 -> gap 1/3) and 1 row has max(gold)=1.0 (gap 0)
        # Expected ECE = (3/4) * (1/3) = 0.25 > 0.
        self.assertGreater(report["gold_matching_predictor_ece"], 0.0)
        self.assertAlmostEqual(report["gold_matching_predictor_ece"], 0.25, places=4)

        # (b) 2 errors total: case 1 has gold top-1 = 2/3 (<= 2/3), case 2 has gold top-1 = 1.0 (> 2/3)
        self.assertAlmostEqual(report["share_errors_gold_top1_le_two_thirds"], 0.5, places=4)

        # (a) Schema accuracy and recoverable errors
        self.assertEqual(report["by_schema"]["wf_a::q1"]["accuracy"], 0.5)
        self.assertEqual(report["by_schema"]["wf_a::q1"]["recoverable_errors"], 0)
        self.assertEqual(report["by_schema"]["wf_b::q2"]["accuracy"], 0.5)
        self.assertEqual(report["by_schema"]["wf_b::q2"]["recoverable_errors"], 1)
        self.assertEqual(report["top_recoverable_schemas"][0]["schema"], "wf_b::q2")

        # (c) Per-schema train-prior accuracy present
        self.assertIn("schema_prior_oof_accuracy", report)
        self.assertGreaterEqual(report["schema_prior_oof_accuracy"], 0.0)

        # (e) Token-length buckets and >2560 share (1 of 4 cases = 25%)
        self.assertIn("by_token_bucket", report)
        self.assertAlmostEqual(report["state_token_stats"]["pct_cases_gt_2560"], 25.0, places=2)


if __name__ == "__main__":
    unittest.main()
