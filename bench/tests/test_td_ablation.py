import unittest


class TestTdAblationRenderer(unittest.TestCase):
    def test_render_td_ablation_md_checks_task7_rc_gate_and_lists_all_ablations(self) -> None:
        from render_td_ablation import check_phase1_rc_gate, render_td_ablation_md

        baseline = {
            "oof_accuracy": 0.8005,
            "oof_kl": 0.0840,
            "oof_brier": 0.0475,
            "oof_ece": 0.1580,
            "per_family_raw_oof": {
                "modernbert_zs": {"backbone_id": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "oof_accuracy": 0.8002, "oof_kl": 0.0875, "oof_brier": 0.0501, "oof_ece": 0.1646},
                "ettin_400m": {"backbone_id": "jhu-clsp/ettin-encoder-400m", "oof_accuracy": 0.7890, "oof_kl": 0.0910, "oof_brier": 0.0520, "oof_ece": 0.1550},
            },
        }
        cand_pass = {"oof_accuracy": 0.8170, "oof_kl": 0.0785, "oof_brier": 0.0420, "oof_ece": 0.0620}
        cand_fail = {"oof_accuracy": 0.8120, "oof_kl": 0.0815, "oof_brier": 0.0440, "oof_ece": 0.0700}

        g_pass = check_phase1_rc_gate(baseline, cand_pass)
        self.assertTrue(g_pass["passed"])
        g_fail = check_phase1_rc_gate(baseline, cand_fail)
        self.assertFalse(g_fail["passed"])

        md = render_td_ablation_md(
            baseline_oof=baseline,
            teacher_cal={"candidates": {"skamalj/signpost-d-4b": {"accuracy": 0.720, "mean_kl": 0.166, "accepted": False}}},
            phase1_runs=[
                {
                    "name": "task5_6spec_legacy_mask",
                    "oof": {
                        "oof_accuracy": 0.8155,
                        "oof_kl": 0.0790,
                        "oof_brier": 0.0428,
                        "oof_ece": 0.0640,
                        "winner_family": "modernbert_zs",
                        "per_family_raw_oof": {
                            "modernbert_zs": {"backbone_id": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "oof_accuracy": 0.8020, "oof_kl": 0.0850, "oof_brier": 0.0480, "oof_ece": 0.1600},
                            "deberta_v3_large_512": {"backbone_id": "microsoft/deberta-v3-large", "oof_accuracy": 0.7910, "oof_kl": 0.0920, "oof_brier": 0.0510, "oof_ece": 0.1500},
                        },
                    },
                },
                {
                    "name": "task6_holdout_lbfgs_soup",
                    "oof": {
                        **cand_pass,
                        "winner_family": "modernbert_zs",
                        "beta_oof": 0.32,
                        "alpha_wise": 0.04,
                        "temperatures": {"noul": 1.42, "choice": 1.18, "score": 1.31},
                        "student_holdout_sweep": [
                            {"beta_oof": 0.0, "alpha_wise": 0.0, "holdout_acc": 0.805, "holdout_kl": 0.082, "holdout_brier": 0.045},
                            {"beta_oof": 0.32, "alpha_wise": 0.04, "holdout_acc": 0.819, "holdout_kl": 0.076, "holdout_brier": 0.041},
                        ],
                        "soup_selection": {"selected": "soup", "delta_acc": 0.002, "delta_kl": -0.0025},
                    },
                },
            ],
        )
        self.assertIn("skamalj/signpost-d-4b", md)
        self.assertIn("deberta_v3_large_512", md)
        self.assertIn("PASS", md)
        self.assertIn("0.8170", md)


if __name__ == "__main__":
    unittest.main()
