import unittest


class TestJevBenchV16Scorer(unittest.TestCase):
    def test_reproduces_published_v161_aggregate_equations_to_1e9(self) -> None:
        from score_v16 import (
            aggregate_calibration_v16,
            aggregate_intelligence_v16,
            capability_score_v16,
            composite_option_a_v16,
        )

        # Exact published Sage 1.3.0 numbers from JevBench v1.6.1 artifact
        per_type_split = {
            "open|choice": {"tiers": {"easy": 96.10810810810811, "standard": 84.93093344495604, "judge": 84.17930125247197, "hard": 83.01352626612143}},
            "open|noul": {"tiers": {"easy": 85.71428571428572, "standard": 55.55555555555556, "judge": 39.99999999999999, "hard": 56.52173913043479}},
            "open|score": {"tiers": {"easy": 71.80397727272727, "standard": 45.94907407407408, "judge": 68.19385593220339, "hard": 52.14120370370371}},
            "sealed|choice": {"tiers": {"easy": 82.70637794693668, "standard": 83.37233649464221, "judge": 85.1024208566108, "hard": 73.02036199095022}},
            "sealed|noul": {"tiers": {"easy": 68.0, "standard": 77.46478873239437, "judge": 57.89473684210527, "hard": 41.747572815533985}},
            "sealed|score": {"tiers": {"easy": 76.96974734042553, "standard": 62.692338197969534, "judge": 64.89053143712576, "hard": 51.94546568627452}},
        }
        intel = aggregate_intelligence_v16(per_type_split, g_med=2.573387642438244)
        self.assertAlmostEqual(intel["I_open"], 65.67742817331194, places=9)
        self.assertAlmostEqual(intel["I_sealed"], 65.50839071719783, places=9)
        self.assertAlmostEqual(intel["score"], 65.59290944525489, places=9)

        cal = aggregate_calibration_v16(
            choice_ece_hard=0.034436035156249975,
            choice_mean_tvd=None,
            noul_ece=0.049375,
            score_nrps=0.06392779541015624,
            score_top_ece=0.05246875,
        )
        self.assertAlmostEqual(cal["score"], 91.59817606608073, places=9)

        cap = capability_score_v16(
            intelligence=intel["score"],
            calibration=cal["score"],
            usd_per_1000=0.024677048070413,
            p50_adjusted_s=0.15245595201849937,
        )
        self.assertTrue(cap["eligible"])
        self.assertAlmostEqual(cap["capability"], 78.59554275566781, places=9)

        comp = composite_option_a_v16(
            intelligence=intel["score"],
            calibration=cal["score"],
            speed=93.61638181239104,
            cost=58.231203787388296,
        )
        self.assertAlmostEqual(comp, 74.04992179175468, places=9)

    def test_o1s_noul_abstention_and_midscale_score_baseline(self) -> None:
        from score_v16 import score_items_v16

        items = [
            # Choice (4 options -> chance 0.25)
            {"id": "c1", "split": "open", "tier": "hard", "type": "choice", "labels": ["a", "b", "c", "d"], "expected": "b", "gold_probs": {"a": 0.1, "b": 0.8, "c": 0.05, "d": 0.05}},
            # Noul: decisive right (P(yes)=0.90 >= 0.80) + abstention wrong (P(yes)=0.65 in (0.20, 0.80))
            {"id": "n1", "split": "open", "tier": "hard", "type": "noul", "labels": ["no", "yes"], "expected": "yes"},
            {"id": "n2", "split": "open", "tier": "hard", "type": "noul", "labels": ["no", "yes"], "expected": "yes"},
            # Score (5 levels 0..4, mid=2.0): flat uniform distribution predicts EV=2.0 -> 0 O1S competence!
            {"id": "s1", "split": "open", "tier": "hard", "type": "score", "labels": ["0", "1", "2", "3", "4"], "expected": "4"},
            {"id": "s2", "split": "open", "tier": "hard", "type": "score", "labels": ["0", "1", "2", "3", "4"], "expected": "0"},
        ]
        preds_flat = {
            "c1": {"probs": {"a": 0.1, "b": 0.8, "c": 0.05, "d": 0.05}, "latency_s": 0.02, "input_tokens": 200, "output_tokens": 0},
            "n1": {"probs": {"no": 0.10, "yes": 0.90}, "latency_s": 0.02, "input_tokens": 200, "output_tokens": 0},
            "n2": {"probs": {"no": 0.35, "yes": 0.65}, "latency_s": 0.02, "input_tokens": 200, "output_tokens": 0},
            "s1": {"probs": {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2}, "latency_s": 0.02, "input_tokens": 200, "output_tokens": 0},
            "s2": {"probs": {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2}, "latency_s": 0.02, "input_tokens": 200, "output_tokens": 0},
        }
        res = score_items_v16(items, preds_flat, usd_per_1m_in=0.04, endpoint_kind="gpu")
        # n1 is decisive right, n2 is abstention wrong -> acc=0.5 -> CC=0.0
        self.assertAlmostEqual(res["intelligence"]["per_type_split"]["open|noul"]["cc"], 0.0, places=6)
        self.assertAlmostEqual(res["calibration"]["parts"]["noul"]["abstention_rate"], 0.5, places=6)
        # s1, s2 flat uniform predicts EV=2.0 -> equals mid-scale chance error -> O1S Score CC == 0.0!
        self.assertAlmostEqual(res["intelligence"]["per_type_split"]["open|score"]["cc"], 0.0, places=6)
        # c1 exact match -> CC=100.0
        self.assertAlmostEqual(res["intelligence"]["per_type_split"]["open|choice"]["cc"], 100.0, places=6)

    def test_eval_public_benchmarks_wires_score_v16(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from eval_public_benchmarks import compute_public_v16_from_records

        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td)
            (pdir / "easy.jsonl").write_text(
                json.dumps({
                    "id": "e1",
                    "family": "intent",
                    "labels": ["a", "b"],
                    "expected": "a",
                    "question": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B"}},
                }) + "\n",
                encoding="utf-8",
            )
            (pdir / "original.jsonl").write_text(
                json.dumps({
                    "id": "o1",
                    "family": "policy",
                    "labels": ["no", "yes"],
                    "expected": "yes",
                    "question": {"type": "noul", "instructions": "Allow?", "criteria": {"false": "No", "true": "Yes"}},
                }) + "\n",
                encoding="utf-8",
            )
            (pdir / "hard.jsonl").write_text(
                json.dumps({
                    "id": "h1",
                    "family": "ordinal",
                    "labels": ["0", "1", "2", "3", "4"],
                    "expected": "4",
                    "question": {"type": "score", "instructions": "Score", "criteria": ["0", "1", "2", "3", "4"]},
                }) + "\n",
                encoding="utf-8",
            )
            records = [
                {"task_id": "e1", "probs": {"a": 0.95, "b": 0.05}, "latency_s": 0.03, "input_tokens": 200, "output_tokens": 0},
                {"task_id": "o1", "probs": {"no": 0.05, "yes": 0.95}, "latency_s": 0.03, "input_tokens": 200, "output_tokens": 0},
                {"task_id": "h1", "probs": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0}, "latency_s": 0.04, "input_tokens": 300, "output_tokens": 0},
            ]
            v16 = compute_public_v16_from_records(str(pdir), records)
            self.assertEqual(v16["n_items"], 3)
            self.assertAlmostEqual(v16["intelligence"]["score"], 100.0, places=6)
            self.assertTrue(v16["capability"]["eligible"])
            self.assertGreater(v16["capability"]["capability"], 90.0)

    def test_render_jevbench_submission_md_includes_required_disclosures(self) -> None:
        from score_v16 import render_jevbench_submission_md

        fake_v16 = {
            "protocol": "jevbench::v1.6.1-O1S-local",
            "n_items": 203,
            "intelligence": {"score": 47.25, "I_open": 47.25},
            "calibration": {"score": 88.40},
            "speed": {"p50_s_raw": 0.018, "p95_s_raw": 0.036, "p50_s_adjusted": 0.186, "score": 94.61},
            "cost": {"usd_per_1000": 0.0195, "usd_per_1m_in": 0.04, "usd_per_1m_out": 0.0, "score": 61.29},
            "capability": {"eligible": True, "capability": 67.825},
            "composite_option_a": 68.12,
            "per_tier_accuracy": {"easy": 1.0, "standard": 0.812, "hard": 0.491},
        }
        md = render_jevbench_submission_md(
            version="v1.3.0",
            hf_repo="gnumanth/nullspan",
            hf_revision="51acf9e097ade3fb1068b3511490ef2756485927",
            git_sha="de900ac4801c9d9f4b55abff68b3511490ef2756",
            backbone_desc="ModernBERT-large (395M encoder, encoder_classifier_<=0.6B)",
            v16_readout=fake_v16,
            training_disclosure="BoolQ, ARC-Challenge, SciQ, LocalLLaMA/typed-decisions train; zero JevBench items.",
            issue_ref="fstandhartinger/jevbench#238",
            snapshot_date="2026-11-11",
        )
        self.assertIn("gnumanth/nullspan", md)
        self.assertIn("51acf9e097ade3fb1068b3511490ef2756485927", md)
        self.assertIn("encoder_classifier_<=0.6B", md)
        self.assertIn("fstandhartinger/jevbench#238", md)
        self.assertIn("67.83", md)
        self.assertIn("zero JevBench", md)
        self.assertIn("$0.04", md)

    def test_score_items_v16_reads_provenance_tier_when_top_level_tier_absent(self) -> None:
        from score_v16 import score_items_v16

        items = [
            {
                "id": "dev-easy-1",
                "split": "public",
                "type": "choice",
                "labels": ["a", "b"],
                "expected": "a",
                "provenance": {"tier": "easy"},
            },
            {
                "id": "dev-hard-1",
                "split": "public",
                "type": "choice",
                "labels": ["a", "b"],
                "expected": "b",
                "provenance": {"tier": "hard"},
            },
        ]
        preds = {
            "dev-easy-1": {"probs": {"a": 0.9, "b": 0.1}, "latency_s": 0.02, "input_tokens": 100, "output_tokens": 0},
            "dev-hard-1": {"probs": {"a": 0.8, "b": 0.2}, "latency_s": 0.02, "input_tokens": 100, "output_tokens": 0},
        }
        res = score_items_v16(items, preds, usd_per_1m_in=0.04, endpoint_kind="gpu")
        self.assertIn("easy", res["per_tier_accuracy"])
        self.assertIn("hard", res["per_tier_accuracy"])
        self.assertAlmostEqual(res["per_tier_accuracy"]["easy"], 1.0, places=5)
        self.assertAlmostEqual(res["per_tier_accuracy"]["hard"], 0.0, places=5)

    def test_jevbench_submission_md_matches_public_v16_artifact(self) -> None:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        sub_p = root / "bench" / "jevbench_submission.md"
        self.assertTrue(sub_p.exists(), "bench/jevbench_submission.md must exist for Task 8")
        md = sub_p.read_text(encoding="utf-8")
        pub = json.loads((root / "bench" / "public_benchmarks_results.json").read_text(encoding="utf-8"))
        v16 = pub["jevbench_public"]["public_v16"]
        cap_str = f"{v16['capability']['capability']:.2f}"
        intel_str = f"{v16['intelligence']['score']:.2f}"
        cal_str = f"{v16['calibration']['score']:.2f}"
        self.assertIn("gnumanth/nullspan", md)
        self.assertIn("encoder_classifier_<=0.6B", md)
        self.assertIn("fstandhartinger/jevbench#238", md)
        self.assertIn("2026-11-11", md)
        self.assertIn(cap_str, md)
        self.assertIn(intel_str, md)
        self.assertIn(cal_str, md)
        self.assertIn("Zero JevBench items", md)


if __name__ == "__main__":
    unittest.main()
