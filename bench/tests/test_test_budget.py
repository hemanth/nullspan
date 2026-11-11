import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestBudgetAndOofGate(unittest.TestCase):
    def test_typed_decisions_refuses_unknown_or_mismatched_ledger_before_dataset_load(self):
        import bench.eval_public_benchmarks as epb

        with mock.patch("datasets.load_dataset") as mocked_load:
            mocked_load.side_effect = AssertionError("load_dataset must NOT be called before ledger check")

            # 1. Missing --ledger-entry
            with self.assertRaises(SystemExit):
                epb.main(["--suite", "typed_decisions"])

            # 2. Unknown ledger id
            with self.assertRaises(SystemExit):
                epb.main(["--suite", "typed_decisions", "--ledger-entry", "nonexistent-id"])

            # 3. Historical entry cannot be reused
            with self.assertRaises(SystemExit):
                epb.main(["--suite", "typed_decisions", "--ledger-entry", "hist-v1.3.0"])

            # 4. Empty / PENDING slot cannot be used without matching git SHA
            with self.assertRaises(SystemExit):
                epb.main(["--suite", "typed_decisions", "--ledger-entry", "slot-1"])

            self.assertEqual(mocked_load.call_count, 0)

    def test_valid_ledger_entry_with_matching_git_sha_passes_guard(self):
        import bench.eval_public_benchmarks as epb

        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tf:
            tf.write(
                "# Test Ledger\n\n"
                "| ID | Git SHA | Status | OOF Acc | OOF KL | Test Acc | Notes |\n"
                "|---|---|---|---|---|---|---|\n"
                f"| slot-1 | {head_sha} | READY | 0.8180 | 0.0810 | PENDING | Phase 1 RC |\n"
            )
            ledger_path = tf.name
        try:
            entry = epb.verify_ledger_entry("slot-1", ledger_path=ledger_path, expected_sha=head_sha)
            self.assertEqual(entry["id"], "slot-1")
            self.assertEqual(entry["git_sha"], head_sha)
        finally:
            os.remove(ledger_path)

    def test_jevbench_requires_jevbench_dir_and_has_no_hardcoded_user_path(self):
        import bench.eval_public_benchmarks as epb

        src = (REPO_ROOT / "bench/eval_public_benchmarks.py").read_text(encoding="utf-8")
        self.assertNotIn("/Users/lika/.gemini", src)
        self.assertNotIn("def train_and_eval_typed_decisions", src)

        env_clean = {k: v for k, v in os.environ.items() if k != "JEVBENCH_DIR"}
        with mock.patch.dict(os.environ, env_clean, clear=True):
            with self.assertRaises(SystemExit):
                epb.main(["--suite", "jevbench"])

    def test_train_systemone_exists_and_never_touches_test_split(self):
        train_file = REPO_ROOT / "training/train_systemone.py"
        self.assertTrue(train_file.exists(), "training/train_systemone.py must exist")
        src = train_file.read_text(encoding="utf-8")
        self.assertNotIn('split="test"', src)
        self.assertNotIn("split='test'", src)

        from training.train_systemone import build_oof_report

        dummy_decisions = [
            {
                "case_idx": 0,
                "workflow": "invoice_processing",
                "qid": "action",
                "qtype": "choice",
                "state_tokens": 410,
                "probs": {"pay": 0.8, "hold": 0.2},
                "gold_probs": {"pay": 1.0, "hold": 0.0},
                "correct": True,
                "kl": 0.2231,
                "brier": 0.08,
                "confidence": 0.8,
            },
            {
                "case_idx": 0,
                "workflow": "invoice_processing",
                "qid": "valid",
                "qtype": "noul",
                "state_tokens": 410,
                "probs": {"true": 0.3, "false": 0.7},
                "gold_probs": {"true": 1.0, "false": 0.0},
                "correct": False,
                "kl": 1.2040,
                "brier": 0.98,
                "confidence": 0.7,
            },
        ]
        rep = build_oof_report(dummy_decisions)
        for key in (
            "oof_accuracy",
            "oof_kl",
            "oof_brier",
            "oof_ece",
            "per_type",
            "per_workflow",
            "per_decision",
        ):
            self.assertIn(key, rep)
        self.assertAlmostEqual(rep["oof_accuracy"], 0.5)
        self.assertEqual(len(rep["per_decision"]), 2)

    def test_compare_oof_paired_cluster_bootstrap_and_kl_regression_gate(self):
        from bench.compare_oof import compare_oof_reports

        base_decisions = []
        better_decisions = []
        kl_regressed_decisions = []
        for c_idx in range(60):
            for q_idx in range(5):
                qid = f"q{q_idx}"
                base_ok = (c_idx % 4) != 0  # 0.75 acc
                better_ok = (c_idx % 10) != 0  # 0.90 acc
                base_decisions.append(
                    {
                        "case_idx": c_idx,
                        "qid": qid,
                        "correct": base_ok,
                        "kl": 0.12 if base_ok else 0.45,
                        "brier": 0.05 if base_ok else 0.25,
                    }
                )
                better_decisions.append(
                    {
                        "case_idx": c_idx,
                        "qid": qid,
                        "correct": better_ok,
                        "kl": 0.08 if better_ok else 0.35,
                        "brier": 0.03 if better_ok else 0.20,
                    }
                )
                # Review Focus #5: slightly higher accuracy, but severe KL blow-up
                kl_regressed_decisions.append(
                    {
                        "case_idx": c_idx,
                        "qid": qid,
                        "correct": better_ok,
                        "kl": 0.55,
                        "brier": 0.18,
                    }
                )

        res_good = compare_oof_reports(
            {"per_decision": base_decisions},
            {"per_decision": better_decisions},
            n_boot=2000,
            seed=42,
        )
        self.assertTrue(res_good["accepted"])
        self.assertGreater(res_good["accuracy"]["ci95"][0], 0.0)
        self.assertLess(res_good["kl"]["ci95"][1], 0.0)

        res_bad_kl = compare_oof_reports(
            {"per_decision": base_decisions},
            {"per_decision": kl_regressed_decisions},
            n_boot=2000,
            seed=42,
        )
        self.assertFalse(
            res_bad_kl["accepted"],
            "Must reject candidate that improves accuracy while significantly regressing KL",
        )

    def test_launch_script_bundles_train_systemone_and_defaults_to_train(self):
        script = (REPO_ROOT / "training/launch_gcp_public_bench.sh").read_text(encoding="utf-8")
        self.assertIn("training/train_systemone.py", script)
        self.assertIn("bench/TEST_LEDGER.md", script)
        self.assertIn("bench/compare_oof.py", script)
        self.assertIn('MODE="${MODE:-train}"', script)


if __name__ == "__main__":
    unittest.main()
