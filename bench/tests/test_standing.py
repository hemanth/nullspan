import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.standing import load, rank  # noqa: E402


class TestStanding(unittest.TestCase):
    def test_sub1b_accuracy_rank(self):
        rows = load(str(REPO_ROOT / "bench/leaderboards/typed_decisions_2026-11-11.json"))
        fitted = [x for x in rows if x["mode"] == "fitted"]
        r = rank(fitted, "accuracy", True, max_params_m=1000)
        self.assertEqual(r[0][1], "sysone-typed-decisions-0.4B")
        self.assertEqual([m for _, m, _ in r].index("nullspan"), 4)
        nullspan_entry = next(entry for entry in r if entry[1] == "nullspan")
        self.assertEqual(nullspan_entry[0], 5)
        self.assertAlmostEqual(nullspan_entry[2], 0.790)

        overall = rank(fitted, "accuracy", True)
        self.assertEqual(overall[0][1], "ZhiPan-Z3-Pro")
        for row in rows:
            self.assertFalse(
                row["model"].startswith("#"),
                f"Discussion tag parsed as model name: {row['model']}",
            )

    def test_null_metric_is_skipped(self):
        rows = [
            {"model": "a", "params_m": 1, "kl": None},
            {"model": "b", "params_m": 1, "kl": 0.1},
        ]
        self.assertEqual([m for _, m, _ in rank(rows, "kl", False)], ["b"])

    def test_sub1b_and_overall_kl_and_brier_ranks_with_ties(self):
        rows = load(str(REPO_ROOT / "bench/leaderboards/typed_decisions_2026-11-11.json"))
        fitted = [x for x in rows if x["mode"] == "fitted"]

        kl_sub = rank(fitted, "kl", False, max_params_m=1000)
        self.assertEqual(kl_sub[0][1], "nullspan")
        self.assertEqual(kl_sub[0][0], 1)

        kl_all = rank(fitted, "kl", False)
        self.assertEqual(kl_all[0][1], "pramana-2b")
        ns_kl_all = next(e for e in kl_all if e[1] == "nullspan")
        self.assertEqual(ns_kl_all[0], 2)

        brier_sub = rank(fitted, "brier", False, max_params_m=1000)
        self.assertEqual(brier_sub[0][1], "nullspan")
        self.assertEqual(brier_sub[0][0], 1)

        brier_all = rank(fitted, "brier", False)
        self.assertEqual(brier_all[0][1], "pramana-2b")
        ns_brier_all = next(e for e in brier_all if e[1] == "nullspan")
        m3_brier_all = next(
            e for e in brier_all if e[1] == "ministral3-3b-reasoning-typed-decisions"
        )
        self.assertEqual(ns_brier_all[0], 2)
        self.assertEqual(m3_brier_all[0], 2)

    def test_unknown_params_excluded_from_sub1b_but_kept_overall(self):
        rows = [
            {"model": "closed-api", "params_m": None, "ece": 0.019},
            {"model": "small-enc", "params_m": 395.0, "ece": 0.1665},
        ]
        sub = rank(rows, "ece", False, max_params_m=1000)
        overall = rank(rows, "ece", False, max_params_m=None)
        self.assertEqual([m for _, m, _ in sub], ["small-enc"])
        self.assertEqual([m for _, m, _ in overall], ["closed-api", "small-enc"])

    def test_cli_outputs_dated_standing_and_handles_unlisted_model(self):
        td_snap = str(REPO_ROOT / "bench/leaderboards/typed_decisions_2026-11-11.json")
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "bench/standing.py"), "--snapshot", td_snap, "--model", "nullspan"],
            capture_output=True,
            text=True,
            check=True,
        )
        out = proc.stdout
        self.assertIn("2026-11-11", out)
        self.assertIn("accuracy", out)
        self.assertIn("kl", out)
        self.assertIn("brier", out)
        self.assertIn("ece", out)
        self.assertIn("sysone-typed-decisions-0.4B", out)

        jb_snap = str(REPO_ROOT / "bench/leaderboards/jevbench_v1.6.1_2026-11-11.json")
        proc_jb = subprocess.run(
            [sys.executable, str(REPO_ROOT / "bench/standing.py"), "--snapshot", jb_snap, "--model", "nullspan"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("unlisted", proc_jb.stdout.lower())

    def test_readme_standing_claims_are_dated_and_truthful(self):
        banned_phrases = [
            "worldwide",
            "82.05%",
            ".eval_results` widget",
            "#1 Hub widget",
            "Arbiter-4B` (`78.6%`)",
            "Arbiter-4B` `78.6%`",
        ]
        for rel_path in [
            "README.md",
            "python/README.md",
            "js/README.md",
            "artifacts/nullspan/README.md",
        ]:
            content = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
            self.assertIn("2026-11-11", content, f"{rel_path} must cite snapshot date 2026-11-11")
            for phrase in banned_phrases:
                self.assertNotIn(phrase, content, f"Stale claim {phrase!r} found in {rel_path}")


if __name__ == "__main__":
    unittest.main()
