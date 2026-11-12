import json
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestNullspanBenchIntegrity(unittest.TestCase):
    def test_every_ranked_row_is_measured_with_predictions_sha256(self):
        results_path = REPO_ROOT / "bench" / "results.json"
        data = json.loads(results_path.read_text(encoding="utf-8"))
        systems = data["systems"]
        self.assertGreaterEqual(len(systems), 12)

        hex64 = re.compile(r"^[0-9a-f]{64}$")
        nullspan_hashes = set()
        for row in systems:
            self.assertTrue(
                row.get("measured") is True,
                f"Ranked system {row.get('id')} must have measured=True",
            )
            sha = row.get("predictions_sha256", "")
            self.assertRegex(
                sha,
                hex64,
                f"Ranked system {row.get('id')} missing valid 64-hex predictions_sha256",
            )
            if row["id"].startswith("nullspan-"):
                nullspan_hashes.add(sha)

        # No ranked row outside nullspan-* may share a predictions hash with nullspan-*
        for row in systems:
            if not row["id"].startswith("nullspan-"):
                self.assertNotIn(
                    row["predictions_sha256"],
                    nullspan_hashes,
                    f"Non-nullspan row {row['id']} shares predictions hash with nullspan",
                )

    def test_no_simulated_perturbation_or_ranked_jevbench_anchors(self):
        run_bench_src = (REPO_ROOT / "bench" / "run_bench.py").read_text(encoding="utf-8")
        self.assertNotIn("N/A-guessed", run_bench_src)
        self.assertNotIn("rng_llm", run_bench_src)
        self.assertNotIn("ans_copy = dict(p_pt", run_bench_src)

        data = json.loads((REPO_ROOT / "bench" / "results.json").read_text(encoding="utf-8"))
        ranked_ids = {row["id"] for row in data["systems"]}
        for anchor_id in ("jev-1.13.0-ref", "h2o-lightning-4b", "basal-omni-1.0-ultra"):
            self.assertNotIn(
                anchor_id,
                ranked_ids,
                f"External JevBench anchor {anchor_id} must NOT appear in ranked systems",
            )

        anchors = data.get("reference_external_anchors", [])
        self.assertEqual(len(anchors), 3)
        for anc in anchors:
            self.assertNotIn("official_rank", anc)
            self.assertNotIn("by_domain", anc)
            self.assertNotIn("by_request_type", anc)
            self.assertIn("not comparable", anc.get("note", "").lower())

    def test_api_decoder_rows_if_present_are_backed_by_complete_real_cache(self):
        data = json.loads((REPO_ROOT / "bench" / "results.json").read_text(encoding="utf-8"))
        ranked_ids = {row["id"] for row in data["systems"]}
        api_ids = {"claude-haiku-4.5-json", "gpt-5.4-nano-json"} & ranked_ids
        self.assertIn(
            "claude-haiku-4.5-json",
            api_ids,
            "claude-haiku-4.5-json must be measured in results.json",
        )
        cache_path = REPO_ROOT / "bench" / "llm_decoder_cache.json"
        self.assertTrue(cache_path.exists(), "bench/llm_decoder_cache.json must exist")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for mid in sorted(api_ids):
            self.assertIn(mid, cache)
            self.assertEqual(len(cache[mid]), 500, f"Expected 500 cached item responses for {mid}")

    def test_sealed_pool_is_headline_and_open_pool_discloses_shared_generator(self):
        data = json.loads((REPO_ROOT / "bench" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data.get("headline_pool"), "sealed")
        self.assertIn("sealed_leaderboard", data)
        self.assertIn("open_leaderboard", data)
        self.assertIn("generate_dataset.py", data.get("open_pool_disclosure", ""))

        report_md = (REPO_ROOT / "bench" / "SCIENCE_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("Sealed Pool", report_md)
        self.assertIn("Open Pool", report_md)
        self.assertIn("generate_dataset.py", report_md)
        self.assertIn("not comparable", report_md.lower())

        readme_md = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Sealed Pool", readme_md)
        self.assertIn("generate_dataset.py", readme_md)
        self.assertNotIn("H2O-Lightning-4B v1.1 (JevBench #1 Comp) | `72.5`", readme_md)

    def test_squad_v2_reconciled_and_full_validation_recorded(self):
        pub = json.loads((REPO_ROOT / "bench" / "public_benchmarks_results.json").read_text(encoding="utf-8"))
        self.assertIn("squad_v2", pub)
        self.assertIn("squad_v2_full", pub)
        self.assertGreaterEqual(pub["squad_v2_full"]["n_evaluated"], 11800)

        yaml_txt = (REPO_ROOT / "artifacts" / "nullspan" / ".eval_results" / "nullspan-bench.yaml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("value: 0.87\n", yaml_txt)
        self.assertNotIn("0.8936", yaml_txt)
        full_em_ratio = round(pub["squad_v2_full"]["exact"] / 100.0, 4)
        self.assertIn(str(full_em_ratio), yaml_txt)

        for rel in ("docs/bench.html", "space/bench.html", "docs/index.html", "space/index.html"):
            p = REPO_ROOT / rel
            if p.exists():
                txt = p.read_text(encoding="utf-8")
                self.assertNotIn("Bayes ceiling", txt, f"Stale Bayes ceiling claim in {rel}")
                self.assertNotIn("#1 Hub widget", txt, f"Stale #1 Hub widget claim in {rel}")

    def test_llm_decoder_prompt_and_normalization(self):
        from bench.eval_llm_decoders import build_item_prompt, normalize_llm_prediction

        rec_null = {
            "doc_id": "inc-1",
            "field_name": "cve_id",
            "state": "Incident INC-1: CVE not yet assigned.",
            "field_spec": {"type": "text", "instructions": "Assigned CVE ID"},
        }
        prompt = build_item_prompt(rec_null)
        self.assertIn("cve_id", prompt)
        pred_null = normalize_llm_prediction(
            {"is_null": True, "p_null": 0.96, "value_str": None, "value_num": None, "value_bool": None, "value_list_str": None, "value_list_num": None, "p10": None, "p90": None, "verified": None},
            rec_null["field_spec"],
        )
        self.assertIsNone(pred_null["value"])
        self.assertAlmostEqual(pred_null["p_null"], 0.96)

        pred_num = normalize_llm_prediction(
            {"is_null": False, "p_null": 0.04, "value_str": None, "value_num": 1450.5, "value_bool": None, "value_list_str": None, "value_list_num": None, "p10": None, "p90": None, "verified": True},
            {"type": "number", "constraint": "subtotal == qty * price"},
        )
        self.assertEqual(pred_num["value"], 1450.5)
        self.assertIs(pred_num["verified"], True)


if __name__ == "__main__":
    unittest.main()
