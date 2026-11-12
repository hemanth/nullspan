import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
JEVBENCH_TMP = Path("/tmp/jevbench_v161")


class TestSynthJevCases(unittest.TestCase):
    def test_planned_jev_mix_matches_v15_ratios_within_one_item(self):
        from training.synth_jev_cases import HARD_FAMILIES, build_jev_allocation_plan

        plan = build_jev_allocation_plan(n_total=400)
        self.assertEqual(len(plan), 400)

        by_type = {}
        by_tier = {}
        families = set()
        for spec in plan:
            by_type[spec["qtype"]] = by_type.get(spec["qtype"], 0) + 1
            by_tier[spec["tier"]] = by_tier.get(spec["tier"], 0) + 1
            families.add(spec["family"])

        self.assertLessEqual(abs(by_type.get("choice", 0) - 200), 1)
        self.assertLessEqual(abs(by_type.get("noul", 0) - 100), 1)
        self.assertLessEqual(abs(by_type.get("score", 0) - 100), 1)

        self.assertLessEqual(abs(by_tier.get("easy", 0) - 56), 1)
        self.assertLessEqual(abs(by_tier.get("standard", 0) - 112), 1)
        self.assertLessEqual(abs(by_tier.get("judge", 0) - 112), 1)
        self.assertLessEqual(abs(by_tier.get("hard", 0) - 120), 1)

        for hf in HARD_FAMILIES:
            self.assertIn(hf, families, f"Missing hard family {hf} in 400-item plan")
        self.assertIn("state_tracking", families)
        self.assertIn("tool_call_guardrail", families)
        self.assertIn("lead_qualification", families)

    def test_generated_records_load_with_jevbench_load_jsonl_and_score_task(self):
        from training.synth_jev_cases import format_jev_task_record

        records = [
            format_jev_task_record(
                item_id="dev-choice-001",
                tier="hard",
                family="probability",
                state="Historical cohort of 20 matching tickets: 14 resolved via tier2_network, 4 via billing, 2 via auth.",
                question={
                    "type": "choice",
                    "instructions": "Select the most likely resolution queue and assign empirical probabilities.",
                    "criteria": {
                        "tier2_network": "Network engineering queue.",
                        "billing": "Billing adjustments queue.",
                        "auth": "Identity and access queue.",
                    },
                },
                expected="tier2_network",
                author_model="claude-haiku-4-5",
                rationale="14/20 = 0.70, 4/20 = 0.20, 2/20 = 0.10.",
                gold_probs={"tier2_network": 0.70, "billing": 0.20, "auth": 0.10},
            ),
            format_jev_task_record(
                item_id="dev-noul-001",
                tier="standard",
                family="tool_call_guardrail",
                state={"tool": "drop_database", "env": "production", "approval_ticket": None},
                question={
                    "type": "noul",
                    "instructions": "Should this tool execution be allowed under production change control?",
                    "criteria": {
                        "false": "Block execution because approval or safety checks fail.",
                        "true": "Allow execution because all safety checks pass.",
                    },
                },
                expected="no",
                author_model="claude-haiku-4-5",
                rationale="Destructive production call lacks an approval ticket.",
            ),
            format_jev_task_record(
                item_id="dev-score-001",
                tier="standard",
                family="lead_qualification",
                state={"company_size": 450, "budget_confirmed": True, "timeline_days": 21},
                question={
                    "type": "score",
                    "instructions": "Score enterprise lead readiness.",
                    "criteria": [
                        "0 - Disqualified",
                        "1 - Cold",
                        "2 - Marketing qualified",
                        "3 - Sales ready",
                    ],
                },
                expected=3,
                author_model="claude-haiku-4-5",
                rationale="Confirmed budget + 450 employees + 21-day timeline = level 3.",
            ),
        ]

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dev_sample.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

            if JEVBENCH_TMP.exists():
                if str(JEVBENCH_TMP) not in sys.path:
                    sys.path.insert(0, str(JEVBENCH_TMP))
                from jevbench.scoring import score_task
                from jevbench.tasks import load_jsonl
                from jevbench.validate_data import validate_paths

                loaded = load_jsonl(str(p))
                self.assertEqual(len(loaded), 3)
                _, problems = validate_paths([str(p)])
                self.assertEqual(problems, [])

                s0 = score_task({"tier2_network": 0.7, "billing": 0.2, "auth": 0.1}, loaded[0])
                self.assertTrue(s0["valid"])
                self.assertTrue(s0["correct"])
                s1 = score_task({"no": 0.9, "yes": 0.1}, loaded[1])
                self.assertTrue(s1["correct"])
                s2 = score_task({"0": 0.02, "1": 0.03, "2": 0.15, "3": 0.80}, loaded[2])
                self.assertTrue(s2["correct"])
            else:
                self.assertEqual(len(records), 3)

    def test_shingle_guard_rejects_12_token_overlap_with_jevbench_public(self):
        from training.synth_jev_cases import (
            build_jev_public_shingles_from_records,
            has_jev_shingle_overlap,
        )

        pub_item = {
            "state": (
                "Under Section 14.2 of the Meridian Cloud Enterprise Master Addendum, "
                "scheduled maintenance windows never count toward monthly uptime credits."
            ),
            "question": {"instructions": "Is the customer eligible for a service credit?"},
        }
        shingles = build_jev_public_shingles_from_records([pub_item], k=12)
        self.assertTrue(
            has_jev_shingle_overlap(
                {
                    "state": (
                        "Note: Under Section 14.2 of the Meridian Cloud Enterprise Master Addendum, "
                        "scheduled maintenance windows never apply here."
                    ),
                    "question": {"instructions": "Decide refund."},
                },
                shingles,
                k=12,
            )
        )
        self.assertFalse(
            has_jev_shingle_overlap(
                {
                    "state": "Northwind Logistics freight tariff clause 9 caps demurrage at USD 450 per day.",
                    "question": {"instructions": "Does the invoice exceed the demurrage cap?"},
                },
                shingles,
                k=12,
            )
        )

    def test_dual_teacher_agreement_filter_handles_standard_and_ambiguous_items(self):
        from training.synth_jev_cases import resolve_dual_teacher_runs

        normal_item = {
            "family": "multi_hop",
            "labels": ["approve", "deny", "escalate"],
            "question": {"type": "choice"},
        }
        kept_agree = resolve_dual_teacher_runs(
            normal_item,
            {"approve": 0.70, "deny": 0.20, "escalate": 0.10},
            {"approve": 0.60, "deny": 0.25, "escalate": 0.15},
        )
        self.assertIsNotNone(kept_agree)
        self.assertEqual(kept_agree["expected"], "approve")

        dropped_disagree = resolve_dual_teacher_runs(
            normal_item,
            {"approve": 0.70, "deny": 0.20, "escalate": 0.10},
            {"approve": 0.20, "deny": 0.65, "escalate": 0.15},
        )
        self.assertIsNone(dropped_disagree)

        ambig_item = {
            "family": "ambiguous",
            "labels": ["approve", "deny", "no_clear_answer"],
            "question": {"type": "choice"},
        }
        mapped_ambig = resolve_dual_teacher_runs(
            ambig_item,
            {"approve": 0.65, "deny": 0.25, "no_clear_answer": 0.10},
            {"approve": 0.20, "deny": 0.70, "no_clear_answer": 0.10},
        )
        self.assertIsNotNone(mapped_ambig)
        self.assertEqual(mapped_ambig["expected"], "no_clear_answer")
        self.assertGreater(mapped_ambig["soft_probs"]["no_clear_answer"], 0.5)

    def test_long_policy_state_reaches_2k_to_6k_tokens_and_preserves_governing_clauses(self):
        from training.synth_jev_cases import expand_long_policy_state
        from training.synth_td_cases import tokenize_for_shingles

        core_sections = [
            "Governing Clause A: Cold-chain biologics delayed beyond 14 hours require QA quarantine sign-off.",
            "Exception Clause B: Shipments equipped with validated dry-ice logger telemetry up to 22 hours are exempt unless excursion exceeds -60C.",
            "Case Facts: Lot BIO-884 arrived at 18.5 hours with continuous logger telemetry peaking at -68C.",
        ]
        doc = expand_long_policy_state(
            core_sections=core_sections,
            domain_title="Global Biologics Cold-Chain SOP",
            target_tokens=2800,
            seed=19,
        )
        n_toks = len(tokenize_for_shingles(doc))
        self.assertGreaterEqual(n_toks, 2000)
        self.assertLessEqual(n_toks, 6000)
        for cs in core_sections:
            self.assertIn(cs, doc)

    def test_blind_reviewer_and_dev_tier_alignment_gate(self):
        from training.synth_jev_cases import check_dev_tier_alignment, verify_dev_item_blind

        rec = {
            "id": "dev-010",
            "family": "trap",
            "state": "Account tier is Silver. Waiver applies only to Gold.",
            "question": {
                "type": "noul",
                "instructions": "Does fee waiver apply?",
                "criteria": {"false": "No waiver", "true": "Waiver applies"},
            },
            "labels": ["no", "yes"],
            "expected": "no",
            "provenance": {"tier": "hard", "author_model": "claude-haiku-4-5"},
        }
        accepted = verify_dev_item_blind(
            rec,
            reviewer_fn=lambda st, q, lbs: {"predicted": "no", "probs": {"no": 0.9, "yes": 0.1}},
            reviewer_model="Qwen/Qwen3-4B-Instruct-2507",
        )
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted["provenance"]["reviewer_model"], "Qwen/Qwen3-4B-Instruct-2507")

        rejected = verify_dev_item_blind(
            rec,
            reviewer_fn=lambda st, q, lbs: {"predicted": "yes", "probs": {"no": 0.2, "yes": 0.8}},
            reviewer_model="Qwen/Qwen3-4B-Instruct-2507",
        )
        self.assertIsNone(rejected)

        ok_align = check_dev_tier_alignment(
            {"easy": 0.964, "standard": 0.705, "hard": 0.383},
            ref_accs={"easy": 1.00, "standard": 0.6806, "hard": 0.3604},
            tol=0.10,
        )
        self.assertTrue(ok_align["usable"])

        bad_align = check_dev_tier_alignment(
            {"easy": 0.964, "standard": 0.830, "hard": 0.383},
            ref_accs={"easy": 1.00, "standard": 0.6806, "hard": 0.3604},
            tol=0.10,
        )
        self.assertFalse(bad_align["usable"])

    def test_select_reviewed_dev_cells_from_pool_falls_back_to_second_candidate(self):
        from training.synth_jev_cases import select_reviewed_dev_cells_from_pool

        c0_bad = {
            "id": "jev-hard-noul-00000",
            "cell_index": 0,
            "family": "trap",
            "state": "S1",
            "question": {"type": "noul", "instructions": "Q?", "criteria": {"false": "N", "true": "Y"}},
            "labels": ["no", "yes"],
            "expected": "no",
            "provenance": {"tier": "hard", "author_model": "claude-haiku-4-5"},
        }
        c0_good = {
            "id": "jev-hard-noul-00000",
            "cell_index": 0,
            "family": "trap",
            "state": "S2",
            "question": {"type": "noul", "instructions": "Q2?", "criteria": {"false": "N", "true": "Y"}},
            "labels": ["no", "yes"],
            "expected": "yes",
            "provenance": {"tier": "hard", "author_model": "claude-haiku-4-5"},
        }
        reviews = [{"predicted": "yes"}, {"predicted": "yes"}]
        accepted, missing = select_reviewed_dev_cells_from_pool(
            candidates=[c0_bad, c0_good],
            review_outputs=reviews,
            expected_indices=[0, 1],
            reviewer_model="microsoft/Phi-4-mini-instruct",
        )
        self.assertIn(0, accepted)
        self.assertEqual(accepted[0]["state"], "S2")
        self.assertNotIn("cell_index", accepted[0])
        self.assertEqual(accepted[0]["provenance"]["reviewer_model"], "microsoft/Phi-4-mini-instruct")
        self.assertEqual(missing, [1])

    def test_family_c_train_generator_covers_all_families_and_passes_jevbench_validation(self):
        import random
        from training.synth_jev_cases import (
            build_jev_allocation_plan,
            has_jev_shingle_overlap,
            load_jevbench_public_shingles,
        )
        from training.synth_jev_train import (
            build_family_c_candidate,
            default_family_c_seed_bank,
            parse_rationale_teacher_output,
        )
        from training.synth_td_cases import tokenize_for_shingles

        plan = build_jev_allocation_plan(n_total=120)
        bank = default_family_c_seed_bank()
        rng = random.Random(42)
        shingles, _ = (
            load_jevbench_public_shingles(JEVBENCH_TMP, k=12)
            if JEVBENCH_TMP.exists()
            else (set(), 0)
        )

        recs = [build_family_c_candidate(sp, bank, rng, author_model="Qwen/Qwen3-8B") for sp in plan]
        self.assertEqual(len(recs), 120)

        seen_fams = {r["family"] for r in recs}
        self.assertGreaterEqual(len(seen_fams), 15)

        for r in recs:
            if shingles:
                self.assertFalse(has_jev_shingle_overlap(r, shingles, k=12))
            if r["family"] == "long_policy":
                n_toks = len(tokenize_for_shingles(r["state"]))
                self.assertGreaterEqual(n_toks, 2000)
                self.assertLessEqual(n_toks, 6000)
            if r["family"] == "probability":
                gp = (r.get("provenance") or {}).get("gold_probs") or r.get("gold_probs")
                self.assertIsNotNone(gp)
                self.assertAlmostEqual(sum(gp.values()), 1.0, places=4)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train_sample.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
            if JEVBENCH_TMP.exists():
                if str(JEVBENCH_TMP) not in sys.path:
                    sys.path.insert(0, str(JEVBENCH_TMP))
                from jevbench.validate_data import validate_paths

                _, problems = validate_paths([str(p)])
                self.assertEqual(problems, [])

        raw_out = (
            '<think>internal</think>{"rationale": "Cohort A has 18/24 = 0.75 vs 6/24 = 0.25.", '
            '"probs": {"alpha": 0.75, "beta": 0.25}}'
        )
        parsed = parse_rationale_teacher_output(raw_out, ["alpha", "beta"])
        self.assertIn("18/24", parsed["rationale"])
        self.assertAlmostEqual(parsed["probs"]["alpha"], 0.75, places=2)
        self.assertAlmostEqual(sum(parsed["probs"].values()), 1.0, places=5)

    def test_generate_and_distil_jev_train_reuses_single_teacher_runner_across_rounds(self):
        from training.synth_jev_train import generate_and_distil_jev_train

        init_calls = []

        class FakeTeacherRunner:
            def __init__(self, teacher_model: str, temperature: float = 0.7, batch_size: int = 32):
                init_calls.append(teacher_model)

            def distil_batch(self, records):
                outs = []
                for r in records:
                    lbs = [str(x) for x in r["labels"]]
                    exp = str(r["expected"])
                    p = {lb: (0.85 if lb == exp else 0.15 / max(1, len(lbs) - 1)) for lb in lbs}
                    outs.append(({"rationale": "ok", "probs": p}, {"rationale": "ok2", "probs": p}))
                return outs

        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "train.jsonl"
            manifest = generate_and_distil_jev_train(
                out_path=out_p,
                jevbench_dir=Path(td),
                n_target=600,
                teacher_model="Qwen/Qwen3-8B",
                max_round_batch=260,
                runner_cls=FakeTeacherRunner,
            )
            self.assertEqual(manifest["n_items"], 600)
            self.assertEqual(init_calls, ["Qwen/Qwen3-8B"])

    def test_format_jev_task_record_parses_prefixed_score_and_schedules_missing_copies(self):
        from training.synth_jev_cases import (
            build_todo_specs_for_pool_round,
            format_jev_task_record,
        )

        rec = format_jev_task_record(
            item_id="jev-standard-score-00012",
            tier="standard",
            family="lead_qualification",
            state={"seats": 150, "budget": True},
            question={
                "type": "score",
                "instructions": "Score readiness 0..3.",
                "criteria": ["0 - Low", "1 - Fair", "2 - Good", "3 - Ready"],
            },
            expected="2 - Good",
            author_model="claude-haiku-4-5",
        )
        self.assertEqual(rec["expected"], 2)

        plan = [{"index": 0, "qtype": "choice", "tier": "easy", "family": "routing"},
                {"index": 1, "qtype": "noul", "tier": "hard", "family": "trap"}]
        todo = build_todo_specs_for_pool_round(plan, counts_by_cell={0: 2, 1: 0}, candidates_per_cell=3)
        # Cell 0 needs 1 more; Cell 1 needs 3 -> total 4 scheduled specs in one round!
        self.assertEqual([sp["index"] for sp in todo].count(0), 1)
        self.assertEqual([sp["index"] for sp in todo].count(1), 3)

    def test_family_c_candidates_never_leak_expected_label_or_verbatim_rubric_into_state(self):
        import random
        from training.synth_jev_cases import build_jev_allocation_plan
        from training.synth_jev_train import build_family_c_candidate, default_family_c_seed_bank

        plan = build_jev_allocation_plan(n_total=120)
        bank = default_family_c_seed_bank()
        rng = random.Random(101)
        for sp in plan:
            rec = build_family_c_candidate(sp, bank, rng, author_model="Qwen/Qwen3-8B")
            if rec["question"]["type"] != "choice" or rec["family"] in ("probability", "tradeoff", "multi_hop", "state_tracking"):
                continue
            state_blob = rec["state"] if isinstance(rec["state"], str) else json.dumps(rec["state"])
            exp = str(rec["expected"])
            crit_map = rec["question"]["criteria"]
            self.assertNotIn(exp, state_blob, f"Leaked option key {exp} into state for family={rec['family']}")
            self.assertNotIn(
                crit_map[exp],
                state_blob,
                f"Leaked verbatim rubric into state for family={rec['family']}",
            )

    def test_format_jev_task_record_parses_json_string_criteria_and_case_insensitive_expected(self):
        from training.synth_jev_cases import format_jev_task_record

        rec = format_jev_task_record(
            item_id="jev-standard-choice-00045",
            tier="standard",
            family="policy",
            state='{"ticket_id": "T-10", "hours_elapsed": 14}',
            question={
                "type": "choice",
                "instructions": "Select the policy action.",
                "criteria": '{"approve_credit": "Approve standard credit.", "deny_late": "Deny late filing."}',
            },
            expected="Approve_Credit",
            author_model="claude-haiku-4-5",
        )
        self.assertEqual(rec["expected"], "approve_credit")
        self.assertEqual(rec["labels"], ["approve_credit", "deny_late"])
        self.assertIsInstance(rec["state"], dict)

    def test_review_dev_pool_hf_allow_partial_writes_missing_cells_json(self):
        from unittest.mock import patch
        from training.synth_jev_cases import review_dev_pool_hf

        with tempfile.TemporaryDirectory() as td:
            pool_p = Path(td) / "pool.jsonl"
            out_p = Path(td) / "dev.jsonl"
            cand0 = {
                "id": "jev-easy-choice-00000",
                "cell_index": 0,
                "family": "routing",
                "state": "Ticket T1",
                "question": {"type": "choice", "instructions": "Route?", "criteria": {"a": "A", "b": "B"}},
                "labels": ["a", "b"],
                "expected": "a",
                "split": "public",
                "group": "jev-easy-choice-00000",
                "provenance": {"tier": "easy", "author_model": "claude-haiku-4-5"},
            }
            pool_p.write_text(json.dumps(cand0) + "\n", encoding="utf-8")
            with patch("training.synth_jev_cases.HFBlindReviewer") as mock_rev:
                mock_rev.return_value.review_batch.return_value = [{"predicted": "a"}]
                res = review_dev_pool_hf(
                    pool_path=pool_p,
                    out_path=out_p,
                    jevbench_dir=Path(td),
                    n_total=2,
                    allow_partial=True,
                )
            self.assertEqual(res["missing_cells"], [1])
            self.assertTrue((Path(td) / "missing_cells.json").exists())

    def test_hf_blind_reviewer_length_sorts_isolates_long_prompts_and_recovers_from_oom(self):
        import torch
        from training.synth_jev_cases import HFBlindReviewer

        class FakeTok:
            pad_token_id = 0

            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
                return msgs[0]["content"]

            def __call__(self, chunk, return_tensors="pt", padding=True, truncation=True, max_length=6144):
                class FakeEnc(dict):
                    def to(self, _dev):
                        return self

                lens = [min(max_length, max(4, len(c) // 4)) for c in chunk]
                max_l = max(lens)
                ids = torch.ones((len(chunk), max_l), dtype=torch.long)
                for i, c in enumerate(chunk):
                    ids[i, 0] = 101 if "EXPECT_ALPHA" in c else 102
                return FakeEnc(input_ids=ids, attention_mask=torch.ones_like(ids))

            def decode(self, ids, skip_special_tokens=True):
                return '{"reason": "ok", "predicted": "alpha"}' if int(ids[0]) == 101 else '{"reason": "ok", "predicted": "beta"}'

        seen_batch_shapes = []
        oom_triggered = {"done": False}

        class FakeModel:
            def generate(self, **kwargs):
                inp = kwargs["input_ids"]
                bsz, seq_l = inp.shape
                seen_batch_shapes.append((bsz, seq_l))
                # Simulate the exact L4 failure: batch >= 2 with seq_l > 1500 OR first batch >= 4 OOMs!
                if bsz >= 2 and seq_l > 1500:
                    raise RuntimeError("CUDA out of memory: long prompt was batched with short prompts!")
                if bsz >= 4 and not oom_triggered["done"]:
                    oom_triggered["done"] = True
                    raise RuntimeError("CUDA out of memory. Tried to allocate 778.00 MiB.")
                gen_col = inp[:, :1].clone()
                return torch.cat([inp, gen_col], dim=1)

        rev = object.__new__(HFBlindReviewer)
        rev.model_id = "fake"
        rev.device = "cpu"
        rev.tok = FakeTok()
        rev.model = FakeModel()

        records = [
            {"state": "short_0 EXPECT_ALPHA", "question": {"type": "choice", "instructions": "Q", "criteria": {"alpha": "A", "beta": "B"}}, "labels": ["alpha", "beta"]},
            {"state": ("LONG_POLICY " * 900) + " EXPECT_BETA", "question": {"type": "choice", "instructions": "Q", "criteria": {"alpha": "A", "beta": "B"}}, "labels": ["alpha", "beta"]},
            {"state": "short_2 EXPECT_BETA", "question": {"type": "choice", "instructions": "Q", "criteria": {"alpha": "A", "beta": "B"}}, "labels": ["alpha", "beta"]},
            {"state": "short_3 EXPECT_ALPHA", "question": {"type": "choice", "instructions": "Q", "criteria": {"alpha": "A", "beta": "B"}}, "labels": ["alpha", "beta"]},
            {"state": "short_4 EXPECT_BETA", "question": {"type": "choice", "instructions": "Q", "criteria": {"alpha": "A", "beta": "B"}}, "labels": ["alpha", "beta"]},
        ]
        outs = rev.review_batch(records, batch_size=4)
        self.assertEqual([o["predicted"] for o in outs], ["alpha", "beta", "beta", "alpha", "beta"])
        self.assertTrue(oom_triggered["done"])
        # Verify the long prompt ran with batch_size == 1!
        long_batches = [bsz for bsz, seq_l in seen_batch_shapes if seq_l > 1500]
        self.assertEqual(long_batches, [1])

    def test_dual_run_teacher_runner_selects_auto_dtype_for_fp8_and_caps_prompt_length(self):
        from training.synth_jev_train import select_vllm_dtype_for_teacher, truncate_chat_prompt_chars

        self.assertEqual(select_vllm_dtype_for_teacher("Qwen/Qwen3-8B-FP8"), "auto")
        self.assertEqual(select_vllm_dtype_for_teacher("Qwen/Qwen3-8B"), "bfloat16")
        huge = "HEAD " + ("mid " * 9000) + " TAIL_END"
        capped = truncate_chat_prompt_chars(huge, max_chars=20000)
        self.assertLessEqual(len(capped), 20050)
        self.assertIn("HEAD", capped)
        self.assertIn("TAIL_END", capped)

    def test_vllm_runtime_env_disables_flashinfer_jit_sampler_and_exposes_venv_bin(self):
        import os
        import sys
        from training.label_td_teacher import configure_vllm_runtime_env as cfg_td
        from training.synth_jev_train import configure_vllm_runtime_env as cfg_jev

        old_fi = os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
        try:
            cfg_td()
            self.assertEqual(os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"), "0")
            self.assertIn(os.path.dirname(sys.executable), os.environ.get("PATH", "").split(os.pathsep))
            os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
            cfg_jev()
            self.assertEqual(os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"), "0")
        finally:
            if old_fi is not None:
                os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = old_fi

    def test_validate_tier_calibration_hygiene_rejects_leaked_criteria_and_verbose_easy(self):
        from training.synth_jev_cases import validate_tier_calibration_hygiene

        bad_leaked_noul = {
            "id": "jev-easy-noul-00201",
            "family": "policy",
            "state": "Facility capacity is 3.8 MW, notice was 28 minutes, credit is 12.5 MWh of 285 MWh.",
            "question": {
                "type": "noul",
                "instructions": "May the solar farm claim curtailment credits?",
                "criteria": {
                    "true": "Approve: capacity is 3.8 MW, notice 28 minutes, credit 12.5 MWh of 285 MWh.",
                    "false": "Deny: capacity is 3.8 MW, notice 28 minutes, credit 12.5 MWh of 285 MWh.",
                },
            },
            "labels": ["no", "yes"],
            "expected": "yes",
            "provenance": {"tier": "easy"},
        }
        self.assertFalse(validate_tier_calibration_hygiene(bad_leaked_noul))

        bad_verbose_easy = {
            "id": "jev-easy-choice-00001",
            "family": "routing",
            "state": {"blob": "x" * 400},
            "question": {
                "type": "choice",
                "instructions": "Route ticket.",
                "criteria": {"billing": "Invoices", "shipping": "Delivery"},
            },
            "labels": ["billing", "shipping"],
            "expected": "billing",
            "provenance": {"tier": "easy"},
        }
        self.assertFalse(validate_tier_calibration_hygiene(bad_verbose_easy))

        good_easy = {
            "id": "jev-easy-noul-00201",
            "family": "policy",
            "state": "Parcel weighs 2.4 kg; courier limit is 5.0 kg.",
            "question": {
                "type": "noul",
                "instructions": "Is the parcel within the courier weight limit?",
                "criteria": {"false": "Exceeds the weight limit.", "true": "Within the weight limit."},
            },
            "labels": ["no", "yes"],
            "expected": "yes",
            "provenance": {"tier": "easy"},
        }
        self.assertTrue(validate_tier_calibration_hygiene(good_easy))

    def test_anthropic_5_series_models_omit_deprecated_temperature_parameter(self):
        import io
        import json
        from unittest import mock
        from training.synth_jev_cases import _author_batch_anthropic
        from bench.eval_llm_decoders import _call_anthropic_json

        captured_bodies: list[dict] = []

        class _Resp:
            def __init__(self, payload: dict):
                self._raw = json.dumps(payload).encode("utf-8")

            def read(self) -> bytes:
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def _fake_urlopen(req, timeout=60):
            body = json.loads(req.data.decode("utf-8"))
            captured_bodies.append(body)
            if any(t.get("name") == "emit_jev_items" for t in body.get("tools", [])):
                return _Resp(
                    {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "emit_jev_items",
                                "input": {
                                    "items": [
                                        {
                                            "state": "Order total is $42; free shipping threshold is $50.",
                                            "instructions": "Does the order qualify for free shipping?",
                                            "criteria": {
                                                "false": "Order total is below threshold.",
                                                "true": "Order total meets threshold.",
                                            },
                                            "expected": "no",
                                            "rationale": "42 < 50.",
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                )
            return _Resp(
                {
                    "usage": {"input_tokens": 12, "output_tokens": 8},
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "submit_extraction",
                            "input": {"value": None, "null_probability": 0.95},
                        }
                    ],
                }
            )

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            _author_batch_anthropic(
                [{"index": 0, "cell_id": "c0", "family": "policy", "tier": "easy", "qtype": "noul"}],
                author_model="claude-haiku-5-5",
                api_key="test-key",
                seed=1,
            )
            _author_batch_anthropic(
                [{"index": 0, "cell_id": "c0", "family": "policy", "tier": "easy", "qtype": "noul"}],
                author_model="claude-haiku-4-5-20251001",
                api_key="test-key",
                seed=1,
            )
            _call_anthropic_json("claude-opus-5-5", "Extract amount", "test-key")

        self.assertNotIn("temperature", captured_bodies[0])
        self.assertIn("temperature", captured_bodies[1])
        self.assertNotIn("temperature", captured_bodies[2])

    def test_hf_blind_reviewer_preserves_full_35k_char_long_policy_middle_sections(self):
        import torch
        from training.synth_jev_cases import HFBlindReviewer

        rev = HFBlindReviewer.__new__(HFBlindReviewer)
        rev.device = "cpu"
        seen_calls: list[tuple[list[str], int]] = []

        class _FakeTok:
            pad_token = "<pad>"
            eos_token = "<eos>"
            pad_token_id = 0
            padding_side = "left"

            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kwargs):
                return f"<|user|>\n{msgs[0]['content']}\n<|assistant|>\n"

            def __call__(self, txts, return_tensors="pt", padding=True, truncation=True, max_length=6144):
                seen_calls.append((list(txts), int(max_length)))

                class _Batch(dict):
                    def to(self, _dev):
                        return self

                return _Batch({"input_ids": torch.ones((len(txts), 8), dtype=torch.long)})

            def decode(self, ids, skip_special_tokens=True):
                return '{"reason": "Clause 7.4 applies.", "predicted": "2"}'

        class _FakeModel:
            def generate(self, **kwargs):
                b = kwargs["input_ids"].shape[0]
                return torch.ones((b, 12), dtype=torch.long)

        rev.tok = _FakeTok()
        rev.model = _FakeModel()

        middle_needle = "GOVERNING_CARVEOUT_CLAUSE_7_4_SCORE_BAND_2"
        long_state = ("A" * 16500) + middle_needle + ("B" * 16500)
        rec = {
            "id": "jev-hard-score-00388",
            "family": "long_policy",
            "state": long_state,
            "question": {
                "type": "score",
                "instructions": "Score the compliance handling from 0 to 3.",
                "criteria": ["0 - Bad", "1 - Weak", "2 - Compliant", "3 - Over"],
            },
            "labels": ["0", "1", "2", "3"],
            "expected": 2,
        }
        outs = rev.review_batch([rec], batch_size=4)
        self.assertEqual(outs[0]["predicted"], "2")
        self.assertEqual(len(seen_calls), 1)
        prompt_txt, max_len = seen_calls[0]
        self.assertIn(middle_needle, prompt_txt[0])
        self.assertGreaterEqual(max_len, 12288)

    def test_easy_tier_prompt_overrides_multi_step_family_guidance(self):
        import json
        from unittest import mock
        from training.synth_jev_cases import _author_batch_anthropic

        captured_prompts: list[str] = []

        class _Resp:
            def read(self) -> bytes:
                return json.dumps({"content": []}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def _fake_urlopen(req, timeout=60):
            body = json.loads(req.data.decode("utf-8"))
            captured_prompts.append(body["messages"][0]["content"])
            return _Resp()

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            _author_batch_anthropic(
                [
                    {"index": 0, "cell_id": "e0", "family": "state_tracking", "tier": "easy", "qtype": "noul"},
                    {"index": 1, "cell_id": "e1", "family": "policy", "tier": "easy", "qtype": "noul"},
                ],
                author_model="claude-haiku-5-5",
                api_key="test-key",
                seed=7,
            )

        self.assertEqual(len(captured_prompts), 1)
        self.assertNotIn("3-6 chronological events", captured_prompts[0])
        self.assertNotIn("2-3 explicit policy rules", captured_prompts[0])
        self.assertIn("single explicit fact", captured_prompts[0].lower())


if __name__ == "__main__":
    unittest.main()
