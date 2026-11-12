import json
from pathlib import Path
import random
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class _DummyTokenizer:
    """Minimal deterministic tokenizer stub for testing `prepare_td_cases` on CPU."""

    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        toks = [1000 + (abs(hash(tok)) % 5000) for tok in str(text).split()]
        if not toks:
            toks = [1000]
        if add_special_tokens:
            return [self.cls_token_id] + toks + [self.sep_token_id]
        return toks


class TestSynthTypedDecisions(unittest.TestCase):
    def test_shingle_guard_rejects_planted_12_token_near_duplicate_and_never_reads_gold(self):
        from training.synth_td_cases import (
            build_shingle_set,
            has_shingle_overlap,
            tokenize_for_shingles,
        )

        test_state = json.dumps(
            {
                "alert": {
                    "rule": "dormant_account_use",
                    "evidence": (
                        "The unprivileged service account svc_task_alpha initiated a session "
                        "from IP 198.51.100.24 following 180 days of zero activity without MFA."
                    ),
                }
            }
        )
        shingles = build_shingle_set([test_state], k=12)
        self.assertGreater(len(shingles), 0)

        # Plant a 12+ token verbatim span from test_state inside an otherwise novel state
        planted_state = json.dumps(
            {
                "alert": {
                    "rule": "token_reuse",
                    "evidence": (
                        "During routine audit, the unprivileged service account svc_task_alpha "
                        "initiated a session from IP 198.51.100.24 following 180 days of silence."
                    ),
                }
            }
        )
        self.assertGreaterEqual(len(tokenize_for_shingles(planted_state)), 12)
        self.assertTrue(
            has_shingle_overlap(planted_state, shingles, k=12),
            "12-token shingle guard must reject planted near-duplicate from test state",
        )

        # Novel state with < 12 consecutive shared tokens must be accepted
        novel_state = json.dumps(
            {
                "alert": {
                    "rule": "impossible_travel",
                    "evidence": (
                        "Employee login succeeded from Frankfurt and Tokyo within nine minutes "
                        "using hardware passkey credentials rotated twelve days ago."
                    ),
                }
            }
        )
        self.assertFalse(has_shingle_overlap(novel_state, shingles, k=12))

        src = (REPO_ROOT / "training/synth_td_cases.py").read_text(encoding="utf-8")
        self.assertNotIn('["gold"]', src)
        self.assertNotIn("['gold']", src)
        self.assertNotIn('.get("gold"', src)
        self.assertNotIn(".get('gold'", src)

    def test_recombine_workflow_factors_produces_valid_tuples_with_exact_schemas(self):
        from training.synth_td_cases import (
            build_workflow_factor_pools,
            sample_recombined_factor_tuple,
        )

        q_schema = {
            "disposition": {
                "type": "choice",
                "instructions": "Pick disposition.",
                "criteria": {"close_benign": "Close.", "investigate": "Investigate."},
            }
        }
        dummy_train = []
        for idx in range(160):
            dummy_train.append(
                {
                    "workflow": "security_incidents",
                    "factors": json.dumps(
                        {
                            "principal_type": "employee" if idx % 2 == 0 else "service_account",
                            "rule": "token_reuse" if idx % 3 == 0 else "dormant_account_use",
                            "mfa_enrolled": bool(idx % 2),
                            "prior_alerts_90d": idx % 6,
                        }
                    ),
                    "questions": json.dumps(q_schema),
                    "state": json.dumps({"alert": {"rule": "token_reuse"}, "idx": idx}),
                }
            )

        pools = build_workflow_factor_pools(dummy_train)
        self.assertIn("security_incidents", pools)
        rng = random.Random(7)
        fac = sample_recombined_factor_tuple(pools["security_incidents"], rng)
        self.assertIn(fac["principal_type"], {"employee", "service_account"})
        self.assertIn(fac["rule"], {"token_reuse", "dormant_account_use"})
        self.assertIsInstance(fac["mfa_enrolled"], bool)
        self.assertEqual(pools["security_incidents"]["questions"], q_schema)

    def test_label_td_teacher_aggregates_three_samples_and_parses_with_prepare_td_cases(self):
        from training.label_td_teacher import aggregate_teacher_samples
        from training.train_systemone import prepare_td_cases

        questions = {
            "disposition": {
                "type": "choice",
                "instructions": "Select disposition.",
                "criteria": {
                    "close_benign": "Expected activity.",
                    "investigate": "Open investigation.",
                },
            },
            "true_positive": {
                "type": "noul",
                "instructions": "This alert is malicious.",
                "criteria": {"false": "Benign.", "true": "Malicious."},
            },
            "severity": {
                "type": "score",
                "instructions": "Rate severity.",
                "criteria": ["0: Low", "1: Medium", "2: High"],
            },
        }
        samples = [
            {
                "disposition": {"close_benign": 0.20, "investigate": 0.80},
                "true_positive": {"false": 0.25, "true": 0.75},
                "severity": {"0": 0.10, "1": 0.60, "2": 0.30},
            },
            {
                "disposition": {"close_benign": 0.30, "investigate": 0.70},
                "true_positive": {"false": 0.35, "true": 0.65},
                "severity": {"0": 0.20, "1": 0.50, "2": 0.30},
            },
            {
                "disposition": {"close_benign": 0.10, "investigate": 0.90},
                "true_positive": {"false": 0.30, "true": 0.70},
                "severity": {"0": 0.00, "1": 0.70, "2": 0.30},
            },
        ]

        gold = aggregate_teacher_samples(questions, samples)
        self.assertEqual(gold["disposition"]["label"], "investigate")
        self.assertAlmostEqual(gold["disposition"]["probabilities"]["investigate"], 0.80, places=5)
        self.assertEqual(gold["true_positive"]["label"], "true")
        self.assertAlmostEqual(gold["true_positive"]["noul"], 0.70, places=5)
        self.assertEqual(gold["severity"]["label"], "1")
        self.assertAlmostEqual(gold["severity"]["score"], 0.10 * 0 + 0.60 * 1 + 0.30 * 2, places=5)

        row = {
            "workflow": "security_incidents",
            "state": json.dumps({"alert": {"rule": "token_reuse", "evidence": "Two IPs used token."}}),
            "questions": json.dumps(questions),
            "gold": json.dumps(gold),
            "factors": json.dumps({"principal_type": "service_account", "rule": "token_reuse"}),
        }
        factor_vocabs = {
            "principal_type": ["employee", "service_account"],
            "rule": ["dormant_account_use", "token_reuse"],
        }
        packed = prepare_td_cases([row], _DummyTokenizer(), factor_vocabs)
        self.assertEqual(len(packed), 1)
        self.assertEqual(len(packed[0]["questions"]), 3)
        self.assertEqual(packed[0]["factor_targets"], {"principal_type": 1, "rule": 1})
        for qm in packed[0]["questions"]:
            self.assertAlmostEqual(sum(qm["gold_probs"]), 1.0, places=5)

    def test_teacher_kl_agreement_gate_enforces_0_10_stop_threshold(self):
        from training.label_td_teacher import evaluate_teacher_kl_against_gold

        ref_gold = [
            {
                "q1": {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}},
                "q2": {"type": "noul", "probabilities": {"false": 0.2, "true": 0.8}},
            }
        ]
        good_pred = [
            {
                "q1": {"type": "choice", "probabilities": {"a": 0.66, "b": 0.34}},
                "q2": {"type": "noul", "probabilities": {"false": 0.24, "true": 0.76}},
            }
        ]
        bad_pred = [
            {
                "q1": {"type": "choice", "probabilities": {"a": 0.15, "b": 0.85}},
                "q2": {"type": "noul", "probabilities": {"false": 0.85, "true": 0.15}},
            }
        ]

        res_ok = evaluate_teacher_kl_against_gold(good_pred, ref_gold, max_kl=0.10)
        self.assertTrue(res_ok["accepted"])
        self.assertLess(res_ok["mean_kl"], 0.10)

        res_bad = evaluate_teacher_kl_against_gold(bad_pred, ref_gold, max_kl=0.10)
        self.assertFalse(res_bad["accepted"])
        self.assertGreaterEqual(res_bad["mean_kl"], 0.10)

    def test_fold_split_adds_synthetic_cases_to_train_side_only(self):
        from training.train_systemone import build_fold_splits_with_synth

        real_cases = [{"case_id": i, "orig_train_idx": i, "source": "td"} for i in range(10)]
        synth_cases = [{"case_id": 1000 + j, "orig_train_idx": j, "source": "td_synth"} for j in range(15)]
        folds = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        splits = build_fold_splits_with_synth(real_cases, folds, synth_cases)
        self.assertEqual(len(splits), 5)
        for f_idx, (f_train, f_val) in enumerate(splits):
            self.assertEqual([c["orig_train_idx"] for c in f_val], folds[f_idx])
            self.assertTrue(all(c["source"] == "td" for c in f_val))
            self.assertEqual(len(f_train), 8 + 15)
            self.assertEqual(sum(1 for c in f_train if c["source"] == "td_synth"), 15)

    def test_label_rows_caches_gpu_runner_across_multiple_chunks(self):
        from training.label_td_teacher import label_rows

        load_count = []

        class FakeGPURunner:
            def __init__(self, teacher_id: str, n_samples: int = 3, temperature: float = 0.7, batch_size: int = 16):
                load_count.append(teacher_id)

            def label_batch(self, rows):
                outs = []
                for r in rows:
                    qs = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
                    g = {}
                    for qid, qspec in qs.items():
                        g[qid] = {"type": qspec["type"], "label": "true", "confidence": 0.6, "probabilities": {"false": 0.2, "true": 0.8}}
                    outs.append(g)
                return outs

        sample_row = {
            "state": '{"ok": true}',
            "questions": json.dumps({"q1": {"type": "noul", "instructions": "Ok?", "criteria": {"false": "N", "true": "Y"}}}),
        }
        cache = {}
        r1 = label_rows([sample_row], teacher="Qwen/Qwen3-4B-Instruct-2507", runner_cache=cache, runner_cls=FakeGPURunner)
        r2 = label_rows([sample_row], teacher="Qwen/Qwen3-4B-Instruct-2507", runner_cache=cache, runner_cls=FakeGPURunner)
        self.assertEqual(len(r1), 1)
        self.assertEqual(len(r2), 1)
        self.assertEqual(load_count, ["Qwen/Qwen3-4B-Instruct-2507"])


if __name__ == "__main__":
    unittest.main()
