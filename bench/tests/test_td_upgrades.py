import math
import unittest

import torch
import torch.nn.functional as F


class TestTypedDecisionsUpgrades(unittest.TestCase):
    def test_choose_max_state_train_respects_2pct_threshold(self) -> None:
        from train_systemone import choose_max_state_train

        # Fewer than 2% exceed 2,560 -> keep 2560
        r_low = {"state_length": {"pct_cases_gt_2560": 0.008, "p99": 2100}}
        self.assertEqual(choose_max_state_train(r_low), 2560)

        # >= 2% exceed 2,560 -> raise to cover p99 (rounded up to multiple of 64, <= 8192)
        r_high = {"state_length": {"pct_cases_gt_2560": 0.035, "p99": 3410}}
        chosen = choose_max_state_train(r_high)
        self.assertGreaterEqual(chosen, 3410)
        self.assertLessEqual(chosen, 8192)

    def test_candidate_backbones_phase1_registry(self) -> None:
        from train_systemone import PHASE1_CANDIDATE_SPECS

        bb_ids = [spec["backbone_id"] for spec in PHASE1_CANDIDATE_SPECS]
        self.assertIn("MoritzLaurer/ModernBERT-large-zeroshot-v2.0", bb_ids)
        self.assertIn("answerdotai/ModernBERT-large", bb_ids)
        self.assertIn("jhu-clsp/ettin-encoder-400m", bb_ids)
        self.assertIn("microsoft/deberta-v3-large", bb_ids)
        ettin_specs = [s for s in PHASE1_CANDIDATE_SPECS if s["backbone_id"] == "jhu-clsp/ettin-encoder-400m"]
        self.assertEqual(len(ettin_specs), 3)
        deb_specs = [s for s in PHASE1_CANDIDATE_SPECS if s["backbone_id"] == "microsoft/deberta-v3-large"]
        self.assertEqual(deb_specs[0].get("max_total_tokens"), 512)

    def test_header_attends_options_mask_and_permutation_invariance(self) -> None:
        from transformers import ModernBertConfig, ModernBertForSequenceClassification
        from systemone_decision import (
            ParallelSpanDecisionHead,
            build_parallel_tree_mask,
            configure_modernbert_full_bidir,
            encode_case_parallel,
            forward_parallel_backbone,
        )

        # 1. Mask check: q_t=[0, 1, 1, 1, 2, 2], o_t=[0, 0, 1, 2, 0, 1]
        q_t = torch.tensor([[0, 1, 1, 1, 2, 2]], dtype=torch.long)
        o_t = torch.tensor([[0, 0, 1, 2, 0, 1]], dtype=torch.long)
        m_legacy = build_parallel_tree_mask(q_t, o_t, dtype=torch.float32, return_bool=True, header_attends_options=False)
        self.assertFalse(bool(m_legacy[0, 0, 1, 2]))  # Q1 header cannot see Q1 opt1 in legacy
        m_deep = build_parallel_tree_mask(q_t, o_t, dtype=torch.float32, return_bool=True, header_attends_options=True)
        self.assertTrue(bool(m_deep[0, 0, 1, 2]))   # Q1 header sees Q1 opt1
        self.assertTrue(bool(m_deep[0, 0, 1, 3]))   # Q1 header sees Q1 opt2
        self.assertFalse(bool(m_deep[0, 0, 1, 5]))  # Q1 header NEVER sees Q2 opt1
        self.assertFalse(bool(m_deep[0, 0, 0, 2]))  # State NEVER sees options
        self.assertFalse(bool(m_deep[0, 0, 2, 3]))  # Q1 opt1 NEVER directly sees Q1 opt2 (no RoPE collision)

        # 2. End-to-end ModernBERT + ParallelSpanDecisionHead option-order invariance
        torch.manual_seed(7)
        cfg = ModernBertConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_labels=2,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
        )
        bb = ModernBertForSequenceClassification(cfg).eval()
        configure_modernbert_full_bidir(bb, None)
        head = ParallelSpanDecisionHead(hidden_size=64, comp_dim=32, zs_dim=2, dropout=0.0).eval()

        class DummyTok:
            cls_token_id = 1
            sep_token_id = 2
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 0

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [3 + (ord(ch) % 240) for ch in text]

        tok = DummyTok()
        state = "Customer ticket: enterprise SSO renewal failed after certificate rotation."
        q_order_1 = {
            "route": {
                "type": "choice",
                "instructions": "Select the best escalation queue.",
                "options": {
                    "identity": "Route to Identity & SSO engineering",
                    "billing": "Route to Billing operations",
                    "general": "Keep in Tier-1 general support",
                },
            }
        }
        q_order_2 = {
            "route": {
                "type": "choice",
                "instructions": "Select the best escalation queue.",
                "options": {
                    "general": "Keep in Tier-1 general support",
                    "identity": "Route to Identity & SSO engineering",
                    "billing": "Route to Billing operations",
                },
            }
        }
        pc1 = encode_case_parallel(tok, state, q_order_1, max_state_tokens=128, case_id=1)
        pc2 = encode_case_parallel(tok, state, q_order_2, max_state_tokens=128, case_id=1)
        with torch.no_grad():
            seq1, zs1 = forward_parallel_backbone(bb, [pc1], pad_id=0, device="cpu", header_attends_options=True)
            lgs1, meta1 = head.forward_packed_cases(seq1, [pc1], zs_head_fn=zs1, apply_temp=False)
            p1 = {k: float(v) for k, v in zip(meta1[0]["option_keys"], F.softmax(lgs1[0], dim=-1).tolist())}

            seq2, zs2 = forward_parallel_backbone(bb, [pc2], pad_id=0, device="cpu", header_attends_options=True)
            lgs2, meta2 = head.forward_packed_cases(seq2, [pc2], zs_head_fn=zs2, apply_temp=False)
            p2 = {k: float(v) for k, v in zip(meta2[0]["option_keys"], F.softmax(lgs2[0], dim=-1).tolist())}

        for k in ("identity", "billing", "general"):
            self.assertAlmostEqual(p1[k], p2[k], places=5)

    def test_deberta_512_control_packing_and_forward(self) -> None:
        from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
        from systemone_decision import encode_case_parallel, forward_parallel_backbone

        class DummyTok:
            cls_token_id = 1
            sep_token_id = 2
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 0

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [3 + (ord(ch) % 200) for ch in text]

        tok = DummyTok()
        long_state = "Policy paragraph. " * 80
        qs = {
            f"q{i}": {
                "type": "choice",
                "instructions": "Pick one option carefully.",
                "options": {"a": "First detailed option text", "b": "Second detailed option text"},
            }
            for i in range(5)
        }
        pc = encode_case_parallel(tok, long_state, qs, max_state_tokens=512, max_total_tokens=512, case_id=9)
        self.assertLessEqual(len(pc["input_ids"]), 512)
        self.assertLess(max(pc["position_ids"]), 512)

        cfg = DebertaV2Config(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            relative_attention=True,
            position_buckets=64,
            num_labels=2,
        )
        deb = DebertaV2ForSequenceClassification(cfg).eval()
        with torch.no_grad():
            seq_f, _ = forward_parallel_backbone(deb, [pc], pad_id=0, device="cpu", max_seq_pad=512)
        self.assertEqual(seq_f.shape[0], 1)
        self.assertLessEqual(seq_f.shape[1], 512)
        self.assertEqual(seq_f.shape[2], 64)

    def test_carve_student_holdout_is_disjoint_from_teacher_folds(self) -> None:
        from train_systemone import carve_student_holdout_and_folds

        ds_mock = [
            {"workflow": f"wf_{i % 4}", "factors": {"issue": f"iss_{i % 3}"}}
            for i in range(100)
        ]
        holdout_idx, teacher_pool_idx, teacher_folds = carve_student_holdout_and_folds(ds_mock, seed=42)
        self.assertEqual(len(holdout_idx), 20)
        self.assertEqual(len(teacher_pool_idx), 80)
        self.assertEqual(set(holdout_idx) & set(teacher_pool_idx), set())
        self.assertEqual(set(holdout_idx) | set(teacher_pool_idx), set(range(100)))

        seen_val: list[int] = []
        for tr_i, va_i in teacher_folds:
            self.assertEqual(set(tr_i) & set(va_i), set())
            self.assertEqual(set(tr_i) & set(holdout_idx), set())
            self.assertEqual(set(va_i) & set(holdout_idx), set())
            seen_val.extend(va_i)
        self.assertEqual(sorted(seen_val), sorted(teacher_pool_idx))

    def test_fit_per_type_temperatures_lbfgs_minimizes_kl(self) -> None:
        from train_systemone import fit_per_type_temperatures_lbfgs

        torch.manual_seed(11)
        true_temps = {0: 1.65, 1: 0.72, 2: 1.35}
        qtypes = {0: "noul", 1: "choice", 2: "score"}
        pairs = []
        for qt_id, t_star in true_temps.items():
            K = 2 if qt_id == 0 else (4 if qt_id == 1 else 5)
            opt_keys = ["false", "true"] if qt_id == 0 else [str(i) for i in range(K)]
            for _ in range(80):
                raw_logits = torch.randn(K) * 1.8
                gold_p = F.softmax(raw_logits / t_star, dim=-1).tolist()
                qm = {
                    "qtype": qtypes[qt_id],
                    "qtype_id": qt_id,
                    "option_keys": opt_keys,
                    "gold_probs": gold_p,
                    "gold_raw": {"type": qtypes[qt_id], "probabilities": dict(zip(opt_keys, gold_p))},
                }
                pairs.append((raw_logits, qm))

        temps = fit_per_type_temperatures_lbfgs(pairs)
        self.assertEqual(len(temps), 3)
        for qt_id, t_star in true_temps.items():
            self.assertAlmostEqual(temps[qt_id], t_star, delta=0.03)

    def test_select_soup_or_best_seed_uses_oof_gate(self) -> None:
        from train_systemone import average_state_dicts, select_soup_or_best_seed

        sd1 = {"w": torch.tensor([1.0, 3.0])}
        sd2 = {"w": torch.tensor([3.0, 5.0])}
        avg = average_state_dicts([sd1, sd2])
        self.assertTrue(torch.allclose(avg["w"], torch.tensor([2.0, 4.0])))

        seeds = [
            {"seed": 1, "oof_accuracy": 0.812, "oof_kl": 0.0800, "oof_brier": 0.0440},
            {"seed": 2, "oof_accuracy": 0.815, "oof_kl": 0.0780, "oof_brier": 0.0430},
            {"seed": 3, "oof_accuracy": 0.813, "oof_kl": 0.0790, "oof_brier": 0.0435},
        ]
        # Soup beats best seed on KL by >= 0.002 with non-worse accuracy -> accepted
        soup_good = {"oof_accuracy": 0.816, "oof_kl": 0.0755, "oof_brier": 0.0420}
        choice_good = select_soup_or_best_seed(seeds, soup_good)
        self.assertEqual(choice_good["selected"], "soup")

        # Soup regresses accuracy / fails gate -> keep best single seed (seed 2)
        soup_bad = {"oof_accuracy": 0.814, "oof_kl": 0.0778, "oof_brier": 0.0431}
        choice_bad = select_soup_or_best_seed(seeds, soup_bad)
        self.assertEqual(choice_bad["selected"], "best_seed")
        self.assertEqual(choice_bad["best_seed_idx"], 1)

    def test_resolve_candidate_specs_and_normalize_jev_train_items(self) -> None:
        from train_systemone import (
            normalize_extra_gen_item,
            resolve_candidate_specs,
            select_best_holdout_grid,
        )

        specs = resolve_candidate_specs({"use_phase1_specs": True})
        self.assertEqual(len(specs), 6)
        deb = [s for s in specs if s["key"] == "deberta_v3_large_512"][0]
        self.assertEqual(deb["max_total_tokens"], 512)

        # JevBench Noul task JSONL -> canonical ("false", "true") probabilities
        jev_noul = {
            "id": "jev-tr-1",
            "family": "policy",
            "state": "Refund allowed within 30 days. Bought 10 days ago with receipt.",
            "labels": ["no", "yes"],
            "expected": "yes",
            "soft_probs": {"no": 0.04, "yes": 0.96},
            "question": {
                "type": "noul",
                "instructions": "Is refund permitted?",
                "criteria": {"false": "Not permitted", "true": "Permitted"},
            },
        }
        st, qid, q, probs = normalize_extra_gen_item(jev_noul)
        self.assertEqual(qid, "decision")
        self.assertEqual(q["type"], "noul")
        self.assertAlmostEqual(probs["true"], 0.96, places=6)
        self.assertAlmostEqual(probs["false"], 0.04, places=6)

        # JevBench probability family item with exact gold_probs
        jev_prob = {
            "id": "jev-tr-2",
            "family": "probability",
            "state": "Urn has 3 red, 1 blue.",
            "labels": ["red", "blue"],
            "expected": "red",
            "gold_probs": {"red": 0.75, "blue": 0.25},
            "question": {
                "type": "choice",
                "instructions": "Posterior over drawn ball?",
                "criteria": {"red": "Red ball", "blue": "Blue ball"},
            },
        }
        _, _, _, probs_p = normalize_extra_gen_item(jev_prob)
        self.assertAlmostEqual(probs_p["red"], 0.75, places=6)
        self.assertAlmostEqual(probs_p["blue"], 0.25, places=6)

        grid = [
            {"beta_oof": 0.0, "alpha_wise": 0.0, "holdout_acc": 0.805, "holdout_kl": 0.0820, "holdout_brier": 0.0450},
            {"beta_oof": 0.32, "alpha_wise": 0.04, "holdout_acc": 0.818, "holdout_kl": 0.0750, "holdout_brier": 0.0410},
            {"beta_oof": 0.50, "alpha_wise": 0.10, "holdout_acc": 0.812, "holdout_kl": 0.0770, "holdout_brier": 0.0420},
        ]
        best = select_best_holdout_grid(grid)
        self.assertEqual(best["beta_oof"], 0.32)
        self.assertEqual(best["alpha_wise"], 0.04)

    def test_long_state_truncation_keeps_question_relevant_middle_clauses(self) -> None:
        from systemone_decision import encode_case_parallel

        class WordTok:
            cls_token_id = 1
            sep_token_id = 2
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 0

            def __init__(self) -> None:
                self._vocab: dict[str, int] = {}
                self._rev: dict[int, str] = {}

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                ids: list[int] = []
                for w in text.replace("\n", " \n ").split(" "):
                    if not w:
                        continue
                    tid = self._vocab.get(w)
                    if tid is None:
                        tid = len(self._vocab) + 10
                        self._vocab[w] = tid
                        self._rev[tid] = w
                    ids.append(tid)
                return ids

            def decode(self, ids: list[int]) -> str:
                return " ".join(self._rev.get(i, "") for i in ids)

        tok = WordTok()
        sections = []
        for idx in range(40):
            if idx == 20:
                sections.append(
                    f"Section {idx}: LITHIUM_HAZMAT_SURCHARGE_EXEMPTION — shipments under 4.5kg "
                    "certified UN3481_PACKING_PI967 are strictly exempt from hazmat surcharges."
                )
            else:
                filler = " ".join(f"boilerplate_clause_{idx}_{w}" for w in range(55))
                sections.append(f"Section {idx}: {filler}")
        long_policy_state = "\n\n".join(sections)

        questions = {
            "exempt": {
                "type": "noul",
                "instructions": "Under UN3481_PACKING_PI967, is a 3.2kg battery parcel exempt from hazmat surcharges?",
                "criteria": {
                    "false": "Not exempt from hazmat surcharge",
                    "true": "Exempt under LITHIUM_HAZMAT_SURCHARGE_EXEMPTION",
                },
            }
        }
        pc = encode_case_parallel(tok, long_policy_state, questions, max_state_tokens=420, case_id=5)
        ss, se = pc["state_span"]
        self.assertLessEqual(se - ss, 420)
        kept_state_text = tok.decode(pc["input_ids"][ss:se])
        self.assertIn("LITHIUM_HAZMAT_SURCHARGE_EXEMPTION", kept_state_text)
        self.assertIn("UN3481_PACKING_PI967", kept_state_text)
        # Head and tail anchors must still appear in original document order
        self.assertIn("Section 0:", kept_state_text)
        self.assertIn("Section 39:", kept_state_text)
        self.assertLess(
            kept_state_text.index("Section 0:"),
            kept_state_text.index("LITHIUM_HAZMAT_SURCHARGE_EXEMPTION"),
        )
        self.assertLess(
            kept_state_text.index("LITHIUM_HAZMAT_SURCHARGE_EXEMPTION"),
            kept_state_text.index("Section 39:"),
        )

    def test_load_jev_dev_hard_cases_filters_provenance_tier_and_splits_long_batches(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from train_systemone import load_jev_dev_hard_cases, split_long_subbatch

        class DummyTok:
            cls_token_id = 1
            sep_token_id = 2
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 0

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [3 + (ord(ch) % 200) for ch in text]

        rows = [
            {
                "id": f"jev-dev-{tier}",
                "family": "policy",
                "state": f"Policy rule for {tier}.",
                "labels": ["no", "yes"],
                "expected": "yes",
                "provenance": {"tier": tier, "author_model": "claude-haiku-4-5"},
                "question": {
                    "type": "noul",
                    "instructions": "Is it permitted?",
                    "criteria": {"false": "No", "true": "Yes"},
                },
            }
            for tier in ("easy", "standard", "judge", "hard")
        ]
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "dev.jsonl"
            fp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            hard_cases = load_jev_dev_hard_cases(str(fp), DummyTok(), max_state_tokens=256)
            self.assertEqual(len(hard_cases), 1)

        short_batch = [{"input_ids": [1] * 400} for _ in range(4)]
        self.assertEqual(len(split_long_subbatch(short_batch)), 1)

        long_batch = [{"input_ids": [1] * 4200} for _ in range(4)]
        micro = split_long_subbatch(long_batch)
        self.assertEqual(len(micro), 4)
        self.assertTrue(all(len(m) == 1 for m in micro))

    def test_nested_fold_lbfgs_temperatures_and_wise_soup_merge_have_zero_val_leakage(self) -> None:
        from train_systemone import (
            fit_nested_fold_temperatures_lbfgs,
            merge_wise_and_seed_soup,
        )

        # 1. Verify WiSE-FT + multi-seed weight soup collapses to a single state dict cleanly
        gen_sd = {"w": torch.tensor([10.0, 20.0])}
        seed_sds = [
            {"w": torch.tensor([0.0, 10.0])},
            {"w": torch.tensor([2.0, 14.0])},
        ]
        merged = merge_wise_and_seed_soup(seed_sds, gen_sd, alpha_wise=0.10)
        # Seed mean = [1.0, 12.0]; 0.9 * [1.0, 12.0] + 0.1 * [10.0, 20.0] = [1.9, 12.8]
        self.assertTrue(torch.allclose(merged["w"], torch.tensor([1.9, 12.8])))

        # 2. Verify nested fold L-BFGS fits temperatures for fold f strictly from tr_idx (never va_idx!)
        torch.manual_seed(77)
        n_cases = 20
        folds = [
            ([i for i in range(n_cases) if i % 5 != f], [i for i in range(n_cases) if i % 5 == f])
            for f in range(5)
        ]
        td_cases = []
        oof_logits = {}
        for i in range(n_cases):
            raw_l = torch.tensor([0.4, 2.0])
            # Fold 0 val cases have extreme corrupted gold; if leaked into fold 0 temp fit, fold 0 temp would flip!
            gp = [0.99, 0.01] if (i % 5 == 0) else F.softmax(raw_l / 1.5, dim=-1).tolist()
            qm = {
                "qid": "q0",
                "qtype": "noul",
                "qtype_id": 0,
                "option_keys": ["false", "true"],
                "gold_probs": gp,
                "gold_raw": {"type": "noul", "probabilities": {"false": gp[0], "true": gp[1]}},
            }
            td_cases.append({"questions": [qm]})
            oof_logits[i] = [raw_l]

        temps_by_case = fit_nested_fold_temperatures_lbfgs(oof_logits, td_cases, folds)
        self.assertEqual(len(temps_by_case), n_cases)
        # Fold 0 cases (i % 5 == 0) were calibrated ONLY on folds 1..4 (where true T = 1.5)!
        for i in range(0, n_cases, 5):
            self.assertAlmostEqual(temps_by_case[i][0], 1.5, delta=0.05)

    def test_family_cache_key_aligns_baseline_and_phase1_and_invalidates_on_config_change(self) -> None:
        import tempfile
        from pathlib import Path
        from train_systemone import (
            compute_family_cache_key,
            load_cached_family_result,
            resolve_candidate_specs,
            save_cached_family_result,
        )

        base_specs = resolve_candidate_specs({})
        p1_specs = resolve_candidate_specs({"use_phase1_specs": True})

        # Baseline modernbert_zs and ettin_400m match Phase 1 modernbert_zs and ettin_400m_s0 cache keys!
        k_base_mb = compute_family_cache_key(base_specs[0], max_state_train=2560, header_attends_options=False)
        k_p1_mb = compute_family_cache_key(p1_specs[0], max_state_train=2560, header_attends_options=False)
        self.assertEqual(k_base_mb, k_p1_mb)

        k_base_et = compute_family_cache_key(base_specs[1], max_state_train=2560, header_attends_options=False)
        k_p1_et0 = compute_family_cache_key(p1_specs[2], max_state_train=2560, header_attends_options=False)
        self.assertEqual(k_base_et, k_p1_et0)

        # Changing header_attends_options or max_state_train invalidates cache key
        k_hdr = compute_family_cache_key(p1_specs[0], max_state_train=2560, header_attends_options=True)
        self.assertNotEqual(k_p1_mb, k_hdr)

        with tempfile.TemporaryDirectory() as td:
            cdir = Path(td)
            self.assertIsNone(load_cached_family_result(cdir, k_p1_mb))
            payload = {
                "backbone_id": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
                "max_total_tokens": None,
                "max_state_tokens": 2560,
                "gen_bb_sd": {"w": torch.tensor([1.0, 2.0])},
                "gen_hd_sd": {"h": torch.tensor([3.0])},
                "gen_acc": 0.83,
                "gen_kl": 0.22,
                "oof_acc": 0.8002,
                "oof_kl": 0.0875,
                "oof_br": 0.0501,
                "oof_ece": 0.1646,
                "oof_logits_by_case": {0: [torch.tensor([0.2, 0.8])]},
            }
            save_cached_family_result(cdir, k_p1_mb, payload)
            loaded = load_cached_family_result(cdir, k_p1_mb)
            self.assertIsNotNone(loaded)
            self.assertAlmostEqual(loaded["oof_acc"], 0.8002, places=6)
            self.assertTrue(torch.allclose(loaded["oof_logits_by_case"][0][0], torch.tensor([0.2, 0.8])))


if __name__ == "__main__":
    unittest.main()
