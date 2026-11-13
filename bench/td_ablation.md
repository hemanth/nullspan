# Phase 1 Typed Decisions Out-of-Fold Ablation Report (Snapshot 2026-11-13)

All numbers below are 5-fold case-clustered out-of-fold (`train`, `N=1,200` cases / `6,000` decisions) metrics.
`Typed Decisions test` (`N=400` cases / `2,000` decisions) was **not** touched during model selection.

## 1. Baseline OOF (Task 3 — v1.3.0 Recipe)

| Run / Family | Backbone | OOF Accuracy | OOF KL | OOF Brier | OOF ECE |
|---|---|---:|---:|---:|---:|
| **v1.3.0 Calibrated Ensemble (Baseline)** | ModernBERT-large-zs + Ettin-400M | **0.7990** | **0.0868** | **0.0496** | 0.1757 |
| `modernbert_zs` (raw 5-fold) | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | 0.7990 | 0.0875 | 0.0500 | 0.1633 |
| `ettin_400m` (raw 5-fold) | `jhu-clsp/ettin-encoder-400m` | 0.7920 | 0.0888 | 0.0516 | 0.1569 |

## 2. Task 4 Teacher Agreement Gate (`mean_kl < 0.10` on Real `train`)

| Teacher Candidate | Train 3-Sample Accuracy | Train Mean KL vs Gold | Gate (`KL < 0.10`) | Notes |
|---|---:|---:|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | 0.5227 | 0.3724 | STOP (`>= 0.10`) | Measured 3-sample vLLM calibration across all 1,200 train cases (3,600 prompts). |
| `skamalj/signpost-d-4b` | 0.7200 | 0.1660 | STOP (`>= 0.10`) | Published HF model card evaluation on all 1,200 LocalLLaMA/typed-decisions train cases. |

## 3. Tasks 5 & 6 Backbone, Masking, Holdout Sweep & Calibration Ablations

| Run | Winner Backbone | OOF Accuracy | Δ Acc vs v1.3.0 | OOF KL | Δ KL vs v1.3.0 | OOF Brier | Task 7 RC Gate |
|---|---|---:|---:|---:|---:|---:|---|
| `task5_6specs` | `modernbert_zs` | **0.7955** | -0.0035 | **0.0888** | +0.0021 | 0.0504 | FAIL |
| `task5_hdropt` | `modernbert_zs` | **0.7947** | -0.0043 | **0.0890** | +0.0022 | 0.0512 | FAIL |
| `phase1_rc` | `modernbert_zs` | **0.7938** | -0.0052 | **0.0875** | +0.0008 | 0.0501 | FAIL |
| `unified_rc` | `ettin_400m_s2` | **0.7912** | -0.0078 | **0.0883** | +0.0016 | 0.0508 | FAIL |

### Per-Backbone Raw 5-Fold OOF (`task5_6specs`)

| Candidate Spec | Hugging Face ID | Raw OOF Acc | Raw OOF KL | Raw OOF Brier | Raw OOF ECE |
|---|---|---:|---:|---:|---:|
| `modernbert_zs` | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | 0.7955 | 0.0899 | 0.0509 | 0.1567 |
| `modernbert_base` | `answerdotai/ModernBERT-large` | 0.7563 | 0.1324 | 0.0749 | 0.1426 |
| `ettin_400m_s0` | `jhu-clsp/ettin-encoder-400m` | 0.7958 | 0.0880 | 0.0507 | 0.1601 |
| `ettin_400m_s1` | `jhu-clsp/ettin-encoder-400m` | 0.7935 | 0.0901 | 0.0516 | 0.1614 |
| `ettin_400m_s2` | `jhu-clsp/ettin-encoder-400m` | 0.7965 | 0.0861 | 0.0495 | 0.1620 |
| `deberta_v3_large_512` | `microsoft/deberta-v3-large` | 0.7517 | 0.1477 | 0.0836 | 0.1227 |

- **L-BFGS Per-Type Temperatures (`task5_6specs`)**: `noul=1.0766`, `choice=1.0792`, `score=1.0762`
- **3-Seed Weight Soup Gate (`task5_6specs`)**: selected `single_seed` (`Δacc=+0.0000`, `Δkl=+0.0000`)

### Per-Backbone Raw 5-Fold OOF (`task5_hdropt`)

| Candidate Spec | Hugging Face ID | Raw OOF Acc | Raw OOF KL | Raw OOF Brier | Raw OOF ECE |
|---|---|---:|---:|---:|---:|
| `modernbert_zs` | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | 0.7947 | 0.0902 | 0.0518 | 0.1556 |

- **L-BFGS Per-Type Temperatures (`task5_hdropt`)**: `noul=1.0981`, `choice=1.0771`, `score=1.0837`
- **3-Seed Weight Soup Gate (`task5_hdropt`)**: selected `single_seed` (`Δacc=+0.0000`, `Δkl=+0.0000`)

### Per-Backbone Raw 5-Fold OOF (`phase1_rc`)

| Candidate Spec | Hugging Face ID | Raw OOF Acc | Raw OOF KL | Raw OOF Brier | Raw OOF ECE |
|---|---|---:|---:|---:|---:|
| `ettin_400m_s2` | `jhu-clsp/ettin-encoder-400m` | 0.7937 | 0.0918 | 0.0529 | 0.1610 |
| `modernbert_zs` | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | 0.7965 | 0.0875 | 0.0500 | 0.1579 |

### Task 6 Carved Real-Train Holdout Sweep (`phase1_rc`)

| `beta_oof` | `alpha_wise` | Holdout Acc | Holdout KL | Holdout Brier |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.7900 | 0.0874 | 0.0508 |
| 0.00 | 0.04 | 0.7900 | 0.0876 | 0.0509 |
| 0.00 | 0.10 | 0.7950 | 0.0881 | 0.0512 |
| 0.16 | 0.00 | 0.7933 | 0.0845 | 0.0491 |
| 0.16 | 0.04 | 0.7950 | 0.0850 | 0.0494 |
| 0.16 | 0.10 | 0.7892 | 0.0860 | 0.0499 |
| 0.32 | 0.00 | 0.7958 | 0.0857 | 0.0490 |
| 0.32 | 0.04 | 0.7992 | 0.0857 | 0.0490 |
| 0.32 | 0.10 | 0.7950 | 0.0865 | 0.0494 |
| 0.50 | 0.00 | 0.7867 | 0.0847 | 0.0487 |
| 0.50 | 0.04 | 0.7850 | 0.0851 | 0.0489 |
| 0.50 | 0.10 | 0.7875 | 0.0858 | 0.0493 |

- **L-BFGS Per-Type Temperatures (`phase1_rc`)**: `noul=1.0780`, `choice=1.0595`, `score=1.0688`
- **3-Seed Weight Soup Gate (`phase1_rc`)**: selected `best_seed` (`Δacc=+0.0017`, `Δkl=+0.0008`)

### Per-Backbone Raw 5-Fold OOF (`unified_rc`)

| Candidate Spec | Hugging Face ID | Raw OOF Acc | Raw OOF KL | Raw OOF Brier | Raw OOF ECE |
|---|---|---:|---:|---:|---:|
| `ettin_400m_s2` | `jhu-clsp/ettin-encoder-400m` | 0.7903 | 0.0906 | 0.0524 | 0.1552 |
| `modernbert_zs` | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | 0.7938 | 0.0911 | 0.0518 | 0.1569 |

- **L-BFGS Per-Type Temperatures (`unified_rc`)**: `noul=1.0924`, `choice=1.0626`, `score=1.0635`
- **3-Seed Weight Soup Gate (`unified_rc`)**: selected `single_seed` (`Δacc=+0.0000`, `Δkl=+0.0000`)

## 4. Task 7 Release Candidate Gate Summary

- **Required OOF Accuracy**: `>= 0.8150` (Observed: `0.7912` -> `FAIL`)
- **Required OOF KL**: `<= 0.0828` (Observed: `0.0883` -> `FAIL`)
- **Overall Task 7 Gate**: **FAIL — Do NOT spend `slot-1`**
