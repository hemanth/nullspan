# JevBench v1.6.1 Evaluation Request — `gnumanth/nullspan` (v1.3.0)

- **Channel / Tracking Issue**: `fstandhartinger/jevbench#238` (`https://github.com/fstandhartinger/jevbench/issues/238`)
- **Snapshot Date**: `2026-11-11`
- **Scorer Protocol**: `jevbench::v1.6.1-O1S-local` (`N = 231` public items; official sealed evaluation is run by the JevBench maintainers)

## Model & Pinned Artifacts

| Field | Value |
| :--- | :--- |
| Hugging Face Repository | `https://huggingface.co/gnumanth/nullspan` |
| Pinned HF Revision | `51acf9e097ade3fb1068b3511490ef2756485927` |
| Pinned Source Git SHA | `de900ac4801c9d9f4b55abff68b4c593123d4ed8` |
| Architecture & Category | `ModernBERT-large 395M + ParallelSpanDecisionHead` (`encoder_classifier_<=0.6B`) |
| Execution Mode | Single bidirectional encoder forward pass per request, `0` generated output tokens, zero ensembling |
| Cost Basis (`endpoint_kind=gpu`) | `$0.04 / 1M` input tokens, `$0.00 / 1M` output tokens (`$0.02416 / 1,000` requests) |

## Local JevBench v1.6.1 (O1S) Public Readout

> Note: Public split readout only (`easy.jsonl`, `original.jsonl`, `hard.jsonl`). Official JevBench leaderboard standing requires maintainer execution on the sealed split.

| Metric | Public v1.6.1 Readout |
| :--- | ---: |
| **Capability Score** `(I + C) / 2` | **41.99** (`eligible=True`) |
| **Intelligence** (O1S equal-type) | **12.82** |
| **Calibration** (O1S equal-type) | **71.15** |
| **Speed Score** (`p50_raw=36.8ms`, `p50_adj=223.5ms`) | **92.96** |
| **Cost Score** (`$0.02416 / 1k`) | **58.51** |
| **Composite (Option A)** | **2.19** |
| Public Tier Accuracy (Easy / Original / Hard) | `97.9% / 47.2% / 27.0%` |

## Training & Data Hygiene Disclosure

- **Stage 1 General System One Pool**: `google/boolq` (`train`), `allenai/ai2_arc` (`ARC-Challenge train`), `allenai/sciq` (`train`), and deterministic procedural rubric generators (`training/train_systemone.py`).
- **Stage 2 Workflow Distillation**: `LocalLLaMA/typed-decisions` (`train` split only, 1,200 cases / 6,000 decisions; cross-backbone 5-fold OOF distillation between `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` and `jhu-clsp/ettin-encoder-400m` into a single 395M `ModernBERT-large` student checkpoint with WiSE-FT interpolation).
- **Maintainer Sealed-Split Execution**: Compatible out-of-the-box with `training/serve_systemone.py` (`/v1/decide` HTTP endpoint) or `NullspanSystemOneEngine.load_from_dir(...)` (`1` bidirectional encoder forward pass, `0` generated output tokens).

- **Strict Contamination Guarantee**: Zero JevBench items (public or sealed) were used in pretraining, distillation, hyperparameter tuning, early stopping, or temperature calibration; all synthetic dev/train generators enforce a 12-token shingle rejection filter against `datasets/public/*.jsonl`.
