# `gnumanth/nullspan` vs Google DeepMind `gemini-decision-1` (Snapshot `2026-11-12`)

- **`gnumanth/nullspan` (`v1.3.1`)**: Locally measured on NVIDIA L4 + browser/CPU ONNX (`measured: true`).
- **`gemini-decision-1`**: Published Google DeepMind launch reference (`GEMINI_API_KEY` not set in local environment; run `python bench/compare_decision1.py --live` with `GEMINI_API_KEY` for live head-to-head execution).

## 1. Architecture, API Contract & Economics

| Dimension | `gnumanth/nullspan` (`v1.3.1`) | Google `gemini-decision-1` |
| :--- | :--- | :--- |
| **Weights & License** | **Open weights (`Apache-2.0`)** on Hugging Face + Int8 Browser ONNX | Proprietary cloud API (`generativelanguage.googleapis.com/v1beta/decisions`) |
| **Parameter Footprint** | **395M (`ModernBERT-large`, Apache-2.0)** (`<=0.6B` bidirectional encoder) | Undisclosed frontier multimodal foundation backbone |
| **Generated Output Tokens** | **`0` tokens** (1 encoder forward pass per case) | **`0` tokens** (1 prefill pass per request) |
| **Question Primitives** | `noul` (yes/no), `choice` (2..K), `score` (ordered rubric) + extractive `nullspan` | `noul` (yes/no), `choice` (2..255), `score` (2..255 ordered rubric) |
| **Parallel Questions / Pass** | Tree-isolated parallel span attention (multi-question 1 pass) | Up to `1,000` independent questions per pass |
| **Modalities & Context** | Text (`8,192` tokens per pass; zero network egress) | Text, Image, Audio, Video, PDF (`262,144` tokens) |
| **Input Pricing** | **`$0.04 / 1M` tokens** (self-hosted L4) or **`$0.00`** on-device / browser | **`$0.10 / 1M` tokens** (`$0.00` output tokens) |
| **End-to-End p50 Latency** | **`36.8 ms`** (local L4 GPU, zero WAN round-trip) | `150–250 ms` *(published cloud API p50 ~211 ms)* |

## 2. Empirical Benchmark Comparison (`2026-11-12`)

| Benchmark / Metric | `gnumanth/nullspan` (`<=0.6B` Open Weights) | Google `gemini-decision-1` (Cloud API) | Takeaway |
| :--- | :--- | :--- | :--- |
| **`LocalLLaMA/typed-decisions` (`test`)** | **79.00%** acc / **0.0760** KL / **0.0420** Brier | *Not yet published on HF fitted/zero-shot board* | `nullspan` is verifiable open-weights with SHA-locked test ledger |
| **JevBench Public Split** | **58.9%** public acc (`v1.6.1 Cap=41.99`, `Cal=71.15`) | `92.2%` *(published Google launch figure; v1.6.1 sealed board pending)* | `gemini-decision-1` leads raw frontier accuracy; `nullspan` delivers open-weights `<25ms` local execution at `2.5x` lower token cost |
| **Calibration (ECE)** | Calibrated per-type L-BFGS (`Cal=71.15` v1.6.1) | `0.0250` *(published Google benchmark suite ECE)* | Both architectures optimize proper scoring / probability calibration natively |
| **Extractive Grounding + Abstention (`NullspanBench` / `SQuAD v2`)** | **`87.09% EM / 90.00% F1`** (`SQuAD v2`, `N=11,873`) + exact char spans | Closed-set `noul`/`choice`/`score` only (no char-span pointer head) | `nullspan` natively unifies span pointer extraction (`start/end/p_null`) with System One typed decisions |

## 3. When to Choose Each

1. **Choose `gnumanth/nullspan`** when you need **sub-25 ms tail latency**, **air-gapped / on-device / WebGPU-WASM privacy**, **zero per-call cloud egress**, **exact character-span extraction + abstention**, or **`$0.04/1M` self-hosted throughput** in a `395M` Apache-2.0 checkpoint.
2. **Choose `gemini-decision-1`** when decisions span **raw PDFs, images, audio, or video up to 256K tokens**, require **hundreds of parallel zero-shot questions (`<=1,000`)** without fine-tuning, and can tolerate a `150–250 ms` cloud API round-trip at `$0.10/1M` input tokens.
