---
title: Nullspan
emoji: ⚡
colorFrom: gray
colorTo: indigo
sdk: static
pinned: false
license: mit
models:
  - gnumanth/nullspan
datasets:
  - gnumanth/nullspan-dataset
  - LocalLLaMA/typed-decisions
  - rajpurkar/squad_v2
short_description: System One extraction, typed decisions, p_null & quantiles
---

# `nullspan` — System One Structured Extraction, Typed Decisions, Calibrated Abstention (`p_null`) & Quantiles

Interactive Hugging Face Space for **[`gnumanth/nullspan`](https://huggingface.co/gnumanth/nullspan)** (`345.3M` `NullspanForExtraction` ELECTRA-Large + `331.1 MB` Int8 ONNX + `NullspanSystemOneEngine` `/v1/systemone` DeBERTa-v3-Large cross-encoder).

- **In-Browser Neural Execution**: Click **Tier 1: In-Browser Neural ONNX (WASM SIMD · 331.1 MB Int8)** in the toolbar to stream and run [`gnumanth/nullspan/onnx/model_quantized.onnx`](https://huggingface.co/gnumanth/nullspan/blob/main/onnx/model_quantized.onnx) directly inside your browser via WebAssembly SIMD.
- **Interactive 19-System Leaderboard**: Open [`bench.html`](./bench.html) to explore `NullspanBench` (`N = 500` zero-leakage observations).
- **Local Gradio Python App**: Also includes [`app.py`](./app.py) (`python app.py`) for PyTorch FP32 Gradio execution.
- **GitHub Repository**: [`hemanth/nullspan`](https://github.com/hemanth/nullspan)
- **Dataset**: [`gnumanth/nullspan-dataset`](https://huggingface.co/datasets/gnumanth/nullspan-dataset)
