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
short_description: System One structured extraction, p_null & quantiles
---

# `nullspan` — System One Structured Extraction, Calibrated Abstention (`p_null`) & Quantile Distribution Space

Interactive Hugging Face Space for **[`gnumanth/nullspan`](https://huggingface.co/gnumanth/nullspan)** (`344.2M` parameters, 512-token `ELECTRA-Large` SQuAD2 cross-encoder + 2-layer `SchemaAdapterBlock` + 10-way System One operation router + calibrated Pinball-loss quantile head).

- **In-Browser Neural Execution**: Click **Tier 1: In-Browser Neural ONNX (WASM SIMD · 330.1 MB Int8)** in the toolbar to stream and run [`gnumanth/nullspan/onnx/model_quantized.onnx`](https://huggingface.co/gnumanth/nullspan/blob/main/onnx/model_quantized.onnx) directly inside your browser via WebAssembly SIMD.
- **Local Gradio Python App**: Also includes [`app.py`](./app.py) (`python app.py`) for PyTorch FP32 Gradio execution.
- **GitHub Repository**: [`hemanth/nullspan`](https://github.com/hemanth/nullspan)
- **Dataset**: [`gnumanth/nullspan-dataset`](https://huggingface.co/datasets/gnumanth/nullspan-dataset)
