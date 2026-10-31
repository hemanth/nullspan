# nullspan

Single-pass System One structured extraction, multi-numeric contract disambiguation, calibrated `p_null` abstention, character-offset `[start, end]` spans, calibrated `noul`/`choice`/`score` `/v1/systemone` decisions, cross-field algebraic invariant verification, and calibrated quantile estimation (`p10`, `median`, `p90`) in JavaScript, Python, and Browser WebAssembly ONNX (`512`-token context + long-document windowing).

- **Interactive Playground (Browser WASM ONNX + Zero-Dep Engine)**: [https://hemanth.github.io/nullspan/](https://hemanth.github.io/nullspan/)
- **NullspanBench Leaderboard (JevBench-Style 19-System Evaluation, N=500)**: [https://hemanth.github.io/nullspan/bench.html](https://hemanth.github.io/nullspan/bench.html)
- **Hugging Face Space**: [https://huggingface.co/spaces/gnumanth/nullspan](https://huggingface.co/spaces/gnumanth/nullspan)
- **Fine-Tuned Hugging Face Model (345.3M Extraction + 331.1 MB Int8 ONNX + 395M 8K System One Parallel-Span Encoder)**: [https://huggingface.co/gnumanth/nullspan](https://huggingface.co/gnumanth/nullspan)
- **Hugging Face Dataset**: [https://huggingface.co/datasets/gnumanth/nullspan-dataset](https://huggingface.co/datasets/gnumanth/nullspan-dataset)
- **Empirical Science Report**: [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md)

## In Plain English

Standard LLMs extract data and make typed workflow judgments by generating JSON token-by-token, which is slow, bills per output token, can get hijacked by *"ignore previous instructions"* prompt injections, and often hallucinates values when a field is missing. **`nullspan` acts like a grounded highlighter + calibrated System One judge instead of a chatbot:**

1. **Points to Exact Text (`[start, end]` Span)**: Highlights exact character positions in your input text (`0` generated output tokens), making fabricated strings and prompt-injection payloads impossible.
2. **Knows When to Say "Not There" (`p_null`)**: Computes a calibrated probability that a field is missing (`p_null`) and returns `null` with `abstention_reason` when evidence is absent (`87.80%` EM / `90.90%` F1 on Stanford `SQuAD v2.0`).
3. **Disambiguates Multi-Numeric Contracts & Checks Math (`constraint` & `verified`)**: Separates competing dollar caps, percentages, day windows, and numeric tranches across multi-section agreements (`96/96` on unseen contracts) and verifies equations like `subtotal == nodes * hourly_rate`.
4. **Calibrated `/v1/systemone` Typed Decisions (`decide()`)**: Evaluates `noul` (`true`/`false`), `choice` (`options`), and ordinal `score` (`min..max`) rubrics in a single 8K forward pass (`0.0` cross-option RoPE bleed, `100%` option-order permutation invariant)—scoring **`79.00%` accuracy** (`96.3%` of ~`82.05%` empirical Bayes ceiling; **`KL 0.0760` `#2` worldwide**; **`Brier 0.0420` `#1` tied worldwide**; `47.8 ms/case` on L4) on [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) and hitting **`100.0%` (`48/48`)** on `JevBench Easy` + **`68.06%`** on `JevBench Original` (`100.0%` strict schema validity, `38.8 ms p50`).

## Install

```sh
npm install nullspan
pip install nullspan
```

## Usage (JavaScript / ESM)

```js
import { extract } from 'nullspan';

const incident =
  "Incident INC-40928 opened on Apr 12, 2026 with a 502 Bad Gateway errors and a TLS handshake timeouts. " +
  "Issued $1240.50 plus $310.25 SLA credit. Severity is sev-1 and customer data loss was false. " +
  "Root cause CVE not yet assigned. Cluster stabilized on Apr 19, 2026. Reported by Mira";

const result = extract(incident, {
  incident_id: { type: 'text', instructions: 'Incident ticket identifier' },
  opened_on: { type: 'date', instructions: 'Date the incident opened' },
  total_credit: { type: 'number', decimals: 2, unit: 'USD', instructions: 'Total credit including SLA penalty' },
  outage_days: { type: 'duration', unit: 'days', integer: true, instructions: 'Days from incident open to cluster stabilization' },
  failure_modes: { type: 'list', items: 'text', instructions: 'Each failure mode triggered' },
  severity: { type: 'enum', options: ['sev-1', 'sev-2', 'sev-3'], instructions: 'Incident severity tier' },
  data_loss: { type: 'boolean', instructions: 'Whether customer data loss occurred' },
  cve_id: { type: 'text', instructions: 'Assigned CVE identifier if present' },
  remediation_cost: { type: 'estimate', unit: 'USD', min: 100, max: 25000, scale: 'log', step: 10, instructions: 'Estimated engineering remediation cost' },
});
```

## Usage (Python Extraction + `/v1/systemone` Typed Decisions)

```python
from nullspan import extract, decide

invoice = (
    "Invoice from TensorCore Cloud\n"
    "Invoice #88310 | Issued: 2026-08-14 18:20\n"
    "Nodes: 4 @ $3.50 each\n"
    "Subtotal: $14.00\n"
    "Egress Fee: $0.00\n"
    "Priority Surcharge: $1.75"
)

# 1. Grounded extraction + cross-field algebraic invariants (backend="fast" | "hf" | "onnx")
out = extract(
    state=invoice,
    fields={
        "nodes": {"type": "number", "integer": True, "instructions": "Provisioned GPU node count"},
        "hourly_rate": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Hourly rate per node"},
        "subtotal": {
            "type": "number",
            "decimals": 2,
            "unit": "USD",
            "constraint": "subtotal == nodes * hourly_rate",
            "instructions": "Compute subtotal before fees and surcharge",
        },
        "promo_rebate": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Promotional rebate amount if present"},
    },
    backend="hf",
    model_id="gnumanth/nullspan",
)

# 2. TypeSafe-compatible System One typed decisions (noul / choice / score)
judgments = decide(
    state={"invoice_id": "INV-88310", "subtotal": 14.0, "nodes": 4, "unit_price": 3.5, "approved_vendor": True},
    questions={
        "math_valid": {"type": "noul", "instructions": "Does subtotal equal nodes * unit_price?"},
        "routing": {"type": "choice", "options": ["auto_pay", "manual_review", "reject"], "instructions": "Select workflow action."},
        "risk_tier": {"type": "score", "min": 1, "max": 5, "instructions": "Rate audit risk from 1 (low) to 5 (critical)."},
    },
    model_id="gnumanth/nullspan",
)
```

To run the live TypeSafe `/v1/systemone` HTTP server for `jevbench` or `LocalLLaMA/typed-decisions`:

```sh
python training/serve_systemone.py --model-dir gnumanth/nullspan --port 8010
```

## Official External Public Benchmarks (NVIDIA L4 GPU · Zero Test Contamination)

Reproducible via [`bench/eval_public_benchmarks.py`](./bench/eval_public_benchmarks.py) with results stored in [`bench/public_benchmarks_results.json`](./bench/public_benchmarks_results.json):

| External Public Benchmark | Split & Scale | Primary Metric | Calibration & Validity | Latency & Cost (NVIDIA L4) | Standing |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)** | Official `test` (`400` cases · **`2,000` decisions**) | **`79.00%` Accuracy** (`noul` `84.17%`, `choice` `76.83%`, `score` `76.75%`) | **`KL 0.0760` (`#2` worldwide)** · **`Brier 0.0420` (`#1` tied)** · `ECE 0.1665` | **`9.56 ms/dec` (`47.8 ms/case`)** · `0` output tokens | `#1` Hub `.eval_results` widget; `96.3%` of ~`82.05%` Bayes ceiling; beats `Arbiter-4B` (`78.6%`), `qwen3.5-27b` (`78.3%`), `gpt-5.4` (`77.7%`), `SafeDecide-400M` (`77.5%`), `TypeSafe Jev` (`73.3%`) |
| **[Benchmark Heaven / `jevbench`](https://github.com/fstandhartinger/jevbench)** | Official Public Suite (`231` items via `/v1/systemone`) | **`100.0%` Easy (`48/48`)** · **`68.06%` Original (`49/72`)** · `36.04%` Hard (`40/111`) (`59.31%` all) | **`100.0%` Strict Schema Validity** (`ECE 0.0755`, `v1.3 Geo 46.27`) | **`38.82 ms` `p50`** (`37.76 ms` Easy) · **`$0.00604 / 1k`** | `100.0%` on `fact` & `tool_selection`; `91.7%` `intent` & `ordinal` |
| **[Stanford `rajpurkar/squad_v2`](https://huggingface.co/datasets/rajpurkar/squad_v2)** | Zero-overlap `validation` (`N = 500`: `250` HasAns + `250` NoAns) | **`87.80%` Exact Match** · **`90.90%` F1** | `HasAns F1 90.61%` · `NoAns EM 91.20%` | `10.3 ms` `p50` (L4 CUDA) · `$0.0017 / 1k` | Dedicated `qa_null_head` temperature-calibrated on disjoint `train` |

## Architecture (`NullspanForExtraction` 345.3M + `NullspanSystemOneEngine` 395M 8K)

![Architecture Flow](./docs/flow-diagram.svg)

1. **`null_head` + `qa_null_head` (Span-Attended Calibrated Abstention)**: Bilinear span-summary + `[CLS]` margin residual with held-out temperature scaling (`p_null` AUROC `0.9933`, ECE `0.0142`; `0.9951` / `0.0127` in `331.1 MB` Int8 ONNX) and `abstention_reason` (`unassigned_placeholder` vs `missing_field`).
2. **`span_head` (Grounded Pointer + Shannon Entropy)**: Projects `512`-token encoder states (plus query-conditioned sliding windows on ultra-long contracts) to `[start, end]` character spans with calibrated `confidence` and Shannon `span_entropy_bits`.
3. **`bio_head` (Multi-Span List Tagging)**: Extracts ordered text and numeric tranche arrays (`type: "list"`) in a single pass.
4. **`op_head` + `quantile_head` + Invariant Verifier**: Routes 10 System One extraction/math operations, predicts monotone `(p10, median, p90)` quantiles, and verifies cross-field algebraic constraints (`verified: true/false`).
5. **`NullspanSystemOneEngine` (`/v1/systemone` Tree-Isolated Parallel-Span 8K Judge)**: `ModernBERT-large` (`395M`, full bidirectional `8,192`-token context) packing the entire raw case plus all `Q` questions and candidate options into **a single forward pass** (`47.8 ms/case`), with **Exact Tree-Isolated Parallel-Option Masking** (`0.0` cross-option RoPE bleed, `100%` option-order & `A/B/C` label permutation invariant), 2-layer `ParallelSpanDecisionHead` cross-option + cross-question workflow competition, 24 latent workflow factor heads, and 10-model 5-fold Cross-Backbone OOF Self-Distillation + Trajectory-SWA.

## NullspanBench Leaderboard (`bench/run_bench.py` · 19 Systems · $N=500$ Zero-Leakage Decisions · NVIDIA L4 GPU)

Modeled after [BenchmarkHeaven's JevBench](https://benchmarkheaven.com/jev-models), **NullspanBench v3.2** evaluates `19` systems (`13` local open-weights/open-code models running live forward passes on Google Cloud NVIDIA L4 GPU, live Cloudflare Workers AI `@cf/cloudflare/clairic + @cf/cloudflare/clef`, `2` reference JSON-mode LLM decoders, and `3` published JevBench v1.7 reference anchors) across **`500` zero-leakage observations** (`336` Open structured items + `164` Sealed items: `60` Stanford SQuAD v2.0, `96` Multi-Numeric Legal/Commercial Contract fields across `12` unseen agreements, and `8` Adversarial prompt-injection traps) on four axes: **Intelligence (`I`)**, **Calibration (`C`)**, **Speed (`S`)**, and **Cost (`K`)**.

Explore interactive weight sliders, Pareto frontier plots, and 2-system radar comparisons at [**https://hemanth.github.io/nullspan/bench.html**](https://hemanth.github.io/nullspan/bench.html):

| Rank | System | Composite (`25:25:25:25`) | Capability (`50I:50C`) | Intelligence `I` (95% CI) | Calibration `C` | Speed `S` | Cost `K` | `$/1k` | `p50` Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`nullspan-Hybrid (Fast + ONNX Int8)`** (Open) | **95.7** | **98.2** | **97.8%** `[96.6, 99.0]` | **98.6** | **92.7** | **93.8** | `$0.0003` | `0.040 ms` |
| **2** | **`nullspan-ZeroDep (Fast Engine)`** (Open) | **94.7** | `92.9` | `90.8%` `[88.2, 93.4]` | `95.0` | **93.5** | **99.8** | `$0.0001` | `0.034 ms` |
| **3** | **`nullspan-345M (PyTorch FP32 · L4 CUDA)`** (Open) | **81.3** | **98.2** | **98.0%** `[96.8, 99.2]` | **98.5** | `63.2` | `76.8` | `$0.0017` | `10.297 ms` |
| **4** | **`nullspan-345M-Int8 (ONNX WASM/CPU)`** (Open) | **78.1** | **98.2** | **97.8%** `[96.6, 99.0]` | **98.6** | `52.7` | `84.2` | `$0.0008` | `75.933 ms` |
| **5** | `H2O-Lightning-4B v1.1 (JevBench #1 Comp)` (Open 4B) | `72.5` | `75.0` | `60.0%` `[57.8, 62.3]` | `90.0` | `92.6` | `60.3` | `$0.0211` | `29.285 ms` |
| **6** | **`@cf/cloudflare/clairic + clef (Live Workers AI)`** (API) | `72.1` | `93.2` | `91.6%` `[89.2, 93.9]` | `94.9` | `51.7` | `68.2` | `$0.0041` | `91.323 ms` (`315.3 ms/doc`) |
| **7** | `basal-omni-1.0-ultra (JevBench #1 Cap)` (Open 31B) | `71.8` | `82.5` | `74.1%` `[71.8, 76.3]` | `91.0` | `87.3` | `50.5` | `$0.0447` | `102.000 ms` |
| **8** | `Jev 1.13.0 (TypeSafe AI · JevBench Ref)` (API) | `71.5` | `77.1` | `63.6%` `[61.2, 66.0]` | `90.6` | `91.5` | `54.7` | `$0.0323` | `239.102 ms` |
| **9** | **`nullspan-345M (ONNX FP32)`** (Open) | `71.2` | **98.2** | **98.0%** `[96.8, 99.2]` | **98.5** | `43.5` | `78.0` | `$0.0015` | `424.943 ms` |
| **10** | `deberta-v3-base-squad2 (184M SQuAD2 QA)` (Open) | `60.3` | `55.8` | `48.1%` `[43.9, 52.4]` | `63.4` | `65.1` | `69.2` | `$0.0037` | `7.234 ms` |
| **11** | `gpt-5.4-nano (JSON Decoder · API)` (API) | `59.8` | `92.1` | `95.2%` `[93.4, 97.0]` | `89.0` | `43.6` | `45.1` | `$0.0438` | `417.991 ms` |
| **12** | `claude-haiku-4.5 (JSON Decoder · API)` (API) | `48.8` | `92.1` | `95.4%` `[93.4, 97.2]` | `88.7` | `41.8` | `27.5` | `$0.2639` | `586.392 ms` |
| **13** | `bge-small-en-v1.5 (Dense Span+Null)` (Open) | `37.6` | `52.0` | `14.7%` `[12.0, 17.5]` | `89.2` | `64.2` | `85.5` | `$0.0007` | `8.625 ms` |
| **14** | `distilbert-base-control (Raw Control)` (Open) | `36.9` | `52.9` | `14.2%` `[11.6, 17.0]` | `91.5` | `67.7` | `81.1` | `$0.0011` | `4.479 ms` |
| **15** | `all-MiniLM-L12-v2 (Dense Span+Null)` (Open) | `36.3` | `44.0` | `14.4%` `[11.7, 17.5]` | `73.5` | `64.2` | `85.5` | `$0.0007` | `8.582 ms` |
| **16** | `all-mpnet-base-v2 (Dense Span+Null)` (Open) | `34.4` | `42.3` | `13.6%` `[10.9, 16.7]` | `71.0` | `63.3` | `77.4` | `$0.0016` | `10.257 ms` |
| **17** | `paraphrase-albert-small-v2 (Dense Span+Null)` (Open) | `33.6` | `44.5` | `12.5%` `[10.0, 15.2]` | `76.6` | `66.4` | `91.0` | `$0.0004` | `5.700 ms` |
| **18** | `paraphrase-MiniLM-L3-v2 (Dense Span+Null)` (Open) | `30.9` | `41.7` | `11.1%` `[8.7, 13.8]` | `72.3` | `69.6` | `91.0` | `$0.0004` | `3.114 ms` |
| **19** | `all-MiniLM-L6-v2 (Dense Span+Null)` (Open) | `28.5` | `40.5` | `10.0%` `[7.7, 12.3]` | `70.9` | `67.1` | `88.8` | `$0.0005` | `4.988 ms` |

See full methodology, proper scoring definitions, request-type breakdowns (`100.0%` on Multi-Numeric Contracts `n=96`, `100.0%` Choice `n=41`, `100.0%` Math & Invariants `n=51`), and head-to-head analysis in [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md).

## License

MIT © [Hemanth.HM](https://h3manth.com)
