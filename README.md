# nullspan

Single-pass System One structured extraction, calibrated `p_null` abstention, character-offset `[start, end]` spans, zero-shot `boolean`/`enum` judgment, cross-field algebraic invariant verification, and calibrated quantile estimation (`p10`, `median`, `p90`) in JavaScript, Python, and Browser WebAssembly ONNX.

- **Interactive Playground (Browser WASM ONNX + Zero-Dep Engine)**: [https://hemanth.github.io/nullspan/](https://hemanth.github.io/nullspan/)
- **NullspanBench Leaderboard (JevBench-Style 14-System Evaluation)**: [https://hemanth.github.io/nullspan/bench.html](https://hemanth.github.io/nullspan/bench.html)
- **Hugging Face Space**: [https://huggingface.co/spaces/gnumanth/nullspan](https://huggingface.co/spaces/gnumanth/nullspan)
- **Fine-Tuned Hugging Face Model (67M PyTorch + 64.3 MB Int8 ONNX)**: [https://huggingface.co/gnumanth/nullspan](https://huggingface.co/gnumanth/nullspan)
- **Hugging Face Dataset**: [https://huggingface.co/datasets/gnumanth/nullspan-dataset](https://huggingface.co/datasets/gnumanth/nullspan-dataset)
- **Empirical Science Report**: [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md)

## In Plain English

Standard LLMs extract data by generating JSON token-by-token—which is slow, costs money per output token, can get hijacked by *"ignore previous instructions"* prompt injections, and often hallucinates values when a field is missing. **`nullspan` acts like a smart highlighter + calculator instead of a chatbot:**

1. **Points to Exact Text (`[start, end]` Span)**: Highlights exact character positions in your input text (`0` output tokens generated), making fabricated strings and prompt-injection payloads impossible.
2. **Knows When to Say "Not There" (`p_null`)**: Computes a calibrated probability that a field is missing (`p_null`) and returns `null` when evidence is absent.
3. **Checks the Math Automatically (`constraint` & `verified`)**: Computes sums, durations, and products and verifies cross-field equations like `subtotal == nodes * hourly_rate`.
4. **Gives Realistic Ranges (`p10 / median / p90`)**: Returns calibrated low/middle/high quantile distributions for estimates and forecasts.

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

## Usage (Python + Cross-Field Invariants + Hugging Face Backend)

```python
from nullspan import extract

invoice = (
    "Invoice from TensorCore Cloud\n"
    "Invoice #88310 | Issued: 2026-08-14 18:20\n"
    "Nodes: 4 @ $3.50 each\n"
    "Subtotal: $14.00\n"
    "Egress Fee: $0.00\n"
    "Priority Surcharge: $1.75"
)

# Zero-dependency fast engine or fine-tuned neural checkpoint (backend="hf")
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
# out["invariants_passed"] -> True, out["answers"]["subtotal"]["verified"] -> True
```

## Architectural Enhancements

![Architecture Flow](./docs/flow-diagram.svg)

1. **`null_head` (Calibrated Abstention & Reason Diagnostics)**: Predicts calibrated `p_null`, nearest decoy `guess`, and `abstention_reason` (`unassigned_placeholder` vs `missing_field`).
2. **`span_head` (Grounded Pointer + Shannon Entropy)**: Projects encoder tokens to `[start, end]` character spans with calibrated `confidence` and Shannon `span_entropy_bits`.
3. **`bio_head` (Class-Weighted Multi-Span Tagging)**: Extracts arbitrary-length ordered arrays (`type: "list"`) in a single pass.
4. **`op_head` (10-Way System One Router)**: Routes fields across `SPAN_TEXT`, `SPAN_NUMBER`, `ARITH_SUM`, `ARITH_DIFF`, `ARITH_MUL`, `ISO_DATETIME`, `DATE_DURATION`, `DIST_ESTIMATE_FORECAST`, `BOOL_JUDGE` (`p_true`), and `ENUM_CLASSIFY` (`probs`).
5. **`quantile_head` + Invariant Verifier**: Predicts monotone `(p10, median, p90)` quantiles and verifies cross-field algebraic constraints (`verified: true/false`).

## NullspanBench Leaderboard (`bench/run_bench.py` · 14 Systems · $N=404$ Scored Decisions)

Modeled after [BenchmarkHeaven's JevBench](https://benchmarkheaven.com/jev-models), **NullspanBench** evaluates `14` systems (`12` local open-weights/open-code models running live forward passes + `2` reference JSON-mode LLM decoders) across **`404` scored observations** (`336` Open structured items + `68` Sealed Stanford SQuAD v2.0 & adversarial prompt-injection items) on four axes: **Intelligence (`I`)**, **Calibration (`C`)**, **Speed (`S`)**, and **Cost (`K`)**.

Explore interactive weight sliders, Pareto frontier plots, and 2-system radar comparisons at [**https://hemanth.github.io/nullspan/bench.html**](https://hemanth.github.io/nullspan/bench.html):

| Rank | System | Composite (`25:25:25:25`) | Capability (`50I:50C`) | Intelligence `I` (95% CI) | Calibration `C` | Speed `S` | Cost `K` | `$/1k` | `p50` Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`nullspan-Hybrid (Fast + ONNX Int8)`** (Open) | **97.7** | **98.7** | **98.0%** `[96.5, 99.3]` | **99.3** | **99.8** | **93.8** | `$0.0003` | `0.005 ms` |
| **2** | **`nullspan-ZeroDep (Fast Engine)`** (Open) | **95.3** | **91.2** | `88.6%` `[85.6, 91.6]` | `93.8` | **99.8** | **99.8** | `$0.0001` | `0.005 ms` |
| **3** | **`nullspan-67M (PyTorch FP32)`** (Open) | **83.3** | **99.2** | **98.8%** `[97.5, 99.8]` | **99.7** | `66.5` | `78.0` | `$0.0015` | `5.575 ms` |
| **4** | **`nullspan-67M-Int8 (ONNX WASM/CPU)`** (Open) | **83.0** | **98.7** | **98.0%** `[96.5, 99.3]` | **99.3** | `62.3` | `84.2` | `$0.0008` | `12.268 ms` |
| **5** | **`nullspan-67M (ONNX FP32)`** (Open) | **82.1** | **99.2** | **98.8%** `[97.5, 99.8]` | **99.7** | `63.0` | `78.7` | `$0.0014` | `10.832 ms` |
| **6** | `gpt-5.4-nano (JSON Decoder · API)` | `60.3` | `91.5` | `95.3%` `[93.1, 97.3]` | `87.7` | `43.6` | `46.5` | `$0.0380` | `415.9 ms` |
| **7** | `claude-haiku-4.5 (JSON Decoder · API)` | `49.7` | `92.3` | `96.3%` `[94.3, 98.0]` | `88.3` | `41.8` | `28.7` | `$0.2328` | `588.0 ms` |
| **8** | `bge-small-en-v1.5 (Dense Span+Null)` (Open) | `42.0` | `53.3` | `17.5%` `[13.8, 21.3]` | `89.1` | `64.7` | `87.0` | `$0.0006` | `7.855 ms` |
| **9** | `distilbert-base-control (Raw Control)` (Open) | `41.6` | `53.6` | `17.3%` `[13.6, 21.1]` | `90.0` | `65.3` | `84.2` | `$0.0008` | `6.947 ms` |
| **10** | `all-MiniLM-L12-v2 (Dense Span+Null)` (Open) | `37.6` | `44.8` | `15.1%` `[11.9, 18.6]` | `74.6` | `65.4` | `87.0` | `$0.0006` | `6.894 ms` |
| **11** | `all-mpnet-base-v2 (Dense Span+Null)` (Open) | `35.3` | `43.1` | `14.1%` `[10.9, 17.5]` | `72.2` | `62.8` | `80.2` | `$0.0012` | `11.098 ms` |
| **12** | `paraphrase-albert-small-v2 (Dense Span+Null)` (Open) | `34.8` | `45.1` | `13.1%` `[9.9, 16.4]` | `77.2` | `66.8` | `93.8` | `$0.0003` | `5.224 ms` |
| **13** | `paraphrase-MiniLM-L3-v2 (Dense Span+Null)` (Open) | `31.5` | `42.2` | `11.3%` `[8.4, 14.5]` | `73.1` | `70.6` | `93.8` | `$0.0003` | `2.586 ms` |
| **14** | `all-MiniLM-L6-v2 (Dense Span+Null)` (Open) | `28.5` | `41.2` | `9.9%` `[7.1, 13.0]` | `72.6` | `68.0` | `91.0` | `$0.0004` | `4.200 ms` |

See full methodology, proper scoring definitions, request-type breakdowns, and ablation analysis in [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md).

## License

MIT © [Hemanth.HM](https://h3manth.com)
