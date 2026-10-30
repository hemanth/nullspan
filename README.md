# nullspan

Single-pass System One structured extraction, multi-numeric contract disambiguation, calibrated `p_null` abstention, character-offset `[start, end]` spans, zero-shot `boolean`/`enum` judgment, cross-field algebraic invariant verification, and calibrated quantile estimation (`p10`, `median`, `p90`) in JavaScript, Python, and Browser WebAssembly ONNX (`512`-token context + long-document windowing).

- **Interactive Playground (Browser WASM ONNX + Zero-Dep Engine)**: [https://hemanth.github.io/nullspan/](https://hemanth.github.io/nullspan/)
- **NullspanBench Leaderboard (JevBench-Style 19-System Evaluation, N=500)**: [https://hemanth.github.io/nullspan/bench.html](https://hemanth.github.io/nullspan/bench.html)
- **Hugging Face Space**: [https://huggingface.co/spaces/gnumanth/nullspan](https://huggingface.co/spaces/gnumanth/nullspan)
- **Fine-Tuned Hugging Face Model (344.2M PyTorch + 330.1 MB Int8 ONNX)**: [https://huggingface.co/gnumanth/nullspan](https://huggingface.co/gnumanth/nullspan)
- **Hugging Face Dataset**: [https://huggingface.co/datasets/gnumanth/nullspan-dataset](https://huggingface.co/datasets/gnumanth/nullspan-dataset)
- **Empirical Science Report**: [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md)

## In Plain English

Standard LLMs extract data by generating JSON token-by-token, which is slow, bills per output token, can get hijacked by *"ignore previous instructions"* prompt injections, and often hallucinates values when a field is missing. **`nullspan` acts like a grounded highlighter + calculator instead of a chatbot:**

1. **Points to Exact Text (`[start, end]` Span)**: Highlights exact character positions in your input text (`0` output tokens generated), making fabricated strings and prompt-injection payloads impossible.
2. **Knows When to Say "Not There" (`p_null`)**: Computes a calibrated probability that a field is missing (`p_null`) and returns `null` with `abstention_reason` when evidence is absent.
3. **Disambiguates Multi-Numeric Contracts & Checks Math (`constraint` & `verified`)**: Separates competing dollar caps, percentages, day windows, and numeric tranches across multi-section agreements (`96/96` on unseen contracts) and verifies equations like `subtotal == nodes * hourly_rate`.
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

# Zero-dependency fast engine or GCP V100 fine-tuned 344.2M checkpoint (backend="hf" or "onnx")
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

## Architecture (`NullspanForExtraction` v2.0 · 344.2M ELECTRA-Large + Adapter)

![Architecture Flow](./docs/flow-diagram.svg)

1. **`null_head` (Span-Attended Calibrated Abstention)**: Bilinear span-summary + `[CLS]` margin residual with held-out temperature scaling (`p_null` AUROC `0.9983`, ECE `0.0121`) and `abstention_reason` (`unassigned_placeholder` vs `missing_field`).
2. **`span_head` (Grounded Pointer + Shannon Entropy)**: Projects `512`-token encoder states (plus query-conditioned sliding windows on ultra-long contracts) to `[start, end]` character spans with calibrated `confidence` and Shannon `span_entropy_bits`.
3. **`bio_head` (Multi-Span List Tagging)**: Extracts ordered text and numeric tranche arrays (`type: "list"`) in a single pass.
4. **`op_head` (10-Way System One Router)**: Routes fields across `SPAN_TEXT`, `SPAN_NUMBER`, `ARITH_SUM`, `ARITH_DIFF`, `ARITH_MUL`, `ISO_DATETIME`, `DATE_DURATION`, `DIST_ESTIMATE_FORECAST`, `BOOL_JUDGE` (`p_true`), and `ENUM_CLASSIFY` (`probs`).
5. **`quantile_head` + Invariant Verifier**: Predicts monotone `(p10, median, p90)` quantiles and verifies cross-field algebraic constraints (`verified: true/false`).

## NullspanBench Leaderboard (`bench/run_bench.py` · 19 Systems · $N=500$ Zero-Leakage Decisions)

Modeled after [BenchmarkHeaven's JevBench](https://benchmarkheaven.com/jev-models), **NullspanBench v3.2** evaluates `19` systems (`13` local open-weights/open-code models running live forward passes, live Cloudflare Workers AI `@cf/cloudflare/clairic + @cf/cloudflare/clef`, `2` reference JSON-mode LLM decoders, and `3` published JevBench v1.7 reference anchors) across **`500` zero-leakage observations** (`336` Open structured items + `164` Sealed items: `60` Stanford SQuAD v2.0, `96` Multi-Numeric Legal/Commercial Contract fields across `12` unseen agreements, and `8` Adversarial prompt-injection traps) on four axes: **Intelligence (`I`)**, **Calibration (`C`)**, **Speed (`S`)**, and **Cost (`K`)**.

Explore interactive weight sliders, Pareto frontier plots, and 2-system radar comparisons at [**https://hemanth.github.io/nullspan/bench.html**](https://hemanth.github.io/nullspan/bench.html):

| Rank | System | Composite (`25:25:25:25`) | Capability (`50I:50C`) | Intelligence `I` (95% CI) | Calibration `C` | Speed `S` | Cost `K` | `$/1k` | `p50` Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`nullspan-Hybrid (Fast + ONNX Int8)`** (Open) | **97.2** | **98.3** | **97.8%** `[96.6, 99.0]` | **98.8** | **98.6** | **93.8** | `$0.0003` | `0.013 ms` |
| **2** | **`nullspan-ZeroDep (Fast Engine)`** (Open) | **95.9** | **92.9** | `90.8%` `[88.2, 93.4]` | `95.0` | **98.7** | **99.8** | `$0.0001` | `0.013 ms` |
| **3** | **`nullspan-344M (PyTorch FP32)`** (Open) | **78.7** | **98.4** | **98.0%** `[96.8, 99.2]` | **98.8** | `57.2` | `76.8` | `$0.0017` | `32.367 ms` |
| **4** | **`nullspan-344M-Int8 (ONNX WASM/CPU)`** (Open) | **78.4** | **98.3** | **97.8%** `[96.6, 99.0]` | **98.9** | `53.2` | `84.2` | `$0.0008` | `68.483 ms` |
| **5** | **`nullspan-344M (ONNX FP32)`** (Open) | **75.5** | **98.4** | **98.0%** `[96.8, 99.2]` | **98.8** | `50.5` | `78.0` | `$0.0015` | `113.137 ms` |
| **6** | `H2O-Lightning-4B v1.1 (JevBench #1 Comp)` (Open 4B) | `72.5` | `75.0` | `60.0%` `[57.8, 62.3]` | `90.0` | `92.6` | `60.3` | `$0.0211` | `29.285 ms` |
| **7** | **`@cf/cloudflare/clairic + clef (Live Workers AI)`** (API) | `72.1` | `93.2` | `91.6%` `[89.2, 93.9]` | `94.9` | `51.7` | `68.2` | `$0.0041` | `91.323 ms` (`315.3 ms/doc`) |
| **8** | `basal-omni-1.0-ultra (JevBench #1 Cap)` (Open 31B) | `71.8` | `82.5` | `74.1%` `[71.8, 76.3]` | `91.0` | `87.3` | `50.5` | `$0.0447` | `102.000 ms` |
| **9** | `Jev 1.13.0 (TypeSafe AI · JevBench Ref)` (API) | `71.5` | `77.1` | `63.6%` `[61.2, 66.0]` | `90.6` | `91.5` | `54.7` | `$0.0323` | `239.102 ms` |
| **10** | `gpt-5.4-nano (JSON Decoder · API)` (API) | `59.8` | `92.1` | `95.2%` `[93.4, 97.0]` | `89.0` | `43.6` | `45.1` | `$0.0438` | `417.991 ms` |
| **11** | `deberta-v3-base-squad2 (184M SQuAD2 QA)` (Open) | `58.0` | `55.8` | `48.1%` `[43.9, 52.4]` | `63.4` | `55.7` | `69.2` | `$0.0037` | `42.496 ms` |
| **12** | `claude-haiku-4.5 (JSON Decoder · API)` (API) | `48.8` | `92.1` | `95.4%` `[93.4, 97.2]` | `88.7` | `41.8` | `27.5` | `$0.2639` | `586.392 ms` |
| **13** | `bge-small-en-v1.5 (Dense Span+Null)` (Open) | `37.6` | `52.0` | `14.7%` `[12.0, 17.5]` | `89.2` | `64.5` | `85.5` | `$0.0007` | `8.098 ms` |
| **14** | `distilbert-base-control (Raw Control)` (Open) | `36.8` | `52.9` | `14.2%` `[11.6, 17.0]` | `91.5` | `66.0` | `81.1` | `$0.0011` | `6.133 ms` |
| **15** | `all-MiniLM-L12-v2 (Dense Span+Null)` (Open) | `35.9` | `44.0` | `14.4%` `[11.7, 17.5]` | `73.5` | `60.2` | `85.5` | `$0.0007` | `18.356 ms` |
| **16** | `all-mpnet-base-v2 (Dense Span+Null)` (Open) | `34.3` | `42.3` | `13.6%` `[10.9, 16.7]` | `71.0` | `62.8` | `77.4` | `$0.0016` | `11.263 ms` |
| **17** | `paraphrase-albert-small-v2 (Dense Span+Null)` (Open) | `33.6` | `44.5` | `12.5%` `[10.0, 15.2]` | `76.6` | `65.9` | `91.0` | `$0.0004` | `6.263 ms` |
| **18** | `paraphrase-MiniLM-L3-v2 (Dense Span+Null)` (Open) | `30.9` | `41.7` | `11.1%` `[8.7, 13.8]` | `72.3` | `68.9` | `91.0` | `$0.0004` | `3.519 ms` |
| **19** | `all-MiniLM-L6-v2 (Dense Span+Null)` (Open) | `28.3` | `40.5` | `10.0%` `[7.7, 12.3]` | `70.9` | `62.3` | `88.8` | `$0.0005` | `12.322 ms` |

See full methodology, proper scoring definitions, request-type breakdowns (`100.0%` on Multi-Numeric Contracts `n=96`, `100.0%` Choice `n=41`, `100.0%` Math & Invariants `n=51`), and head-to-head analysis in [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md).

## License

MIT © [Hemanth.HM](https://h3manth.com)
