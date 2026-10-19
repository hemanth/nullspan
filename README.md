# nullspan

Single-pass System One structured extraction, calibrated `p_null` abstention, character-offset `[start, end]` spans, zero-shot `boolean`/`enum` judgment, cross-field algebraic invariant verification, and calibrated quantile estimation (`p10`, `median`, `p90`) in JavaScript, Python, and Browser WebAssembly ONNX.

- **Interactive Playground (Browser WASM ONNX + Zero-Dep Engine)**: [https://hemanth.github.io/nullspan/](https://hemanth.github.io/nullspan/)
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

## Multi-Tier Scientific Evaluation (`bench/run_bench.py` · 95% Bootstrap CIs, $B=1,000$)

| Evaluation Axis / Metric | Tier 1A: PyTorch FP32 (67M Neural) | Tier 1B: ONNX Int8 (64.3 MB Browser/Edge) | Tier 2: Zero-Dep Fast Engine |
|---|---|---|---|
| **Structured Multi-Task Accuracy** ($N=42$ fields, 95% CI) | **100.0%** `[100.0, 100.0]` | **100.0%** `[100.0, 100.0]` | **100.0%** `[100.0, 100.0]` |
| **Grounded Span Exact Match & Invariant Verification** | **100.0% / 100.0%** | **100.0% / 100.0%** | **100.0% / 100.0%** |
| **Structured Null Calibration** (`AUROC` / `ECE` / `Brier`) | **1.0000 / 0.00002 / 0.0000** | **1.0000 / 0.00001 / 0.0000** | **1.0000 / 0.00005 / 0.0000** |
| **Quantile Monotonicity & Winkler Score** ($\alpha=0.20$) | **100.0% (0.3723)** | **100.0% (0.3726)** | **100.0% (0.3671)** |
| **Adversarial Prompt-Injection & Decoy Suite** ($N=8$, FPHR) | **100.0% (0.0% FPHR)** | **100.0% (0.0% FPHR)** | **100.0% (0.0% FPHR)** |
| **Stanford SQuAD v2.0 Open-Domain QA** ($N=60$, 95% CI) | **91.67%** `[83.33, 98.33]` | **90.00%** `[81.67, 96.67]` | **23.33%** `[13.33, 35.00]` |
| **SQuAD v2.0 Abstention AUROC / ECE / Brier** | **0.9989 / 0.0135 / 0.0145** | **0.9956 / 0.0402 / 0.0250** | **0.4039 / 0.4194 / 0.4283** |
| **Footprint & Latency (`p50` / `p95` per field)** | **255.8 MB · 12.3 / 13.8 ms** | **64.3 MB (3.98x) · 25.3 / 27.3 ms** | **0 MB · 0.023 / 0.081 ms** |

See full methodology, proper scoring definitions, and ablation analysis in [`bench/SCIENCE_REPORT.md`](./bench/SCIENCE_REPORT.md).

## License

MIT © [Hemanth.HM](https://h3manth.com)
