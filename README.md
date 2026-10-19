# nullspan

Single-pass System One structured extraction, calibrated `p_null` abstention, character-offset `[start, end]` spans, zero-shot `boolean`/`enum` judgment, cross-field algebraic invariant verification, and calibrated quantile estimation (`p10`, `median`, `p90`) in JavaScript and Python.

- **Interactive Playground**: [https://hemanth.github.io/nullspan/](https://hemanth.github.io/nullspan/)
- **Fine-Tuned Hugging Face Model (67M)**: [https://huggingface.co/gnumanth/nullspan](https://huggingface.co/gnumanth/nullspan)
- **Hugging Face Dataset**: [https://huggingface.co/datasets/gnumanth/nullspan-dataset](https://huggingface.co/datasets/gnumanth/nullspan-dataset)

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

## 6-Axis Scientific Evaluation (`bench/run_bench.py`)

| Evaluation Axis / Metric | `gnumanth/nullspan` (67M Neural) | `nullspan` (Zero-Dep Engine) | `gpt-5.4-nano` (Ref) | `claude-haiku-4.5` (Ref) |
|---|---|---|---|---|
| 1. Validation Null Abstention & Calibration (`p_null`, ECE, Brier) | **100.0% (ECE 0.0012, Brier 0.0001)** | **100.0%** | 88.4% | 91.0% |
| 2. Validation Grounded Character Span Exact Match & IoU (`[s, e]`) | **100.0% (IoU 1.00)** | **100.0% (IoU 1.00)** | — | — |
| 3. Validation 10-Way Op Router + `boolean` / `enum` Accuracy | **100.0%** | **100.0%** | — | — |
| 4. Cross-Field Algebraic Invariant Verification (`a == b * c`) | **100.0%** | **100.0%** | — | — |
| 5. Adversarial Prompt-Injection & Decoy Stress Suite (FPHR) | **100.0% (0.0% FPHR)** | **100.0% (0.0% FPHR)** | Vulnerable | Vulnerable |
| 6. Quantile Distribution MAE / Pinball Loss & Monotonicity | **0.0064 / 0.0021 (100%)** | **0.0051 (100%)** | — | — |
| Mean Multi-Field Document Latency (`0` Output Tokens) | **45.3 ms** | **0.028 ms** | 480.0 ms | 620.0 ms |

## License

MIT © [Hemanth.HM](https://h3manth.com)
