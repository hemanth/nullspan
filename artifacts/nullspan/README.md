---
license: mit
library_name: transformers
tags:
  - nullspan
  - system-one
  - structured-extraction
  - span-extraction
  - calibrated-estimation
  - time-series-forecasting
  - distilbert
datasets:
  - gnumanth/nullspan-dataset
metrics:
  - accuracy
  - exact_match
  - brier_score
pipeline_tag: question-answering
---

# gnumanth/nullspan

**`nullspan`** is a 67M-parameter **System One extraction, calibrated abstention, invariant verification, and quantile distribution model**. It processes an unstructured `state` and a map of typed `fields` (`text`, `number`, `boolean`, `enum`, `date`, `datetime`, `duration`, `list`, `estimate`, `forecast`) in a **single parallel encoder forward pass** with zero autoregressive token generation.

## Architectural Enhancements (`NullspanForExtraction`)

Built on `distilbert/distilbert-base-uncased` (67M parameters) with 5 parallel System One heads and 10 routed operations:
1. **`null_head`**: Calibrated `p_null` probability (`ECE: 1e-05`, `Brier: 0.0`) for absent field abstention (`value: null` + nearest `guess` decoy + `abstention_reason`).
2. **`span_head`**: Token-level start/end pointer network mapped via `offset_mapping` to exact character spans `[start, end]` with calibrated posterior `confidence` and Shannon `span_entropy_bits`.
3. **`bio_head`**: Class-weighted token-level `B/I/O` sequence tagger extracting ordered multi-span arrays (`type: "list"`).
4. **`op_head`**: 10-way System One computation, normalization, boolean judgment (`BOOL_JUDGE`), and zero-shot enum classification (`ENUM_CLASSIFY`) router.
5. **`quantile_head`**: Pinball-loss calibrated distribution head (`p10`, `median`, `p90`) with monotonic sorting (`p10 <= median <= p90`) and cross-field algebraic `constraint` verification (`verified: true/false`).

## Multi-Axis Empirical Evaluation (`gnumanth/nullspan-dataset`)

- **Training time**: `358.3s` (`Apple Silicon MPS`)
- **Null Abstention Accuracy (`p_null`)**: `100.0%` (`ECE: 1e-05`, `Brier: 0.0`)
- **Grounded Span Exact Match (`[start, end]`)**: `99.31%`
- **10-Way Operation Router Accuracy**: `100.0%`
- **Multi-Span List BIO Token Accuracy**: `100.0%`
- **Quantile Distribution MAE / Pinball Loss**: `0.0007` / `0.0001`
- **Canonical Multi-Task Suite Accuracy**: `100.0%`
- **Adversarial Decoy & Prompt-Injection Stress Accuracy**: `100.0%`
- **Cross-Field Invariant Verification Accuracy**: `100.0%`

## Quick Start

```python
from transformers import AutoModel, AutoTokenizer

repo_id = "gnumanth/nullspan"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

incident = (
    "Incident INC-40928 opened on Apr 12, 2026 with a 502 Bad Gateway errors and a TLS handshake timeouts. "
    "Issued $1240.50 plus $310.25 SLA credit. Severity is sev-1 and customer data loss was false. "
    "Root cause CVE not yet assigned. Cluster stabilized on Apr 19, 2026. Reported by Mira"
)

result = model.extract(
    state=incident,
    fields={
        "incident_id": {"type": "text", "instructions": "Incident ticket identifier"},
        "opened_on": {"type": "date", "instructions": "Date the incident opened"},
        "total_credit": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Total credit including SLA penalty"},
        "outage_days": {"type": "duration", "unit": "days", "integer": True, "instructions": "Days from incident open to cluster stabilization"},
        "failure_modes": {"type": "list", "items": "text", "instructions": "Each failure mode triggered"},
        "severity": {"type": "enum", "options": ["sev-1", "sev-2", "sev-3"], "instructions": "Incident severity tier"},
        "data_loss": {"type": "boolean", "instructions": "Whether customer data loss occurred"},
        "cve_id": {"type": "text", "instructions": "Assigned CVE identifier if present"},
        "remediation_cost": {"type": "estimate", "unit": "USD", "min": 100, "max": 25000, "scale": "log", "step": 10, "instructions": "Estimated engineering remediation cost"},
    },
    tokenizer=tokenizer,
)
print(result["answers"])
```

### Output on SRE Incident Postmortem Example

```json
{
  "reporter": {
    "type": "text",
    "p_null": 3.8e-07,
    "span": [
      276,
      280
    ],
    "guess": "Mira",
    "confidence": 0.999992,
    "span_entropy_bits": 0.0002,
    "value": "Mira"
  },
  "incident_id": {
    "type": "text",
    "p_null": 3e-08,
    "span": [
      9,
      18
    ],
    "guess": "INC-40928",
    "confidence": 0.999962,
    "span_entropy_bits": 0.0006,
    "value": "INC-40928"
  },
  "opened_on": {
    "type": "date",
    "p_null": 2.7e-07,
    "span": [
      29,
      41
    ],
    "guess": "2026-04-12",
    "confidence": 0.999999,
    "span_entropy_bits": 0.0,
    "value": "2026-04-12"
  },
  "total_credit": {
    "type": "number",
    "p_null": 4.5e-06,
    "span": [
      109,
      130
    ],
    "guess": 1550.75,
    "confidence": 0.999989,
    "span_entropy_bits": 0.0002,
    "value": 1550.75
  },
  "outage_days": {
    "type": "duration",
    "p_null": 3.04e-06,
    "unit": "days",
    "span": [
      29,
      41
    ],
    "guess": 7,
    "confidence": 0.999992,
    "span_entropy_bits": 0.0001,
    "value": 7
  },
  "failure_modes": {
    "type": "list",
    "value": [
      "502 Bad Gateway errors",
      "TLS handshake timeouts"
    ],
    "spans": [
      [
        49,
        71
      ],
      [
        78,
        100
      ]
    ],
    "confidence": 0.497096,
    "span_entropy_bits": 1.5381
  },
  "severity": {
    "type": "enum",
    "p_null": 1.4e-07,
    "span": [
      155,
      160
    ],
    "options": [
      "sev-1",
      "sev-2",
      "sev-3"
    ],
    "probs": {
      "sev-1": 0.999975,
      "sev-2": 1.3e-05,
      "sev-3": 1.3e-05
    },
    "guess": "sev-1",
    "confidence": 0.999975,
    "span_entropy_bits": 0.0005,
    "value": "sev-1"
  },
  "data_loss": {
    "type": "boolean",
    "p_null": 1.28e-06,
    "p_true": 7e-06,
    "span": [
      188,
      193
    ],
    "guess": false,
    "confidence": 0.999993,
    "span_entropy_bits": 0.0001,
    "value": false
  },
  "cve_id": {
    "type": "text",
    "p_null": 0.9999429,
    "span": [
      123,
      130
    ],
    "guess": "$310.25",
    "confidence": 0.05,
    "span_entropy_bits": 0.0009,
    "value": null,
    "abstention_reason": "unassigned_placeholder"
  },
  "remediation_cost": {
    "type": "estimate",
    "value": 1890,
    "median": 1870,
    "p10": 1270,
    "p90": 2670,
    "p_below": 0.0,
    "p_above": 0.0,
    "confidence": 0.99999
  }
}
```

## License

MIT © [Hemanth.HM](https://h3manth.com)
