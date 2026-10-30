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
  - electra
  - onnx
datasets:
  - gnumanth/nullspan-dataset
  - rajpurkar/squad_v2
metrics:
  - accuracy
  - exact_match
  - brier_score
pipeline_tag: question-answering
---

# gnumanth/nullspan

**`nullspan` v1.2** is a `344.2M`-parameter (`330.1 MB` Int8 ONNX, `512`-token context + long-document windowing) **System One extraction, calibrated abstention, multi-numeric contract disambiguation, invariant verification, and quantile distribution model**. It processes an unstructured `state` and a map of typed `fields` (`text`, `number`, `boolean`, `enum`, `date`, `datetime`, `duration`, `list`, `estimate`, `forecast`) in a **single parallel encoder forward pass** with zero autoregressive token generation.

## Architectural Design (`NullspanForExtraction` v2.0)

Warm-started from a 24-layer `1024`-d ELECTRA-Large SQuAD 2.0 transformer (`ahotrod/electra_large_discriminator_squad2_512`) with a non-destructive 2-layer Pre-LN `SchemaAdapterBlock` and trained across `8,290` zero-leakage multi-task field instances (`SQuAD v2.0 train`, multi-section legal/commercial contracts, SRE postmortems, GPU ledgers, cluster comps, and time-series forecasts) at full `512`-token context length:
1. **`null_head`**: Span-attended bilinear null classifier + SQuAD2 `[CLS]` margin residual with held-out temperature scaling (`AUROC: 0.99273`, `ECE: 0.01553`, `Brier: 0.02711`).
2. **`span_head`**: Query-conditioned token start/end pointer network mapped via `offset_mapping` to exact character spans `[start, end]` with calibrated posterior `confidence` and Shannon `span_entropy_bits`.
3. **`bio_head`**: Class-weighted token-level `B/I/O` sequence tagger extracting ordered multi-span arrays (`type: "list"`).
4. **`op_head`**: 10-way System One computation, normalization, boolean judgment (`BOOL_JUDGE`), and zero-shot enum classification (`ENUM_CLASSIFY`) router.
5. **`quantile_head`**: Pinball-loss calibrated distribution head (`p10`, `median`, `p90`) with monotonic sorting (`p10 <= median <= p90`) and cross-field algebraic `constraint` verification (`verified: true/false`).

## Zero-Leakage Empirical Evaluation (`N = 500` `NullspanBench`, `19` Systems)

- **Training time**: `1439.2s` (`NVIDIA Tesla V100-SXM2-16GB` on Google Cloud)
- **NullspanBench Global Rank**: **#1 Capability (`98.4`)** & **#1 Intelligence (`98.0%` `[96.8, 99.2]`)** for `nullspan-344M` (`98.3` Capability / `98.9` Calibration for `330.1 MB` Int8 ONNX); **#1 Composite (`97.2`)** for `nullspan-Hybrid`
- **Held-Out Calibration-Dev Null Accuracy**: `96.94%` (`AUROC: 0.99273`, `ECE: 0.01553`, `Brier: 0.02711`)
- **Open Structured Pool (`S = 336`, `64` Docs)**: `100.0%` (`100.0%` Choice, `100.0%` Math/Invariants, `100.0%` Quantiles)
- **10-Way Operation Router & Multi-Span List BIO Accuracy**: `100.0%` / `100.0%`
- **Sealed Multi-Numeric Legal/Commercial Contracts (`n = 96`, 12 Unseen Agreements)**: `100.0%`
- **Sealed Stanford SQuAD v2.0 Accuracy (`n = 60`, Zero-Overlap)**: `83.33%`
- **Adversarial Decoy & Prompt-Injection Stress Accuracy (`n = 8`)**: `100.0%`
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
    "p_null": 0.0,
    "span": [
      276,
      280
    ],
    "guess": "Mira",
    "confidence": 0.999999,
    "span_entropy_bits": 0.0,
    "value": "Mira"
  },
  "incident_id": {
    "type": "text",
    "p_null": 0.0,
    "span": [
      9,
      18
    ],
    "guess": "INC-40928",
    "confidence": 0.999999,
    "span_entropy_bits": 0.0,
    "value": "INC-40928"
  },
  "opened_on": {
    "type": "date",
    "p_null": 0.0,
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
    "p_null": 0.0,
    "span": [
      109,
      130
    ],
    "guess": 1550.75,
    "confidence": 0.999999,
    "span_entropy_bits": 0.0,
    "value": 1550.75
  },
  "outage_days": {
    "type": "duration",
    "p_null": 0.0,
    "unit": "days",
    "span": [
      29,
      41
    ],
    "guess": 7,
    "confidence": 0.999999,
    "span_entropy_bits": 0.0,
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
    "confidence": 0.999999,
    "span_entropy_bits": 0.0
  },
  "severity": {
    "type": "enum",
    "p_null": 0.0,
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
      "sev-1": 0.999999,
      "sev-2": 1e-06,
      "sev-3": 1e-06
    },
    "guess": "sev-1",
    "confidence": 0.999999,
    "span_entropy_bits": 0.0,
    "value": "sev-1"
  },
  "data_loss": {
    "type": "boolean",
    "p_null": 0.0,
    "p_true": 1e-06,
    "span": [
      188,
      193
    ],
    "guess": false,
    "confidence": 0.999999,
    "span_entropy_bits": 0.0,
    "value": false
  },
  "cve_id": {
    "type": "text",
    "p_null": 1.0,
    "span": [
      109,
      117
    ],
    "guess": "$1240.50",
    "confidence": 0.999999,
    "span_entropy_bits": 2.2523,
    "value": null,
    "abstention_reason": "unassigned_placeholder"
  },
  "remediation_cost": {
    "type": "estimate",
    "value": 1890,
    "median": 1870,
    "p10": 1270,
    "p90": 2660,
    "p_below": 0.0,
    "p_above": 0.0,
    "confidence": 0.999999
  }
}
```

## License

MIT © [Hemanth.HM](https://h3manth.com)
