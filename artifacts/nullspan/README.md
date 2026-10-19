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
pipeline_tag: question-answering
---

# gnumanth/nullspan

**`nullspan`** is a lightweight **System One extraction, abstention, and distribution model**. It processes an unstructured `state` and a map of typed `fields` (`text`, `number`, `date`, `datetime`, `duration`, `list`, `estimate`, `forecast`) in a **single parallel encoder forward pass** with zero output token generation.

## Architecture (`NullspanForExtraction`)

Built on `distilbert/distilbert-base-uncased` (67M parameters) with 5 parallel System One heads:
1. **`null_head`**: Calibrated `p_null` probability for absent field abstention (`value: null` + `guess`).
2. **`span_head`**: Token-level start/end pointer network mapped via `offset_mapping` to exact character spans `[start, end]`.
3. **`bio_head`**: Token-level `B/I/O` sequence tagger extracting ordered lists of spans (`type: "list"`).
4. **`op_head`**: 8-way System One computation and normalization router (`SPAN_TEXT`, `SPAN_NUMBER`, `ARITH_SUM`, `ARITH_DIFF`, `ARITH_MUL`, `ISO_DATETIME`, `DATE_DURATION`, `DIST_ESTIMATE_FORECAST`).
5. **`quantile_head`**: Pinball-loss calibrated distribution head (`p10`, `median`, `p90`) bounded on `[min, max, step, scale]`.

## Empirical Benchmarks

- **Training time**: `565.8s` (`Apple Silicon MPS`)
- **Null Abstention Accuracy (`p_null`)**: `100.0%`
- **Grounded Span Exact Match (`[start, end]`)**: `100.0%`
- **Operation Router Accuracy**: `100.0%`
- **Multi-Span List BIO Accuracy**: `100.0%`
- **Quantile Distribution MAE (normalized `[0, 1]`)**: `0.0068`
- **Canonical Evaluation Suite Accuracy**: `100.0%` (`119.35 ms` per multi-field pass)

## Quick Start

```python
from transformers import AutoModel, AutoTokenizer

repo_id = "gnumanth/nullspan"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

email = (
    "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented corner. "
    "I paid $649.00 plus $52.10 tax. It's now Oct 9, 2026 and I'd like this fixed or refunded. Thanks, Dana"
)

result = model.extract(
    state=email,
    fields={
        "order_id": {"type": "text", "instructions": "Order number"},
        "delivered_on": {"type": "date", "instructions": "Delivery date"},
        "amount_paid": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Total paid including tax"},
        "days_since_delivery": {"type": "duration", "unit": "days", "integer": True, "instructions": "Days from delivery to the date the customer wrote"},
        "damage": {"type": "list", "items": "text", "instructions": "Each kind of damage described"},
        "tracking_number": {"type": "text", "instructions": "Carrier tracking number"},
        "repair_cost": {"type": "estimate", "unit": "USD", "min": 10, "max": 5000, "scale": "log", "step": 1, "instructions": "Cost to repair the damage described"},
    },
    tokenizer=tokenizer,
)
print(result["answers"])
```

### Output on Customer Claim Example

```json
{
  "customer_name": {
    "type": "text",
    "p_null": 2.956e-05,
    "span": [
      184,
      188
    ],
    "guess": "Dana",
    "confidence": 0.999852,
    "value": "Dana"
  },
  "order_id": {
    "type": "text",
    "p_null": 6.76e-06,
    "span": [
      10,
      17
    ],
    "guess": "A-88412",
    "confidence": 0.999975,
    "value": "A-88412"
  },
  "delivered_on": {
    "type": "date",
    "p_null": 6.19e-06,
    "guess": "2026-09-28",
    "confidence": 0.999977,
    "value": "2026-09-28"
  },
  "amount_paid": {
    "type": "number",
    "p_null": 1.171e-05,
    "guess": 701.1,
    "confidence": 0.999961,
    "value": 701.1
  },
  "days_since_delivery": {
    "type": "duration",
    "p_null": 6.101e-05,
    "unit": "days",
    "guess": 11,
    "confidence": 0.99992,
    "value": 11
  },
  "damage": {
    "type": "list",
    "value": [
      "cracked screen",
      "dented corner"
    ],
    "spans": [
      [
        50,
        64
      ],
      [
        71,
        84
      ]
    ]
  },
  "tracking_number": {
    "type": "text",
    "p_null": 0.99896169,
    "span": [
      106,
      112
    ],
    "guess": "$52.10",
    "confidence": 0.05,
    "value": null
  },
  "repair_cost": {
    "type": "estimate",
    "value": 250,
    "median": 247,
    "p10": 166,
    "p90": 424,
    "p_below": 0.0,
    "p_above": 0.0
  }
}
```

## License

MIT © [Hemanth.HM](https://h3manth.com)
