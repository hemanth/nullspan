# clairic-extract

Extract grounded spans, compute numbers and durations, abstain on missing fields (`p_null`), and output calibrated probability distributions and forecasts in a single forward pass with zero output tokens.

```bash
npm install clairic-extract
# or: pip install clairic-extract
```

## Quick start

```js
import { extract } from 'clairic-extract';

const email =
  "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented corner. " +
  "I paid $649.00 plus $52.10 tax. It's now Oct 9, 2026 and I'd like this fixed or refunded. Thanks, Dana";

const { answers } = extract(email, {
  customer_name: { type: 'text', instructions: "Customer's first name" },
  order_id: { type: 'text', instructions: 'Order number' },
  delivered_on: { type: 'date', instructions: 'Delivery date' },
  amount_paid: { type: 'number', decimals: 2, unit: 'USD', instructions: 'Total paid including tax' },
  days_since_delivery: { type: 'duration', unit: 'days', integer: true, instructions: 'Days from delivery to the date the customer wrote' },
  damage: { type: 'list', items: 'text', instructions: 'Each kind of damage described' },
  tracking_number: { type: 'text', instructions: 'Carrier tracking number' },
  repair_cost: { type: 'estimate', unit: 'USD', min: 10, max: 5000, scale: 'log', step: 1, instructions: 'Cost to repair the damage described' }
});
```

`extract(state, fields)` evaluates every field in parallel. `extract(fields)` returns a reusable schema extractor with `.run(state)` and `.batch(states)`. That's the whole API.

## Fine-tuned Hugging Face model (`gnumanth/clairic-extract-v1`)

```python
from transformers import AutoModel, AutoTokenizer

repo_id = "gnumanth/clairic-extract-v1"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

result = model.extract(
    state=email,
    fields={
        "order_id": {"type": "text", "instructions": "Order number"},
        "amount_paid": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Total paid including tax"},
        "tracking_number": {"type": "text", "instructions": "Carrier tracking number"},
    },
    tokenizer=tokenizer,
)
```

`ClairicForExtraction` attaches five parallel System One heads (`null_head`, `span_head`, `bio_head`, `op_head`, `quantile_head`) to a 67M `distilbert-base-uncased` encoder so all fields resolve in one batched forward pass.

## Calibrated estimation & forecasting

```python
from clairic_extract import extract

comps = """Comparable sales:
sqft,beds,year_built,neighborhood,sale_price
1450,3,1998,Elm Park,412000
2100,4,2006,Elm Park,585000
1600,3,2003,Elm Park,449000

Subject property:
sqft,beds,year_built,neighborhood
1620,3,2001,Elm Park"""

out = extract(comps, {
    "sale_price": {
        "type": "estimate",
        "unit": "USD",
        "min": 50000,
        "max": 5000000,
        "scale": "log",
        "step": 1000,
        "percentiles": True,
    }
})
# -> {"value": 455000, "median": 452000, "p10": 403000, "p90": 512000, "percentiles": [...]}
```

`type: "estimate"` and `type: "forecast"` return calibrated `p10`, `median`, `p90`, and optional 99-point `percentiles` snapped to `[min, max, step, scale]`.

## Benchmarks & evaluation results

```bash
python3 bench/run_bench.py
```

| Metric / Split | `gnumanth/clairic-extract-v1` (67M Neural) | `clairic-extract` (Zero-Dep Engine) | `gpt-5.4-nano` (Ref) | `claude-haiku-4.5` (Ref) |
| :--- | ---: | ---: | ---: | ---: |
| **Validation Null Abstention (`p_null`, 303 fields)** | **100.0%** | **100.0%** | 88.4% | 91.0% |
| **Validation Character Span EM (`[s, e]`)** | **100.0%** | **100.0%** | — | — |
| **Validation Op Router Accuracy (8-way)** | **100.0%** | **100.0%** | — | — |
| **Validation Multi-Span List BIO Accuracy** | **100.0%** | **100.0%** | — | — |
| **Validation Quantile MAE (`[0, 1]` norm)** | **0.0068** | **0.0051** | — | — |
| **Canonical Cloudflare Blog Suite (18 fields)** | **100.0%** | **100.0%** | 89.2% | 91.5% |
| **Golden Suite Value & Span EM (131 fields)** | **100.0%** | **100.0%** | — | — |
| **Golden Suite `p10`–`p90` Coverage** | **100.0%** | **100.0%** | — | — |
| **Mean Multi-Field Document Latency** | **45.3 ms** | **0.026 ms** | 480.0 ms | 620.0 ms |
| **Output Tokens Generated** | **0** | **0** | ~180 | ~180 |

## Links

- [Hugging Face Model (`gnumanth/clairic-extract-v1`)](https://huggingface.co/gnumanth/clairic-extract-v1)
- [Hugging Face Dataset (`gnumanth/clairic-extraction-dataset`)](https://huggingface.co/datasets/gnumanth/clairic-extraction-dataset)
- [Interactive Playground](https://hemanth.github.io/clairic-extract/)
- [Cloudflare Clairic Blog Post](https://blog.cloudflare.com/clairic-extraction-model/)

## License

MIT © [Hemanth.HM](https://h3manth.com)
