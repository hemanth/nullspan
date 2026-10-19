---
license: mit
task_categories:
  - question-answering
  - token-classification
  - tabular-regression
language:
  - en
tags:
  - clairic
  - system-one
  - structured-extraction
  - abstention
  - span-extraction
  - calibrated-estimation
  - time-series-forecasting
size_categories:
  - 1K<n<10K
---

# clairic-extraction-dataset

Multi-task **System One structured extraction, abstention, estimation, and time-series forecasting** dataset designed for training and evaluating [Clairic-style](https://blog.cloudflare.com/clairic-extraction-model/) single-forward-pass encoder models (`gnumanth/clairic-extract-v1`).

## Splits & Files

| File | Instances | Description |
| :--- | ---: | :--- |
| `train.jsonl` | 1,854 | Field-level multi-task training instances across 392 documents |
| `val.jsonl` | 303 | Held-out validation instances across 64 documents |
| `golden_eval.json` | 4 canonical + 64 val | Canonical Cloudflare Clairic blog evaluation suite + validation documents |

## Supported Field Tasks

1. **`text`**: Grounded character-offset span extraction (`span: [start, end]`) with calibrated null abstention (`is_null`).
2. **`number`**: Direct numeric span extraction and 1–2 step arithmetic (`ARITH_SUM`, `ARITH_DIFF`, `ARITH_MUL`) honoring `decimals` and `integer`.
3. **`date` / `datetime`**: Normalized ISO-8601 date and timestamp targets.
4. **`duration`**: Date-delta calculation in requested units (`days`).
5. **`list`**: Ordered multi-span sequence tagging (`spans: [[s1, e1], [s2, e2], ...]`).
6. **`estimate`**: Calibrated quantile regression (`p10`, `median`, `p90`) over bounded linear/log grids (`[min, max, step]`).
7. **`forecast`**: Multi-horizon time-series forecasting with `p10`, `median`, and `p90` intervals.

## License

MIT © [Hemanth HM](https://h3manth.com)
