---
license: mit
task_categories:
  - question-answering
  - token-classification
  - tabular-regression
  - time-series-forecasting
language:
  - en
tags:
  - nullspan
  - system-one
  - structured-extraction
  - calibrated-abstention
  - prompt-injection-resistance
size_categories:
  - 1K<n<10K
---

# `gnumanth/nullspan-dataset`

Multi-task **System One structured extraction, calibrated abstention, boolean/enum judgment, cross-field invariant verification, and distributional quantile estimation** dataset for [`gnumanth/nullspan`](https://huggingface.co/gnumanth/nullspan).

## Splits & Composition

- `train.jsonl`: `2,090` field-level training instances across `400` documents (`incident_report`, `compute_ledger`, `cluster_throughput_comps`, `edge_traffic_forecast`, `adversarial_stress`).
- `val.jsonl`: `336` field-level validation instances across `64` documents.
- `golden_eval.json`: `4` canonical multi-field scenarios + `2` adversarial prompt-injection & decoy stress scenarios + `24` held-out validation documents.

## Supervision Targets per Instance

1. **`is_null`**: Binary abstention label (`1` when requested field is absent, unassigned, or negated in `state`) paired with `abstention_reason` and nearest decoy `guess`.
2. **`span`**: Exact character offsets `[start, end]` in `state`.
3. **`spans`**: Multi-span list offsets `[[s1, e1], [s2, e2], ...]` for `type: "list"` (`B/I/O` token sequence tags).
4. **`op`**: 10-way System One operation routing target:
   - `0: SPAN_TEXT`
   - `1: SPAN_NUMBER`
   - `2: ARITH_SUM`
   - `3: ARITH_DIFF`
   - `4: ARITH_MUL`
   - `5: ISO_DATETIME`
   - `6: DATE_DURATION`
   - `7: DIST_ESTIMATE_FORECAST`
   - `8: BOOL_JUDGE`
   - `9: ENUM_CLASSIFY`
5. **`quantiles_norm`**: Log/linear normalized `[p10, median, p90]` targets in `[0, 1]` trained with multi-quantile pinball loss.

## License

MIT © [Hemanth.HM](https://h3manth.com)
