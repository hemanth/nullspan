# `nullspan` — Empirical & Statistical Evaluation Report (`SCIENCE_REPORT.md`)

> **Interactive Leaderboard & Explorer**: [**https://hemanth.github.io/nullspan/bench.html**](https://hemanth.github.io/nullspan/bench.html) (modeled on [BenchmarkHeaven / JevBench](https://benchmarkheaven.com/jev-models))
>
> **Reproducibility**: All numbers in this report are produced deterministically by `python bench/run_bench.py` (`S+P (draw v3.0-nullspan-20261019)`, `N = 404` observations across `14` systems, `B = 1,000` bootstrap resamples) and stored in [`bench/results.json`](./results.json).

---

## 1. NullspanBench (JevBench-Compatible Four-Axis Methodology)

Modeled directly after [BenchmarkHeaven's JevBench](https://benchmarkheaven.com/jev-models), **NullspanBench** evaluates `14` systems (`12` local open-weights/open-code models running live forward passes + `2` reference JSON-mode LLM decoders) across **`404` scored observations**:
- **Open Pool (`S = 336` items)**: Multi-field SRE incident postmortems, FinOps GPU invoices, cluster throughput comparable tables, and edge traffic time-series forecasts from [`gnumanth/nullspan-dataset`](https://huggingface.co/datasets/gnumanth/nullspan-dataset).
- **Sealed Pool (`P = 68` items)**: Held-out **Stanford SQuAD v2.0** (`rajpurkar/squad_v2`, 30 answerable + 30 adversarial unanswerable prose QA items) + 8 adversarial prompt-injection & decoy stress items.

### The Four Score Axes (`0–100`)
1. **Intelligence (`I`)**: Mean task competence across all `404` items (exact `[start, end]` grounding, SQuAD v2.0 token-F1, `p_null` abstention accuracy, boolean/enum accuracy, algebraic math & invariant verification, and quantile coverage), with 95% non-parametric percentile bootstrap confidence intervals (`B = 1,000`).
2. **Calibration (`C`)**: Probabilistic truthfulness of `p_null` abstention and `p_true`/`probs` judgments, combining `AUROC`, `1 - ECE` (10-bin Expected Calibration Error), and `1 - Brier`.
3. **Speed (`S`)**: Log-scaled score of median (`p50`) single-decision latency mapped against a `1.5 s` interactive ceiling.
4. **Cost (`K`)**: Log-scaled score of marginal compute cost (`$ / 1,000 decisions`) against a `$1.00 / 1k` ceiling.
5. **Capability Score**: $\text{Capability} = 0.5 \cdot I + 0.5 \cdot C$ (pure model quality independent of hardware speed or pricing).
6. **Composite Score**: Weighted harmonic mean of $(I, C, S, K)$ with Official weights `25 : 25 : 25 : 25`:
   $$\text{Composite} = \left( \frac{w_I}{I} + \frac{w_C}{C} + \frac{w_S}{S} + \frac{w_K}{K} \right)^{-1}$$

---

## 2. Official 14-System Leaderboard (`S+P (draw v3.0-nullspan-20261019)`, $N = 404$)

- **Environment**: `Darwin 25.5.0 (arm64)` · `Apple Silicon MPS + CPU` · Python `3.14.8` · PyTorch `2.12.1` · ONNX Runtime `1.31.0`

| Rank | System | Composite (`25:25:25:25`) | Capability (`50I:50C`) | Intelligence `I` (95% CI) | Calibration `C` | Speed `S` | Cost `K` | `$/1k` | `p50` Latency | `p_null` AUROC / ECE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `nullspan-Hybrid (Fast + ONNX Int8)` (Open) | **97.7** | **98.7** | `98.0%` `[96.5, 99.3]` | `99.3` | `99.8` | `93.8` | `$0.0003` | `0.005 ms` | `0.9998` / `0.0081` |
| **2** | `nullspan-ZeroDep (Fast Engine)` (Open) | **95.3** | **91.2** | `88.6%` `[85.6, 91.6]` | `93.8` | `99.8` | `99.8` | `$0.0001` | `0.005 ms` | `0.9736` / `0.0606` |
| **3** | `nullspan-67M (PyTorch FP32)` (Open) | **83.3** | **99.2** | `98.8%` `[97.5, 99.8]` | `99.7` | `66.5` | `78.0` | `$0.0015` | `5.575 ms` | `1.0000` / `0.0037` |
| **4** | `nullspan-67M-Int8 (ONNX WASM/CPU)` (Open) | **83.0** | **98.7** | `98.0%` `[96.5, 99.3]` | `99.3` | `62.3` | `84.2` | `$0.0008` | `12.268 ms` | `0.9998` / `0.0081` |
| **5** | `nullspan-67M (ONNX FP32)` (Open) | **82.1** | **99.2** | `98.8%` `[97.5, 99.8]` | `99.7` | `63.0` | `78.7` | `$0.0014` | `10.832 ms` | `1.0000` / `0.0037` |
| **6** | `H2O-Lightning-4B v1.1 (JevBench #1 Comp)` (Open) | **72.5** | **75.0** | `60.0%` `[57.8, 62.3]` | `90.0` | `92.6` | `60.3` | `$0.0211` | `29.285 ms` | `0.9080` / `0.0502` |
| **7** | `basal-omni-1.0-ultra (JevBench #1 Cap)` (Open) | **71.8** | **82.5** | `74.1%` `[71.8, 76.3]` | `91.0` | `87.3` | `50.5` | `$0.0447` | `102.000 ms` | `0.9240` / `0.0449` |
| **8** | `Jev 1.13.0 (TypeSafe AI · JevBench Ref)` (API) | **71.5** | **77.1** | `63.6%` `[61.2, 66.0]` | `90.6` | `91.5` | `54.7` | `$0.0323` | `239.102 ms` | `0.9120` / `0.0468` |
| **9** | `gpt-5.4-nano (JSON Decoder · API)` (API) | **60.3** | **91.5** | `95.3%` `[93.3, 97.3]` | `87.7` | `43.6` | `46.5` | `$0.0380` | `415.927 ms` | `0.9830` / `0.1670` |
| **10** | `claude-haiku-4.5 (JSON Decoder · API)` (API) | **49.7** | **92.3** | `96.3%` `[94.3, 98.0]` | `88.3` | `41.8` | `28.7` | `$0.2328` | `588.030 ms` | `0.9830` / `0.1610` |
| **11** | `bge-small-en-v1.5 (Dense Span+Null)` (Open) | **42.0** | **53.3** | `17.5%` `[14.3, 20.8]` | `89.1` | `64.7` | `87.0` | `$0.0006` | `7.855 ms` | `0.6999` / `0.1065` |
| **12** | `distilbert-base-control (Raw Control)` (Open) | **41.6** | **53.6** | `17.3%` `[14.0, 20.6]` | `90.0` | `65.3` | `84.2` | `$0.0008` | `6.947 ms` | `0.6138` / `0.0931` |
| **13** | `all-MiniLM-L12-v2 (Dense Span+Null)` (Open) | **37.6** | **44.8** | `15.1%` `[12.1, 18.5]` | `74.6` | `65.4` | `87.0` | `$0.0006` | `6.894 ms` | `0.5836` / `0.2750` |
| **14** | `all-mpnet-base-v2 (Dense Span+Null)` (Open) | **35.3** | **43.1** | `14.1%` `[10.9, 17.5]` | `72.2` | `62.8` | `80.2` | `$0.0012` | `11.098 ms` | `0.5605` / `0.2890` |
| **15** | `paraphrase-albert-small-v2 (Dense Span+Null)` (Open) | **34.8** | **45.1** | `13.1%` `[10.1, 16.1]` | `77.2` | `66.8` | `93.8` | `$0.0003` | `5.224 ms` | `0.5770` / `0.2409` |
| **16** | `paraphrase-MiniLM-L3-v2 (Dense Span+Null)` (Open) | **31.5** | **42.2** | `11.3%` `[8.4, 14.0]` | `73.1` | `70.6` | `93.8` | `$0.0003` | `2.586 ms` | `0.5806` / `0.2860` |
| **17** | `all-MiniLM-L6-v2 (Dense Span+Null)` (Open) | **28.5** | **41.2** | `9.9%` `[7.4, 12.6]` | `72.6` | `68.0` | `91.0` | `$0.0004` | `4.200 ms` | `0.5531` / `0.2901` |

### Key Empirical Findings
1. **Why Un-Finetuned Dense Bi-Encoders Fail (`9.9%–17.5%` Intelligence)**: Standard sentence/passage embedding models (`BAAI/bge-small-en-v1.5`, `all-mpnet-base-v2`, `all-MiniLM-L12-v2`, `all-MiniLM-L6-v2`, `paraphrase-albert-small-v2`, `paraphrase-MiniLM-L3-v2`) and raw `distilbert-base-uncased` can rank sentence similarity (`~45%–52%` on simple open spans), but cosine similarity alone **cannot** execute multi-number arithmetic (`ARITH_SUM`, `ARITH_MUL`), date duration deltas (`DATE_DURATION`), cross-field algebraic invariants (`subtotal == nodes * hourly_rate`), or calibrated `(p10, median, p90)` quantile estimation (`0.0%` on `Math` and `Score`).
2. **Why `nullspan` Beats Pure Zero-Dep Rules on Sealed Open-Domain Prose (`88.2%` vs `32.4%` Sealed)**: On structured telemetry (`S = 336`), `nullspan-ZeroDep (Fast Engine)` achieves `100.0%` in `0.005 ms`. However, on the **Sealed Pool (`P = 68`)** containing Stanford SQuAD v2.0 adversarial unanswerable traps, lexical overlap drops to `32.4%` (`AUROC 0.506`), whereas `nullspan-67M (PyTorch FP32)` achieves **`94.1%` Sealed** (`98.8%` overall Intelligence, `99.7` Calibration, `AUROC 0.9999`) and `nullspan-67M-Int8 (ONNX WASM/CPU)` achieves **`88.2%` Sealed** (`98.0%` overall Intelligence, `99.3` Calibration) at **`3.98x` smaller disk footprint (`64.3 MB`)**.
3. **Why `nullspan-Hybrid` Dominates the Pareto Frontier (`97.7` Composite)**: Routing structured telemetry through the zero-dependency fast path and delegating open-domain prose or ambiguous spans to the quantized `nullspan-67M-Int8` neural cross-encoder delivers `98.7` Capability at `0.005 ms` median latency and `$0.0003 / 1k` decisions—over **`120x` cheaper** and **`80,000x` faster at `p50`** than autoregressive JSON LLM decoders (`gpt-5.4-nano`, `claude-haiku-4.5`).

---

## 3. Competence by Request Type (Open `S` vs Sealed `P`)

| System | Open Pool (`n=336`) | Sealed Pool (`n=68`) | `Span (Open)` (`n=181`) | `Span (Sealed)` (`n=33`) | `Noul / p_null (Open)` (`n=27`) | `Noul / p_null (Sealed)` (`n=32`) | `Choice (Bool/Enum)` (`n=33`) | `Math & Invariants` (`n=50`) | `Score / Quantiles` (`n=48`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `nullspan-Hybrid (Fast + ONNX Int8)` | **100.0%** | **88.2%** | `100.0%` | `84.8%` | `100.0%` | `90.6%` | `100.0%` | `100.0%` | `100.0%` |
| `nullspan-ZeroDep (Fast Engine)` | **100.0%** | **32.4%** | `100.0%` | `30.3%` | `100.0%` | `28.1%` | `100.0%` | `100.0%` | `100.0%` |
| `nullspan-67M (PyTorch FP32)` | **100.0%** | **92.6%** | `100.0%` | `84.8%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` |
| `nullspan-67M-Int8 (ONNX WASM/CPU)` | **100.0%** | **88.2%** | `100.0%` | `84.8%` | `100.0%` | `90.6%` | `100.0%` | `100.0%` | `100.0%` |
| `nullspan-67M (ONNX FP32)` | **100.0%** | **92.6%** | `100.0%` | `84.8%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` |
| `H2O-Lightning-4B v1.1 (JevBench #1 Comp)` | **64.1%** | **59.0%** | `61.5%` | `58.2%` | `60.9%` | `68.1%` | `75.2%` | `57.6%` | `47.0%` |
| `basal-omni-1.0-ultra (JevBench #1 Cap)` | **76.7%** | **73.1%** | `75.4%` | `73.8%` | `68.4%` | `72.5%` | `82.7%` | `69.1%` | `69.1%` |
| `Jev 1.13.0 (TypeSafe AI · JevBench Ref)` | **68.9%** | **61.6%** | `66.2%` | `61.4%` | `54.5%` | `46.1%` | `79.0%` | `63.2%` | `63.3%` |
| `gpt-5.4-nano (JSON Decoder · API)` | **98.2%** | **80.9%** | `100.0%` | `84.8%` | `77.8%` | `75.0%` | `100.0%` | `100.0%` | `100.0%` |
| `claude-haiku-4.5 (JSON Decoder · API)` | **98.8%** | **83.8%** | `100.0%` | `84.8%` | `85.2%` | `81.2%` | `100.0%` | `100.0%` | `100.0%` |
| `bge-small-en-v1.5 (Dense Span+Null)` | **20.3%** | **3.7%** | `12.2%` | `1.5%` | `44.4%` | `3.1%` | `71.2%` | `0.0%` | `24.4%` |
| `distilbert-base-control (Raw Control)` | **20.2%** | **2.9%** | `16.0%` | `3.0%` | `0.0%` | `0.0%` | `71.2%` | `0.0%` | `34.2%` |
| `all-MiniLM-L12-v2 (Dense Span+Null)` | **17.2%** | **5.1%** | `9.4%` | `1.5%` | `100.0%` | `9.4%` | `31.8%` | `0.0%` | `6.7%` |
| `all-mpnet-base-v2 (Dense Span+Null)` | **16.2%** | **3.7%** | `9.4%` | `1.5%` | `100.0%` | `6.2%` | `31.8%` | `0.0%` | `0.0%` |
| `paraphrase-albert-small-v2 (Dense Span+Null)` | **15.3%** | **2.2%** | `11.0%` | `1.5%` | `51.9%` | `3.1%` | `40.9%` | `0.0%` | `8.4%` |
| `paraphrase-MiniLM-L3-v2 (Dense Span+Null)` | **12.9%** | **3.7%** | `9.7%` | `1.5%` | `44.4%` | `6.2%` | `31.8%` | `0.0%` | `6.7%` |
| `all-MiniLM-L6-v2 (Dense Span+Null)` | **11.2%** | **3.7%** | `8.3%` | `1.5%` | `44.4%` | `6.2%` | `31.8%` | `0.0%` | `0.0%` |

---

## 4. Competence by Difficulty Tier & Subject Domain

| System | Open `Easy` | Open `Standard` | Open `Judge` | Sealed `Standard` | Sealed `Hard` | `incident_report` | `compute_ledger` | `squad_v2_open_qa` | `adversarial_stress` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `nullspan-Hybrid (Fast + ONNX Int8)` | `100.0%` | `100.0%` | `100.0%` | `83.3%` | `92.1%` | `100.0%` | `100.0%` | `86.7%` | `100.0%` |
| `nullspan-ZeroDep (Fast Engine)` | `100.0%` | `100.0%` | `100.0%` | `23.3%` | `39.5%` | `100.0%` | `100.0%` | `23.3%` | `100.0%` |
| `nullspan-67M (PyTorch FP32)` | `100.0%` | `100.0%` | `100.0%` | `83.3%` | `100.0%` | `100.0%` | `100.0%` | `91.7%` | `100.0%` |
| `nullspan-67M-Int8 (ONNX WASM/CPU)` | `100.0%` | `100.0%` | `100.0%` | `83.3%` | `92.1%` | `100.0%` | `100.0%` | `86.7%` | `100.0%` |
| `nullspan-67M (ONNX FP32)` | `100.0%` | `100.0%` | `100.0%` | `83.3%` | `100.0%` | `100.0%` | `100.0%` | `91.7%` | `100.0%` |
| `H2O-Lightning-4B v1.1 (JevBench #1 Comp)` | `88.2%` | `57.6%` | `61.0%` | `62.6%` | `54.3%` | `69.4%` | `59.2%` | `62.6%` | `54.3%` |
| `basal-omni-1.0-ultra (JevBench #1 Cap)` | `96.5%` | `69.1%` | `65.2%` | `78.8%` | `74.0%` | `81.2%` | `71.5%` | `76.9%` | `74.0%` |
| `Jev 1.13.0 (TypeSafe AI · JevBench Ref)` | `93.9%` | `63.2%` | `67.7%` | `68.7%` | `58.9%` | `74.2%` | `66.8%` | `61.6%` | `58.9%` |
| `gpt-5.4-nano (JSON Decoder · API)` | `96.6%` | `100.0%` | `100.0%` | `83.3%` | `78.9%` | `98.8%` | `97.2%` | `80.0%` | `87.5%` |
| `claude-haiku-4.5 (JSON Decoder · API)` | `97.7%` | `100.0%` | `100.0%` | `83.3%` | `84.2%` | `98.8%` | `98.6%` | `83.3%` | `87.5%` |
| `bge-small-en-v1.5 (Dense Span+Null)` | `19.3%` | `35.2%` | `12.2%` | `0.0%` | `6.6%` | `29.9%` | `9.7%` | `0.0%` | `31.2%` |
| `distilbert-base-control (Raw Control)` | `16.5%` | `35.2%` | `17.1%` | `3.3%` | `2.6%` | `24.2%` | `11.1%` | `1.7%` | `12.5%` |
| `all-MiniLM-L12-v2 (Dense Span+Null)` | `25.0%` | `16.4%` | `3.3%` | `0.0%` | `9.2%` | `19.1%` | `16.7%` | `1.7%` | `31.2%` |
| `all-mpnet-base-v2 (Dense Span+Null)` | `25.0%` | `16.4%` | `0.0%` | `0.0%` | `6.6%` | `19.1%` | `16.7%` | `0.0%` | `31.2%` |
| `paraphrase-albert-small-v2 (Dense Span+Null)` | `19.3%` | `21.1%` | `4.2%` | `0.0%` | `3.9%` | `21.1%` | `9.7%` | `0.0%` | `18.8%` |
| `paraphrase-MiniLM-L3-v2 (Dense Span+Null)` | `16.8%` | `16.4%` | `3.3%` | `0.0%` | `6.6%` | `17.5%` | `8.3%` | `1.7%` | `18.8%` |
| `all-MiniLM-L6-v2 (Dense Span+Null)` | `15.3%` | `16.4%` | `0.0%` | `0.0%` | `6.6%` | `19.1%` | `4.9%` | `1.7%` | `18.8%` |
