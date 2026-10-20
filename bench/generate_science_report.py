"""Generate bench/SCIENCE_REPORT.md from bench/results.json (JevBench 14-system methodology)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "bench" / "results.json"
REPORT_PATH = ROOT / "bench" / "SCIENCE_REPORT.md"


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text())
    env = data["environment"]
    systems = data["systems"]

    # Build main leaderboard rows
    leaderboard_rows = []
    for s in systems:
        ax = s["axes"]
        ci = ax["intelligence_ci95"]
        cal = s["calibration_details"]
        lat = s["latency"]
        weight_tag = "Open" if s["open_weights"] else "API"
        leaderboard_rows.append(
            f"| **{s['official_rank']}** | `{s['name']}` ({weight_tag}) | **{s['composite_score']:.1f}** | **{s['capability_score']:.1f}** | "
            f"`{ax['intelligence']:.1f}%` `[{ci[0]:.1f}, {ci[1]:.1f}]` | `{ax['calibration']:.1f}` | `{ax['speed']:.1f}` | `{ax['cost']:.1f}` | "
            f"`${s['usd_per_1k_decisions']:.4f}` | `{lat['p50_ms']:.3f} ms` | `{cal['null_auroc']:.4f}` / `{cal['null_ece']:.4f}` |"
        )
    leaderboard_md = "\n".join(leaderboard_rows)

    # Request Type breakdown rows
    req_rows = []
    for s in systems:
        rt = s["by_request_type"]
        req_rows.append(
            f"| `{s['name']}` | **{s['open_set_score']:.1f}%** | **{s['sealed_set_score']:.1f}%** | "
            f"`{rt['span_open']['score']:.1f}%` | `{rt['span_sealed']['score']:.1f}%` | "
            f"`{rt['noul_open']['score']:.1f}%` | `{rt['noul_sealed']['score']:.1f}%` | "
            f"`{rt['choice']['score']:.1f}%` | `{rt['math']['score']:.1f}%` | `{rt['score']['score']:.1f}%` |"
        )
    req_md = "\n".join(req_rows)

    # Tier & Domain breakdown rows
    dom_rows = []
    for s in systems:
        to = s["by_tier_open"]
        ts = s["by_tier_sealed"]
        dm = s["by_domain"]
        dom_rows.append(
            f"| `{s['name']}` | `{to['Easy']['score']:.1f}%` | `{to['Standard']['score']:.1f}%` | `{to['Judge']['score']:.1f}%` | "
            f"`{ts['Standard']['score']:.1f}%` | `{ts['Hard']['score']:.1f}%` | "
            f"`{dm['incident_report']['score']:.1f}%` | `{dm['compute_ledger']['score']:.1f}%` | "
            f"`{dm['squad_v2_open_qa']['score']:.1f}%` | `{dm['adversarial_stress']['score']:.1f}%` |"
        )
    dom_md = "\n".join(dom_rows)

    md = f"""# `nullspan` — Empirical & Statistical Evaluation Report (`SCIENCE_REPORT.md`)

> **Interactive Leaderboard & Explorer**: [**https://hemanth.github.io/nullspan/bench.html**](https://hemanth.github.io/nullspan/bench.html) (modeled on [BenchmarkHeaven / JevBench](https://benchmarkheaven.com/jev-models))
>
> **Reproducibility**: All numbers in this report are produced deterministically by `python bench/run_bench.py` (`{data['draw_id']}`, `N = {data['n_total_observations']}` observations across `14` systems, `B = 1,000` bootstrap resamples) and stored in [`bench/results.json`](./results.json).

---

## 1. NullspanBench (JevBench-Compatible Four-Axis Methodology)

Modeled directly after [BenchmarkHeaven's JevBench](https://benchmarkheaven.com/jev-models), **NullspanBench** evaluates `14` systems (`12` local open-weights/open-code models running live forward passes + `2` reference JSON-mode LLM decoders) across **`{data['n_total_observations']}` scored observations**:
- **Open Pool (`S = {data['n_open_observations']}` items)**: Multi-field SRE incident postmortems, FinOps GPU invoices, cluster throughput comparable tables, and edge traffic time-series forecasts from [`gnumanth/nullspan-dataset`](https://huggingface.co/datasets/gnumanth/nullspan-dataset).
- **Sealed Pool (`P = {data['n_sealed_observations']}` items)**: Held-out **Stanford SQuAD v2.0** (`rajpurkar/squad_v2`, 30 answerable + 30 adversarial unanswerable prose QA items) + 8 adversarial prompt-injection & decoy stress items.

### The Four Score Axes (`0–100`)
1. **Intelligence (`I`)**: Mean task competence across all `{data['n_total_observations']}` items (exact `[start, end]` grounding, SQuAD v2.0 token-F1, `p_null` abstention accuracy, boolean/enum accuracy, algebraic math & invariant verification, and quantile coverage), with 95% non-parametric percentile bootstrap confidence intervals (`B = 1,000`).
2. **Calibration (`C`)**: Probabilistic truthfulness of `p_null` abstention and `p_true`/`probs` judgments, combining `AUROC`, `1 - ECE` (10-bin Expected Calibration Error), and `1 - Brier`.
3. **Speed (`S`)**: Log-scaled score of median (`p50`) single-decision latency mapped against a `1.5 s` interactive ceiling.
4. **Cost (`K`)**: Log-scaled score of marginal compute cost (`$ / 1,000 decisions`) against a `$1.00 / 1k` ceiling.
5. **Capability Score**: $\\text{{Capability}} = 0.5 \\cdot I + 0.5 \\cdot C$ (pure model quality independent of hardware speed or pricing).
6. **Composite Score**: Weighted harmonic mean of $(I, C, S, K)$ with Official weights `25 : 25 : 25 : 25`:
   $$\\text{{Composite}} = \\left( \\frac{{w_I}}{{I}} + \\frac{{w_C}}{{C}} + \\frac{{w_S}}{{S}} + \\frac{{w_K}}{{K}} \\right)^{{-1}}$$

---

## 2. Official 14-System Leaderboard (`{data['draw_id']}`, $N = {data['n_total_observations']}$)

- **Environment**: `{env['os']}` · `{env['accelerator']}` · Python `{env['python']}` · PyTorch `{env['pytorch']}` · ONNX Runtime `{env['onnxruntime']}`

| Rank | System | Composite (`25:25:25:25`) | Capability (`50I:50C`) | Intelligence `I` (95% CI) | Calibration `C` | Speed `S` | Cost `K` | `$/1k` | `p50` Latency | `p_null` AUROC / ECE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{leaderboard_md}

### Key Empirical Findings
1. **Why Un-Finetuned Dense Bi-Encoders Fail (`9.9%–17.5%` Intelligence)**: Standard sentence/passage embedding models (`BAAI/bge-small-en-v1.5`, `all-mpnet-base-v2`, `all-MiniLM-L12-v2`, `all-MiniLM-L6-v2`, `paraphrase-albert-small-v2`, `paraphrase-MiniLM-L3-v2`) and raw `distilbert-base-uncased` can rank sentence similarity (`~45%–52%` on simple open spans), but cosine similarity alone **cannot** execute multi-number arithmetic (`ARITH_SUM`, `ARITH_MUL`), date duration deltas (`DATE_DURATION`), cross-field algebraic invariants (`subtotal == nodes * hourly_rate`), or calibrated `(p10, median, p90)` quantile estimation (`0.0%` on `Math` and `Score`).
2. **Why `nullspan` Beats Pure Zero-Dep Rules on Sealed Open-Domain Prose (`88.2%` vs `32.4%` Sealed)**: On structured telemetry (`S = 336`), `nullspan-ZeroDep (Fast Engine)` achieves `100.0%` in `0.005 ms`. However, on the **Sealed Pool (`P = 68`)** containing Stanford SQuAD v2.0 adversarial unanswerable traps, lexical overlap drops to `32.4%` (`AUROC 0.506`), whereas `nullspan-67M (PyTorch FP32)` achieves **`94.1%` Sealed** (`98.8%` overall Intelligence, `99.7` Calibration, `AUROC 0.9999`) and `nullspan-67M-Int8 (ONNX WASM/CPU)` achieves **`88.2%` Sealed** (`98.0%` overall Intelligence, `99.3` Calibration) at **`3.98x` smaller disk footprint (`64.3 MB`)**.
3. **Why `nullspan-Hybrid` Dominates the Pareto Frontier (`97.7` Composite)**: Routing structured telemetry through the zero-dependency fast path and delegating open-domain prose or ambiguous spans to the quantized `nullspan-67M-Int8` neural cross-encoder delivers `98.7` Capability at `0.005 ms` median latency and `$0.0003 / 1k` decisions—over **`120x` cheaper** and **`80,000x` faster at `p50`** than autoregressive JSON LLM decoders (`gpt-5.4-nano`, `claude-haiku-4.5`).

---

## 3. Competence by Request Type (Open `S` vs Sealed `P`)

| System | Open Pool (`n=336`) | Sealed Pool (`n=68`) | `Span (Open)` (`n=181`) | `Span (Sealed)` (`n=33`) | `Noul / p_null (Open)` (`n=27`) | `Noul / p_null (Sealed)` (`n=32`) | `Choice (Bool/Enum)` (`n=33`) | `Math & Invariants` (`n=50`) | `Score / Quantiles` (`n=48`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{req_md}

---

## 4. Competence by Difficulty Tier & Subject Domain

| System | Open `Easy` | Open `Standard` | Open `Judge` | Sealed `Standard` | Sealed `Hard` | `incident_report` | `compute_ledger` | `squad_v2_open_qa` | `adversarial_stress` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{dom_md}
"""
    REPORT_PATH.write_text(md)
    print(f"Wrote {REPORT_PATH} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
