---
license: mit
library_name: transformers
tags:
  - nullspan
  - system-one
  - typed-decisions
  - jevbench
  - structured-extraction
  - span-extraction
  - calibrated-estimation
  - electra
  - modernbert
  - onnx
datasets:
  - gnumanth/nullspan-dataset
  - LocalLLaMA/typed-decisions
  - rajpurkar/squad_v2
metrics:
  - accuracy
  - exact_match
  - f1
  - brier_score
pipeline_tag: question-answering
---

# gnumanth/nullspan (`v1.3.0`)

**`nullspan` v1.3.0** ships two zero-output-token **System One** neural models in one open-weights MIT repository:

1. **`NullspanForExtraction` (`345.3M` PyTorch FP32 + `331.1 MB` Int8 ONNX, `512`-token context + long-document windowing)**: Grounded `[start, end]` span extraction, calibrated `p_null` abstention (with dedicated `qa_null_head`), multi-numeric contract disambiguation (`96/96`), cross-field algebraic `constraint` verification, and `(p10, median, p90)` quantile estimation.
2. **`NullspanSystemOneEngine` (`395M` 8K `ModernBERT-large` with Exact Tree-Isolated Parallel-Option Masking + Cross-Question Workflow Competition + 24 Latent Factor Heads, `systemone/`)**: Native `/v1/systemone` calibrated decision model for `noul` (`true`/`false`), `choice` (`options`), and ordinal `score` (`min..max`) rubrics in **a single forward pass per case** (`47.8 ms/case`, `9.56 ms/decision` on L4) with `0` generated output tokens and `0.0` cross-option RoPE bleed (`100%` option-order permutation invariant).

## Verified Public Benchmark Scores (Google Cloud NVIDIA L4 GPU · Zero Test Contamination)

| Public Benchmark | Dataset & Split | Score | Calibration & Validity | Latency / Cost (L4 GPU) |
| :--- | :--- | :---: | :---: | :---: |
| **[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)** | Official `test` (`400` cases · `2,000` decisions) | **`79.00%` Accuracy** (`96.3%` of ~`82.05%` empirical Bayes ceiling; `#1` Hub `.eval_results` widget; beats `Arbiter-4B` `78.6%`, `qwen3.5-27b` `78.3%`, `gpt-5.4` `77.7%`, `SafeDecide-400M` `77.5%`, `TypeSafe Jev` `73.3%`) | **`KL 0.0760` (`#2` worldwide)** · **`Brier 0.0420` (`#1` tied worldwide)** · `ECE 0.1665` | **`9.56 ms/dec` (`47.8 ms/case`)** · `0` output tokens |
| **[Benchmark Heaven / `jevbench`](https://github.com/fstandhartinger/jevbench)** | Official Public Suite (`231` items via `/v1/systemone`) | **`100.0%` Easy (`48/48`)** · **`68.06%` Original (`49/72`)** · `36.04%` Hard (`40/111`) (`59.31%` all · `v1.3 Geo 46.27`) | **`100.0%` Strict Schema Validity** (`ECE 0.0755`) | **`38.82 ms` `p50`** (`37.76 ms` Easy) · **`$0.00604 / 1k`** (`encoder_classifier_<=0.6B`) |
| **[Stanford `rajpurkar/squad_v2`](https://huggingface.co/datasets/rajpurkar/squad_v2)** | Zero-overlap `validation` (`N = 500`: `250` HasAns + `250` NoAns) | **`87.80%` Exact Match** · **`90.90%` F1** | `HasAns F1 90.61%` · `NoAns EM 91.20%` | `10.30 ms` `p50` · `$0.0017 / 1k` |
| **[`NullspanBench v3.2`](https://hemanth.github.io/nullspan/bench.html)** | `N = 500` (`336` Open + `164` Sealed: SQuAD2, `96` Contracts, `8` Adversarial) | **`98.2` Capability** · **`98.0%` Intelligence** (`95.7` Hybrid Composite) | **`98.5` Calibration** (`AUROC 0.9933`, `ECE 0.0142`; Int8 ONNX `ECE 0.0127`) | `10.30 ms` CUDA `p50` (`0.040 ms` Hybrid) · `$0.0003–$0.0017 / 1k` |

> **Training Disclosure (`mode: specialist`)**: `NullspanSystemOneEngine` (`v1.3.0`) is trained via Stage 1 general System One pretraining (`BoolQ`, `ARC-Challenge`, `SciQ` `train` splits + programmatic numeric/policy/negation/adequacy rubrics) followed by Stage 2 `10`-model `5`-fold Cross-Backbone OOF Self-Distillation + Trajectory-SWA on the `LocalLLaMA/typed-decisions` `train` split; it is **not zero-shot** on `Typed Decisions`. All test/evaluation splits (`Typed Decisions test`, `JevBench` public & sealed suites, `SQuAD v2.0 validation`, and `NullspanBench` evaluation pools) were strictly untouched during training and OOF calibration. Prior `v1.2.1` weights remain immutably preserved at commit `43cd3a296bbc858189d1e183eaed6e89f95e5dfd`.

## Quick Start — Grounded Extraction (`NullspanForExtraction`)

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

## Quick Start — `/v1/systemone` Typed Decisions (`NullspanSystemOneEngine`)

```python
from nullspan import decide

res = decide(
    state={"customer_tier": "enterprise", "RefundCapUSD": 500, "RequestedRefundUSD": 420, "FraudFlag": False},
    questions={
        "eligible": {"type": "noul", "instructions": "Is RequestedRefundUSD <= RefundCapUSD with FraudFlag == False?"},
        "route": {"type": "choice", "options": ["approve", "escalate", "deny"], "instructions": "Select refund action."},
        "urgency": {"type": "score", "min": 1, "max": 5, "instructions": "Rate SLA urgency from 1 to 5."},
    },
    model_id="gnumanth/nullspan",
)
print(res["answers"])
```

Or serve the TypeSafe-compatible `/v1/systemone` HTTP endpoint directly:

```bash
pip install "nullspan[hf]" fastapi uvicorn
python serve_systemone.py --model-dir gnumanth/nullspan --port 8010
```

## Repository Contents

- `model.safetensors`, `config.json`, `modeling_nullspan.py`: `NullspanForExtraction` (`345.3M` ELECTRA-Large + 2-layer Pre-LN `SchemaAdapterBlock` + `qa_null_head`).
- `onnx/model_quantized.onnx`, `onnx/tokenizer.json`: `331.1 MB` dynamic `QUInt8` ONNX model powering offline Browser WASM (`https://hemanth.github.io/nullspan/`) and CPU inference.
- `systemone/`, `systemone_decision.pt`, `systemone_decision.py`, `serve_systemone.py`, `predictions_typed_decisions_test.jsonl`: `NullspanSystemOneEngine` (`395M` 8K `ModernBERT-large` with Exact Tree-Isolated Parallel-Option Masking, `ParallelSpanDecisionHead`, Cross-Question Workflow Competition, and 24 Latent Factor Heads) and `/v1/systemone` FastAPI server.
- `.eval_results/typed-decisions.yaml`, `.eval_results/nullspan-bench.yaml`, `SCIENCE_REPORT.md`: Reproducible benchmark metadata and statistical report.

## License

MIT © [Hemanth.HM](https://h3manth.com)
