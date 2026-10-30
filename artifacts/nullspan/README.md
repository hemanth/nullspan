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
  - deberta-v3
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

# gnumanth/nullspan (`v1.2.1`)

**`nullspan` v1.2.1** ships two zero-output-token **System One** neural models in one open-weights MIT repository:

1. **`NullspanForExtraction` (`345.3M` PyTorch FP32 + `331.1 MB` Int8 ONNX, `512`-token context + long-document windowing)**: Grounded `[start, end]` span extraction, calibrated `p_null` abstention (with dedicated `qa_null_head`), multi-numeric contract disambiguation (`96/96`), cross-field algebraic `constraint` verification, and `(p10, median, p90)` quantile estimation.
2. **`NullspanSystemOneEngine` (`435M` DeBERTa-v3-Large + `r=32` LoRA + Cross-Option & Cross-Question Workflow Competition Head, `systemone/`)**: Native `/v1/systemone` calibrated decision model for `noul` (`true`/`false`), `choice` (`options`), and ordinal `score` (`min..max`) rubrics with `0` generated output tokens.

## Verified Public Benchmark Scores (Google Cloud NVIDIA L4 GPU · Zero Test Contamination)

| Public Benchmark | Dataset & Split | Score | Calibration & Validity | Latency / Cost (L4 GPU) |
| :--- | :--- | :---: | :---: | :---: |
| **[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)** | Official `test` (`400` cases · `2,000` decisions) | **`75.40%` Accuracy** (Table 2 fitted cross-encoder; `#2` vs Hub `.eval_results` widget) | `KL 0.1515` · `Brier 0.0770` · `ECE 0.0944` | `78.0 ms/dec` (`390.0 ms/case`) · `0` output tokens |
| **[Benchmark Heaven / `jevbench`](https://github.com/fstandhartinger/jevbench)** | Official Public Suite (`231` items via `/v1/systemone`) | **`100.0%` Easy (`48/48`)** · `62.50%` Original (`45/72`) · `37.84%` Hard (`42/111`) | **`100.0%` Strict Schema Validity** (`Easy Brier 0.0207`, `ECE 0.0605`) | **`64.8 ms` `p50`** · **`$0.00667 / 1k`** (`encoder_classifier_<=0.6B`) |
| **[Stanford `rajpurkar/squad_v2`](https://huggingface.co/datasets/rajpurkar/squad_v2)** | Zero-overlap `validation` (`N = 500`: `250` HasAns + `250` NoAns) | **`87.00%` Exact Match** · **`89.36%` F1** | `HasAns F1 91.52%` · `NoAns EM 87.20%` | `10.30 ms` `p50` · `$0.0017 / 1k` |
| **[`NullspanBench v3.2`](https://hemanth.github.io/nullspan/bench.html)** | `N = 500` (`336` Open + `164` Sealed: SQuAD2, `96` Contracts, `8` Adversarial) | **`98.2` Capability** · **`98.0%` Intelligence** (`95.7` Hybrid Composite) | **`98.5` Calibration** (`AUROC 0.9933`, `ECE 0.0142`; Int8 ONNX `ECE 0.0127`) | `10.30 ms` CUDA `p50` (`0.040 ms` Hybrid) · `$0.0003–$0.0017 / 1k` |

> **Training Disclosure**: `NullspanSystemOneEngine` is fine-tuned on the `LocalLLaMA/typed-decisions` `train` split alongside public `BoolQ`, `ARC-Challenge`, and `SciQ` `train` splits plus programmatic synthetic System One rubrics; it is **not zero-shot** on `Typed Decisions`. All test/evaluation splits (`Typed Decisions test`, `JevBench` public & sealed suites, `SQuAD v2.0 validation`, and `NullspanBench` evaluation pools) were strictly untouched during training and post-hoc temperature calibration.

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
pip install "nullspan[hf]" fastapi uvicorn peft
python serve_systemone.py --model-dir gnumanth/nullspan --port 8010
```

## Repository Contents

- `model.safetensors`, `config.json`, `modeling_nullspan.py`: `NullspanForExtraction` (`345.3M` ELECTRA-Large + 2-layer Pre-LN `SchemaAdapterBlock` + `qa_null_head`).
- `onnx/model_quantized.onnx`, `onnx/tokenizer.json`: `331.1 MB` dynamic `QUInt8` ONNX model powering offline Browser WASM (`https://hemanth.github.io/nullspan/`) and CPU inference.
- `systemone/`, `systemone_decision.pt`, `systemone_decision.py`, `serve_systemone.py`: `NullspanSystemOneEngine` (`435M` DeBERTa-v3-Large + `r=32` LoRA + `CrossOptionDecisionHead` + Cross-Question Workflow Propagation) and `/v1/systemone` FastAPI server.
- `.eval_results/typed-decisions.yaml`, `.eval_results/nullspan-bench.yaml`, `SCIENCE_REPORT.md`: Reproducible benchmark metadata and statistical report.

## License

MIT © [Hemanth.HM](https://h3manth.com)
