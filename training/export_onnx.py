"""Export NullspanForExtraction v2.0 to ONNX (FP32 + Dynamic Int8 Quantized) for Browser WASM/WebGPU & Edge Runtime."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from modeling_nullspan import NullspanConfig, NullspanForExtraction

ARTIFACT_DIR = ROOT / "artifacts" / "nullspan"
ONNX_DIR = ARTIFACT_DIR / "onnx"


class NullspanOnnxWrapper(nn.Module):
    """Clean tensor-in, tensor-tuple-out wrapper sharing exact NullspanForExtraction.forward() math & temperature scaling."""

    def __init__(self, model: NullspanForExtraction) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_mask: torch.Tensor,
        num_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # In eval() mode, self.model.forward() computes token_type_ids internally, runs span-attended null_head,
        # and applies null_temperature & span_temperature scaling!
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            state_mask=state_mask,
            num_feats=num_feats,
        )
        return (
            out["null_logits"],
            out["start_logits"],
            out["end_logits"],
            out["bio_logits"],
            out["op_logits"],
            out["quantile_preds"],
        )


def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    cfg = NullspanConfig.from_pretrained(str(ARTIFACT_DIR))
    model = NullspanForExtraction.from_pretrained(str(ARTIFACT_DIR), config=cfg)
    model.encoder.config._attn_implementation = "eager"
    if hasattr(model.encoder, "encoder") and hasattr(model.encoder.encoder, "layer"):
        for layer in model.encoder.encoder.layer:
            if hasattr(layer, "attention") and hasattr(layer.attention, "self") and hasattr(layer.attention.self, "config"):
                layer.attention.self.config._attn_implementation = "eager"
    elif hasattr(model.encoder, "transformer") and hasattr(model.encoder.transformer, "layer"):
        for layer in model.encoder.transformer.layer:
            if hasattr(layer.attention, "config"):
                layer.attention.config._attn_implementation = "eager"
    model.eval()

    wrapper = NullspanOnnxWrapper(model)
    wrapper.eval()

    B, L = 2, 64
    dummy_ids = torch.ones((B, L), dtype=torch.long)
    dummy_ids[:, 0] = 101
    dummy_ids[:, 12] = 102
    dummy_ids[:, -1] = 102
    dummy_attn = torch.ones((B, L), dtype=torch.long)
    dummy_state = torch.zeros((B, L), dtype=torch.long)
    dummy_state[:, 13:-1] = 1
    dummy_feats = torch.zeros((B, 8), dtype=torch.float32)

    fp32_path = ONNX_DIR / "model.onnx"
    int8_path = ONNX_DIR / "model_quantized.onnx"

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_attn, dummy_state, dummy_feats),
        str(fp32_path),
        input_names=["input_ids", "attention_mask", "state_mask", "num_feats"],
        output_names=["null_logits", "start_logits", "end_logits", "bio_logits", "op_logits", "quantile_preds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "state_mask": {0: "batch_size", 1: "seq_len"},
            "num_feats": {0: "batch_size"},
            "null_logits": {0: "batch_size"},
            "start_logits": {0: "batch_size", 1: "seq_len"},
            "end_logits": {0: "batch_size", 1: "seq_len"},
            "bio_logits": {0: "batch_size", 1: "seq_len"},
            "op_logits": {0: "batch_size"},
            "quantile_preds": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Consolidate external data if any into single model.onnx
    onnx_model = onnx.load(str(fp32_path), load_external_data=True)
    onnx.save_model(onnx_model, str(fp32_path), save_as_external_data=False)
    for ext_file in ONNX_DIR.glob("*.data"):
        ext_file.unlink()

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
    )

    # Copy tokenizer.json into ONNX_DIR, docs/, and space/ for browser tokenizer
    if (ARTIFACT_DIR / "tokenizer.json").exists():
        shutil.copyfile(ARTIFACT_DIR / "tokenizer.json", ONNX_DIR / "tokenizer.json")
        shutil.copyfile(ARTIFACT_DIR / "tokenizer.json", ROOT / "docs" / "tokenizer.json")
        if (ROOT / "space").exists():
            shutil.copyfile(ARTIFACT_DIR / "tokenizer.json", ROOT / "space" / "tokenizer.json")

    fp32_mb = round(fp32_path.stat().st_size / (1024 * 1024), 2)
    int8_mb = round(int8_path.stat().st_size / (1024 * 1024), 2)
    print(json.dumps({
        "onnx_fp32_path": str(fp32_path),
        "onnx_fp32_size_mb": fp32_mb,
        "onnx_int8_path": str(int8_path),
        "onnx_int8_size_mb": int8_mb,
        "compression_ratio": round(fp32_mb / max(0.1, int8_mb), 2),
    }, indent=2))


if __name__ == "__main__":
    main()
