"""Export NullspanForExtraction to ONNX (FP32 + Dynamic Int8 Quantized) for Browser WASM/WebGPU & Edge Runtime."""
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
    """Clean tensor-in, tensor-tuple-out wrapper for ONNX export (WASM/WebGPU compatible ops)."""

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
        enc_out = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = enc_out.last_hidden_state  # [B, L, H]
        cls_out = seq_out[:, 0, :]  # [B, H]

        null_logits = self.model.null_head(cls_out).squeeze(-1)  # [B]
        span_logits = self.model.span_head(seq_out)  # [B, L, 2]
        start_logits = span_logits[:, :, 0] + (1.0 - state_mask.float()) * (-1e4)
        end_logits = span_logits[:, :, 1] + (1.0 - state_mask.float()) * (-1e4)

        bio_logits = self.model.bio_head(seq_out)  # [B, L, 3]
        op_logits = self.model.op_head(cls_out)  # [B, 10]

        q_in = torch.cat([cls_out, num_feats], dim=-1)
        q_delta = torch.tanh(self.model.quantile_head(q_in)) * 0.15
        anchors = torch.stack([num_feats[:, 1], num_feats[:, 0], num_feats[:, 2]], dim=-1)
        quantile_preds = torch.clamp(anchors + q_delta, 0.0, 1.0)

        return null_logits, start_logits, end_logits, bio_logits, op_logits, quantile_preds


def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    cfg = NullspanConfig.from_pretrained(str(ARTIFACT_DIR))
    model = NullspanForExtraction.from_pretrained(str(ARTIFACT_DIR), config=cfg)
    model.encoder.config._attn_implementation = "eager"
    for layer in model.encoder.transformer.layer:
        layer.attention.config._attn_implementation = "eager"
    model.eval()

    wrapper = NullspanOnnxWrapper(model)
    wrapper.eval()

    B, L = 2, 64
    dummy_ids = torch.ones((B, L), dtype=torch.long)
    dummy_attn = torch.ones((B, L), dtype=torch.long)
    dummy_state = torch.ones((B, L), dtype=torch.long)
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

    # Copy tokenizer.json into ONNX_DIR and docs/ for browser tokenizer
    if (ARTIFACT_DIR / "tokenizer.json").exists():
        shutil.copyfile(ARTIFACT_DIR / "tokenizer.json", ONNX_DIR / "tokenizer.json")
        shutil.copyfile(ARTIFACT_DIR / "tokenizer.json", ROOT / "docs" / "tokenizer.json")

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
