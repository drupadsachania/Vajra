"""ONNX export for Vajra encoder-fusion and decoder graphs (§10.4).

Exports:
- vajra_encoder_fusion.onnx  — encoder stack + epistemic fusion; opset 17
- vajra_decoder.onnx          — constrained decoder; opset 17

Dynamic axes: batch_size, sequence_length.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


ONNX_OPSET = 17


def export_encoder_fusion(
    encoder_fusion_model: nn.Module,
    path: str | Path = "vajra_encoder_fusion.onnx",
    d_model: int = 1024,
    seq_len: int = 8,
    n_domains: int = 7,
) -> Path:
    """Export the encoder+fusion graph to ONNX opset 17.

    encoder_fusion_model: a module whose forward accepts (batch, n_domains, d_model)
                          and returns (batch, 4) decision logits.
    """
    path = Path(path)
    dummy = torch.randn(1, n_domains, d_model)
    torch.onnx.export(
        encoder_fusion_model,
        (dummy,),
        str(path),
        opset_version=ONNX_OPSET,
        input_names=["domain_cls_tokens"],
        output_names=["decision_logits"],
        dynamic_axes={
            "domain_cls_tokens": {0: "batch_size"},
            "decision_logits": {0: "batch_size"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    return path


def export_decoder(
    decoder_model: nn.Module,
    path: str | Path = "vajra_decoder.onnx",
    vocab_size: int = 50428,
    afn_top_k: int = 16,
    d_model: int = 1024,
    tgt_len: int = 4,
) -> Path:
    """Export the constrained decoder graph to ONNX opset 17."""
    path = Path(path)
    dummy_ids   = torch.randint(0, 100, (1, tgt_len))
    dummy_enc   = torch.randn(1, afn_top_k, d_model)
    torch.onnx.export(
        decoder_model,
        (dummy_ids, dummy_enc),
        str(path),
        opset_version=ONNX_OPSET,
        input_names=["input_ids", "afn_encoder_states"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":          {0: "batch_size", 1: "sequence_length"},
            "afn_encoder_states": {0: "batch_size"},
            "logits":             {0: "batch_size", 1: "sequence_length"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    return path
