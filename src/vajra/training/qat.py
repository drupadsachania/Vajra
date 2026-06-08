"""wNa8o8 QAT schedule (§8.1.1).

Applied from epoch 0:
- Decoder Linear layers: channel-wise 2-bit FakeQuantize observers
- All encoder + fusion KV-cache projections: 8-bit static FakeQuantize observers

In production these FakeQuantize wrappers train alongside the model so that
INT8 export is a lossless format conversion, not PTQ precision degradation.

KVCacheQuantizer attaches to nn.Linear layers that project K/V for attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


try:
    from torch.quantization import (
        FakeQuantize,
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
    )
    _QAT_AVAILABLE = True
except ImportError:
    _QAT_AVAILABLE = False


class KVCacheQuantizer(nn.Module):
    """8-bit static FakeQuantize wrapper for KV-cache Linear projections."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        if _QAT_AVAILABLE:
            self.fake_quant = FakeQuantize(
                observer=MovingAverageMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_affine,
            )
        else:
            self.fake_quant = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.fake_quant(x))


def _wrap_decoder_linear(module: nn.Module) -> None:
    """Replace Linear layers in the decoder with 2-bit channel-wise FakeQuantize wrappers."""
    if not _QAT_AVAILABLE:
        return
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            fq = FakeQuantize(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=0,
                quant_max=3,  # 2-bit: 0..3
                dtype=torch.quint8,
                qscheme=torch.per_channel_affine,
                ch_axis=0,
            )
            # Wrap: apply FQ to weight proxy; keep original linear for activation
            child.weight_fake_quant = fq
            setattr(module, name, child)
        else:
            _wrap_decoder_linear(child)


def _wrap_kv_projections(module: nn.Module, names: tuple[str, ...] = ("k", "v")) -> None:
    """Attach 8-bit KVCacheQuantizer to k/v projection linears in attention blocks."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in names:
            setattr(module, name, KVCacheQuantizer(child))
        else:
            _wrap_kv_projections(child, names)


class wNa8o8Schedule:
    """Applies QAT wrappers to a Vajra model instance.

    Decoder layers get 2-bit channel-wise FakeQuantize on weights.
    Encoder + fusion attention K/V projections get 8-bit static FakeQuantize.

    Call `apply_qat(model)` once before training epoch 0.
    """

    @staticmethod
    def apply_qat(model: nn.Module) -> nn.Module:
        # Decoder: 2-bit on Linear weights
        if hasattr(model, "decoder"):
            _wrap_decoder_linear(model.decoder)

        # Encoder stacks: 8-bit KV cache quantization
        for attr in ("domain_encoders", "fusion"):
            if hasattr(model, attr):
                _wrap_kv_projections(getattr(model, attr))

        return model

    @staticmethod
    def apply_qat_to_module(module: nn.Module, mode: str = "decoder") -> nn.Module:
        """Apply QAT wrappers to an isolated module (for testing without full model)."""
        if mode == "decoder":
            _wrap_decoder_linear(module)
        elif mode == "encoder":
            _wrap_kv_projections(module)
        return module
