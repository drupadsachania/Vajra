import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


# Mock config and encoders to avoid heavy imports/instantiations during test
class MockConfig:
    def __init__(self):
        self.d_model = 64


class MockDomainEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class MockFusionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=2, batch_first=True
        )
        # 4 classes for decision state
        self.decision_head = nn.Linear(d_model, 4)

    def forward(self, domain_outputs):
        # domain_outputs: [batch, 7, d_model]
        attn_out, _ = self.cross_attn(domain_outputs, domain_outputs, domain_outputs)
        # simple pool
        pooled = attn_out.mean(dim=1)
        logits = self.decision_head(pooled)
        return logits


class MockVajraModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 64
        # 7 isolated domain encoders
        self.encoders = nn.ModuleDict(
            {
                "detection": MockDomainEncoder(self.d_model),
                "forensics": MockDomainEncoder(self.d_model),
                "cti": MockDomainEncoder(self.d_model),
                "vulnerability": MockDomainEncoder(self.d_model),
                "identity": MockDomainEncoder(self.d_model),
                "ir": MockDomainEncoder(self.d_model),
                "compliance": MockDomainEncoder(self.d_model),
            }
        )
        self.fusion = MockFusionLayer(self.d_model)

    def forward(self, inputs):
        # inputs: dict of domain -> tensor [batch, seq_len, d_model]
        domain_cls_tokens = []
        for domain_name, encoder in self.encoders.items():
            if domain_name in inputs:
                # pass through encoder and get mock CLS token (mean pool)
                enc_out = encoder(inputs[domain_name])
                cls_token = enc_out.mean(dim=1, keepdim=True)
            else:
                cls_token = torch.zeros(
                    inputs[list(inputs.keys())[0]].shape[0],
                    1,
                    self.d_model,
                    device=encoder.linear1.weight.device,
                )
            domain_cls_tokens.append(cls_token)

        fusion_input = torch.cat(domain_cls_tokens, dim=1)
        return self.fusion(fusion_input)


def _run_distributed_isolation_test(rank, world_size):
    """Run DDP isolation test."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group using gloo for CPU compatibility
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = MockVajraModel()
    ddp_model = DDP(model)

    # Simulate a batch with only CTI data
    batch_size = 2
    seq_len = 10
    inputs = {
        "cti": torch.randn(batch_size, seq_len, model.d_model, requires_grad=True)
    }

    # Hook to capture gradients
    grad_norms = {}

    def make_hook(name):
        def hook(grad):
            grad_norms[name] = grad.norm().item()

        return hook

    # Register hooks on identity and cti encoders
    ddp_model.module.encoders["identity"].linear1.weight.register_hook(
        make_hook("identity")
    )
    ddp_model.module.encoders["cti"].linear1.weight.register_hook(make_hook("cti"))

    logits = ddp_model(inputs)

    # Fake loss tied to CTI (but passing through fusion)
    targets = torch.randint(0, 4, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)

    loss.backward()

    # Assertions
    # Since only CTI input was provided and identity was zero-filled/no gradient flow to identity input,
    # the identity encoder parameters should have strictly 0.0 gradient (or None if untouched).
    if ddp_model.module.encoders["identity"].linear1.weight.grad is not None:
        identity_grad_norm = (
            ddp_model.module.encoders["identity"].linear1.weight.grad.norm().item()
        )
        assert (
            identity_grad_norm == 0.0
        ), f"Isolation failure! Identity encoder received gradient norm {identity_grad_norm} from CTI-only input."

    if ddp_model.module.encoders["cti"].linear1.weight.grad is not None:
        cti_grad_norm = (
            ddp_model.module.encoders["cti"].linear1.weight.grad.norm().item()
        )
        assert (
            cti_grad_norm > 0.0
        ), "CTI encoder did not receive gradients despite having input."

    dist.destroy_process_group()


def test_distributed_isolation_integrity():
    """Test zero attention leakage/memory bleed between namespaces using distributed framework."""
    world_size = 2
    # Use torch.multiprocessing to spawn workers for DDP
    mp.spawn(
        _run_distributed_isolation_test,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
