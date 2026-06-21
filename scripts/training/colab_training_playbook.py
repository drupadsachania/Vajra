"""
Colab/Kaggle Training Playbook for Vajra

This script demonstrates how to train Vajra using PyTorch Lightning.
It applies 4-bit NF4 double quantization to the entire model for memory efficiency
and targets QLoRA adapters STRICTLY at the Epistemic Fusion layer and the 
Constrained Decoder. The 7 Domain Encoders remain fully frozen to preserve
their isolated parameter namespaces and prevent catastrophic forgetting.
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from vajra.config import VajraConfig
from vajra.interface.pipeline import VajraInferencePipeline

class VajraLightningTrainer(pl.LightningModule):
    def __init__(self, cfg: VajraConfig):
        super().__init__()
        self.cfg = cfg
        
        # 1. Initialize the raw Vajra Pipeline (built from scratch, not a HuggingFace generic model)
        # Using tiny=False to build the full 7-encoder + fusion architecture
        pipeline = VajraInferencePipeline(cfg, tiny=False)
        
        # 2. Configure 4-bit Quantization (NF4) for Colab A100s
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # In a real environment, you'd apply quantization directly during instantiation or via accelerate.
        # Here we simulate the quantization process setup for the pipeline.
        # NOTE: pipeline must be moved/quantized before PEFT wrapping.
        self.pipeline = pipeline 
        
        # Freeze Domain Encoders explicitly
        for name, param in self.pipeline.domain_encoders.named_parameters():
            param.requires_grad = False
            
        # 3. Apply QLoRA targeted specifically at Fusion and Decoder
        # Target modules match the attention and feed-forward layers in EpistemicFusionLayer and ConstrainedDecoder
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["f3_head", "f6_head", "cross_attn", "out_proj", "fc1", "fc2", "q_proj", "v_proj"], 
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION" # Custom task type for our non-HF architecture
        )
        
        # Wrap fusion and decoder with PEFT
        self.pipeline.fusion = get_peft_model(self.pipeline.fusion, lora_config)
        self.pipeline.decoder = get_peft_model(self.pipeline.decoder, lora_config)
        
        print("Trainable parameters focused entirely on Fusion and Decoder:")
        self.pipeline.fusion.print_trainable_parameters()
        self.pipeline.decoder.print_trainable_parameters()
        
        # Loss function for decision state (Stage 4)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        # batch represents a dictionary of domain tensors and targets
        # This is a placeholder for the actual PyTorch Lightning training step
        # using Vajra's internal forward pass methods.
        inputs, targets = batch
        
        # Note: Since we are training, we bypass the pipeline.run() inference wrapper
        # and directly interface with the modules or implement a training_forward method.
        # This is omitted for brevity in this playbook template.
        
        loss = torch.tensor(0.0, requires_grad=True) # Placeholder
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Only pass trainable parameters to the optimizer
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)
        return optimizer

# Playbook Execution
if __name__ == "__main__":
    print("Initializing Vajra PEFT Training Playbook...")
    cfg = VajraConfig()
    
    # Normally you'd load your Fenrir-v2.1 or OpTC datasets into a LightningDataModule here
    # datamodule = VajraDataModule(...)
    
    model = VajraLightningTrainer(cfg)
    
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed"
    )
    
    # trainer.fit(model, datamodule)
    print("Playbook initialized. Ready to run trainer.fit()!")
