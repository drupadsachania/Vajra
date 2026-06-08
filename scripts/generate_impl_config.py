#!/usr/bin/env python3
"""Generate impl_config.json by reading locked values from REQUIREMENTS.lock."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "REQUIREMENTS.lock"
OUTPUT = ROOT / "impl_config.json"


def main():
    with open(LOCK) as f:
        lock = json.load(f)

    arch = lock["architecture"]
    tok = lock["tokenization"]
    enc_shared = arch["encoder_shared_config"]
    dec = arch["constrained_decoder"]
    fus = arch["epistemic_fusion_layer"]
    afn = arch["afn"]
    sc = lock["success_criteria"]
    tp = lock["training_pipeline"]

    domain_encoders = {}
    for name, enc in arch["domain_encoders"].items():
        domain_encoders[name] = {
            "layers": enc["layers"],
            "d_model": enc["d_model"],
            "attention_heads": enc["attention_heads"],
            "ffn_width": enc["ffn_width"],
        }

    config = {
        "total_parameters": int(arch["total_parameters_estimated"].replace("_", "")),
        "exploit_synthesis_in_weights": lock["hard_constraints"]["exploit_synthesis_in_weights"],
        "free_text_attack_tooling_output": lock["hard_constraints"]["free_text_attack_tooling_output"],
        "decoder_vocabulary_mask_level": lock["hard_constraints"]["decoder_vocabulary_mask_level"],
        "real_ips_in_training": lock["hard_constraints"]["real_ips_in_training"],
        "shared_d_model": arch["shared_d_model"],
        "domain_encoders": domain_encoders,
        "decoder": {
            "layers": dec["layers"],
            "d_model": dec["d_model"],
            "max_output_tokens": dec["max_output_tokens"],
            "cross_attention_source": dec["cross_attention_source"],
            "afn_top_k": dec["afn_top_k"],
            "cross_attention_raw_encoder_states_allowed": dec["cross_attention_raw_encoder_states_allowed"],
        },
        "fusion": {
            "blocks": fus["blocks"],
            "d_model": fus["d_model"],
            "attention_heads": fus["attention_heads"],
            "ecl_tap_fast": fus["ecl_tap_fast"],
            "ecl_tap_full": fus["ecl_tap_full"],
            "ecl_tap_fast_condition_defer_confidence_gt": fus["ecl_tap_fast_condition_defer_confidence_gt"],
        },
        "afn": {
            "max_latency_ms": afn["max_latency_ms"],
            "top_k": afn["top_k_evidence_tokens"],
        },
        "inference_latency_ms_no_decoder": sc["inference_latency_ms_4096_tokens_no_decoder"]["target"],
        "tokenization": {
            "cvss_bpe_allowed": tok["cvss_bpe_allowed"],
            "ip_addresses_bpe_allowed": tok["ip_addresses_bpe_allowed"],
            "ecl_tokens_encoder_input_allowed": tok["ecl_tokens_encoder_input_allowed"],
        },
        "encoder_shared_config": {
            "shared_weights_across_encoders": enc_shared["shared_weights_across_encoders"],
            "cross_encoder_attention_within_stacks": enc_shared["cross_encoder_attention_within_stacks"],
        },
        "training": {
            "stage_1": {"sentinel_positions_masked": tp["stage_1"]["sentinel_positions_masked"]},
            "stage_2": {"verbose_NL_CoT_in_training_signal": tp["stage_2"]["verbose_NL_CoT_in_training_signal"]},
            "stage_3": {"policy_gradient_RL_used": tp["stage_3"]["policy_gradient_RL_used"]},
        },
        "integration": {
            "argus_xdr": {"onnx_opset": lock["interface"]["onnx_opset"]}
        },
        "baked_in_knowledge": lock["operator_locked_decisions"]["knowledge_boundary"]["baked_in"]["content"],
    }

    with open(OUTPUT, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    main()
