#!/usr/bin/env python3
"""
Validate that ARCHITECTURE.md has not drifted from REQUIREMENTS.lock.

Usage:
    python validate_requirements.py                 # checks spec integrity only
    python validate_requirements.py --impl <path>   # also validates an implementation config JSON
"""

import json
import hashlib
import sys
import argparse
from pathlib import Path

LOCK_FILE = Path(__file__).parent / "REQUIREMENTS.lock"
SPEC_FILE = Path(__file__).parent / "ARCHITECTURE.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def check_spec_integrity() -> list[str]:
    errors = []
    with open(LOCK_FILE) as f:
        lock = json.load(f)

    expected = lock["_meta"]["spec_sha256"]
    actual = sha256_file(SPEC_FILE)
    if expected != actual:
        errors.append(
            f"SPEC INTEGRITY FAILURE: ARCHITECTURE.md has changed since lock-in.\n"
            f"  Locked SHA-256 : {expected}\n"
            f"  Current SHA-256: {actual}\n"
            f"  Any change to ARCHITECTURE.md requires operator sign-off and a new REQUIREMENTS.lock."
        )
    else:
        print(f"[OK] ARCHITECTURE.md integrity verified (SHA-256: {actual[:16]}...)")
    return errors


def check_impl(impl_path: Path) -> list[str]:
    errors = []
    warnings = []

    with open(LOCK_FILE) as f:
        lock = json.load(f)
    with open(impl_path) as f:
        impl = json.load(f)

    def get(d, *keys, default=None):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    def check_eq(impl_val, lock_val, label):
        if impl_val is None:
            warnings.append(f"[MISSING] {label} not found in implementation config")
        elif impl_val != lock_val:
            errors.append(f"[MISMATCH] {label}: impl={impl_val!r}, locked={lock_val!r}")
        else:
            print(f"[OK] {label}: {impl_val}")

    def check_lte(impl_val, lock_val, label):
        if impl_val is None:
            warnings.append(f"[MISSING] {label} not found in implementation config")
        elif impl_val > lock_val:
            errors.append(f"[VIOLATION] {label}: impl={impl_val} exceeds locked max={lock_val}")
        else:
            print(f"[OK] {label}: {impl_val} <= {lock_val}")

    def check_gte(impl_val, lock_val, label):
        if impl_val is None:
            warnings.append(f"[MISSING] {label} not found in implementation config")
        elif impl_val < lock_val:
            errors.append(f"[VIOLATION] {label}: impl={impl_val} is below locked min={lock_val}")
        else:
            print(f"[OK] {label}: {impl_val} >= {lock_val}")

    arch = lock["architecture"]
    hc = lock["hard_constraints"]
    sc = lock["success_criteria"]

    # Hard constraints
    total_params = get(impl, "total_parameters")
    if total_params is not None:
        limit = 4_000_000_000
        if total_params > limit:
            errors.append(f"[VIOLATION] total_parameters: {total_params:,} exceeds hard limit {limit:,}")
        else:
            print(f"[OK] total_parameters: {total_params:,} <= {limit:,}")

    check_eq(get(impl, "exploit_synthesis_in_weights"),        False,  "exploit_synthesis_in_weights")
    check_eq(get(impl, "free_text_attack_tooling_output"),     False,  "free_text_attack_tooling_output")
    check_eq(get(impl, "decoder_vocabulary_mask_level"),
             "logit_p_neg_infinity",                                   "decoder_vocabulary_mask_level")
    check_eq(get(impl, "real_ips_in_training"),                False,  "real_ips_in_training")

    # Shared projection dimension
    check_eq(get(impl, "shared_d_model"),                      1024,   "shared_d_model")

    # Domain encoders
    for domain, locked_enc in arch["domain_encoders"].items():
        for field in ("layers", "d_model", "attention_heads", "ffn_width"):
            impl_val = get(impl, "domain_encoders", domain, field)
            check_eq(impl_val, locked_enc[field], f"domain_encoders.{domain}.{field}")

    # Decoder
    dec = arch["constrained_decoder"]
    check_eq(get(impl, "decoder", "layers"),              dec["layers"],              "decoder.layers")
    check_eq(get(impl, "decoder", "d_model"),             dec["d_model"],             "decoder.d_model")
    check_eq(get(impl, "decoder", "max_output_tokens"),   dec["max_output_tokens"],   "decoder.max_output_tokens")
    check_eq(get(impl, "decoder", "cross_attention_source"),
             "AFN_top_k_evidence_tokens_only",                                        "decoder.cross_attention_source")
    check_eq(get(impl, "decoder", "afn_top_k"),           dec["afn_top_k"],           "decoder.afn_top_k")
    check_eq(get(impl, "decoder", "cross_attention_raw_encoder_states_allowed"),
             False,                                                                   "decoder.cross_attention_raw_encoder_states_allowed")

    # Epistemic fusion
    fus = arch["epistemic_fusion_layer"]
    check_eq(get(impl, "fusion", "blocks"),          fus["blocks"],           "fusion.blocks")
    check_eq(get(impl, "fusion", "d_model"),         fus["d_model"],          "fusion.d_model")
    check_eq(get(impl, "fusion", "attention_heads"), fus["attention_heads"],   "fusion.attention_heads")

    # ECL tap points
    check_eq(get(impl, "fusion", "ecl_tap_fast"),    "F3",                    "fusion.ecl_tap_fast")
    check_eq(get(impl, "fusion", "ecl_tap_full"),    "F6",                    "fusion.ecl_tap_full")
    check_eq(get(impl, "fusion", "ecl_tap_fast_condition_defer_confidence_gt"),
             0.92,                                                            "fusion.ecl_tap_fast_condition_defer_confidence_gt")

    # AFN
    afn = arch["afn"]
    check_lte(get(impl, "afn", "max_latency_ms"),  afn["max_latency_ms"],    "afn.max_latency_ms")
    check_eq( get(impl, "afn", "top_k"),           afn["top_k_evidence_tokens"], "afn.top_k")

    # Latency targets
    check_lte(get(impl, "inference_latency_ms_no_decoder"),
              sc["inference_latency_ms_4096_tokens_no_decoder"]["target"],
              "inference_latency_ms_no_decoder")

    # Tokenization constraints
    check_eq(get(impl, "tokenization", "cvss_bpe_allowed"),         False, "tokenization.cvss_bpe_allowed")
    check_eq(get(impl, "tokenization", "ip_addresses_bpe_allowed"), False, "tokenization.ip_addresses_bpe_allowed")
    check_eq(get(impl, "tokenization", "ecl_tokens_encoder_input_allowed"), False,
             "tokenization.ecl_tokens_encoder_input_allowed")

    # Encoder isolation (no shared weights, no cross-encoder attention)
    check_eq(get(impl, "encoder_shared_config", "shared_weights_across_encoders"),   False,
             "encoder_config.shared_weights_across_encoders")
    check_eq(get(impl, "encoder_shared_config", "cross_encoder_attention_within_stacks"), False,
             "encoder_config.cross_encoder_attention_within_stacks")

    # Training pipeline constraints
    check_eq(get(impl, "training", "stage_3", "policy_gradient_RL_used"), False,
             "training.stage_3.policy_gradient_RL_used")
    check_eq(get(impl, "training", "stage_2", "verbose_NL_CoT_in_training_signal"), False,
             "training.stage_2.verbose_NL_CoT_in_training_signal")
    check_eq(get(impl, "training", "stage_1", "sentinel_positions_masked"), False,
             "training.stage_1.sentinel_positions_masked")

    # ONNX export
    check_eq(get(impl, "integration", "argus_xdr", "onnx_opset"), 17, "integration.argus_xdr.onnx_opset")

    # Knowledge boundary
    check_eq(get(impl, "baked_in_knowledge"),
             "MITRE_ATT&CK_Enterprise_v16_tactics_and_techniques_only",
             "baked_in_knowledge")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate REQUIREMENTS.lock compliance")
    parser.add_argument("--impl", type=Path, help="Path to implementation config JSON to validate")
    args = parser.parse_args()

    all_errors = []
    all_warnings = []

    print("=" * 60)
    print("SecureFoundation Requirements Lock Validator")
    print("=" * 60)

    # Always check spec integrity first
    integrity_errors = check_spec_integrity()
    all_errors.extend(integrity_errors)

    if args.impl:
        print(f"\nValidating implementation config: {args.impl}")
        print("-" * 60)
        if not args.impl.exists():
            all_errors.append(f"Implementation config not found: {args.impl}")
        else:
            impl_errors, impl_warnings = check_impl(args.impl)
            all_errors.extend(impl_errors)
            all_warnings.extend(impl_warnings)

    print("\n" + "=" * 60)

    if all_warnings:
        print(f"\nWARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  {w}")

    if all_errors:
        print(f"\nFAILURES ({len(all_errors)}):")
        for e in all_errors:
            print(f"  {e}")
        print("\nResult: NON-COMPLIANT")
        sys.exit(1)
    else:
        print("\nResult: COMPLIANT" if args.impl else "\nResult: SPEC INTEGRITY OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
