"""VajraConfig — typed dataclass mirroring all locked values from REQUIREMENTS.lock."""

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    layers: int
    d_model: int
    attention_heads: int
    ffn_width: int
    dcat_layers: list[int] = field(default_factory=list)
    context_window_tokens: int = 4096


@dataclass
class DCATConfig:
    dual_stream_alpha: float = 0.3
    ema_beta_baseline_update: float = 0.9
    baseline_frozen_on_decision_classes: tuple[int, ...] = (1, 3)
    # θ_divergence is EXPERIMENT-1 (§12); this is a placeholder default until the
    # sweep resolves it. The override rule (§6.4) fires when mean DCAT divergence
    # exceeds this and the decision class is not already 0/1.
    theta_divergence: float = 1.0


@dataclass
class FusionConfig:
    blocks: int = 6
    d_model: int = 1024
    attention_heads: int = 16
    ffn_width: int = 4096
    ecl_tap_fast: str = "F3"
    ecl_tap_full: str = "F6"
    ecl_tap_fast_condition_defer_confidence_gt: float = 0.92
    ecl_tap_fast_condition_fail_safe_confidence_gt: float = 0.85


@dataclass
class AFNConfig:
    top_k: int = 16
    max_latency_ms: int = 15
    layer_target_12l: int = 8
    layer_target_8l: int = 6
    layer_target_6l: int = 4
    layer_target_4l: int = 3


@dataclass
class DecoderConfig:
    layers: int = 8
    d_model: int = 1024
    attention_heads: int = 16
    ffn_width: int = 4096
    max_output_tokens: int = 256
    afn_top_k: int = 16
    cross_attention_raw_encoder_states_allowed: bool = False


@dataclass
class QATConfig:
    schema: str = "wNa8o8"
    initialized_from_epoch: int = 0
    post_training_quantization_used: bool = False
    decoder_channel_wise_bits: int = 2
    kv_cache_bits: int = 8
    vram_target_encoder_fusion_gb: int = 4
    vram_target_decoder_gb: int = 2


@dataclass
class MTPDrafterConfig:
    decoder_layers: int = 2
    d_model: int = 512
    inference_device: str = "CPU"
    decoder_latency_ms_with_drafter: int = 50
    decoder_latency_ms_without_drafter: int = 90


@dataclass
class EmbeddingConfig:
    vocabulary_size: int = 50428
    source_type_count: int = 16
    temporal_frequency_pairs: int = 128
    temporal_encoding_dim: int = 256
    temporal_projection_dim: int = 1024
    ontology_embedding_dim_pre_projection: int = 256
    ontology_embedding_dim_post_projection: int = 1024
    ontology_frozen_after_stage: int = 1


@dataclass
class KillChainConfig:
    states: tuple[str, ...] = (
        "RECON",
        "WEAPONIZE",
        "DELIVER",
        "EXPLOIT",
        "INSTALL",
        "C2",
        "EXFIL",
    )
    skip_stages_allowed: bool = False


@dataclass
class VajraConfig:
    """Single source of truth for all locked architecture values."""

    shared_d_model: int = 1024
    total_parameters_estimated: int = 1_175_000_000

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    dcat: DCATConfig = field(default_factory=DCATConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    afn: AFNConfig = field(default_factory=AFNConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    qat: QATConfig = field(default_factory=QATConfig)
    mtp_drafter: MTPDrafterConfig = field(default_factory=MTPDrafterConfig)
    kill_chain: KillChainConfig = field(default_factory=KillChainConfig)

    domain_encoders: dict[str, EncoderConfig] = field(
        default_factory=lambda: {
            "detection_network": EncoderConfig(
                layers=12,
                d_model=1024,
                attention_heads=16,
                ffn_width=4096,
                dcat_layers=[7, 8],
                context_window_tokens=32768,
            ),
            "forensics_provenance": EncoderConfig(
                layers=12,
                d_model=1024,
                attention_heads=16,
                ffn_width=4096,
                dcat_layers=[7, 8],
                context_window_tokens=32768,
            ),
            "cti_stix": EncoderConfig(
                layers=8,
                d_model=768,
                attention_heads=12,
                ffn_width=3072,
            ),
            "vulnerability_risk": EncoderConfig(
                layers=8,
                d_model=768,
                attention_heads=12,
                ffn_width=3072,
            ),
            "identity_access": EncoderConfig(
                layers=6,
                d_model=768,
                attention_heads=12,
                ffn_width=3072,
            ),
            "incident_response": EncoderConfig(
                layers=6,
                d_model=768,
                attention_heads=12,
                ffn_width=3072,
            ),
            "compliance": EncoderConfig(
                layers=4,
                d_model=512,
                attention_heads=8,
                ffn_width=2048,
            ),
        }
    )

    # Hard constraints (non-mutable by design)
    exploit_synthesis_in_weights: bool = False
    free_text_attack_tooling_output: bool = False
    decoder_vocabulary_mask_level: str = "logit_p_neg_infinity"
    real_ips_in_training: bool = False
    baked_in_knowledge: str = "MITRE_ATT&CK_Enterprise_v16_tactics_and_techniques_only"

    # Inference targets
    inference_latency_ms_no_decoder: int = 100
    inference_latency_ms_with_decoder: int = 120

    # Sentinel tokens (Path A tokenization)
    sentinel_tokens: tuple[str, ...] = (
        "<|S_START|>",
        "<|S_END|>",
        "<|S_KEY|>",
        "<|S_VAL|>",
        "<|S_ARR|>",
        "<|S_NULL|>",
        "<|S_DOMAIN:detection|>",
        "<|S_DOMAIN:forensics|>",
        "<|S_DOMAIN:cti|>",
        "<|S_DOMAIN:vuln|>",
        "<|S_DOMAIN:identity|>",
        "<|S_DOMAIN:ir|>",
        "<|S_DOMAIN:compliance|>",
        "<|S_EVTX|>",
        "<|S_STIX|>",
        "<|S_SIGMA|>",
        "<|S_CLOUD|>",
        "<|S_LDAP|>",
        "<|S_BASELINE|>",
        "<|S_OBSERVED|>",
    )

    @property
    def sentinel_count(self) -> int:
        return len(self.sentinel_tokens)

    def afn_layer_for_encoder(self, domain: str) -> int:
        """Return the AFN target layer index for a given domain encoder."""
        enc = self.domain_encoders[domain]
        mapping = {
            12: self.afn.layer_target_12l,
            8: self.afn.layer_target_8l,
            6: self.afn.layer_target_6l,
            4: self.afn.layer_target_4l,
        }
        return mapping[enc.layers]
