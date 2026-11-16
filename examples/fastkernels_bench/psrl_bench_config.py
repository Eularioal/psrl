"""Serializable bounded configuration for the original ConvNeXt implementation."""

from dataclasses import dataclass


@dataclass
class VisionConfig:
    num_channels: int = 3
    hidden_sizes: tuple = (16, 32, 64, 128)
    depths: tuple = (1, 1, 1, 1)
    patch_size: int = 4
    layer_norm_eps: float = 1e-12
    num_labels: int = 4
    hidden_act: str = "gelu"


@dataclass
class TextConfig:
    vocab_size: int = 128
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    max_position_embeddings: int = 32
    layer_norm_eps: float = 1e-5
    eos_token_id: int = 2
