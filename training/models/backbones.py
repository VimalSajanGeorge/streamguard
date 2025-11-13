"""
Utilities for loading transformer backbones used in training.
"""

from typing import Dict, Tuple

from transformers import AutoModel, AutoTokenizer

BACKBONE_REGISTRY: Dict[str, Dict[str, str]] = {
    'codebert': {
        'model_name': 'microsoft/codebert-base',
        'pooling': 'cls'
    },
    'graphcodebert': {
        'model_name': 'microsoft/graphcodebert-base',
        'pooling': 'cls'
    },
    'unixcoder': {
        'model_name': 'microsoft/unixcoder-base',
        'pooling': 'cls'
    }
}


def load_backbone(model_key: str):
    """
    Load tokenizer and encoder for a supported backbone.

    Args:
        model_key: One of BACKBONE_REGISTRY keys.

    Returns:
        Tuple of (tokenizer, encoder, hidden_size, pooling_strategy).
    """
    key = (model_key or '').lower()
    if key not in BACKBONE_REGISTRY:
        available = ', '.join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(f"Unknown backbone '{model_key}'. Available: {available}")

    backbone_cfg = BACKBONE_REGISTRY[key]
    model_name = backbone_cfg['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    hidden_size = getattr(encoder.config, 'hidden_size', None)

    if hidden_size is None:
        raise ValueError(f"Backbone '{model_name}' is missing hidden_size metadata.")

    pooling = backbone_cfg.get('pooling', 'cls')
    return tokenizer, encoder, hidden_size, pooling
