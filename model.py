"""
Skip-gram model: parameter initialization.
"""

import numpy as np


def create_model(
    vocab_size: int,
    embedding_dim: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create skip-gram model. Returns (W_in, W_out). Xavier init."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(embedding_dim)
    W_in = rng.standard_normal((vocab_size, embedding_dim)).astype(np.float64) * scale
    W_out = rng.standard_normal((vocab_size, embedding_dim)).astype(np.float64) * scale
    return W_in, W_out
