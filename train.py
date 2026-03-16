"""
Training loop for skip-gram with negative sampling.
"""

from data import generate_training_pairs


def train(model: tuple, ids, window_size: int = 5, epochs: int = 5):
    """Train skip-gram. Returns W_in, W_out."""
    W_in, W_out = model
    pairs = generate_training_pairs(ids, window_size)

    for epoch in range(epochs):
        pass  # stub

    return W_in, W_out
