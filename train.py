"""
Training loop for skip-gram with negative sampling.
"""

import numpy as np

from data import generate_training_pairs, build_unigram_table, sample_negatives
from neighbors import nearest_neighbors


def _sigmoid(x: np.ndarray, clip_value: float = 500.0) -> np.ndarray:
    x = np.clip(x, -clip_value, clip_value)
    return 1.0 / (1.0 + np.exp(-x))


def _forward(
    center_id: int,
    context_id: int,
    neg_ids: np.ndarray,
    W_in: np.ndarray,
    W_out: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass. Returns: loss, grad_v_c, grad_v_w, grad_v_negs."""
    v_c = W_in[center_id]
    v_w = W_out[context_id]
    v_negs = W_out[neg_ids]

    s_pos = np.dot(v_c, v_w)
    s_negs = v_negs @ v_c

    sigma_pos = _sigmoid(np.array([s_pos]))[0]
    sigma_negs = _sigmoid(s_negs)

    # Loss: -log σ(s_pos) - Σ log σ(-s_neg)
    loss_pos = -np.log(sigma_pos + 1e-10)
    loss_negs = -np.sum(np.log(1.0 - sigma_negs + 1e-10))
    loss = loss_pos + loss_negs

    # Gradients: dL/ds_pos = σ-1, dL/ds_neg = σ
    d_pos = sigma_pos - 1.0
    d_negs = sigma_negs

    grad_v_c = d_pos * v_w + np.dot(d_negs, v_negs)
    grad_v_w = d_pos * v_c
    grad_v_negs = d_negs[:, np.newaxis] * v_c

    return loss, grad_v_c, grad_v_w, grad_v_negs


def _train_step(center_id: int, context_id: int, neg_ids: np.ndarray, W_in: np.ndarray, W_out: np.ndarray, lr: float) -> float:
    loss, grad_v_c, grad_v_w, grad_v_negs = _forward(center_id, context_id, neg_ids, W_in, W_out)
    # Parameter update (SGD)
    W_in[center_id] -= lr * grad_v_c
    W_out[context_id] -= lr * grad_v_w
    for k, n in enumerate(neg_ids):
        W_out[n] -= lr * grad_v_negs[k]
    return loss


def train(
    model: tuple,
    ids,
    word_counts: list,
    window_size: int = 5,
    num_negatives: int = 5,
    epochs: int = 5,
    lr: float = 0.025,
    seed: int = 42,
    show_neighbors: bool = False,
    neighbor_words: list = None,
    word2idx: dict = None,
    idx2word: dict = None,
):
    """Train skip-gram with negative sampling. Returns W_in, W_out."""
    W_in, W_out = model
    rng = np.random.default_rng(seed)

    unigram_table = build_unigram_table(word_counts, table_size=int(1e6), power=0.75, seed=seed)
    table_idx = rng.integers(0, len(unigram_table))

    pairs = generate_training_pairs(ids, window_size)
    n_pairs = len(pairs)

    for epoch in range(epochs):
        perm = rng.permutation(n_pairs)
        total_loss = 0.0

        for i in perm:
            center_id, context_id = pairs[i]
            neg_ids, table_idx = sample_negatives(
                num_negatives, {context_id}, unigram_table, table_idx
            )
            loss = _train_step(center_id, context_id, neg_ids, W_in, W_out, lr)
            total_loss += loss

        avg_loss = total_loss / n_pairs
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
        if show_neighbors and neighbor_words and word2idx and idx2word:
            print(f"  Epoch {epoch + 1} neighbors:")
            for word in neighbor_words:
                nearest_neighbors(W_in, word, word2idx, idx2word)

    return W_in, W_out
