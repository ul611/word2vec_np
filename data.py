"""
Data preprocessing for word2vec: tokenization and vocabulary.
"""

import re

import numpy as np
from collections import Counter

from config import config


def tokenize(text: str, remove_stopwords: bool = True, min_length: int = 1) -> tuple[list[str], int, int]:
    """Lowercase and extract words (letters + apostrophe). If remove_stopwords: drop stopwords and words shorter than min_length.
    Returns: (tokens, n_total, n_after_filters)."""
    tokens_raw = re.findall(r"[a-z']+", text.lower())
    n_total = len(tokens_raw)
    if remove_stopwords:
        tokens = [t for t in tokens_raw if t not in config.stopwords and len(t) >= min_length]
    else:
        tokens = tokens_raw
    return tokens, n_total, len(tokens)


def build_vocab(tokens: list[str], min_count: int = 1) -> tuple[dict, dict, list]:
    """
    Build vocabulary from tokens.
    Returns: word2idx, idx2word, word_counts.
    """
    counts = Counter(tokens)
    word2idx = {"<pad>": 0, "<unk>": 1}
    idx2word = {0: "<pad>", 1: "<unk>"}
    word_counts = [0, 0]

    for w, c in counts.most_common():
        if c < min_count:
            break
        idx = len(word2idx)
        word2idx[w] = idx
        idx2word[idx] = w
        word_counts.append(c)

    return word2idx, idx2word, word_counts


def tokens_to_ids(tokens: list[str], word2idx: dict) -> np.ndarray:
    """Convert tokens to integer IDs (unk for OOV)."""
    return np.array([word2idx.get(w, word2idx["<unk>"]) for w in tokens], dtype=np.int64)


def build_unigram_table(
    word_counts: list,
    table_size: int = int(1e6),
    power: float = 0.75,
    seed: int | None = None,
) -> np.ndarray:
    """Unigram table for negative sampling. P(w) ∝ count^power."""
    rng = np.random.default_rng(seed)
    counts = np.array(word_counts, dtype=np.float64)
    counts = np.power(counts, power)
    probs = counts / counts.sum()
    return rng.choice(len(probs), size=table_size, p=probs)


def sample_negatives(
    num_negatives: int,
    exclude: set[int],
    unigram_table: np.ndarray,
    table_idx: int,
) -> tuple[np.ndarray, int]:
    """Sample negative indices from unigram table."""
    negs = []
    table_len = len(unigram_table)
    while len(negs) < num_negatives:
        idx = int(unigram_table[table_idx % table_len])
        table_idx += 1
        if idx not in exclude:
            negs.append(idx)
    return np.array(negs, dtype=np.int64), table_idx


def generate_training_pairs(ids: np.ndarray, window_size: int) -> list[tuple[int, int]]:
    """Generate (center, context) pairs for skip-gram."""
    pairs = []
    n = len(ids)
    for i in range(n):
        center = ids[i]
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        for j in range(start, end):
            if j != i:
                pairs.append((center, ids[j]))
    return pairs
