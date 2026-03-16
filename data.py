"""
Data preprocessing for word2vec: tokenization and vocabulary.
"""

import re

import numpy as np
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase and extract words (letters + apostrophe for contractions)."""
    return re.findall(r"[a-z']+", text.lower())


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
