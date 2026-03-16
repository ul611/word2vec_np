"""
Nearest neighbors for embeddings.
"""

import numpy as np


def nearest_neighbors(embeddings: np.ndarray, word: str, word2idx: dict, idx2word: dict, top_k: int = 5) -> None:
    """Print nearest neighbors for a word (cosine similarity)."""
    if word not in word2idx:
        print(f"  '{word}' not in vocab")
        return
    idx = word2idx[word]
    vec = embeddings[idx]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sims = (embeddings @ vec) / (norms.squeeze() * np.linalg.norm(vec))
    sims[idx] = -np.inf
    top = np.argsort(sims)[::-1][:top_k]
    print(f"  {word}: {', '.join(idx2word[i] for i in top)}")
