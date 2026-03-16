#!/usr/bin/env python3
"""
Entry point for word2vec pipeline.
"""

import argparse
from pathlib import Path

from data import build_vocab, tokenize, tokens_to_ids


def create_model(vocab_size: int, embedding_dim: int):
    """Create skip-gram model (W_in, W_out)."""
    raise NotImplementedError("create_model")


def train(model, ids, word_counts, word2idx, idx2word, **kwargs):
    """Train model on token ids."""
    raise NotImplementedError("train")


def main():
    parser = argparse.ArgumentParser(description="Word2vec pipeline")
    parser.add_argument("--corpus", type=Path, default=None, help="Path to corpus txt (default: trump_rallies.txt)")
    parser.add_argument("--min-count", type=int, default=1, help="Min word count for vocabulary")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args()

    corpus_path = args.corpus or (Path(__file__).parent / "trump_rallies.txt")
    if not corpus_path.exists():
        print("Run merge_corpus.py first to create trump_rallies.txt")
        return

    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    tokens = tokenize(text)
    word2idx, idx2word, word_counts = build_vocab(tokens, min_count=args.min_count)
    ids = tokens_to_ids(tokens, word2idx)
    vocab_size = len(word2idx)

    print(f"Tokens: {len(tokens):,}, vocab size: {vocab_size}")

    model = create_model(vocab_size, args.dim)
    train(
        model,
        ids,
        word_counts=word_counts,
        word2idx=word2idx,
        idx2word=idx2word,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
