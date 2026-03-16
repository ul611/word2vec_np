#!/usr/bin/env python3
"""
Entry point for word2vec pipeline.
"""

import argparse
from pathlib import Path

from data import build_vocab, tokenize, tokens_to_ids
from model import create_model
from train import train


def main():
    parser = argparse.ArgumentParser(description="Word2vec pipeline")
    parser.add_argument("--corpus", type=Path, default=None, help="Path to corpus txt (default: trump_rallies.txt)")
    parser.add_argument("--min-count", type=int, default=1, help="Min word count for vocabulary")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--negatives", type=int, default=5, help="Negative samples per pair")
    parser.add_argument("--lr", type=float, default=0.025, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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

    model = create_model(vocab_size, args.dim, seed=args.seed)
    train(model, ids, window_size=args.window, epochs=args.epochs)


if __name__ == "__main__":
    main()
