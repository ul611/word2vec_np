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
    parser.add_argument("--min-count", type=int, default=5, help="Min word count for vocabulary")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--negatives", type=int, default=5, help="Negative samples per pair")
    parser.add_argument("--lr", type=float, default=0.025, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-tokens", type=int, default=None, help="Use first N tokens (for quick runs)")
    parser.add_argument("--show-neighbors", action="store_true", help="Print nearest neighbors after each epoch")
    parser.add_argument("--neighbor-words", nargs="*", default=["election", "trump", "america", "people", "great"], help="Words for --show-neighbors")
    args = parser.parse_args()

    corpus_path = args.corpus or (Path(__file__).parent / "trump_rallies.txt")
    if not corpus_path.exists():
        print("Run merge_corpus.py first to create trump_rallies.txt")
        return

    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    tokens = tokenize(text)
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]
    word2idx, idx2word, word_counts = build_vocab(tokens, min_count=args.min_count)
    ids = tokens_to_ids(tokens, word2idx)
    vocab_size = len(word2idx)

    subset = f" (first {args.max_tokens:,})" if args.max_tokens else ""
    print(f"Tokens: {len(tokens):,}{subset}, vocab size: {vocab_size}")

    model = create_model(vocab_size, args.dim, seed=args.seed)
    W_in, W_out = train(
        model,
        ids,
        word_counts=word_counts,
        window_size=args.window,
        num_negatives=args.negatives,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        show_neighbors=args.show_neighbors,
        neighbor_words=args.neighbor_words,
        word2idx=word2idx,
        idx2word=idx2word,
    )


if __name__ == "__main__":
    main()
