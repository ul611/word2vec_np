#!/usr/bin/env python3
"""
Entry point for word2vec pipeline.
"""

import argparse
from pathlib import Path

from config import config
from data import build_vocab, tokenize, tokens_to_ids
from model import create_model
from train import train


def main():
    parser = argparse.ArgumentParser(description="Word2vec pipeline")
    parser.add_argument("--corpus", type=Path, default=config.corpus, help="Path to corpus txt")
    parser.add_argument("--min-count", type=int, default=config.min_count, help="Min word count for vocabulary")
    parser.add_argument("--min-length", type=int, default=config.min_length, help="Min word length (chars), e.g. 3 to drop 1-2 letter words")
    parser.add_argument("--dim", type=int, default=config.dim, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Training epochs")
    parser.add_argument("--window", type=int, default=config.window, help="Context window size")
    parser.add_argument("--negatives", type=int, default=config.negatives, help="Negative samples per pair")
    parser.add_argument("--lr", type=float, default=config.lr, help="Learning rate")
    parser.add_argument("--seed", type=int, default=config.seed, help="Random seed")
    parser.add_argument("--max-tokens", type=int, default=config.max_tokens, help="Use first N tokens (for quick runs)")
    parser.add_argument("--show-neighbors", action="store_true", default=config.show_neighbors, help="Print nearest neighbors after each epoch")
    parser.add_argument("--neighbor-words", nargs="*", default=config.default_neighbor_words, help="Words for --show-neighbors")
    parser.add_argument("--keep-stopwords", action="store_true", default=config.keep_stopwords, help="Do not remove stopwords from corpus")
    args = parser.parse_args()

    corpus_path = args.corpus or (Path(__file__).parent / config.corpus)
    if not corpus_path.exists():
        print("Run merge_corpus.py first to create trump_rallies.txt")
        return

    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    tokens, n_total, n_after_stopwords = tokenize(
        text, remove_stopwords=not args.keep_stopwords, min_length=args.min_length
    )
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]
    word2idx, idx2word, word_counts = build_vocab(tokens, min_count=args.min_count)
    ids = tokens_to_ids(tokens, word2idx)
    vocab_size = len(word2idx)

    print(f"Tokens: {n_total:,} total, {n_after_stopwords:,} after filters, {len(tokens):,} used, vocab size: {vocab_size}")

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
