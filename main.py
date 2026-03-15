#!/usr/bin/env python3
"""
"""

import argparse
from pathlib import Path

from data import build_vocab, tokenize, tokens_to_ids


def main():
    parser = argparse.ArgumentParser(description="Tokenize corpus and build vocabulary")
    parser.add_argument("--corpus", type=Path, default=None, help="Path to corpus txt (default: trump_rallies.txt)")
    parser.add_argument("--min-count", type=int, default=1, help="Min word count for vocabulary")
    args = parser.parse_args()

    corpus_path = args.corpus or (Path(__file__).parent / "trump_rallies.txt")
    if not corpus_path.exists():
        print("Run merge_corpus.py first to create trump_rallies.txt")
        return

    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    tokens = tokenize(text)
    word2idx, idx2word, word_counts = build_vocab(tokens, min_count=args.min_count)
    ids = tokens_to_ids(tokens, word2idx)

    print(f"Tokens: {len(tokens):,}")
    print(f"Vocab size: {len(word2idx)}")
    print(f"Sample tokens: {tokens[:20]}")
    print(f"Sample ids: {ids[:20].tolist()}")


if __name__ == "__main__":
    main()
