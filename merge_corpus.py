#!/usr/bin/env python3
"""
Merge all text files from archive/ into one corpus trump_rallies.txt.

Download a dataset from Kaggle and place the files in archive/.
Then run: python merge_corpus.py
"""

import csv
from pathlib import Path

ARCHIVE_DIR = Path(__file__).parent / "archive"
OUTPUT_TXT = Path(__file__).parent / "trump_rallies.txt"


def read_file(path: Path) -> str:
    """Read txt or csv, return combined text."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        parts = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                for v in row.values():
                    if v and len(str(v).strip()) > 5:
                        parts.append(str(v).strip())
        return "\n\n".join(parts)
    return path.read_text(encoding="utf-8", errors="replace").strip()


def main():

    texts = []
    for path in sorted(ARCHIVE_DIR.glob("**/*")):
        if path.is_file() and path.suffix.lower() in (".txt", ".csv", ""):
            try:
                content = read_file(path)
                if content:
                    texts.append(content)
            except Exception as e:
                print(f"Skipping {path.name}: {e}")

    if not texts:
        print("No text files in archive/. Place Kaggle files there first.")
        return

    combined = "\n\n".join(texts)
    OUTPUT_TXT.write_text(combined, encoding="utf-8")
    print(f"Merged {len(texts)} files → {OUTPUT_TXT} ({len(combined):,} chars)")


if __name__ == "__main__":
    main()
