# Word2Vec in Pure NumPy

Skip-gram with negative sampling, implemented from scratch (no PyTorch/TensorFlow).

## Setup

```bash
pip install -r requirements.txt
```

## Data

1. Download [Donald Trump Rallies](https://www.kaggle.com/datasets/christianlillelund/donald-trumps-rallies) and place files in `archive/`.

2. Merge into one corpus:
```bash
python merge_corpus.py
```
Creates `trump_rallies.txt`.

## Config

Parameters in `config.yaml`. Access via `config.lr`, `config.epochs`, etc. Edit YAML to change defaults.

## Run

```bash
python main.py
```

Quick run (10k tokens, 5 epochs, show neighbors):
```bash
python main.py --max-tokens 10000 --epochs 5 --show-neighbors
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--corpus` | from config | Path to corpus txt |
| `--min-count` | 3 | Min word count for vocab |
| `--min-length` | 3 | Min word length (drops 1–2 letter words when stopwords removed) |
| `--dim` | 128 | Embedding dimension |
| `--epochs` | 10 | Training epochs |
| `--window` | 5 | Context window size |
| `--negatives` | 5 | Negative samples per pair |
| `--lr` | 0.025 | Learning rate |
| `--seed` | 42 | Random seed |
| `--max-tokens` | null | Use first N tokens (for quick runs) |
| `--show-neighbors` | false | Print nearest neighbors after each epoch |
| `--neighbor-words` | trump, america, ... | Words for --show-neighbors |
| `--keep-stopwords` | false | Do not remove stopwords |

## Structure

| File | Role |
|------|------|
| `main.py` | Entry point, CLI |
| `config.py` | Load config.yaml, expose `config` |
| `config.yaml` | All parameters |
| `data.py` | tokenize, build_vocab, training pairs, negative sampling |
| `model.py` | Xavier init for W_in, W_out |
| `train.py` | Forward, loss, gradients, SGD, training loop |
| `neighbors.py` | Cosine similarity, nearest neighbors |
| `merge_corpus.py` | Merge archive/*.txt, *.csv → trump_rallies.txt |

## Algorithm

- **Skip-gram**: predict context from center word
- **Negative sampling**: loss = −log σ(v_c·v_w) − Σ log σ(−v_c·v_neg)
- **Unigram table**: P(neg) ∝ count^0.75
- **SGD** updates on each (center, context, negatives) tuple
