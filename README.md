# Word2Vec in Pure NumPy

## Task

Implement the core training loop of word2vec in pure NumPy (no PyTorch, TensorFlow, or other ML frameworks). The task is to implement the optimization procedure: forward pass, loss, gradients, and parameter updates for a standard word2vec variant (skip-gram with negative sampling or CBOW).

## Data

1. Download a dataset from Kaggle (e.g. [Donald Trump Rallies](https://www.kaggle.com/datasets/christianlillelund/donald-trumps-rallies)) and place the files in the `archive/` folder.

2. Merge all files into one corpus:
```bash
python merge_corpus.py
```

The script creates `trump_rallies.txt` — the combined text from all `.txt` and `.csv` files in `archive/`.
