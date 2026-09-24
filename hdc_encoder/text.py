"""Character trigram hypervector encoding of text (European languages)."""
import re

import numpy as np
import torch
import torch.nn as nn
from torchhd import embeddings, functional

MAX_INPUT_SIZE = 128
PADDING_IDX = 0
ASCII_A, ASCII_Z = ord("a"), ord("z")
NUM_TOKENS = (ASCII_Z - ASCII_A + 1) + 1 + 1  # letters, space/other, padding


def char2int(char):
    a = ord(char)
    if ASCII_A <= a <= ASCII_Z:
        return a - ASCII_A
    return ASCII_Z - ASCII_A + 1  # space and every other character


def tokenize(x):
    """Lower-cased text with collapsed whitespace as MAX_INPUT_SIZE padded token ids."""
    x = re.sub(r"\s+", " ", x.lower())[:MAX_INPUT_SIZE]
    ids = [char2int(ch) + 1 for ch in x]
    return torch.tensor(ids + [PADDING_IDX] * (MAX_INPUT_SIZE - len(ids)), dtype=torch.long)


class TrigramEncoder(nn.Module):
    """Bipolar bundle of the bound, permuted character trigrams of token id sequences."""

    def __init__(self, dim, vocab_size=NUM_TOKENS, padding_idx=PADDING_IDX):
        super().__init__()
        self.symbol = embeddings.Random(vocab_size, dim, padding_idx=padding_idx)

    @torch.no_grad()
    def forward(self, x_ids):
        symbols = self.symbol(x_ids.to(self.symbol.weight.device))  # [B, T, D]
        hv = functional.ngrams(symbols, n=3)                         # [B, D]
        return functional.normalize(hv)                              # bipolarize

    def encode_texts(self, texts, batch_size=256):
        """Bipolar HVs of a list of strings, as a float32 numpy array."""
        return np.concatenate([
            self(torch.stack([tokenize(t) for t in texts[i:i + batch_size]])).cpu().numpy()
            for i in range(0, len(texts), batch_size)
        ])
