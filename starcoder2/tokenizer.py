import json
from typing import List


class Starcoder2Tokenizer:
    """A very small whitespace tokenizer as a placeholder."""

    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}

    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        ids = []
        for token in tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inv_vocab[idx] = token
            ids.append(self.vocab[token])
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.inv_vocab.get(i, "") for i in ids)
