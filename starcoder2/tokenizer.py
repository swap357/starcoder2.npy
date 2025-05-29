import json
import os
from typing import List, Optional


class Starcoder2Tokenizer:
    """Minimal byte pair encoding tokenizer."""

    def __init__(self, vocab_path: Optional[str] = None, merges_path: Optional[str] = None):
        base = os.path.join(os.path.dirname(__file__), "resources")
        vocab_path = vocab_path or os.path.join(base, "vocab.json")
        merges_path = merges_path or os.path.join(base, "merges.txt")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(merges_path, "r", encoding="utf-8") as f:
            merges_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        self.merges = [tuple(line.split()) for line in merges_lines]
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    tokens[i : i + 2] = [tokens[i] + tokens[i + 1]]
                else:
                    i += 1
        return tokens

    def encode(self, text: str) -> List[int]:
        tokens = list(text)
        tokens = self._apply_merges(tokens)
        ids = [self.vocab[token] for token in tokens]
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.inv_vocab.get(i, "") for i in ids)
