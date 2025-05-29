import json
import os
from typing import Dict, List, Tuple, Iterable

try:
    import numpy as np
except ImportError:  # optional dependency
    np = None


class Starcoder2Tokenizer:
    """Byte Pair Encoding tokenizer with optional numpy usage."""

    def __init__(self, vocab: Dict[str, int] = None, merges: Iterable[Tuple[str, str]] = None):
        self.vocab: Dict[str, int] = vocab or {}
        self.inv_vocab: Dict[int, str] = {i: tok for tok, i in self.vocab.items()}
        self.merges = list(merges) if merges else []
        # rank of merge pairs for fast lookup
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            tuple(pair): idx for idx, pair in enumerate(self.merges)
        }
        self.cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # helper utilities
    @staticmethod
    def _get_pairs(tokens: List[str]) -> set:
        """Return set of adjacent token pairs."""
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs

    def _bpe(self, token: str) -> List[str]:
        if token in self.cache:
            return self.cache[token]

        if not self.bpe_ranks:
            # no merges defined -> return characters
            result = list(token)
            self.cache[token] = result
            return result

        word = list(token)
        pairs = self._get_pairs(word)
        while pairs:
            # select candidate pair with smallest rank
            min_pair = None
            min_rank = None
            for p in pairs:
                r = self.bpe_ranks.get(p)
                if r is not None and (min_rank is None or r < min_rank):
                    min_pair = p
                    min_rank = r
            if min_pair is None:
                break
            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = new_word
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        self.cache[token] = word
        return word

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, vocab_file: str, merges_file: str) -> "Starcoder2Tokenizer":
        """Load tokenizer vocabulary and merges from files."""
        with open(vocab_file, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)

        merges = []
        with open(merges_file, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        return cls(vocab=vocab, merges=merges)

    @classmethod
    def from_pretrained(cls, directory: str) -> "Starcoder2Tokenizer":
        """Load ``vocab.json`` and ``merges.txt`` from a directory."""
        vocab_path = os.path.join(directory, "vocab.json")
        merges_path = os.path.join(directory, "merges.txt")
        return cls.from_file(vocab_path, merges_path)

    # ------------------------------------------------------------------
    def _token_to_id(self, token: str) -> int:
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
        return self.vocab[token]

    def encode(self, text: str) -> List[int]:
        tokens: List[str] = []
        words = text.split()
        for i, word in enumerate(words):
            piece = word if i == 0 else "Ġ" + word
            tokens.extend(self._bpe(piece))

        ids = [self._token_to_id(t) for t in tokens]
        if np is not None:
            return np.array(ids, dtype=np.int64).tolist()
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(i, "") for i in ids]
        text = "".join(tokens)
        text = text.replace("Ġ", " ")
        return text
