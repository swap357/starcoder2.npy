from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import math

try:
    import numpy as np
except ImportError:  # Fallback to pure Python if numpy is unavailable
    np = None


@dataclass
class Starcoder2Config:
    """Configuration mirror of the HuggingFace Starcoder2Config."""

    vocab_size: int = 32000
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_layers: int = 4


class Starcoder2Model:
    """Minimal Transformer model loosely following HuggingFace Starcoder2."""

    def __init__(self, config: Starcoder2Config | None = None, weights_path: str | None = None):
        self.config = config or Starcoder2Config()
        self.weights: Dict[str, Any] = {}
        if weights_path:
            self.load_weights(weights_path)
        else:
            self._init_weights()

    def load_weights(self, path: str):
        """Load model weights from an ``.npy`` file.

        The ``.npy`` file should contain a dictionary with numpy arrays or
        lists. If ``numpy`` is not available, a pickle loader is used as a
        fallback.
        """
        if np is not None:
            self.weights = np.load(path, allow_pickle=True).item()
        else:
            import pickle
            with open(path, "rb") as f:
                self.weights = pickle.load(f)

    # ------------------------------------------------------------------
    def _randn(self, *shape):
        """Utility to create random arrays or nested lists."""
        if np is not None:
            return np.random.randn(*shape).astype("float32")
        import random
        return [[random.gauss(0, 1) for _ in range(shape[1])] for _ in range(shape[0])]

    def _init_weights(self):
        """Create random weights similar to HuggingFace initialisation."""
        h = self.config.hidden_size
        v = self.config.vocab_size
        n = self.config.num_layers

        self.weights["embedding"] = self._randn(v, h)
        self.weights["linear"] = self._randn(h, v)

        for i in range(n):
            for name in ("wq", "wk", "wv", "wo"):
                self.weights[f"{name}_{i}"] = self._randn(h, h)
            self.weights[f"ff1_{i}"] = self._randn(h, 4 * h)
            self.weights[f"ff2_{i}"] = self._randn(4 * h, h)

    def _matmul(self, a, b):
        result = []
        for row in a:
            result_row = []
            for col in zip(*b):
                result_row.append(sum(x * y for x, y in zip(row, col)))
            result.append(result_row)
        return result

    def _softmax(self, x: List[List[float]]) -> List[List[float]]:
        result = []
        for row in x:
            m = max(row)
            exps = [math.exp(i - m) for i in row]
            s = sum(exps)
            result.append([e / s for e in exps])
        return result

    def forward(self, input_ids: List[int]) -> List[List[float]]:
        """Forward pass through a tiny Transformer encoder."""
        emb = self.weights["embedding"]
        if np is not None:
            x = emb[input_ids]
        else:
            x = [emb[i] for i in input_ids]

        for i in range(self.config.num_layers):
            wq = self.weights[f"wq_{i}"]
            wk = self.weights[f"wk_{i}"]
            wv = self.weights[f"wv_{i}"]
            wo = self.weights[f"wo_{i}"]
            ff1 = self.weights[f"ff1_{i}"]
            ff2 = self.weights[f"ff2_{i}"]

            if np is not None:
                q = x @ wq
                k = x @ wk
                v = x @ wv
                att = q @ k.transpose() / math.sqrt(self.config.hidden_size)
                att = np.exp(att - att.max(axis=-1, keepdims=True))
                att = att / att.sum(axis=-1, keepdims=True)
                x = x + (att @ v) @ wo
                ff = np.maximum(0, x @ ff1)
                x = x + ff @ ff2
            else:
                q = self._matmul(x, wq)
                k = self._matmul(x, wk)
                v = self._matmul(x, wv)
                kt = list(zip(*k))
                att_scores = []
                for row in q:
                    att_scores.append([sum(a*b for a,b in zip(row,col))/math.sqrt(self.config.hidden_size) for col in kt])
                att = self._softmax(att_scores)
                att_out = self._matmul(att, v)
                proj = self._matmul(att_out, wo)
                x = [[a+b for a,b in zip(rp, pp)] for rp, pp in zip(x, proj)]
                ff_in = self._matmul(x, ff1)
                ff_act = [[max(0, y) for y in row] for row in ff_in]
                ff_out = self._matmul(ff_act, ff2)
                x = [[a+b for a,b in zip(rp, pp)] for rp, pp in zip(x, ff_out)]

        linear = self.weights["linear"]
        if np is not None:
            logits = x @ linear
            return logits.tolist()
        logits = self._matmul(x, linear)
        return logits

    def generate(self, input_ids: List[int], steps: int = 1) -> List[int]:
        ids = input_ids[:]
        for _ in range(steps):
            logits = self.forward(ids)
            last = logits[-1]
            if np is not None:
                next_id = int(np.argmax(last))
            else:
                next_id = last.index(max(last))
            ids.append(next_id)
        return ids
