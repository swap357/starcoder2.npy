from typing import List, Dict, Any

try:
    import numpy as np
except ImportError:  # Fallback to pure Python if numpy is unavailable
    np = None


class Starcoder2Model:
    """Simplified Starcoder2 model implemented in pure Python."""

    def __init__(self, weights_path: str = None):
        self.weights: Dict[str, Any] = {}
        if weights_path:
            self.load_weights(weights_path)

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

    def _matmul(self, a, b):
        result = []
        for row in a:
            result_row = []
            for col in zip(*b):
                result_row.append(sum(x * y for x, y in zip(row, col)))
            result.append(result_row)
        return result

    def forward(self, input_ids: List[int]) -> List[List[float]]:
        """Compute logits for the next token in pure Python."""
        x = [[float(i)] for i in input_ids]
        embedding = self.weights.get("embedding")
        linear = self.weights.get("linear")

        if embedding is not None:
            if np is not None:
                embedded = embedding[input_ids]
            else:
                embedded = [embedding[i] for i in input_ids]
        else:
            embedded = x

        if linear is not None:
            if np is not None:
                logits = embedded @ linear
            else:
                logits = self._matmul(embedded, linear)
        else:
            logits = embedded

        if np is not None:
            return logits.tolist()
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
