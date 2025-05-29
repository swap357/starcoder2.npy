from starcoder2 import Starcoder2Model, Starcoder2Tokenizer


EMBEDDING = [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5, 1.6],
    [1.7, 1.8, 1.9, 2.0],
    [2.1, 2.2, 2.3, 2.4],
    [2.5, 2.6, 2.7, 2.8],
    [2.9, 3.0, 3.1, 3.2],
    [3.3, 3.4, 3.5, 3.6],
    [3.7, 3.8, 3.9, 4.0],
]

LINEAR = [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
    [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
]

WEIGHTS = {"embedding": EMBEDDING, "linear": LINEAR}


class TorchStyleModel:
    """Torch-like reference implementation using Python lists."""

    def __init__(self, weights):
        self.embedding = weights["embedding"]
        self.linear = weights["linear"]

    def _matmul(self, a, b):
        result = []
        for row in a:
            result_row = []
            for col in zip(*b):
                result_row.append(sum(x * y for x, y in zip(row, col)))
            result.append(result_row)
        return result

    def forward(self, input_ids):
        embedded = [self.embedding[i] for i in input_ids]
        return self._matmul(embedded, self.linear)

    def generate(self, ids, steps=1):
        ids = list(ids)
        for _ in range(steps):
            logits = self.forward(ids)
            last = logits[-1]
            next_id = last.index(max(last))
            ids.append(next_id)
        return ids


def test_forward_equivalence():
    tokenizer = Starcoder2Tokenizer()
    input_ids = tokenizer.encode("hello world")

    torch_model = TorchStyleModel(WEIGHTS)
    py_model = Starcoder2Model()
    py_model.weights = WEIGHTS

    assert torch_model.forward(input_ids) == py_model.forward(input_ids)


def test_generate_equivalence():
    tokenizer = Starcoder2Tokenizer()
    input_ids = tokenizer.encode("foo bar")

    torch_model = TorchStyleModel(WEIGHTS)
    py_model = Starcoder2Model()
    py_model.weights = WEIGHTS

    assert torch_model.generate(input_ids, steps=2) == py_model.generate(input_ids, steps=2)
