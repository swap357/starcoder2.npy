try:
    import numpy as np
except Exception:  # noqa: PIE786
    np = None

from starcoder2 import Starcoder2Model, Starcoder2Config


def test_forward_shape():
    config = Starcoder2Config(vocab_size=10, hidden_size=4, num_attention_heads=1, num_layers=1)
    model = Starcoder2Model(config=config)

    # force deterministic weights
    h = config.hidden_size
    v = config.vocab_size
    if np is not None:
        model.weights['embedding'] = np.arange(v*h, dtype=np.float32).reshape(v, h)
        model.weights['linear'] = np.ones((h, v), dtype=np.float32)
        model.weights['wq_0'] = np.ones((h, h), dtype=np.float32)
        model.weights['wk_0'] = np.ones((h, h), dtype=np.float32)
        model.weights['wv_0'] = np.ones((h, h), dtype=np.float32)
        model.weights['wo_0'] = np.ones((h, h), dtype=np.float32)
        model.weights['ff1_0'] = np.ones((h, 4*h), dtype=np.float32)
        model.weights['ff2_0'] = np.ones((4*h, h), dtype=np.float32)
    else:
        model.weights['embedding'] = [[float(i*h + j) for j in range(h)] for i in range(v)]
        model.weights['linear'] = [[1.0 for _ in range(v)] for _ in range(h)]
        ones_hh = [[1.0 for _ in range(h)] for _ in range(h)]
        model.weights['wq_0'] = ones_hh
        model.weights['wk_0'] = ones_hh
        model.weights['wv_0'] = ones_hh
        model.weights['wo_0'] = ones_hh
        model.weights['ff1_0'] = [[1.0 for _ in range(4*h)] for _ in range(h)]
        model.weights['ff2_0'] = [[1.0 for _ in range(h)] for _ in range(4*h)]

    logits = model.forward([0, 1])
    assert len(logits) == 2
    assert len(logits[0]) == v

