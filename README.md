# starcoder2.npy

This repository provides a minimal pure Python implementation of the
Starcoder2 model. The goal is to showcase how one might interact with
pretrained weights stored in a `npy` file without relying on external
libraries like `transformers` or even `numpy`.

If `numpy` is installed the implementation will transparently use it for
array operations. Otherwise it falls back to slower pure Python code. The
package declares `numpy` as an optional dependency in `pyproject.toml`.

The interface is intentionally similar to the Hugging Face usage:

```python
from starcoder2 import Starcoder2Tokenizer, Starcoder2Model

# load tokenizer and model
tokenizer = Starcoder2Tokenizer()
model = Starcoder2Model(weights_path="starcoder2.npy")

# encode some text
ids = tokenizer.encode("hello world")

# generate one additional token
output_ids = model.generate(ids, steps=1)
print(tokenizer.decode(output_ids))
```

`Starcoder2Tokenizer` implements a tiny byte pair encoding (BPE) tokenizer.
It loads its vocabulary from `starcoder2/resources/vocab.json` along with the
merge rules in `merges.txt`. Custom paths for these files can be provided via
the tokenizer constructor if desired.

The weights file `starcoder2.npy` is expected to be a serialized Python
dictionary containing arrays or lists. If `numpy` is available it will be
used, otherwise a pure Python pickle loader is employed. This example
shows the mechanics of loading such a file and performing a minimal
forward pass.
