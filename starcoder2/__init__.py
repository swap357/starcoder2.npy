"""Simple pure Python implementation of Starcoder2.

This package mirrors the high level API of ``transformers`` so that users
can experiment with a tiny model implementation entirely in Python.
"""

from .model import Starcoder2Model
from .tokenizer import Starcoder2Tokenizer

__all__ = ["Starcoder2Model", "Starcoder2Tokenizer"]
