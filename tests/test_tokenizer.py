import json
from starcoder2 import Starcoder2Tokenizer

def test_encode_decode_roundtrip():
    tok = Starcoder2Tokenizer()
    text = "hello world"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_bpe_from_files(tmp_path):
    vocab = {
        "hello": 0,
        "Ġworld": 1,
        "h": 2,
        "e": 3,
        "l": 4,
        "o": 5,
        "Ġ": 6,
        "w": 7,
        "r": 8,
        "d": 9,
    }
    merges = [
        "h e",
        "he l",
        "hel l",
        "hell o",
        "Ġ w",
        "Ġw o",
        "Ġwo r",
        "Ġwor l",
        "Ġworl d",
    ]
    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.txt"
    vocab_file.write_text(json.dumps(vocab))
    merges_file.write_text("\n".join(merges))

    tok = Starcoder2Tokenizer.from_pretrained(tmp_path)
    ids = tok.encode("hello world")
    assert ids == [0, 1]
    assert tok.decode(ids) == "hello world"

