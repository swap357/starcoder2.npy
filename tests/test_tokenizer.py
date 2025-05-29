from starcoder2 import Starcoder2Tokenizer


def test_encode_decode_roundtrip():
    tok = Starcoder2Tokenizer()
    text = "hello world"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_encode_decode_punctuation():
    tok = Starcoder2Tokenizer()
    text = "hello, world!"
    ids = tok.encode(text)
    assert ids == [0, 1]
    assert tok.decode(ids) == text


def test_encode_decode_special_tokens():
    tok = Starcoder2Tokenizer()
    text = "<CLS> hello <PAD>"
    ids = tok.encode(text)
    assert ids == [0, 1, 2]
    assert tok.decode(ids) == text


def test_token_ids_consistency():
    tok = Starcoder2Tokenizer()
    first = tok.encode("hello world")
    assert first == [0, 1]
    # encoding again should reuse the same ids
    second = tok.encode("world hello")
    assert second == [1, 0]
    assert tok.vocab == {"hello": 0, "world": 1}
