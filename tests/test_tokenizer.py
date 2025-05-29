from starcoder2 import Starcoder2Tokenizer

def test_encode_decode_roundtrip():
    tok = Starcoder2Tokenizer()
    text = "hello world"
    ids = tok.encode(text)
    assert tok.decode(ids) == text

