import base64

from talkie_mlx.tokenization_talkie import TalkieTokenizer


def write_byte_vocab(path):
    with path.open("w", encoding="utf-8") as handle:
        for value in range(256):
            token = base64.b64encode(bytes([value])).decode("ascii")
            handle.write(f"{token} {value}\n")


def test_talkie_tokenizer_roundtrip(tmp_path):
    vocab = tmp_path / "vocab.txt"
    write_byte_vocab(vocab)
    tokenizer = TalkieTokenizer(str(vocab), style="base")

    ids = tokenizer.encode("Hello, world!", add_special_tokens=False)
    assert ids
    assert tokenizer.decode(ids) == "Hello, world!"
    assert tokenizer.eos_token_id == 65_535


def test_talkie_it_special_tokens(tmp_path):
    vocab = tmp_path / "vocab.txt"
    write_byte_vocab(vocab)
    tokenizer = TalkieTokenizer(str(vocab), style="it")

    ids = tokenizer.encode("<|user|>Hello<|end|><|assistant|>", add_special_tokens=False)
    assert 65_537 in ids
    assert 65_536 in ids
    assert 65_538 in ids
