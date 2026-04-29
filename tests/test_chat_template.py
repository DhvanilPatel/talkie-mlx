from talkie_mlx.chat_template import Message, format_chat, format_prompt, truncate_at_stop


def test_format_prompt():
    assert format_prompt("Hello") == "<|user|>Hello<|end|><|assistant|>"


def test_format_chat_multiple_roles():
    text = format_chat(
        [
            Message(role="system", content="Be brief."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Sir, hello."),
            Message(role="user", content="Again"),
        ]
    )
    assert text == (
        "<|system|>Be brief.<|end|>"
        "<|user|>Hello<|end|>"
        "<|assistant|>Sir, hello.<|end|>"
        "<|user|>Again<|end|>"
        "<|assistant|>"
    )


def test_truncate_at_stop():
    assert truncate_at_stop("hello<|end|>ignored") == ("hello", True)
    assert truncate_at_stop("hello") == ("hello", False)
