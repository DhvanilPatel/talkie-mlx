import json

import mlx.core as mx
from mlx_lm import load


def test_convert_tiny_checkpoint_unquantized(tiny_export):
    output = tiny_export()
    config = json.loads((output / "config.json").read_text())
    assert config["model_file"] == "modeling_talkie_mlx.py"
    assert config["vocab_size"] == 128
    assert (output / "model.safetensors").exists()
    assert (output / "tokenization_talkie.py").exists()
    assert (output / "tokenizer_config.json").exists()


def test_mlx_lm_loads_converted_directory(tiny_export):
    output = tiny_export()
    model, tokenizer = load(output, tokenizer_config={"trust_remote_code": True})
    ids = tokenizer.encode("Hello", add_special_tokens=False)
    assert ids

    logits = model(mx.array([ids], dtype=mx.int32), cache=model.make_cache())
    mx.eval(logits)
    assert logits.shape == (1, len(ids), 128)


def test_mlx_lm_loads_quantized_converted_directory(tiny_export):
    output = tiny_export(q_bits=4, q_group_size=64)
    config = json.loads((output / "config.json").read_text())
    assert config["quantization"]["bits"] == 4

    model, tokenizer = load(output, tokenizer_config={"trust_remote_code": True})
    ids = tokenizer.encode("Hello", add_special_tokens=False)
    logits = model(mx.array([ids], dtype=mx.int32), cache=model.make_cache())
    mx.eval(logits)
    assert logits.shape == (1, len(ids), 128)
