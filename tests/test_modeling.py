import json
import shutil

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model

from talkie_mlx.modeling_talkie_mlx import Model, ModelArgs


def tiny_args():
    return ModelArgs(
        vocab_size=128,
        n_layer=1,
        n_head=4,
        n_embd=128,
        head_dim=32,
        max_seq_len=16,
    )


def test_tiny_model_forward_with_cache():
    model = Model(tiny_args())
    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, 3, 128)
    assert cache[0].offset == 3

    next_logits = model(mx.array([[4]], dtype=mx.int32), cache=cache)
    mx.eval(next_logits)
    assert next_logits.shape == (1, 1, 128)
    assert cache[0].offset == 4


def test_mlx_lm_load_model_custom_model_file(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = Model(tiny_args())
    mx.eval(model.parameters())
    mx.save_safetensors(model_dir / "model.safetensors", dict(tree_flatten(model.parameters())))
    config = {
        "model_type": "talkie",
        "model_file": "modeling_talkie_mlx.py",
        "vocab_size": 128,
        "n_layer": 1,
        "n_head": 4,
        "n_embd": 128,
        "head_dim": 32,
        "max_seq_len": 16,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    shutil.copyfile("src/talkie_mlx/modeling_talkie_mlx.py", model_dir / "modeling_talkie_mlx.py")

    loaded, loaded_config = load_model(model_dir)
    logits = loaded(mx.array([[1, 2]], dtype=mx.int32), cache=loaded.make_cache())
    mx.eval(logits)
    assert logits.shape == (1, 2, 128)
    assert loaded_config["model_type"] == "talkie"
