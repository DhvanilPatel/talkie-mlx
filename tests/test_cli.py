import json
import sys

from talkie_mlx.cli import batch_main


def test_batch_cli_writes_jsonl(tmp_path, tiny_export, monkeypatch):
    model_dir = tiny_export()
    input_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    input_path.write_text(
        json.dumps({"prompt": "Hello", "mode": "completion"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "talkie-mlx-batch",
            "--model",
            str(model_dir),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--temp",
            "0",
            "--max-tokens",
            "2",
        ],
    )

    assert batch_main() == 0
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["prompt"] == "Hello"
    assert "output" in rows[0]
    assert rows[0]["tokens"] == 2
