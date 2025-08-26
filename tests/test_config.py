from __future__ import annotations

from pathlib import Path
import json
import yaml

from qnn.config import load_config, merge_dicts


def test_load_config_yaml_and_merge(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"spec": "specs/tensor_spec.yaml", "train": {"steps": 10, "batch": 8}}))
    cfg = load_config(cfg_path)
    assert cfg["spec"].endswith("tensor_spec.yaml")
    merged = merge_dicts(cfg, {"train": {"batch": 16, "q": 2}})
    assert merged["train"]["steps"] == 10
    assert merged["train"]["batch"] == 16
    assert merged["train"]["q"] == 2

def test_load_config_json(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"a": 1}))
    cfg = load_config(p)
    assert cfg["a"] == 1
