from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(p.read_text()) or {}
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    # Try YAML by default
    return yaml.safe_load(p.read_text()) or {}


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out
