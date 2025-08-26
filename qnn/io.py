from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_tensor_spec(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    return data["tensor_spec"]


__all__ = ["load_tensor_spec"]
