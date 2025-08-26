#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector


def main():
    spec = load_tensor_spec(Path("specs/tensor_spec.yaml"))
    # demo vector length 16 in [-1,1]
    x = [((i % 8) - 4) / 4.0 for i in range(16)]
    vec, info = preprocess_vector(x, spec)
    out = {
        "len": len(vec),
        "min": min(vec),
        "max": max(vec),
        "angles_sample": (info.get("angles", [])[:4] if "angles" in info else None),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
