#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None


def main(spec_path: str):
    if yaml is None:
        print("PyYAML not installed; cannot parse spec.")
        return 2
    p = Path(spec_path)
    if not p.exists():
        print(f"Spec not found: {p}")
        return 2
    spec = yaml.safe_load(p.read_text())
    ts = spec["tensor_spec"]

    # Minimal acceptance checks
    problems = []
    required = ["id", "purpose", "dtype", "value_range", "shape", "encoding_goal"]
    for k in required:
        if k not in ts:
            problems.append(f"missing: {k}")

    # dtype sanity
    if ts.get("dtype") not in {"float32", "float64", "int8", "uint8", "complex64", "complex128"}:
        problems.append(f"unsupported dtype: {ts.get('dtype')}")

    # shape/layout sanity
    layout = ts.get("shape", {}).get("layout")
    if layout not in {"d", "HxWxC"}:
        problems.append(f"unsupported layout: {layout}")

    # encoding target
    target = ts.get("encoding_goal", {}).get("target")
    if target not in {"angle", "amplitude", "phase", "basis"}:
        problems.append(f"unsupported encoding target: {target}")

    # value range
    vr = ts.get("value_range", {})
    if any(v == "TODO" for v in [vr.get("min"), vr.get("max")]):
        problems.append("value_range.min/max is TODO")

    if problems:
        print("Spec FAIL")
        for pr in problems:
            print(" -", pr)
        return 1
    print("Spec PASS")
    return 0


if __name__ == "__main__":
    sp = sys.argv[1] if len(sys.argv) > 1 else "specs/tensor_spec.yaml"
    sys.exit(main(sp))
