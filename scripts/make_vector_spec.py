#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse


def infer_dim_from_dataset(path: Path) -> int:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if "X" in data and isinstance(data["X"], list) and data["X"]:
            return len(data["X"][0])
        raise SystemExit("JSON dataset must contain key 'X' with non-empty list of vectors")
    elif path.suffix.lower() == ".csv":
        import csv
        with path.open(newline="") as f:
            r = csv.reader(f)
            row = next(r)
            # If labels present, assume last col is label (int)
            try:
                int(row[-1])
                return len(row) - 1
            except Exception:
                return len(row)
    else:
        raise SystemExit("Unsupported dataset extension; use .json or .csv")


def main():
    ap = argparse.ArgumentParser(description="Create a vector tensor_spec with dimension inferred from a dataset")
    ap.add_argument("--data", required=True, help="Path to dataset (.json with X/y or .csv)")
    ap.add_argument("--out", required=True, help="Path to write spec YAML")
    args = ap.parse_args()

    d = infer_dim_from_dataset(Path(args.data))
    spec = {
        "tensor_spec": {
            "id": "qnn.tensor.v1",
            "purpose": "input features for angle-encoding",
            "dtype": "float32",
            "value_range": {"min": -1.0, "max": 1.0, "units": "normalized", "allow_nan_inf": False},
            "shape": {"layout": "d", "H": None, "W": None, "C": None, "d": int(d), "batch_dimension": "separate"},
            "indexing": {"order": "row-major", "mapping_note": "i = i  (vector layout)"},
            "complex": {"representation": "real"},
            "preprocessing": {"normalize": {"type": "none", "params": {}}, "clip": {"enabled": False, "min": None, "max": None}, "quantize": {"enabled": False, "bits": None, "mode": None}, "pad": {"to_power_of_two": {"enabled": False, "axis": "flattened", "pad_value": 0.0}}},
            "encoding_goal": {"target": "angle", "angles": {"alpha": 3.141592653589793}, "amplitude": {"require_l2_norm": True}, "phase": {"beta": None}},
            "observables_plan": [{"name": "Z_chain", "desc": "Z\u2297…\u2297Z on data qubits"}],
            "versioning": {"date": "auto", "author": "auto", "dataset_version": "auto"},
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    out.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(json.dumps({"saved": str(out), "d": d}, indent=2))


if __name__ == "__main__":
    main()
