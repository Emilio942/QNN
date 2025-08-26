#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.models import predict_scores
from qnn.io import load_tensor_spec
from qnn.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Predict scores using trained re-upload classifier params")
    parser.add_argument("--config", type=str, help="Optional YAML/JSON config file")
    parser.add_argument("--params", type=str, help="Path to params JSON produced by training")
    parser.add_argument("--spec", type=str, help="Spec YAML path")
    parser.add_argument("--input", type=str, help="Optional JSON file with list of vectors; otherwise random toy vectors")
    parser.add_argument("--n", type=int, default=None, help="Number of random vectors if no input file")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    pcfg = cfg.get("predict", {}) if isinstance(cfg, dict) else {}
    params_path = args.params or pcfg.get("params")
    spec_path = args.spec or cfg.get("spec") or str(ROOT / "specs" / "tensor_spec.yaml")
    n_rand = int(args.n) if args.n is not None else int(pcfg.get("n", 5))
    input_path = args.input or pcfg.get("input")

    if not params_path:
        raise SystemExit("--params missing (and not found in config)")

    spec = load_tensor_spec(spec_path)
    if spec["shape"]["layout"] == "d":
        d = int(spec["shape"]["d"])
    else:
        d = int(spec["shape"]["H"]) * int(spec["shape"]["W"]) * int(spec["shape"]["C"])

    if input_path:
        vectors = json.loads(Path(input_path).read_text())
    else:
        random.seed(0)
        vectors = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n_rand)]

    scores = predict_scores(vectors, params_path, spec_path)
    out = [{"i": i, "score": float(s)} for i, s in enumerate(scores)]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
