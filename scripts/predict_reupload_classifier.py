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


def main():
    parser = argparse.ArgumentParser(description="Predict scores using trained re-upload classifier params")
    parser.add_argument("--params", type=str, required=True, help="Path to params JSON produced by training")
    parser.add_argument("--spec", type=str, default=str(ROOT / "specs" / "tensor_spec.yaml"), help="Spec YAML path")
    parser.add_argument("--input", type=str, help="Optional JSON file with list of vectors; otherwise random toy vectors")
    parser.add_argument("--n", type=int, default=5, help="Number of random vectors if no input file")
    args = parser.parse_args()

    spec = load_tensor_spec(args.spec)
    if spec["shape"]["layout"] == "d":
        d = int(spec["shape"]["d"])
    else:
        d = int(spec["shape"]["H"]) * int(spec["shape"]["W"]) * int(spec["shape"]["C"])

    if args.input:
        vectors = json.loads(Path(args.input).read_text())
    else:
        random.seed(0)
        vectors = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(args.n)]

    scores = predict_scores(vectors, args.params, args.spec)
    out = [{"i": i, "score": float(s)} for i, s in enumerate(scores)]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
