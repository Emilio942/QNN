#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse
import csv
import random
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.models import predict_scores
from qnn.metrics import accuracy, confusion_counts, roc_auc
from qnn.config import load_config


def load_vectors_labels(path: str | Path) -> Tuple[List[List[float]], List[int]]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        # Expect {"X": [[...], ...], "y": [1,-1,...]}
        return data["X"], data["y"]
    # CSV: rows of values with last column as label
    X: List[List[float]] = []
    y: List[int] = []
    with p.open() as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            *feat, lab = row
            X.append([float(v) for v in feat])
            y.append(int(lab))
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Evaluate re-upload classifier params on a dataset")
    parser.add_argument("--config", type=str, help="Optional YAML/JSON config file")
    parser.add_argument("--params", type=str, help="Path to params JSON")
    parser.add_argument("--spec", type=str, help="Spec YAML path")
    parser.add_argument("--data", type=str, help="Dataset file (.json with X/y or .csv with last col label)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for toy fallback")
    parser.add_argument("--export", type=str, help="Optional path to save predictions (.json/.csv)")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    ecfg = cfg.get("eval", {}) if isinstance(cfg, dict) else {}
    params_path = args.params or ecfg.get("params") or str(ROOT / "reports" / "reupload_params.json")
    spec_path = args.spec or cfg.get("spec") or str(ROOT / "specs" / "tensor_spec.yaml")
    data_path = args.data or ecfg.get("data")
    export_path = args.export or ecfg.get("export")
    seed = args.seed if args.seed is not None else ecfg.get("seed")

    spec = load_tensor_spec(spec_path)
    if spec["shape"]["layout"] == "d":
        d = int(spec["shape"]["d"])
    else:
        d = int(spec["shape"]["H"]) * int(spec["shape"]["W"]) * int(spec["shape"]["C"])

    if data_path:
        X, y = load_vectors_labels(data_path)
    else:
        # Toy fallback
        random.seed(int(seed) if seed is not None else 0)
        n = 64
        X = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
        y = [1 if sum(v) >= 0 else -1 for v in X]

    scores = predict_scores(X, params_path, spec_path)
    preds = [1 if s >= 0 else -1 for s in scores]
    acc = accuracy(y, preds)
    tp, fp, tn, fn = confusion_counts(y, preds)
    auc = roc_auc(y, scores)

    report = {
        "n": len(y),
        "accuracy": round(float(acc), 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "auc": None if auc is None else round(float(auc), 4),
    }
    print(json.dumps(report, indent=2))

    if export_path:
        p = Path(export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"i": i, "score": float(s), "pred": int(preds[i]), "y": int(y[i])} for i, s in enumerate(scores)]
        if p.suffix.lower() == ".csv":
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["i", "score", "pred", "y"]) 
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        else:
            p.write_text(json.dumps(rows, indent=2))
        print(json.dumps({"saved": str(p), "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
