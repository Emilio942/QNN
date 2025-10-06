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
from qnn.metrics import accuracy, confusion_counts, roc_auc, precision_recall_f1, pr_curve
from qnn.config import load_config
from qnn.data import load_vectors_labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate re-upload classifier params on a dataset")
    parser.add_argument("--config", type=str, help="Optional YAML/JSON config file")
    parser.add_argument("--params", type=str, help="Path to params JSON")
    parser.add_argument("--spec", type=str, help="Spec YAML path")
    parser.add_argument("--data", type=str, help="Dataset file (.json with X/y or .csv with last col label)")
    parser.add_argument("--val-data", type=str, help="Optional validation dataset (currently unused)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for toy fallback")
    parser.add_argument("--export", type=str, help="Optional path to save predictions (.json/.csv)")
    parser.add_argument("--plot", type=str, help="Optional path to save PR curve (.png)")
    parser.add_argument("--roc-plot", type=str, help="Optional path to save ROC curve (.png)")
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
    p, r, f1 = precision_recall_f1(y, preds)

    report = {
        "n": len(y),
        "accuracy": round(float(acc), 4),
    "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "auc": None if auc is None else round(float(auc), 4),
    "precision": round(float(p), 4),
    "recall": round(float(r), 4),
    "f1": round(float(f1), 4),
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

    if args.plot:
        import matplotlib.pyplot as plt
        pr_p, pr_r, thr = pr_curve(y, scores, thresholds=50)
        plt.figure(figsize=(4, 3))
        plt.plot(pr_r, pr_p, label="PR curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True, alpha=0.3)
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.plot)
        print(json.dumps({"saved_plot": args.plot}, indent=2))

    if args.roc_plot:
        # Build ROC by sweeping thresholds
        import matplotlib.pyplot as plt
        # thresholds similar to PR; compute FPR/TPR
        ts = [min(scores) + (max(scores) - min(scores)) * i / max(1, 50 - 1) for i in range(50)]
        fpr, tpr = [], []
        for t in ts:
            y_pred = [1 if s >= t else -1 for s in scores]
            tp, fp, tn, fn = confusion_counts(y, y_pred)
            tpr.append(tp / max(1, tp + fn))
            fpr.append(fp / max(1, fp + tn))
        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc if auc is not None else 'NA'})")
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.grid(True, alpha=0.3)
        Path(args.roc_plot).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.roc_plot)
        print(json.dumps({"saved_plot": args.roc_plot}, indent=2))


if __name__ == "__main__":
    main()
