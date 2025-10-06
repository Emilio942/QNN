#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json
import random
import math
from dataclasses import dataclass
from typing import List, Tuple
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector
from qnn.planning import plan_from_spec
from qnn.vqa import build_reupload_variational_circuit, spsa_gradient, spsa_step
from qnn.observables import z_expectation_statevector
from qnn.noise import make_basic_noise_model
from qnn.models import save_params
from qnn.config import load_config
from qnn.data import load_vectors_labels

from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def make_toy_dataset(n: int, d: int) -> Tuple[List[List[float]], List[int]]:
    X = []
    y = []
    for _ in range(n):
        vec = [random.uniform(-1, 1) for _ in range(d)]
        X.append(vec)
        y.append(1 if sum(vec) >= 0 else -1)
    return X, y


def split_data(X, y, val_ratio=0.25):
    idx = list(range(len(X)))
    random.shuffle(idx)
    cut = int(len(X) * (1 - val_ratio))
    train_idx, val_idx = idx[:cut], idx[cut:]
    Xtr = [X[i] for i in train_idx]
    ytr = [y[i] for i in train_idx]
    Xva = [X[i] for i in val_idx]
    yva = [y[i] for i in val_idx]
    return Xtr, ytr, Xva, yva


def batch_iter(X, y, bs: int):
    n = len(X)
    for i in range(0, n, bs):
        yield X[i:i+bs], y[i:i+bs]


def score_qc(qc, use_noise=False, shots=2048, noise_cfg=None):
    if not use_noise:
        sv = Statevector.from_instruction(qc)
        return sv.probabilities()
    nm = make_basic_noise_model(**(noise_cfg or {"p1": 0.005, "p2": 0.02, "readout_p01": 0.02, "readout_p10": 0.02}))
    sim = AerSimulator(noise_model=nm)
    qc_m = qc.copy()
    qc_m.measure_all()
    res = sim.run(qc_m, shots=shots).result()
    counts = res.get_counts()
    # Convert to probs
    total = sum(counts.values())
    max_qubits = qc.num_qubits
    size = 2 ** max_qubits
    probs = [0.0] * size
    for bstr, c in counts.items():
        i = int(bstr[::-1], 2)
        probs[i] += c / total
    return probs


def main():
    parser = argparse.ArgumentParser(description="Train data re-upload classifier with SPSA")
    parser.add_argument("--config", type=str, help="Optional YAML/JSON config file")
    parser.add_argument("--spec", type=str, help="Spec YAML path")
    parser.add_argument("--steps", type=int, default=None, help="SPSA steps")
    parser.add_argument("--batch", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--q", type=int, default=None, help="Number of qubits (cap)")
    parser.add_argument("--L", type=int, default=None, help="Re-upload layers")
    parser.add_argument("--noise", type=str, choices=["true", "false"], default=None, help="Train with noise model")
    parser.add_argument("--out", type=str, default=None, help="Path to save trained params JSON")
    parser.add_argument("--data", type=str, default=None, help="Optional training dataset file (CSV/JSON)")
    parser.add_argument("--val-data", type=str, default=None, help="Optional validation dataset file (CSV/JSON)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    # Resolve spec path
    default_spec = str(ROOT / "specs" / "tensor_spec.yaml")
    spec_path = args.spec or cfg.get("spec") or default_spec
    spec = load_tensor_spec(spec_path)
    spec["encoding_goal"]["target"] = "angle"
    d = spec["shape"]["d"] if spec["shape"]["layout"] == "d" else spec["shape"]["H"]*spec["shape"]["W"]*spec["shape"]["C"]

    if args.seed is not None:
        random.seed(int(args.seed))
    else:
        random.seed(0)

    # Resolve training hyperparams (config -> defaults -> CLI overrides)
    tcfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    steps = args.steps if args.steps is not None else int(tcfg.get("steps", 30))
    batch = args.batch if args.batch is not None else int(tcfg.get("batch", 32))
    q_cap = args.q if args.q is not None else int(tcfg.get("q", 4))
    L = args.L if args.L is not None else int(tcfg.get("L", 2))
    if args.noise is not None:
        noise_flag = True if args.noise.lower() == "true" else False
    else:
        noise_flag = bool(tcfg.get("noise", False))
    out_path = args.out or str(tcfg.get("out", str(ROOT / "reports" / "reupload_params.json")))
    data_path = args.data or tcfg.get("data")
    val_data_path = args.val_data or tcfg.get("val_data")

    # Load data (file or toy fallback) and split
    if data_path:
        X, y = load_vectors_labels(data_path)
        if val_data_path:
            Xva, yva = load_vectors_labels(val_data_path)
            Xtr, ytr = X, y
        else:
            Xtr, ytr, Xva, yva = split_data(X, y, val_ratio=0.25)
    else:
        X, y = make_toy_dataset(256, int(d))
        Xtr, ytr, Xva, yva = split_data(X, y, val_ratio=0.25)

    plan = plan_from_spec(spec)
    q = min(q_cap, plan["d"])  # cap

    # params
    theta = [0.0] * (q * L)

    def predict_vec(vec, local_theta, noise=False):
        v, info = preprocess_vector(vec, spec)
        angles = info["angles"]
        qc = build_reupload_variational_circuit(q, L, angles, local_theta)
        probs = score_qc(qc, use_noise=noise)
        return z_expectation_statevector(probs, 0)

    def loss_on_batch(batchX, batchY, local_theta, noise=False):
        # hinge loss
        loss = 0.0
        for xi, yi in zip(batchX, batchY):
            s = predict_vec(xi, local_theta, noise=noise)
            loss += max(0.0, 1.0 - yi * s)
        return loss / max(1, len(batchX))

    def loss_fn(params):
        # evaluate on a random mini-batch
        bs = int(batch)
        # Single random batch per call (SPSA stochastic estimate)
        start = random.randrange(0, max(1, len(Xtr) - bs + 1))
        bx = Xtr[start:start+bs]
        by = ytr[start:start+bs]
        return loss_on_batch(bx, by, params, noise=noise_flag)

    grad = spsa_gradient(loss_fn, c=0.1)

    log = []
    # per-step CSV
    outdir = ROOT / "reports"
    outdir.mkdir(exist_ok=True)
    csv_path = outdir / "train_steps.csv"
    try:
        import csv as _csv
        with csv_path.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["step", "train_loss", "val_loss", "train_acc", "val_acc"])
    except Exception:
        csv_path = None
    for k in range(1, int(steps) + 1):
        theta = spsa_step(theta, grad, a=0.15, c=0.1)
        if k % 5 == 0 or k == 1:
            # full-batch metrics
            train_loss = loss_on_batch(Xtr, ytr, theta, noise=noise_flag)
            val_loss = loss_on_batch(Xva, yva, theta, noise=noise_flag)
            # accuracy
            def acc(Xs, Ys):
                correct = 0
                for xi, yi in zip(Xs, Ys):
                    s = predict_vec(xi, theta, noise=False)
                    pred = 1 if s >= 0 else -1
                    correct += 1 if pred == yi else 0
                return correct / len(Xs)
            train_acc = acc(Xtr, ytr)
            val_acc = acc(Xva, yva)
            row = {"step": k, "train_loss": round(float(train_loss), 4), "val_loss": round(float(val_loss), 4), "train_acc": round(float(train_acc), 4), "val_acc": round(float(val_acc), 4)}
            print(row)
            log.append(row)
            if csv_path is not None:
                with csv_path.open("a", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow([row["step"], row["train_loss"], row["val_loss"], row["train_acc"], row["val_acc"]])

    # save log
    (outdir / "train_reupload_classifier.json").write_text(json.dumps(log, indent=2))
    # Save parameters
    save_params(out_path, theta, q=q, L=L)
    # Save metadata
    meta = {
        "spec": spec_path,
        "out_params": out_path,
        "steps": int(steps),
        "batch": int(batch),
        "q": int(q),
        "L": int(L),
        "noise": bool(noise_flag),
        "seed": int(args.seed) if args.seed is not None else 0,
        "sizes": {"train": len(Xtr), "val": len(Xva)},
        "data": {"train": data_path, "val": val_data_path},
    }
    (outdir / "train_reupload_classifier_meta.json").write_text(json.dumps(meta, indent=2))
    print({"saved_log": str(outdir / "train_reupload_classifier.json"), "saved_meta": str(outdir / "train_reupload_classifier_meta.json"), "saved_params": out_path})


if __name__ == "__main__":
    main()
