#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import math
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector
from qnn.planning import plan_from_spec
from qnn.vqa import build_reupload_variational_circuit, predict_statevector, spsa_gradient, spsa_step
from qnn.observables import z_expectation_statevector


random.seed(0)


def make_toy_dataset(n: int, d: int):
    # Binary labels based on sign of sum; features in [-1,1]
    X = []
    y = []
    for i in range(n):
        vec = [random.uniform(-1, 1) for _ in range(d)]
        X.append(vec)
        y.append(1 if sum(vec) >= 0 else -1)
    return X, y


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    spec["encoding_goal"]["target"] = "angle"
    d = spec["shape"]["d"] if spec["shape"]["layout"] == "d" else spec["shape"]["H"]*spec["shape"]["W"]*spec["shape"]["C"]

    X, y = make_toy_dataset(64, int(d))

    plan = plan_from_spec(spec)
    q = min(4, plan["d"])  # cap to 4 qubits
    L = 2

    # Initialize params
    params = [0.0] * (q * L)

    def predict(vec):
        v, info = preprocess_vector(vec, spec)
        angles = info["angles"]
        qc = build_reupload_variational_circuit(q, L, angles, params)
        probs = predict_statevector(qc)
        # Use <Z_0> as score; sign -> class
        score = z_expectation_statevector(probs, 0)
        return score

    def loss_fn(theta):
        nonlocal params
        params = theta
        # hinge-like loss: max(0, 1 - y*score)
        loss = 0.0
        for xi, yi in zip(X, y):
            s = predict(xi)
            loss += max(0.0, 1.0 - yi * s)
        return loss / len(X)

    grad = spsa_gradient(loss_fn, c=0.1)

    theta = params[:]
    for k in range(10):
        theta = spsa_step(theta, grad, a=0.2, c=0.1)
        val = loss_fn(theta)
        print({"step": k + 1, "loss": round(float(val), 4)})


if __name__ == "__main__":
    main()
