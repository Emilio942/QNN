#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import math
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.vqa import build_vqa_circuit, spsa_gradient, spsa_step
from qnn.observables import z_expectation_statevector
from qiskit.quantum_info import Statevector


def main():
    random.seed(0)
    q = 4
    L = 2
    theta = [0.1] * (q * L)

    def loss_fn(params):
        qc = build_vqa_circuit(q, L, params)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        # Minimize -<Z_0> to maximize <Z_0>
        return -float(z_expectation_statevector(probs, qubit=0))

    grad = spsa_gradient(loss_fn, c=0.1)

    for k in range(10):
        theta = spsa_step(theta, grad, a=0.2, c=0.1)
        val = loss_fn(theta)
        print({"step": k + 1, "loss": round(val, 6)})


if __name__ == "__main__":
    main()
