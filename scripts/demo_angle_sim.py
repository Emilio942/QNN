#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector
from qnn.planning import plan_from_spec

from qiskit.quantum_info import Statevector

from qnn.circuits import angle_encode_circuit


essential = ("specs/tensor_spec.yaml",)


def z_chain_expectation(statevector, q: int) -> float:
    # For |ψ>, <Z⊗...⊗Z> = sum_{bitstring} p(s) * (-1)^{# of 1s}
    probs = statevector.probabilities()
    exp = 0.0
    for i, p in enumerate(probs):
        parity = bin(i).count("1") % 2
        exp += p * (-1 if parity == 1 else 1)
    return exp


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    # build input vector; use same as demo_preprocess
    x = [((i % 8) - 4) / 4.0 for i in range(16)]
    vec, info = preprocess_vector(x, spec)

    plan = plan_from_spec(spec)
    q, L = plan["q"], plan["L"]
    angles = info["angles"]

    qc = angle_encode_circuit(angles, q=q, L=L, entangler=plan["entangler"])
    sv = Statevector.from_instruction(qc)

    ez = z_chain_expectation(sv, q)
    print({"q": q, "L": L, "<Z..Z>": round(ez, 6)})


if __name__ == "__main__":
    main()
