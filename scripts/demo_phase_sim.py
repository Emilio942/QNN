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
from qnn.circuits import phase_encode_circuit
from qnn.observables import z_chain_expectation_statevector

from qiskit.quantum_info import Statevector


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    spec["encoding_goal"]["target"] = "phase"
    spec["encoding_goal"]["phase"]["beta"] = 1.0

    x = [((i % 8) - 4) / 4.0 for i in range(16)]
    vec, info = preprocess_vector(x, spec)

    plan = plan_from_spec(spec)
    q, L = plan["q"], plan["L"]
    phases = info["phases"]

    qc = phase_encode_circuit(phases, q=q, L=L, entangler=plan["entangler"])
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    ez = z_chain_expectation_statevector(probs)
    print({"q": q, "L": L, "<Z..Z>": round(float(ez), 6)})


if __name__ == "__main__":
    main()
