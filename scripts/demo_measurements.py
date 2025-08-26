#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector
from qnn.planning import plan_from_spec
from qnn.circuits import angle_encode_circuit
from qnn.observables import (
    z_chain_expectation_statevector,
    sample_from_probs,
    z_chain_expectation_from_samples,
)

from qiskit.quantum_info import Statevector


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    x = [((i % 8) - 4) / 4.0 for i in range(16)]
    vec, info = preprocess_vector(x, spec)
    plan = plan_from_spec(spec)

    q, L = plan["q"], plan["L"]
    angles = info["angles"]
    qc = angle_encode_circuit(angles, q=q, L=L, entangler=plan["entangler"])

    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()

    ideal = z_chain_expectation_statevector(np.asarray(probs))

    for shots in (128, 512, 2048):
        samples = sample_from_probs(np.asarray(probs), shots=shots)
        est = z_chain_expectation_from_samples(samples)
        print({"shots": shots, "ideal": round(float(ideal), 6), "estimate": round(float(est), 6)})


if __name__ == "__main__":
    main()
