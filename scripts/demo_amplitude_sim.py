#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector
from qnn.circuits import amplitude_prep_circuit
from qnn.observables import z_chain_expectation_statevector

from qiskit.quantum_info import Statevector


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    # switch to amplitude encoding and enable pad to power-of-two if needed
    spec["encoding_goal"]["target"] = "amplitude"
    spec["preprocessing"]["pad"]["to_power_of_two"]["enabled"] = True

    # create a length-16 vector and normalize; preprocessing will also normalize for amplitude path
    x = [math.sin(i) for i in range(16)]
    vec, info = preprocess_vector(x, spec)

    # Ensure power of two length
    amps = vec
    n = int(math.log2(len(amps)))
    assert 2 ** n == len(amps)
    # Statevector from the prep circuit
    qc = amplitude_prep_circuit(amps)
    sv = Statevector.from_instruction(qc)

    # Validate amplitudes up to sign precision
    target = amps
    got = sv.data.real.tolist()  # amplitudes are real in this demo
    # Compare L2-normed dot product ~ 1
    dot = sum(a * b for a, b in zip(target, got))
    print({"n": n, "len": len(amps), "overlap": round(float(dot), 6)})


if __name__ == "__main__":
    main()
