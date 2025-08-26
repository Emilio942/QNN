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
from qnn.circuits import angle_encode_circuit
from qnn.transpile import summarize_circuit, transpile_for_backend

from qiskit_aer import AerSimulator


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    x = [((i % 8) - 4) / 4.0 for i in range(16)]
    vec, info = preprocess_vector(x, spec)
    plan = plan_from_spec(spec)
    q, L = plan["q"], plan["L"]

    qc = angle_encode_circuit(info["angles"], q=q, L=L)
    base = summarize_circuit(qc)

    # Configure an AerSimulator with a simple line coupling and common basis gates
    coupling_map = [[0, 1], [1, 2], [2, 3]]
    basis_gates = ["rz", "sx", "x", "cx", "id"]
    backend = AerSimulator(method="automatic")
    backend.set_options(coupling_map=coupling_map, basis_gates=basis_gates)

    tqc = transpile_for_backend(qc, backend, optimization_level=1)
    after = summarize_circuit(tqc)

    print({"base": base, "transpiled": after, "backend": "AerSimulator(line)"})


if __name__ == "__main__":
    main()
