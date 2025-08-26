#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector
from qnn.planning import plan_from_spec
from qnn.circuits import angle_encode_circuit
from qnn.observables import z_chain_expectation_statevector
from qnn.transpile import summarize_circuit, transpile_for_backend

from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    # Ensure angle target
    spec["encoding_goal"]["target"] = "angle"

    # Create a deterministic vector in [-1,1]
    d = spec["shape"]["d"] if spec["shape"]["layout"] == "d" else spec["shape"]["H"]*spec["shape"]["W"]*spec["shape"]["C"]
    x = [((i % 8) - 4) / 4.0 for i in range(int(d))]

    alphas = [0.25*math.pi, 0.5*math.pi, 1.0*math.pi]
    Ls = [1, 2, 4]

    # Simple line-coupled Aer backend for depth estimates
    coupling_map = [[i, i+1] for i in range(3)]  # supports up to 4 qubits
    basis_gates = ["rz", "sx", "x", "cx", "id"]
    backend = AerSimulator(method="automatic")
    backend.set_options(coupling_map=coupling_map, basis_gates=basis_gates)

    results = []
    for alpha in alphas:
        # set alpha in spec
        spec["encoding_goal"]["angles"]["alpha"] = float(alpha)
        vec, info = preprocess_vector(x, spec)
        plan = plan_from_spec(spec)
        q = min(4, plan["d"])  # cap at 4 to fit our simple coupling map
        for L in Ls:
            qc = angle_encode_circuit(info["angles"], q=q, L=L, entangler="cz-ring")
            sv = Statevector.from_instruction(qc)
            ez = z_chain_expectation_statevector(sv.probabilities())
            base = summarize_circuit(qc)
            tqc = transpile_for_backend(qc, backend, optimization_level=1)
            after = summarize_circuit(tqc)
            results.append({
                "alpha": round(float(alpha), 6),
                "L": L,
                "q": q,
                "ideal_Z_chain": round(float(ez), 6),
                "depth_before": int(base["depth"]),
                "depth_after": int(after["depth"]),
                "twoq_after": int(after["two_qubit_ops"]),
            })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
