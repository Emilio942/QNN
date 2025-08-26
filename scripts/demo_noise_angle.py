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
from qnn.observables import z_chain_expectation_from_samples
from qnn.noise import make_basic_noise_model

from qiskit_aer import AerSimulator


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    x = [((i % 8) - 4) / 4.0 for i in range(16)]
    vec, info = preprocess_vector(x, spec)
    plan = plan_from_spec(spec)
    q, L = plan["q"], plan["L"]

    qc = angle_encode_circuit(info["angles"], q=q, L=L)
    qc.measure_all()

    for (p1, p2, ro), shots in [((0.001, 0.01, 0.01), 1024), ((0.01, 0.05, 0.02), 4096)]:
        noise_model = make_basic_noise_model(p1=p1, p2=p2, readout_p01=ro, readout_p10=ro)
        sim = AerSimulator(noise_model=noise_model)
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()
        # Convert bitstring counts to integer samples (little-endian)
        samples = []
        for bstr, c in counts.items():
            # Qiskit returns big-endian by default; reverse to little-endian for our parity convention
            i = int(bstr[::-1], 2)
            samples.extend([i] * c)
        ez = z_chain_expectation_from_samples(samples)
        print({"shots": shots, "p1": p1, "p2": p2, "ro": ro, "estimate": round(float(ez), 6)})


if __name__ == "__main__":
    main()
