#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector
from qnn.planning import plan_from_spec
from qnn.circuits import angle_encode_circuit
from qnn.noise import make_basic_noise_model
from qnn.observables import z_chain_expectation_from_samples

from qiskit_aer import AerSimulator


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    spec["encoding_goal"]["target"] = "angle"
    x = [((i % 8) - 4) / 4.0 for i in range(16)]
    vec, info = preprocess_vector(x, spec)
    plan = plan_from_spec(spec)
    q, L = plan["q"], plan["L"]

    qc = angle_encode_circuit(info["angles"], q=q, L=L)
    qc.measure_all()

    configs = [
        {"p1": 0.001, "p2": 0.01, "ro": 0.01, "shots": 1024},
        {"p1": 0.005, "p2": 0.02, "ro": 0.02, "shots": 2048},
        {"p1": 0.01, "p2": 0.05, "ro": 0.02, "shots": 4096},
    ]

    out = []
    for cfg in configs:
        nm = make_basic_noise_model(cfg["p1"], cfg["p2"], cfg["ro"], cfg["ro"])
        sim = AerSimulator(noise_model=nm)
        res = sim.run(qc, shots=cfg["shots"]).result()
        counts = res.get_counts()
        samples = []
        for bstr, c in counts.items():
            i = int(bstr[::-1], 2)
            samples.extend([i] * c)
        ez = z_chain_expectation_from_samples(samples)
        out.append({"shots": cfg["shots"], "p1": cfg["p1"], "p2": cfg["p2"], "ro": cfg["ro"], "est": round(float(ez), 6)})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
