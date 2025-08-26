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
from qnn.observables import zz_grid_expectation_statevector

from qiskit.quantum_info import Statevector


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    # 3x3 kernel (H=3,W=3,C=1)
    spec["shape"]["layout"] = "HxWxC"
    spec["shape"]["H"] = 3
    spec["shape"]["W"] = 3
    spec["shape"]["C"] = 1
    spec["shape"]["d"] = None

    spec["encoding_goal"]["target"] = "angle"
    spec["encoding_goal"]["angles"]["alpha"] = 3.141592653589793

    # simple kernel with a central peak
    import math
    H, W, C = 3, 3, 1
    img = [[[0.0 for _ in range(C)] for _ in range(W)] for _ in range(H)]
    img[1][1][0] = 1.0  # center

    vec, info = preprocess_vector(img, spec)
    # Plan q and L for 9 features
    plan = plan_from_spec(spec)
    q = 9 if plan["d"] >= 9 else plan["q"]
    L = 1 if q >= 9 else plan["L"]

    qc = angle_encode_circuit(info["angles"], q=q, L=L, entangler="cz-grid", entangler_config={"H": 3, "W": 3})
    sv = Statevector.from_instruction(qc)
    ezz = zz_grid_expectation_statevector(sv.probabilities(), 3, 3)
    print({"q": q, "L": L, "E_ZZ_grid": round(float(ezz), 6)})


if __name__ == "__main__":
    main()
