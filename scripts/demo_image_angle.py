#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector
from qnn.planning import plan_from_spec
from qnn.circuits import angle_encode_circuit
from qiskit.quantum_info import Statevector


def make_fake_image(H, W, C):
    # values in [-1,1]
    return [[[random.uniform(-1.0, 1.0) for _ in range(C)] for _ in range(W)] for _ in range(H)]


def main():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    # Switch to image layout 4x4x1 for a small demo
    spec["shape"]["layout"] = "HxWxC"
    spec["shape"]["H"] = 4
    spec["shape"]["W"] = 4
    spec["shape"]["C"] = 1
    spec["shape"]["d"] = None
    spec["encoding_goal"]["target"] = "angle"
    spec["encoding_goal"]["angles"]["alpha"] = 3.141592653589793

    img = make_fake_image(4, 4, 1)
    vec, info = preprocess_vector(img, spec)
    plan = plan_from_spec(spec)
    q, L = plan["q"], plan["L"]
    angles = info["angles"]

    qc = angle_encode_circuit(angles, q=q, L=L)
    sv = Statevector.from_instruction(qc)
    print({"q": q, "L": L, "len": len(vec)})


if __name__ == "__main__":
    main()
