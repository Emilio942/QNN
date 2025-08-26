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
from qiskit.quantum_info import Statevector


def test_end_to_end_angle():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    x = [((i % 8) - 4) / 4.0 for i in range(16)]
    vec, info = preprocess_vector(x, spec)
    plan = plan_from_spec(spec)
    q, L = plan["q"], plan["L"]
    qc = angle_encode_circuit(info["angles"], q=q, L=L)
    sv = Statevector.from_instruction(qc)
    assert abs(sum(sv.probabilities()) - 1.0) < 1e-9
