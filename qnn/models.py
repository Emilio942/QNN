from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from .io import load_tensor_spec
from .preprocess import preprocess_vector
from .vqa import build_reupload_variational_circuit
from .observables import z_expectation_statevector
from .noise import make_basic_noise_model

from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def save_params(path: str | Path, theta: List[float], q: int, L: int) -> None:
    """Save trained parameters to JSON."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"theta": list(map(float, theta)), "q": int(q), "L": int(L)}
    path.write_text(json.dumps(payload, indent=2))


def load_params(path: str | Path) -> Dict[str, Any]:
    import json
    data = json.loads(Path(path).read_text())
    assert "theta" in data and "q" in data and "L" in data, "Invalid params file"
    return data


def _score_qc(qc, use_noise: bool = False, shots: int = 2048, noise_cfg: Dict[str, float] | None = None):
    """Return probabilities for the quantum circuit, ideal or with noise."""
    if not use_noise:
        sv = Statevector.from_instruction(qc)
        return sv.probabilities()
    nm = make_basic_noise_model(**(noise_cfg or {"p1": 0.005, "p2": 0.02, "readout_p01": 0.02, "readout_p10": 0.02}))
    sim = AerSimulator(noise_model=nm)
    qc_m = qc.copy()
    qc_m.measure_all()
    res = sim.run(qc_m, shots=shots).result()
    counts = res.get_counts()
    # Convert to probs
    total = max(1, sum(counts.values()))
    max_qubits = qc.num_qubits
    size = 2 ** max_qubits
    probs = [0.0] * size
    for bstr, c in counts.items():
        i = int(bstr[::-1], 2)
        probs[i] += c / total
    return probs


def predict_scores(vectors: List[List[float]], params_path: str | Path, spec_path: str | Path) -> List[float]:
    """Compute Z0 expectation scores for a list of input vectors using saved params.

    Returns a list of float scores in [-1, 1].
    """
    spec = load_tensor_spec(spec_path)
    p = load_params(params_path)
    q, L, theta = int(p["q"]), int(p["L"]), list(map(float, p["theta"]))

    scores: List[float] = []
    for vec in vectors:
        v, info = preprocess_vector(vec, spec)
        angles = info["angles"]
        qc = build_reupload_variational_circuit(q, L, angles, theta)
        probs = _score_qc(qc, use_noise=False)
        scores.append(z_expectation_statevector(probs, 0))
    return scores
