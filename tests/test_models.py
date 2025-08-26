from __future__ import annotations

from pathlib import Path
import random

from qnn.models import save_params, load_params, predict_scores
from qnn.io import load_tensor_spec


def test_save_load_and_predict(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    spec_path = root / "specs" / "tensor_spec.yaml"
    spec = load_tensor_spec(spec_path)
    if spec["shape"]["layout"] == "d":
        d = int(spec["shape"]["d"])
    else:
        d = int(spec["shape"]["H"]) * int(spec["shape"]["W"]) * int(spec["shape"]["C"])

    params_file = tmp_path / "params.json"
    save_params(params_file, [0.0, 0.1, -0.2, 0.3], q=2, L=2)
    p = load_params(params_file)
    assert p["q"] == 2 and p["L"] == 2 and len(p["theta"]) == 4

    random.seed(0)
    vectors = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(2)]
    scores = predict_scores(vectors, params_file, spec_path)
    assert len(scores) == 2
    for s in scores:
        assert -1.000001 <= float(s) <= 1.000001
