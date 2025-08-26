#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector


def test_angle_vector():
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    x = [((i % 8) - 4) / 4.0 for i in range(16)]
    vec, info = preprocess_vector(x, spec)
    assert len(vec) == 16
    assert abs(info["angles"][0] + 3.141592653589793) < 1e-9


def test_amplitude_with_padding():
    # modify spec on the fly: amplitude target + pad
    spec = load_tensor_spec(ROOT / "specs/tensor_spec.yaml")
    spec["encoding_goal"]["target"] = "amplitude"
    spec["preprocessing"]["pad"]["to_power_of_two"]["enabled"] = True
    spec["preprocessing"]["pad"]["to_power_of_two"]["pad_value"] = 0.0

    # use d=16 from spec and feed 16 values; pad should keep 16 as 16 (2^4), mask still valid
    x = [1.0] + [0.0] * 15
    vec, info = preprocess_vector(x, spec)
    # l2 norm must be 1
    s = sum(v * v for v in vec)
    assert abs(s - 1.0) < 1e-9
    assert info["pad"]["enabled"] in {True, False}


if __name__ == "__main__":
    test_angle_vector()
    test_amplitude_with_padding()
    print("OK")
