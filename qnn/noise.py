from __future__ import annotations

from typing import Optional

from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


def make_basic_noise_model(p1: float = 0.0, p2: float = 0.0, readout_p01: float = 0.0, readout_p10: float = 0.0) -> NoiseModel:
    """Create a simple noise model with single- and two-qubit depolarizing errors and readout error.
    p1: prob for 1q depolarizing; p2: prob for 2q depolarizing; readout flip probs p(0->1), p(1->0)
    """
    nm = NoiseModel()
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["x", "y", "z", "h", "sx", "rx", "ry", "rz"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz", "cx"])
    if readout_p01 > 0 or readout_p10 > 0:
        ro = ReadoutError([[1 - readout_p01, readout_p01], [readout_p10, 1 - readout_p10]])
        nm.add_all_qubit_readout_error(ro)
    return nm


__all__ = ["make_basic_noise_model"]
