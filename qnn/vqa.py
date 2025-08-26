from __future__ import annotations

from typing import List, Callable, Tuple
import math
import random

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def simple_variational_layer(q: int, params: List[float]) -> QuantumCircuit:
    """A shallow variational layer: Ry on each qubit followed by CZ ring."""
    qc = QuantumCircuit(q)
    idx = 0
    for j in range(q):
        qc.ry(params[idx], j)
        idx += 1
    for j in range(q):
        qc.cz(j, (j + 1) % q)
    return qc


def build_vqa_circuit(q: int, L: int, params: List[float]) -> QuantumCircuit:
    qc = QuantumCircuit(q)
    # L layers, each consumes q params
    assert len(params) == q * L
    idx = 0
    for _ in range(L):
        layer = simple_variational_layer(q, params[idx: idx + q])
        idx += q
        qc = qc.compose(layer)
    return qc


def build_reupload_variational_circuit(q: int, L: int, angles: List[float], params: List[float]) -> QuantumCircuit:
    """Interleave data re-upload (Ry angles) and trainable Ry params with CZ ring entanglers.
    angles must have at least q*L entries; params must have q*L entries.
    """
    assert len(angles) >= q * L, "insufficient angles for q*L"
    assert len(params) >= q * L, "insufficient params for q*L"
    qc = QuantumCircuit(q)
    a_idx = 0
    p_idx = 0
    for _ in range(L):
        # data upload
        for j in range(q):
            qc.ry(angles[a_idx], j)
            a_idx += 1
        for j in range(q):
            qc.cz(j, (j + 1) % q)
        # variational
        for j in range(q):
            qc.ry(params[p_idx], j)
            p_idx += 1
        for j in range(q):
            qc.cz(j, (j + 1) % q)
    return qc


def predict_statevector(qc: QuantumCircuit, to_prob: bool = True):
    sv = Statevector.from_instruction(qc)
    return sv.probabilities() if to_prob else sv


def spsa_step(theta: List[float], grad_estimator: Callable[[List[float]], List[float]], a: float, c: float) -> List[float]:
    g = grad_estimator(theta)
    return [t - a * gi for t, gi in zip(theta, g)]


def spsa_gradient(loss_fn: Callable[[List[float]], float], c: float) -> Callable[[List[float]], List[float]]:
    def grad(theta: List[float]) -> List[float]:
        d = len(theta)
        delta = [1 if random.random() > 0.5 else -1 for _ in range(d)]
        theta_plus = [t + c * d_i for t, d_i in zip(theta, delta)]
        theta_minus = [t - c * d_i for t, d_i in zip(theta, delta)]
        l_plus = loss_fn(theta_plus)
        l_minus = loss_fn(theta_minus)
        g_hat = [(l_plus - l_minus) / (2 * c * d_i) for d_i in delta]
        return g_hat
    return grad
