from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class ValueRange:
    min: Optional[float]
    max: Optional[float]
    units: str = "normalized"
    allow_nan_inf: bool = False


@dataclass
class PadConfig:
    to_power_of_two: bool = False
    axis: str = "flattened"
    pad_value: float = 0.0


@dataclass
class Preprocessing:
    normalize: Dict[str, Any]
    clip: Dict[str, Any]
    quantize: Dict[str, Any]
    pad: PadConfig


@dataclass
class EncodingGoal:
    target: str  # angle|amplitude|phase|basis
    angles: Dict[str, Any]
    amplitude: Dict[str, Any]
    phase: Dict[str, Any]


@dataclass
class TensorSpec:
    id: str
    purpose: str
    dtype: str
    value_range: ValueRange
    shape: Dict[str, Any]
    indexing: Dict[str, Any]
    complex: Dict[str, Any]
    preprocessing: Preprocessing
    encoding_goal: EncodingGoal
    observables_plan: Any
    versioning: Dict[str, Any]


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def compute_flat_shape(shape_cfg: Dict[str, Any]) -> Tuple[int, Tuple[int, ...]]:
    layout = shape_cfg.get("layout")
    if layout == "d":
        d = int(shape_cfg.get("d"))
        return d, (d,)
    elif layout == "HxWxC":
        H = int(shape_cfg.get("H"))
        W = int(shape_cfg.get("W"))
        C = int(shape_cfg.get("C"))
        return H * W * C, (H, W, C)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


__all__ = [
    "ValueRange",
    "PadConfig",
    "Preprocessing",
    "EncodingGoal",
    "TensorSpec",
    "compute_flat_shape",
]
