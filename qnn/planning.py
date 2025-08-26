from __future__ import annotations

import math
from typing import Dict, Any, Tuple


def plan_angle_encoding(d: int, preferred_q: int | None = None) -> Tuple[int, int]:
    """Return (q, L) for angle encoding.
    Defaults: q = min(4, d), L = ceil(d/q).
    """
    if d <= 0:
        raise ValueError("d must be > 0")
    q = preferred_q if preferred_q is not None else min(4, d)
    q = max(1, min(q, d))
    L = math.ceil(d / q)
    return q, L


def plan_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    layout = spec.get("shape", {}).get("layout")
    if layout == "d":
        d = int(spec["shape"]["d"])
    elif layout == "HxWxC":
        H = int(spec["shape"]["H"]) ; W = int(spec["shape"]["W"]) ; C = int(spec["shape"]["C"])
        d = H * W * C
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    target = spec.get("encoding_goal", {}).get("target", "angle")
    plan: Dict[str, Any] = {"target": target, "d": d}
    if target == "angle":
        q, L = plan_angle_encoding(d)
        plan.update({"q": q, "L": L, "entangler": "cz-ring"})
    elif target == "amplitude":
        # amplitude will need n qubits s.t. 2^n >= d'
        n = math.ceil(math.log2(max(1, d)))
        plan.update({"n": n})
    elif target == "phase":
        q, L = plan_angle_encoding(d)
        plan.update({"q": q, "L": L, "entangler": "cz-ring"})
    else:
        raise ValueError(f"Unsupported target: {target}")
    return plan


__all__ = ["plan_angle_encoding", "plan_from_spec"]
