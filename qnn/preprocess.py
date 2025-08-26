from __future__ import annotations

from typing import Dict, Tuple, Any, List
import math

from .spec import compute_flat_shape


def l2_norm(x):
    return math.sqrt(sum(float(v) * float(v) for v in x))


def clip_values(x, vmin=None, vmax=None):
    if vmin is None and vmax is None:
        return x
    out = []
    for v in x:
        fv = float(v)
        if vmin is not None and fv < vmin:
            fv = vmin
        if vmax is not None and fv > vmax:
            fv = vmax
        out.append(fv)
    return out


def pad_to_power_of_two(x, pad_value=0.0):
    n = len(x)
    p = 1
    while p < n:
        p <<= 1
    if p == n:
        return x, n
    pad_len = p - n
    return x + [pad_value] * pad_len, p


def flatten_index_mapping(shape: Tuple[int, ...], order: str = "row-major"):
    # Returns a lambda mapping (h,w,c) -> i for HxWxC or identity for (d,)
    if len(shape) == 1:
        return lambda i: i
    if order != "row-major":
        raise NotImplementedError("Only row-major implemented in scaffold")
    H, W, C = shape
    return lambda h, w, c: ((h * W) + w) * C + c


def _flatten(x: Any, shape_cfg: Dict[str, Any]) -> List[float]:
    layout = shape_cfg.get("layout")
    if layout == "d":
        # Expect x as flat iterable of length d
        d = int(shape_cfg.get("d"))
        seq = [float(v) for v in x]
        if len(seq) != d:
            raise ValueError(f"Expected vector of length d={d}, got {len(seq)}")
        return seq
    elif layout == "HxWxC":
        H, W, C = int(shape_cfg["H"]), int(shape_cfg["W"]), int(shape_cfg["C"])
        # Expect x as nested [H][W][C]
        flat: List[float] = []
        for h in range(H):
            row = x[h]
            for w in range(W):
                pix = row[w]
                for c in range(C):
                    flat.append(float(pix[c]))
        return flat
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def preprocess_vector(x, spec: Dict):
    vr = spec["value_range"]
    prep = spec["preprocessing"]
    shape_cfg = spec["shape"]

    # 0) flatten according to shape
    x = _flatten(x, shape_cfg)

    # 1) optional clipping
    x = clip_values(x, vr.get("min"), vr.get("max"))

    # 2) normalization
    norm_type = prep.get("normalize", {}).get("type", "none")
    if norm_type == "l2":
        nrm = l2_norm(x)
        if nrm == 0:
            raise ValueError("L2 norm is zero; cannot normalize for amplitude encoding")
        x = [v / nrm for v in x]
    elif norm_type == "zscore":
        m = sum(x) / len(x)
        var = sum((v - m) ** 2 for v in x) / max(1, len(x) - 1)
        sd = math.sqrt(var) if var > 0 else 1.0
        x = [(v - m) / sd for v in x]

    # 3) padding (flattened)
    pad_cfg = prep.get("pad", {})
    original_len = len(x)
    mask: List[bool] | None = None
    if pad_cfg.get("to_power_of_two", {}).get("enabled", False):
        pad_val = pad_cfg.get("to_power_of_two", {}).get("pad_value", 0.0)
        x, dprime = pad_to_power_of_two(x, pad_val)
        # build mask to allow de-padding later
        mask = [True] * original_len + [False] * (dprime - original_len)

    # 4) optional angle mapping
    target = spec.get("encoding_goal", {}).get("target", "angle")
    info: Dict[str, Any] = {}
    if target == "angle":
        alpha = spec["encoding_goal"].get("angles", {}).get("alpha", math.pi)
        theta = [alpha * v for v in x]
        info["angles"] = theta
    elif target == "amplitude":
        # For amplitude encoding, ensure L2 normalized
        nrm = l2_norm(x)
        if nrm == 0:
            raise ValueError("Zero vector cannot be amplitude-encoded")
        x = [v / nrm for v in x]
    elif target == "phase":
        beta = spec["encoding_goal"].get("phase", {}).get("beta", 1.0)
        phi = [beta * v for v in x]
        info["phases"] = phi

    info["vector"] = x
    info["shape"] = shape_cfg
    info["indexing_order"] = spec.get("indexing", {}).get("order", "row-major")
    info["pad"] = {
        "enabled": mask is not None,
        "original_len": original_len,
        "padded_len": len(x),
        "mask": mask,
    }
    return x, info
