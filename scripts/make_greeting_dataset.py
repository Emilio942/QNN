#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def offline_embedding(text: str, dim: int = 16) -> List[float]:
    v = [0.0] * dim
    s = text.lower()
    for i, ch in enumerate(s):
        h = (ord(ch) * 1315423911 + i * 2654435761) & 0xFFFFFFFF
        idx = h % dim
        sign = -1.0 if ((h >> 1) & 1) else 1.0
        v[idx] += sign
    m = max(1.0, max(abs(x) for x in v))
    return [x / m for x in v]


def main():
    pos = [
        "hi", "hallo", "servus", "guten tag", "hey", "moin", "grüß dich",
        "hallo zusammen", "guten morgen", "guten abend"
    ]
    neg = [
        "hilfe", "banane", "42", "keine ahnung", "foobar", "was ist die zahl",
        "ich weiß es nicht", "random text", "ungefähr"
    ]
    dim = 16
    X: List[List[float]] = []
    y: List[int] = []
    for t in pos:
        X.append(offline_embedding(t, dim))
        y.append(1)
    for t in neg:
        X.append(offline_embedding(t, dim))
        y.append(-1)

    out = ROOT / "reports" / "greetings.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"X": X, "y": y}, indent=2))
    print(json.dumps({"saved": str(out), "n": len(y), "d": dim}, indent=2))


if __name__ == "__main__":
    main()
