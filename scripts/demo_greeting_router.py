#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.models import predict_scores


def fetch_embedding(text: str, model: str, url: str) -> List[float]:
    import requests
    r = requests.post(url, json={"model": model, "prompt": text})
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if emb is None:
        raise SystemExit(f"No 'embedding' in response. Keys: {list(data.keys())}")
    return list(map(float, emb))


def offline_embedding(text: str, dim: int = 16) -> List[float]:
    """Simple deterministic char-ngram hashing into a fixed dim vector.
    Produces a length-dim vector in [-1,1]. Matches default spec d=16.
    """
    import math
    v = [0.0] * dim
    s = text.lower()
    for i, ch in enumerate(s):
        h = (ord(ch) * 1315423911 + i * 2654435761) & 0xFFFFFFFF
        idx = h % dim
        sign = -1.0 if ((h >> 1) & 1) else 1.0
        v[idx] += sign
    # normalize to [-1,1]
    m = max(1.0, max(abs(x) for x in v))
    return [x / m for x in v]


def main():
    ap = argparse.ArgumentParser(description="Greeting router demo using Ollama embeddings + QNN classifier")
    ap.add_argument("--text", required=True, help="Input text")
    ap.add_argument("--params", required=True, help="Path to trained params JSON")
    ap.add_argument("--spec", default=str(ROOT / "specs" / "tensor_spec.yaml"), help="Spec path (d must match embedding size)")
    ap.add_argument("--model", default="nomic-embed-text", help="Ollama embedding model name")
    ap.add_argument("--url", default="http://localhost:11434/api/embeddings", help="Ollama embeddings endpoint URL")
    ap.add_argument("--offline", action="store_true", help="Use built-in toy embedding (no Ollama)")
    args = ap.parse_args()

    if args.offline:
        emb = offline_embedding(args.text, 16)
    else:
        emb = fetch_embedding(args.text, args.model, args.url)
    scores = predict_scores([emb], args.params, args.spec)
    score = float(scores[0])

    # Simple routing: positive -> greeting
    if score >= 0:
        print("Hallo, wie geht's?")
    else:
        print("Ich habe dich nicht verstanden.")
    print(json.dumps({"score": score}, indent=2))


if __name__ == "__main__":
    main()
