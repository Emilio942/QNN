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
    v = [0.0] * dim
    s = text.lower()
    for i, ch in enumerate(s):
        h = (ord(ch) * 1315423911 + i * 2654435761) & 0xFFFFFFFF
        idx = h % dim
        sign = -1.0 if ((h >> 1) & 1) else 1.0
        v[idx] += sign
    m = max(1.0, max(abs(x) for x in v))
    return [x / m for x in v]


def ollama_generate(prompt: str, model: str, url: str) -> str:
    import requests
    r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False})
    r.raise_for_status()
    data = r.json()
    return str(data.get("response", ""))


def main():
    ap = argparse.ArgumentParser(description="Intent router: QNN classify via embeddings; optionally delegate to Ollama for a response")
    ap.add_argument("--text", required=True, help="User input text")
    ap.add_argument("--params", required=True, help="Path to trained QNN params JSON")
    ap.add_argument("--spec", default=str(ROOT / "specs" / "tensor_spec.yaml"), help="Spec path (d must match embedding size)")
    ap.add_argument("--threshold", type=float, default=0.0, help="Score threshold for positive class")
    ap.add_argument("--positive-response", default="Hallo, wie geht's?", help="Response if classified positive")
    ap.add_argument("--model-embed", default="nomic-embed-text", help="Ollama embedding model")
    ap.add_argument("--url-embed", default="http://localhost:11434/api/embeddings", help="Ollama embeddings endpoint")
    ap.add_argument("--offline", action="store_true", help="Use offline toy embeddings (no Ollama)")
    ap.add_argument("--delegate-negative", action="store_true", help="If negative, call Ollama generate for an answer")
    ap.add_argument("--model-generate", default="llama3", help="Ollama generate model for negative route")
    ap.add_argument("--url-generate", default="http://localhost:11434/api/generate", help="Ollama generate endpoint")
    args = ap.parse_args()

    # Build embedding
    if args.offline:
        emb = offline_embedding(args.text, 16)
    else:
        emb = fetch_embedding(args.text, args.model_embed, args.url_embed)

    # Classify
    score = float(predict_scores([emb], args.params, args.spec)[0])

    if score >= args.threshold:
        print(args.positive_response)
        print(json.dumps({"score": score, "route": "positive"}, indent=2))
        return

    # Negative route
    if args.delegate_negative:
        reply = ollama_generate(args.text, args.model_generate, args.url_generate)
        print(reply)
        print(json.dumps({"score": score, "route": "ollama"}, indent=2))
    else:
        print("Ich habe dich nicht verstanden.")
        print(json.dumps({"score": score, "route": "negative"}, indent=2))


if __name__ == "__main__":
    main()
