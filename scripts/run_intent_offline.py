#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run():
    # 1) Make dataset
    from scripts.make_greeting_dataset import main as make_ds
    make_ds()
    reports = ROOT / "reports"
    data_path = reports / "greetings.json"

    # 2) Train quick
    from scripts.train_reupload_classifier import main as train_main
    sys.argv = [
        "train",
        "--data", str(data_path),
        "--steps", "15",
        "--batch", "8",
        "--q", "4",
        "--L", "2",
        "--out", str(reports / "intent_params.json"),
        "--seed", "0",
    ]
    train_main()

    # 3) Router checks
    from scripts.demo_greeting_router import main as router_main
    sys.argv = [
        "router", "--text", "hi",
        "--params", str(reports / "intent_params.json"),
        "--offline",
    ]
    router_main()
    sys.argv = [
        "router", "--text", "hilfe",
        "--params", str(reports / "intent_params.json"),
        "--offline",
    ]
    router_main()

    # 4) Eval with plots
    from scripts.eval_classifier import main as eval_main
    sys.argv = [
        "eval",
        "--params", str(reports / "intent_params.json"),
        "--data", str(data_path),
        "--plot", str(reports / "pr_intent.png"),
        "--roc-plot", str(reports / "roc_intent.png"),
    ]
    eval_main()

    print(json.dumps({
        "dataset": str(data_path),
        "params": str(reports / "intent_params.json"),
        "plots": [str(reports / "pr_intent.png"), str(reports / "roc_intent.png")],
    }, indent=2))


def main():
    run()


if __name__ == "__main__":
    main()
