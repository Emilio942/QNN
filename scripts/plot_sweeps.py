#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load_json_from_cmd(cmd: list[str]) -> list[dict]:
    import subprocess
    res = subprocess.run(cmd, capture_output=True, text=True)
    res.check_returncode()
    return json.loads(res.stdout)


def plot_alpha_L():
    data = load_json_from_cmd([sys.executable, str(ROOT / "scripts/sweep_alpha_L.py")])
    # Group by alpha
    by_alpha = {}
    for row in data:
        by_alpha.setdefault(row["alpha"], []).append(row)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for alpha, rows in by_alpha.items():
        rows = sorted(rows, key=lambda r: r["L"])
        Ls = [r["L"] for r in rows]
        ez = [r["ideal_Z_chain"] for r in rows]
        depth = [r["depth_after"] for r in rows]
        axs[0].plot(Ls, ez, marker='o', label=f"alpha={alpha}")
        axs[1].plot(Ls, depth, marker='o', label=f"alpha={alpha}")
    axs[0].set_title("<Z..Z> vs L")
    axs[0].set_xlabel("L")
    axs[0].set_ylabel("ideal <Z..Z>")
    axs[0].grid(True)
    axs[1].set_title("Depth after transpile vs L")
    axs[1].set_xlabel("L")
    axs[1].set_ylabel("depth")
    axs[1].grid(True)
    axs[0].legend()
    axs[1].legend()
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "alpha_L.png", dpi=150)
    plt.close(fig)


def plot_noise():
    data = load_json_from_cmd([sys.executable, str(ROOT / "scripts/sweep_noise.py")])
    shots = [r["shots"] for r in data]
    est = [r["est"] for r in data]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(shots, est, marker='o')
    ax.set_title("Noisy <Z..Z> vs Shots (varied noise)")
    ax.set_xlabel("shots")
    ax.set_ylabel("estimate <Z..Z>")
    ax.grid(True)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "noise.png", dpi=150)
    plt.close(fig)


def main():
    plot_alpha_L()
    plot_noise()
    print({"saved": ["reports/alpha_L.png", "reports/noise.png"]})


if __name__ == "__main__":
    main()
