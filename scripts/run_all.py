#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys

CMDS = [
    [sys.executable, "scripts/smoke_validate_spec.py"],
    [sys.executable, "scripts/test_preprocess.py"],
    [sys.executable, "scripts/demo_preprocess.py"],
    [sys.executable, "scripts/demo_angle_sim.py"],
    [sys.executable, "scripts/demo_phase_sim.py"],
    [sys.executable, "scripts/demo_image_angle.py"],
    [sys.executable, "scripts/demo_amplitude_sim.py"],
    [sys.executable, "scripts/demo_noise_angle.py"],
]


def main():
    for cmd in CMDS:
        print("$", *cmd)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            sys.exit(proc.returncode)
    print("ALL OK")


if __name__ == "__main__":
    main()
