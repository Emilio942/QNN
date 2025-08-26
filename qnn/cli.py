from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_all():
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts/run_all.py")]
    raise SystemExit(subprocess.call(cmd))
