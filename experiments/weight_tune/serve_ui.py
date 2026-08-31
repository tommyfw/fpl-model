#!/usr/bin/env python3
"""Experimental weight-tune UI. Does not start or replace the live UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from ui.server import serve

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8777)
    args = p.parse_args()
    serve(host=args.host, port=args.port)
