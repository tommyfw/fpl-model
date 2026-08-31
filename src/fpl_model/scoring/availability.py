from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# FPL `news` is the live lineup/injury signal. Historic start% must not override it.
_OUT = re.compile(
    r"will not play|has been ruled out|ruled out|ruled him out|"
    r"out for (the )?(season|rest)|out indefinitely|"
    r"suspended|ineligible|unknown return date|long[- ]term|"
    r"season[- ]ending|not in (the )?squad|left (the )?club|"
    r"joined .* on loan|season[- ]long",
    re.I,
)
_DOUBT = re.compile(
    r"doubtful|a doubt|knock|assessed|late (fitness )?test|"
    r"close to returning|minor injury|fitness test",
    re.I,
)
_BACK = re.compile(
    r"back in training|returned to training|available for|"
    r"expected to (play|start|feature)|fit to (play|start)",
    re.I,
)


def _chance(this: Any, nxt: Any) -> float | None:
    """Prefer this-round chance (as-of is the current unfinished GW)."""
    for v in (this, nxt):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        try:
            if pd.isna(v):
                continue
        except TypeError:
            pass
        return float(v) / 100.0
    return None


def availability_from_news(
    status: Any,
    *,
    chance_this: Any = None,
    chance_next: Any = None,
    news: Any = None,
) -> float:
    """P(plays a meaningful share of the next GW), 0–1."""
    st = ("" if status is None or (isinstance(status, float) and pd.isna(status)) else str(status)).lower().strip()
    text = "" if news is None or (isinstance(news, float) and pd.isna(news)) else str(news)
    chance = _chance(chance_this, chance_next)

    if _OUT.search(text):
        if chance is not None:
            return float(np.clip(min(chance, 0.12), 0.03, 0.15))
        return 0.05

    if st in {"i", "s", "u", "n"}:
        if chance is not None:
            return float(np.clip(chance, 0.03, 0.35))
        return 0.08

    if st == "d":
        if chance is not None:
            return float(np.clip(chance, 0.10, 0.85))
        if _BACK.search(text):
            return 0.55
        return 0.50

    if chance is not None and chance < 0.999:
        return float(np.clip(chance, 0.12, 1.0))

    if _DOUBT.search(text) and not _BACK.search(text):
        return 0.62
    return 1.0
