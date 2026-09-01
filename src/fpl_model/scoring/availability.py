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
_SHORT_BAN = re.compile(
    r"one[- ]match ban|suspended for (one|1) match|misses? (the )?next (game|match)|"
    r"will miss .* through suspension",
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
_LONG = re.compile(
    r"out for (the )?(season|rest)|out indefinitely|unknown return date|"
    r"long[- ]term|season[- ]ending|months",
    re.I,
)


def _flag(v: Any) -> bool:
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except TypeError:
        pass
    if isinstance(v, (int, float, np.floating)) and not np.isfinite(float(v)):
        return False
    return bool(v) and v not in {0, 0.0, "0", ""}


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
    last_red: Any = False,
) -> float:
    """P(plays a meaningful share of the next GW), 0–1."""
    st = ("" if status is None or (isinstance(status, float) and pd.isna(status)) else str(status)).lower().strip()
    text = "" if news is None or (isinstance(news, float) and pd.isna(news)) else str(news)
    chance = _chance(chance_this, chance_next)
    if _flag(last_red) and st not in {"i", "u", "n"} and (chance is None or chance > 0.5):
        return 0.08

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


def availability_later(
    status: Any,
    *,
    chance_this: Any = None,
    chance_next: Any = None,
    news: Any = None,
    this_gw: float | None = None,
    last_red: Any = False,
) -> float:
    """P(plays in later horizon GWs) — recovers from short-term doubts/bans."""
    this = float(this_gw) if this_gw is not None else availability_from_news(
        status, chance_this=chance_this, chance_next=chance_next, news=news, last_red=last_red
    )
    st = ("" if status is None or (isinstance(status, float) and pd.isna(status)) else str(status)).lower().strip()
    text = "" if news is None or (isinstance(news, float) and pd.isna(news)) else str(news)
    chance_n = _chance(chance_next, None)
    chance_t = _chance(chance_this, None)

    if _LONG.search(text) or st in {"u", "n"}:
        return float(np.clip(this, 0.03, 0.20))

    if _flag(last_red) or _SHORT_BAN.search(text) or (st == "s" and not _LONG.search(text)):
        return 0.92

    if chance_n is not None and chance_t is not None and chance_n > chance_t + 0.05:
        return float(np.clip(max(this, chance_n), 0.08, 1.0))

    if this >= 0.85:
        return this

    if st == "d" or _DOUBT.search(text):
        return float(np.clip(max(this, 0.72), 0.15, 0.92))

    if st in {"i", "s"}:
        return float(np.clip(max(this, 0.35), 0.05, 0.70))

    if _OUT.search(text):
        return float(np.clip(max(this, 0.20), 0.05, 0.45))

    return float(np.clip(max(this, 0.90), 0.0, 1.0))


_XI_SLOTS = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}


def minutes_risk(
    *,
    availability: float | None = None,
    status: Any = None,
    news: Any = None,
    chance_this: Any = None,
    chance_next: Any = None,
    slot_security: float | None = None,
    pos_slot_rank: float | None = None,
    position: str | None = None,
    start_pct_l5: float | None = None,
    minutes: float | None = None,
    starts: float | None = None,
    appearances: float | None = None,
) -> dict[str, Any]:
    """UI / transfer flag: ok, watch, or risk (news, chance, rotation, unused)."""
    labels: list[str] = []
    level = "ok"
    av = 1.0
    if availability is not None:
        try:
            if np.isfinite(float(availability)):
                av = float(availability)
        except (TypeError, ValueError):
            av = 1.0
    st = ("" if status is None or (isinstance(status, float) and pd.isna(status)) else str(status)).lower().strip()
    text = "" if news is None or (isinstance(news, float) and pd.isna(news)) else str(news).strip()
    chance = _chance(chance_this, chance_next)

    if av < 0.35 or st in {"i", "u", "n"}:
        level = "risk"
        labels.append("unlikely to play")
    elif av < 0.75 or st == "d" or (chance is not None and chance < 0.75):
        level = "watch"
        labels.append(f"{int(round(av * 100))}% chance")

    if text and (level != "ok" or _DOUBT.search(text) or _OUT.search(text)):
        short = text if len(text) <= 90 else text[:87] + "…"
        labels.append(short)

    pos = (position or "").upper()
    slots = float(_XI_SLOTS.get(pos, 4))
    rank = None
    if pos_slot_rank is not None:
        try:
            if np.isfinite(float(pos_slot_rank)):
                rank = float(pos_slot_rank)
        except (TypeError, ValueError):
            rank = None
    if rank is not None and rank > slots + 0.05:
        labels.append("rotation risk")
        level = "risk"
    else:
        sec = None
        if slot_security is not None:
            try:
                if np.isfinite(float(slot_security)):
                    sec = float(slot_security)
            except (TypeError, ValueError):
                sec = None
        if sec is not None and sec < 0.70:
            labels.append("sharing minutes")
            if level == "ok":
                level = "watch"

    l5 = None
    if start_pct_l5 is not None:
        try:
            if np.isfinite(float(start_pct_l5)):
                l5 = float(start_pct_l5)
        except (TypeError, ValueError):
            l5 = None
    if l5 is not None and l5 < 0.45:
        labels.append("not starting recently")
        if level == "ok":
            level = "watch"

    try:
        mins = float(minutes) if minutes is not None and np.isfinite(float(minutes)) else 0.0
    except (TypeError, ValueError):
        mins = 0.0
    try:
        apps = float(appearances if appearances is not None else (starts or 0) or 0)
        if not np.isfinite(apps):
            apps = 0.0
    except (TypeError, ValueError):
        apps = 0.0
    if mins < 90 and apps <= 1:
        labels.append("new / unused")
        if level == "ok":
            level = "watch"

    seen: set[str] = set()
    uniq: list[str] = []
    for lab in labels:
        if lab and lab not in seen:
            seen.add(lab)
            uniq.append(lab)
    return {"level": level, "labels": uniq[:4], "news": text[:120] if text else None}


def availability_horizon_mean(
    status: Any,
    *,
    chance_this: Any = None,
    chance_next: Any = None,
    news: Any = None,
    horizon: int = 5,
    last_red: Any = False,
) -> tuple[float, float, float]:
    """Return (this_gw, later_gw, horizon_mean) for M / xPts."""
    hz = max(int(horizon), 1)
    this = availability_from_news(
        status, chance_this=chance_this, chance_next=chance_next, news=news, last_red=last_red
    )
    later = availability_later(
        status,
        chance_this=chance_this,
        chance_next=chance_next,
        news=news,
        this_gw=this,
        last_red=last_red,
    )
    if hz <= 1:
        mean = this
    else:
        mean = (this + (hz - 1) * later) / hz
    return float(this), float(later), float(np.clip(mean, 0.0, 1.0))
