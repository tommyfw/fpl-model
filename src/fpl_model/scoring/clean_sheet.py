from __future__ import annotations

import json
from typing import Any

import numpy as np

# Independent Poisson with a home/away tilt. Clip so elite defences vs
# weak attacks are not ~70% CS every week (that overfills cheap DEF in the XI).
HOME_CONCEDE = 0.88
AWAY_CONCEDE = 1.12
DEFAULT_LEAGUE_XG = 1.35
LAMBDA_MIN = 0.35
LAMBDA_MAX = 2.9
PCS_MIN = 0.03
PCS_MAX = 0.45


def expected_goals_against(
    own_xga: float,
    opp_xg: float,
    *,
    home: bool,
    league_xg: float = DEFAULT_LEAGUE_XG,
) -> float:
    """Opponent expected goals vs this team in one match."""
    own = float(np.clip(own_xga if np.isfinite(own_xga) else league_xg, 0.4, 2.6))
    opp = float(np.clip(opp_xg if np.isfinite(opp_xg) else league_xg, 0.4, 2.6))
    league = float(league_xg) if np.isfinite(league_xg) and league_xg > 0.6 else DEFAULT_LEAGUE_XG
    ha = HOME_CONCEDE if home else AWAY_CONCEDE
    return float(np.clip(own * (opp / league) * ha, LAMBDA_MIN, LAMBDA_MAX))


def p_cs_from_lambda(lam: float) -> float:
    return float(np.clip(np.exp(-float(lam)), PCS_MIN, PCS_MAX))


def fixture_p_cs(
    own_xga: float,
    opp_xg: float,
    *,
    home: bool,
    league_xg: float = DEFAULT_LEAGUE_XG,
) -> float:
    return p_cs_from_lambda(expected_goals_against(own_xga, opp_xg, home=home, league_xg=league_xg))


def pack_fixture(
    *,
    home: bool,
    opp_attack: float,
    opp_defence_leak: float,
    own_xga: float,
    league_xg: float,
) -> dict[str, float]:
    att = float(opp_attack) if np.isfinite(opp_attack) else league_xg
    leak = float(opp_defence_leak) if np.isfinite(opp_defence_leak) else league_xg
    own = float(own_xga) if np.isfinite(own_xga) else league_xg
    return {
        "home": 1.0 if home else 0.0,
        "opp_attack": att,
        "opp_defence_leak": leak,
        "own_xga": own,
        "p_cs": fixture_p_cs(own, att, home=home, league_xg=league_xg),
    }


def parse_fixtures_json(fixtures_json: Any) -> dict[str, list]:
    try:
        by_event = json.loads(fixtures_json) if isinstance(fixtures_json, str) else (fixtures_json or {})
    except json.JSONDecodeError:
        return {}
    return by_event if isinstance(by_event, dict) else {}


def iter_horizon_matches(fixtures_json: Any, horizon: int) -> list[dict]:
    by_event = parse_fixtures_json(fixtures_json)
    if not by_event:
        return []
    events = sorted(int(k) for k in by_event.keys())
    min_e = min(events)
    out: list[dict] = []
    for i in range(horizon):
        matches = by_event.get(str(min_e + i), [])
        if not matches:
            continue
        out.extend(matches)
    return out


def horizon_p_cs(fixtures_json: Any, horizon: int, fallback: float = 0.22) -> float:
    """Mean P(CS) over fixtures in the scoring window (blanks omitted, DGW counts both)."""
    matches = iter_horizon_matches(fixtures_json, horizon)
    if not matches:
        return float(fallback)
    vals = [float(m.get("p_cs") or fallback) for m in matches]
    return float(np.mean(vals))
