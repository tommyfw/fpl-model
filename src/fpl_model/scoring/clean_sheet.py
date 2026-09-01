from __future__ import annotations

import json
from typing import Any

import numpy as np

# Residual H/A on P(CS) only. Venue-split xG already carries the real home
# edge; 0.88/1.12 on top of that made away CS ~3× worse than home.
HOME_CONCEDE = 0.97
AWAY_CONCEDE = 1.03
HOME_ATTACK = 1.03
AWAY_ATTACK = 0.97
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


def expected_goals_for(
    own_xg: float,
    opp_leak: float,
    *,
    home: bool,
    league_xg: float = DEFAULT_LEAGUE_XG,
) -> float:
    """This team's expected goals vs this opponent in one match."""
    own = float(np.clip(own_xg if np.isfinite(own_xg) else league_xg, 0.4, 2.8))
    leak = float(np.clip(opp_leak if np.isfinite(opp_leak) else league_xg, 0.45, 2.6))
    league = float(league_xg) if np.isfinite(league_xg) and league_xg > 0.6 else DEFAULT_LEAGUE_XG
    ha = HOME_ATTACK if home else AWAY_ATTACK
    return float(np.clip(own * (leak / league) * ha, LAMBDA_MIN, LAMBDA_MAX))


DC_RHO = 0.10


def dixon_coles_p_cs(lam_against: float, lam_for: float, rho: float = DC_RHO) -> float:
    """P(concede 0) with Dixon–Coles dependence on own goals (rho=0 → Poisson)."""
    mu = float(np.clip(lam_against, LAMBDA_MIN, LAMBDA_MAX))
    lam = float(np.clip(lam_for, LAMBDA_MIN, LAMBDA_MAX))
    r = float(rho) if np.isfinite(rho) else 0.0
    indep = np.exp(-mu)
    adj = 1.0 + lam * r * np.exp(-lam) * (lam - mu)
    return float(np.clip(indep * adj, PCS_MIN, PCS_MAX))


def fixture_p_cs(
    own_xga: float,
    opp_xg: float,
    *,
    home: bool,
    league_xg: float = DEFAULT_LEAGUE_XG,
    own_xg: float | None = None,
    opp_leak: float | None = None,
) -> float:
    lam_ag = expected_goals_against(own_xga, opp_xg, home=home, league_xg=league_xg)
    leak = opp_leak if opp_leak is not None else league_xg
    oxg = own_xg if own_xg is not None else league_xg
    lam_for = expected_goals_for(oxg, leak, home=home, league_xg=league_xg)
    return dixon_coles_p_cs(lam_ag, lam_for)


def team_xg_context_score(
    own_xg: float,
    own_xga: float,
    league_xg: float,
    *,
    position: str = "MID",
    team_tilt: float | None = None,
) -> float:
    """Own team xG vs this opponent type, 0–100 (same scale as leak/2×100)."""
    lg = float(league_xg) if np.isfinite(league_xg) and league_xg > 0.4 else DEFAULT_LEAGUE_XG
    atk = float(own_xg) if np.isfinite(own_xg) else lg
    de = float(own_xga) if np.isfinite(own_xga) else lg
    atk_s = float(np.clip(atk / 2.0 * 100.0, 5.0, 100.0))
    def_s = float(np.clip((2.4 - de) / 2.0 * 100.0, 5.0, 100.0))
    pos = (position or "MID").upper()
    if pos in ("DEF", "GK"):
        ctx = 0.40 * atk_s + 0.60 * def_s
    else:
        ctx = atk_s
    if team_tilt is not None and np.isfinite(team_tilt):
        ctx = 0.75 * ctx + 0.25 * float(np.clip(team_tilt, 0.0, 1.0) * 100.0)
    return float(np.clip(ctx, 5.0, 100.0))


def _lookup_rate(lookup: Any, col: str, team_id: Any, default: float) -> float:
    if team_id is None or (isinstance(team_id, float) and not np.isfinite(team_id)):
        return default
    try:
        tid = int(team_id)
    except (TypeError, ValueError):
        return default
    if isinstance(lookup, dict):
        row = lookup.get(tid) or lookup.get(str(tid)) or {}
        v = row.get(col, default)
    else:
        if col not in getattr(lookup, "columns", []) or tid not in lookup.index:
            return default
        v = lookup[col].loc[tid]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def pack_match_for_team(
    *,
    home: bool,
    team_id: Any,
    opp_id: Any,
    lookup: Any,
    league_xg: float,
    position: str = "MID",
    team_tilt: float | None = None,
) -> dict[str, float]:
    """Venue-correct pack_fixture for one side of a match (shared by features + UI)."""
    lg = float(league_xg) if np.isfinite(league_xg) and league_xg > 0.4 else DEFAULT_LEAGUE_XG
    own_def_col = "team_defence_home" if home else "team_defence_away"
    own_att_col = "team_attack_home" if home else "team_attack_away"
    opp_att_col = "team_attack_away" if home else "team_attack_home"
    opp_def_col = "team_defence_away" if home else "team_defence_home"
    own_def = _lookup_rate(lookup, own_def_col, team_id, _lookup_rate(lookup, "team_defence", team_id, -lg))
    own_att = _lookup_rate(lookup, own_att_col, team_id, _lookup_rate(lookup, "team_attack", team_id, lg))
    opp_att = _lookup_rate(lookup, opp_att_col, opp_id, _lookup_rate(lookup, "team_attack", opp_id, lg))
    opp_def = _lookup_rate(lookup, opp_def_col, opp_id, _lookup_rate(lookup, "team_defence", opp_id, -lg))
    tilt = team_tilt
    if tilt is None:
        tv = _lookup_rate(lookup, "team_tilt", team_id, float("nan"))
        tilt = tv if np.isfinite(tv) else None
    return pack_fixture(
        home=home,
        opp_attack=opp_att,
        opp_defence_leak=float(-opp_def) if np.isfinite(opp_def) else lg,
        own_xga=float(-own_def) if np.isfinite(own_def) else lg,
        league_xg=lg,
        own_xg=own_att if np.isfinite(own_att) else lg,
        team_tilt=tilt,
        position=position,
    )


def attack_fixture_ease(
    leak: float,
    own_xg: float,
    league_xg: float = DEFAULT_LEAGUE_XG,
    *,
    power: float = 0.45,
) -> float:
    """Attack chip score: opponent leak × own venue attack vs league.

    Same leak is easier for a 2.0 xG side than a 1.1 xG side (eye test).
    """
    lg = float(league_xg) if np.isfinite(league_xg) and league_xg > 0.4 else DEFAULT_LEAGUE_XG
    lk = float(leak) if np.isfinite(leak) else lg
    own = float(own_xg) if np.isfinite(own_xg) else lg
    own = float(np.clip(own, 0.55 * lg, 1.55 * lg))
    return float(lk * (own / lg) ** power)


# Absolute PL-scale bands (not forced tertiles of this week's slate).
ATTACK_EASE_HARD = 1.22
ATTACK_EASE_EASY = 1.48
# Ascending thresholds → ratings 5,4,3,2; at/above last cut → 1.
ATTACK_FDR_CUTS = (1.22, 1.35, 1.48, 1.65)
PCS_HARD = 0.18
PCS_EASY = 0.30
PCS_FDR_CUTS = (0.18, 0.24, 0.30, 0.38)
VENUE_THIN_N = 6


def ease_band(value: float, lo: float, hi: float, *, higher_is_easier: bool = True) -> str:
    if not np.isfinite(value):
        return "mid"
    if higher_is_easier:
        if value >= hi:
            return "easy"
        if value <= lo:
            return "hard"
        return "mid"
    if value <= lo:
        return "easy"
    if value >= hi:
        return "hard"
    return "mid"


def tertile_bounds(values: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return 0.0, 0.0
    s = sorted(vals)
    lo = s[min(len(s) - 1, int(len(s) * 0.33))]
    hi = s[min(len(s) - 1, int(len(s) * 0.66))]
    return float(lo), float(hi)


def fdr_like(value: float, values: list[float], *, higher_is_easier: bool = True) -> int:
    """1 = easiest quintile in this set, 5 = hardest (display only; not FPL FDR)."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals or not np.isfinite(value):
        return 3
    s = sorted(vals)
    rank = sum(1 for x in s if x <= value) / len(s)
    p = (1.0 - rank) if higher_is_easier else rank
    return int(np.clip(np.floor(p * 5.0) + 1, 1, 5))


def fdr_from_cuts(value: float, cuts: tuple[float, ...] | list[float], *, higher_is_easier: bool = True) -> int:
    """Map a score onto 1–5 using fixed ascending cut points (display only)."""
    if not np.isfinite(value) or not cuts:
        return 3
    if higher_is_easier:
        for i, c in enumerate(cuts):
            if value < float(c):
                return 5 - i
        return 1
    for i, c in enumerate(cuts):
        if value > float(c):
            return 5 - i
    return 1


def pack_fixture(
    *,
    home: bool,
    opp_attack: float,
    opp_defence_leak: float,
    own_xga: float,
    league_xg: float,
    own_xg: float | None = None,
    team_tilt: float | None = None,
    position: str = "MID",
) -> dict[str, float]:
    att = float(opp_attack) if np.isfinite(opp_attack) else league_xg
    leak = float(opp_defence_leak) if np.isfinite(opp_defence_leak) else league_xg
    own = float(own_xga) if np.isfinite(own_xga) else league_xg
    oxg = float(own_xg) if own_xg is not None and np.isfinite(own_xg) else league_xg
    tilt = float(team_tilt) if team_tilt is not None and np.isfinite(team_tilt) else None
    match_xga = expected_goals_against(own, att, home=home, league_xg=league_xg)
    match_xg = expected_goals_for(oxg, leak, home=home, league_xg=league_xg)
    return {
        "home": 1.0 if home else 0.0,
        "opp_attack": att,
        "opp_defence_leak": leak,
        "own_xga": own,
        "own_xg": oxg,
        "match_xg": match_xg,
        "match_xga": match_xga,
        "team_xg_context": team_xg_context_score(oxg, own, league_xg, position=position, team_tilt=tilt),
        "p_cs": dixon_coles_p_cs(match_xga, match_xg),
    }


def isolate_first_event(fixtures_json: Any) -> str:
    """Keep only the earliest GW in the packed fixture window (for next-GW xPts)."""
    by = parse_fixtures_json(fixtures_json)
    if not by:
        return "{}"
    return isolate_event(fixtures_json, min(int(k) for k in by.keys()))


def blank_events(fixtures_json: Any, horizon: int) -> list[int]:
    """Event numbers in the scoring window with no matches (blanks)."""
    by = parse_fixtures_json(fixtures_json)
    if not by:
        return []
    events = sorted(int(k) for k in by.keys())
    if not events:
        return []
    min_e = min(events)
    blanks = []
    for i in range(horizon):
        ev = min_e + i
        if not by.get(str(ev)):
            blanks.append(ev)
    return blanks


def isolate_event(fixtures_json: Any, event: int) -> str:
    """Keep only one GW so a later fixture is not treated as 'next' after a blank."""
    by = parse_fixtures_json(fixtures_json)
    return json.dumps({str(int(event)): by.get(str(int(event)), [])})


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
