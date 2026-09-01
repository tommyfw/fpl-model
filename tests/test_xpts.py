import json

import pandas as pd

from fpl_model.config import load_settings, load_weights
from fpl_model.scoring.clean_sheet import fixture_p_cs
from fpl_model.scoring.xpts import compute_xpts
from fpl_model.squad.optimize import build_squad
from fpl_model.backtest.squad import evaluate_window, xi_actual_points


def _fx(n_matches: int, leak=1.2, att=1.2, start_gw=2, p_cs=0.3, home=1.0, own_xg=1.2, own_xga=1.2):
    by = {}
    for i in range(n_matches):
        by[str(start_gw + i)] = [
            {
                "home": home,
                "opp_defence_leak": leak,
                "opp_attack": att,
                "p_cs": p_cs,
                "own_xga": own_xga,
                "own_xg": own_xg,
            }
        ]
    return json.dumps(by)


def test_blank_is_zero_and_dgw_beats_single():
    settings = load_settings()
    weights = load_weights()
    base = {
        "position": ["MID", "MID", "MID"],
        "start_pct": [1.0, 1.0, 1.0],
        "start_pct_l5": [1.0, 1.0, 1.0],
        "minutes_pct": [0.95, 0.95, 0.95],
        "availability": [1.0, 1.0, 1.0],
        "matches": [10, 10, 10],
        "npxg90": [0.4, 0.4, 0.4],
        "npxa90": [0.2, 0.4, 0.2],
        "npxgi90_l5": [0.6, 0.6, 0.6],
        "p_clean_sheet": [0.2, 0.2, 0.2],
        "defcon_hit_rate": [0.1, 0.1, 0.1],
        "saves90": [0.0, 0.0, 0.0],
        "fixtures_json": ["{}", _fx(1), _fx(2)],
    }
    xp = compute_xpts(pd.DataFrame(base), weights, settings)
    assert xp.iloc[0] == 0.0
    assert xp.iloc[2] > xp.iloc[1] > 0


def test_backup_gk_far_below_starter():
    settings = load_settings()
    weights = load_weights()
    df = pd.DataFrame(
        {
            "position": ["GK", "GK"],
            "team_id": [1, 1],
            "start_pct": [0.95, 0.08],
            "start_pct_l5": [1.0, 0.0],
            "minutes_pct": [0.95, 0.05],
            "availability": [1.0, 1.0],
            "matches": [8, 8],
            "npxg90": [0.0, 0.0],
            "npxa90": [0.0, 0.0],
            "p_clean_sheet": [0.35, 0.35],
            "defcon_hit_rate": [0.0, 0.0],
            "saves90": [3.0, 3.0],
            "fixtures_json": [_fx(5), _fx(5)],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert xp.iloc[0] > 8
    assert xp.iloc[1] < 2


def test_fixture_tilt_is_mild():
    settings = load_settings()
    weights = load_weights()
    df = pd.DataFrame(
        {
            "position": ["FWD", "FWD"],
            "start_pct": [1.0, 1.0],
            "start_pct_l5": [1.0, 1.0],
            "minutes_pct": [0.9, 0.9],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.5, 0.5],
            "npxa90": [0.1, 0.1],
            "p_clean_sheet": [0.2, 0.2],
            "defcon_hit_rate": [0.0, 0.0],
            "saves90": [0.0, 0.0],
            "fixtures_json": [_fx(5, leak=2.0, att=0.8, p_cs=0.2), _fx(5, leak=0.7, att=2.0, p_cs=0.2)],
        }
    )
    xp = compute_xpts(df, weights, settings)
    ratio = float(xp.iloc[0] / xp.iloc[1])
    assert 1.12 < ratio < 2.2


def test_away_vs_elite_defence_is_not_a_goal():
    """João Pedro vs Arsenal should not look like ~1 expected goal (old clip was ±6%)."""
    settings = load_settings()
    weights = load_weights()
    arsenal = _fx(1, leak=0.76, att=2.0, p_cs=0.14, own_xg=1.90, home=0.0)
    df = pd.DataFrame(
        {
            "position": ["FWD"],
            "start_pct": [0.95],
            "start_pct_l5": [1.0],
            "minutes_pct": [0.90],
            "availability": [1.0],
            "matches": [2],
            "npxg90": [0.62],
            "npxa90": [0.31],
            "saves90": [0.0],
            "defcon_hit_rate": [0.0],
            "bonus90": [1.3],
            "fixtures_json": [arsenal],
        }
    )
    xp = float(compute_xpts(df, weights, settings).iloc[0])
    assert xp < 5.0


def test_pen_taker_lifts_fwd_xpts():
    settings = load_settings()
    weights = load_weights()
    fx = _fx(1, leak=1.35, att=1.2, p_cs=0.2)
    cols = dict(
        position=["FWD", "FWD"],
        start_pct=[1.0, 1.0],
        start_pct_l5=[1.0, 1.0],
        minutes_pct=[0.95, 0.95],
        availability=[1.0, 1.0],
        matches=[10, 10],
        npxg90=[0.45, 0.45],
        npxa90=[0.15, 0.15],
        pen_taker=[1.0, 0.0],
        saves90=[0.0, 0.0],
        defcon_hit_rate=[0.0, 0.0],
        fixtures_json=[fx, fx],
    )
    xp = compute_xpts(pd.DataFrame(cols), weights, settings)
    assert float(xp.iloc[0]) > float(xp.iloc[1]) + 0.15


def test_fwd_attack_more_fixture_sensitive_than_cb():
    settings = load_settings()
    weights = load_weights()
    easy = _fx(1, leak=2.0, att=0.9, p_cs=0.22)
    hard = _fx(1, leak=0.7, att=2.0, p_cs=0.22)
    base = dict(
        start_pct=[1.0, 1.0],
        start_pct_l5=[1.0, 1.0],
        minutes_pct=[0.95, 0.95],
        availability=[1.0, 1.0],
        matches=[10, 10],
        npxg90=[0.4, 0.4],
        npxa90=[0.1, 0.1],
        saves90=[0.0, 0.0],
        defcon_hit_rate=[0.0, 0.0],
        fixtures_json=[easy, hard],
    )
    fwd = compute_xpts(pd.DataFrame({**base, "position": ["FWD", "FWD"]}), weights, settings)
    cb = compute_xpts(
        pd.DataFrame({**base, "position": ["DEF", "DEF"], "def_fb_share": [0.0, 0.0]}),
        weights,
        settings,
    )
    fwd_ratio = float(fwd.iloc[0] / fwd.iloc[1])
    cb_ratio = float(cb.iloc[0] / cb.iloc[1])
    assert fwd_ratio > cb_ratio + 0.08


def test_early_season_cs_keeper_beats_volume_saves():
    """Two-game save volume should not outrank a high-CS #1 (Raya vs Petrović)."""
    settings = load_settings()
    weights = load_weights()
    elite = _fx(5, leak=1.1, att=1.3, p_cs=0.42)
    leaky = _fx(5, leak=1.7, att=2.0, p_cs=0.12)
    df = pd.DataFrame(
        {
            "position": ["GK", "GK"],
            "start_pct": [1.0, 1.0],
            "start_pct_l5": [1.0, 1.0],
            "minutes_pct": [0.95, 0.95],
            "availability": [1.0, 1.0],
            "matches": [2, 2],
            "npxg90": [0.0, 0.0],
            "npxa90": [0.0, 0.0],
            "saves90": [1.2, 3.4],
            "defcon_hit_rate": [0.0, 0.0],
            "fixtures_json": [elite, leaky],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert float(xp.iloc[0]) > float(xp.iloc[1])


def test_home_vs_weak_attack_beats_away_vs_strong():
    easy = fixture_p_cs(0.85, 0.9, home=True, league_xg=1.35)
    hard = fixture_p_cs(0.85, 2.1, home=False, league_xg=1.35)
    assert easy > hard + 0.12
    assert easy <= 0.45
    assert hard >= 0.03


def test_def_xpts_uses_fixture_p_cs_not_season_rate():
    settings = load_settings()
    weights = load_weights()
    df = pd.DataFrame(
        {
            "position": ["DEF", "DEF"],
            "start_pct": [1.0, 1.0],
            "start_pct_l5": [1.0, 1.0],
            "minutes_pct": [0.95, 0.95],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.05, 0.05],
            "npxa90": [0.05, 0.05],
            "p_clean_sheet": [0.40, 0.40],
            "defcon_hit_rate": [0.2, 0.2],
            "saves90": [0.0, 0.0],
            "fixtures_json": [_fx(5, p_cs=0.48), _fx(5, p_cs=0.12)],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert xp.iloc[0] > xp.iloc[1] + 3.0


def test_bench_minutes_do_not_count_as_starts():
    settings = load_settings()
    weights = load_weights()
    df = pd.DataFrame(
        {
            "position": ["DEF", "DEF"],
            "start_pct": [0.95, 0.30],
            "start_pct_l5": [1.0, 0.20],
            "minutes_pct": [0.95, 0.80],
            "minutes_pct_l5": [0.95, 0.75],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.05, 0.05],
            "npxa90": [0.05, 0.05],
            "defcon_hit_rate": [0.3, 0.3],
            "saves90": [0.0, 0.0],
            "fixtures_json": [_fx(5, p_cs=0.35), _fx(5, p_cs=0.35)],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert xp.iloc[0] > xp.iloc[1] * 1.8


def test_rotation_caps_def_cs_ev():
    settings = load_settings()
    weights = load_weights()
    df = pd.DataFrame(
        {
            "position": ["DEF", "DEF"],
            "start_pct": [0.95, 0.90],
            "start_pct_l5": [1.0, 0.80],
            "minutes_pct": [0.95, 0.85],
            "rotation_start_pct": [1.0, 0.15],
            "slot_security": [1.0, 0.5],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.05, 0.05],
            "npxa90": [0.05, 0.05],
            "defcon_hit_rate": [0.4, 0.4],
            "saves90": [0.0, 0.0],
            "fixtures_json": [_fx(5, p_cs=0.4), _fx(5, p_cs=0.4)],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert xp.iloc[0] > xp.iloc[1] + 8.0


def test_haaland_quirk_uses_minutes_when_starts_are_zero():
    settings = load_settings()
    weights = load_weights()
    df = pd.DataFrame(
        {
            "position": ["FWD"],
            "start_pct": [0.52],
            "start_pct_l5": [0.0],
            "minutes_pct": [0.95],
            "minutes_pct_l5": [1.0],
            "rotation_start_pct": [0.0],
            "rival_games": [0],
            "availability": [1.0],
            "matches": [1],
            "npxg90": [0.7],
            "npxa90": [0.1],
            "defcon_hit_rate": [0.0],
            "saves90": [0.0],
            "fixtures_json": [_fx(5)],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert float(xp.iloc[0]) > 20


def test_xi_actual_doubles_captain():
    xi = [{"fpl_id": 1}, {"fpl_id": 2}]
    assert xi_actual_points(xi, {"fpl_id": 1}, {1: 10.0, 2: 5.0}) == 25.0


def _p(fid, name, pos, price, xpts, overall, team_id, selected, status="a"):
    return {
        "fpl_id": fid,
        "web_name": name,
        "position": pos,
        "price": price,
        "xpts": xpts,
        "overall": overall,
        "team_id": team_id,
        "status": status,
        "last_selected": selected,
    }


def test_model_xi_beats_template_when_xpts_are_true():
    players = [
        _p(1, "Raya", "GK", 5.5, 14.0, 75, 1, 100),
        _p(2, "Pope", "GK", 5.0, 12.0, 70, 2, 4_000_000),
        _p(3, "CheapGK", "GK", 4.0, 1.0, 40, 3, 2_000_000),
    ]
    fid = 10
    for team in range(10, 18):
        for i in range(3):
            players.append(_p(fid, f"DEF{fid}", "DEF", 4.5 + i, 10.0 + i, 55 + i, team, 50_000 * (i + 1)))
            fid += 1
        for i in range(3):
            players.append(_p(fid, f"MID{fid}", "MID", 6.0 + i * 0.5, 12.0 + i * 3, 60 + i * 5, team, 80_000 * (3 - i)))
            fid += 1
        players.append(_p(fid, f"FWD{fid}", "FWD", 7.0, 11.0, 62, team, 200_000))
        fid += 1
        players.append(_p(fid, f"Haal{fid}", "FWD", 14.0, 22.0, 85, team, 10_000))
        fid += 1
    scored = pd.DataFrame(players)
    # Actuals equal predicted xPts — model should beat ownership template
    actual = {int(p["fpl_id"]): float(p["xpts"]) for p in players}
    pack = evaluate_window(scored, actual, polish=False)
    assert pack["model_ok"] and pack["template_ok"]
    assert pack["delta"] > 0


def test_cs_ev_needs_sixty_minutes():
    settings = load_settings()
    weights = load_weights()
    fx = _fx(5, p_cs=0.40)
    df = pd.DataFrame(
        {
            "position": ["DEF", "DEF"],
            "start_pct": [1.0, 0.95],
            "start_pct_l5": [1.0, 0.95],
            "minutes_pct": [0.95, 0.40],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.02, 0.02],
            "npxa90": [0.02, 0.02],
            "defcon_hit_rate": [0.2, 0.2],
            "saves90": [0.0, 0.0],
            "fixtures_json": [fx, fx],
        }
    )
    xp, parts = compute_xpts(df, weights, settings, return_parts=True)
    assert float(parts.iloc[0]["cs"]) > float(parts.iloc[1]["cs"]) * 1.5
    assert float(xp.iloc[0]) > float(xp.iloc[1])


def test_mid_creator_xpts_beats_defcon_merchant():
    settings = load_settings()
    weights = load_weights()
    fx = _fx(5)
    df = pd.DataFrame(
        {
            "position": ["MID", "MID"],
            "start_pct": [1.0, 1.0],
            "start_pct_l5": [1.0, 1.0],
            "minutes_pct": [0.95, 0.95],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.42, 0.08],
            "npxa90": [0.32, 0.08],
            "npxgi90_l5": [0.74, 0.16],
            "p_clean_sheet": [0.2, 0.2],
            "defcon_hit_rate": [0.15, 0.90],
            "saves90": [0.0, 0.0],
            "fixtures_json": [fx, fx],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert float(xp.iloc[0]) > float(xp.iloc[1])
