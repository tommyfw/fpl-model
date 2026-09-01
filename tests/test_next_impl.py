import json

import pandas as pd

from fpl_model.config import load_settings, load_weights
from fpl_model.scoring.xpts import compute_xpts
from fpl_model.squad.chips import ft_plan, recommend_chips
from fpl_model.squad.optimize import pick_xi
from fpl_model.ingest.fpl_api import estimate_free_transfers, parse_entry_team
from fpl_model.scoring.availability import availability_from_news
from fpl_model.scoring.engine import score_feature_frame
from fpl_model.backtest.walkforward import calibration_table


def test_per_match_own_xg_not_mean_product():
    settings = load_settings()
    weights = load_weights()
    mixed = json.dumps(
        {
            "2": [{"home": 1, "opp_defence_leak": 0.70, "opp_attack": 1.2, "p_cs": 0.2, "own_xg": 1.4, "own_xga": 1.2}],
            "3": [{"home": 1, "opp_defence_leak": 2.00, "opp_attack": 1.2, "p_cs": 0.2, "own_xg": 1.4, "own_xga": 1.2}],
        }
    )
    avg = json.dumps(
        {
            "2": [{"home": 1, "opp_defence_leak": 1.35, "opp_attack": 1.2, "p_cs": 0.2, "own_xg": 1.4, "own_xga": 1.2}],
            "3": [{"home": 1, "opp_defence_leak": 1.35, "opp_attack": 1.2, "p_cs": 0.2, "own_xg": 1.4, "own_xga": 1.2}],
        }
    )
    df = pd.DataFrame(
        {
            "position": ["MID", "MID"],
            "start_pct": [1.0, 1.0],
            "start_pct_l5": [1.0, 1.0],
            "minutes_pct": [0.95, 0.95],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.3, 0.3],
            "npxa90": [0.2, 0.2],
            "saves90": [0.0, 0.0],
            "fixtures_json": [mixed, avg],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert float(xp.iloc[0]) != float(xp.iloc[1])


def test_gk_saves_rise_vs_strong_attack():
    settings = load_settings()
    weights = load_weights()
    easy = json.dumps({"2": [{"home": 1, "opp_defence_leak": 1.0, "opp_attack": 0.7, "p_cs": 0.2, "own_xg": 1.2, "own_xga": 1.2}]})
    hard = json.dumps({"2": [{"home": 1, "opp_defence_leak": 1.0, "opp_attack": 2.1, "p_cs": 0.2, "own_xg": 1.2, "own_xga": 1.2}]})
    df = pd.DataFrame(
        {
            "position": ["GK", "GK"],
            "team_id": [1, 2],
            "start_pct": [1.0, 1.0],
            "start_pct_l5": [1.0, 1.0],
            "minutes_pct": [0.95, 0.95],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.0, 0.0],
            "npxa90": [0.0, 0.0],
            "saves90": [3.0, 3.0],
            "defcon_hit_rate": [0.0, 0.0],
            "fixtures_json": [easy, hard],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert float(xp.iloc[1]) > float(xp.iloc[0])


def test_open_plus_sp_xg_beats_open_only():
    settings = load_settings()
    weights = load_weights()
    fx = json.dumps({"2": [{"home": 1, "opp_defence_leak": 1.2, "opp_attack": 1.2, "p_cs": 0.2, "own_xg": 1.2}]})
    df = pd.DataFrame(
        {
            "position": ["FWD", "FWD"],
            "start_pct": [1.0, 1.0],
            "start_pct_l5": [1.0, 1.0],
            "minutes_pct": [0.9, 0.9],
            "availability": [1.0, 1.0],
            "matches": [10, 10],
            "npxg90": [0.4, 0.4],
            "open_npxg90": [0.4, 0.3],
            "sp_xg90": [0.0, 0.1],
            "npxa90": [0.1, 0.1],
            "saves90": [0.0, 0.0],
            "fixtures_json": [fx, fx],
        }
    )
    xp = compute_xpts(df, weights, settings)
    assert float(xp.iloc[1]) > float(xp.iloc[0])


def test_last_red_almost_misses_this_gw():
    assert availability_from_news("a", last_red=True) <= 0.1


def test_captain_uses_xpts_next():
    squad = []
    for i in range(2):
        squad.append({"fpl_id": i + 1, "web_name": f"GK{i}", "position": "GK", "price": 4.5, "xpts": 10, "xpts_next": 2, "team_id": i + 1})
    for i in range(5):
        squad.append({"fpl_id": 10 + i, "web_name": f"D{i}", "position": "DEF", "price": 4.5, "xpts": 8, "xpts_next": 2, "team_id": 10 + i})
    for i in range(5):
        nxt = 9.0 if i == 0 else 3.0
        xp = 30.0 if i == 1 else 12.0
        squad.append({"fpl_id": 20 + i, "web_name": f"M{i}", "position": "MID", "price": 6, "xpts": xp, "xpts_next": nxt, "team_id": 10 + i})
    for i in range(3):
        squad.append({"fpl_id": 30 + i, "web_name": f"F{i}", "position": "FWD", "price": 7, "xpts": 10, "xpts_next": 3, "team_id": 10 + i})
    pack = pick_xi(squad, objective="xpts")
    assert pack["captain"]["web_name"] == "M0"


def test_ft_plan_banks_weak_week():
    p = ft_plan(free_transfers=1, best_single_net=0.4, best_double_gain=5.0)
    assert p["action"] == "bank"


def test_bb_when_bench_is_fat():
    squad = []
    for i in range(2):
        squad.append({"fpl_id": i + 1, "position": "GK", "web_name": f"G{i}", "price": 4.5, "xpts": 8, "xpts_next": 4, "team_id": i + 1})
    for i in range(5):
        squad.append({"fpl_id": 10 + i, "position": "DEF", "web_name": f"D{i}", "price": 4.5, "xpts": 8, "xpts_next": 4, "team_id": 10 + i})
    for i in range(5):
        squad.append({"fpl_id": 20 + i, "position": "MID", "web_name": f"M{i}", "price": 6, "xpts": 8, "xpts_next": 4, "team_id": 10 + i})
    for i in range(3):
        squad.append({"fpl_id": 30 + i, "position": "FWD", "web_name": f"F{i}", "price": 7, "xpts": 8, "xpts_next": 4, "team_id": 10 + i})
    chips = recommend_chips(squad)
    assert any(c["chip"] == "bboost" for c in chips)


def test_estimate_fts_after_unused_week():
    hist = {"past": [{"event": 1, "event_transfers": 0}], "chips": []}
    assert estimate_free_transfers(hist) == 2


def test_parse_entry_team():
    entry = {"id": 9, "name": "Test", "last_deadline_bank": 12, "player_first_name": "A", "player_last_name": "B"}
    history = {"past": [], "chips": [{"name": "wildcard", "event": 1}]}
    picks = {"picks": [{"element": i} for i in range(1, 16)]}
    out = parse_entry_team(entry, history, picks)
    assert out["ok"]
    assert out["bank"] == 1.2
    assert out["chips_used"] == ["wildcard"]


def test_mid_subpos_splits_percentiles():
    settings = load_settings()
    weights = load_weights()
    n = 24
    pos = ["MID"] * n
    pitch = [80] * 12 + [40] * 12
    npxgi = [0.5] * 6 + [0.1] * 6 + [0.5] * 6 + [0.1] * 6
    df = pd.DataFrame(
        {
            "position": pos,
            "team_id": list(range(n)),
            "price": [6.5] * n,
            "start_pct": [0.9] * n,
            "minutes_pct": [0.9] * n,
            "start_pct_l5": [0.9] * n,
            "availability": [1.0] * n,
            "matches": [10] * n,
            "pitch_attack": pitch,
            "pitch_central": [70] * n,
            "npxgi90": npxgi,
            "npxg90": [0.2] * n,
            "npxa90": [0.2] * n,
            "defcon_hit_rate": [0.2] * n,
            "points90": [4.0] * n,
            "goals90": [0.15] * n,
            "assists90": [0.15] * n,
            "bonus90": [0.2] * n,
            "bps90": [18.0] * n,
            "team_attack": [1.4] * n,
            "team_defence": [-1.2] * n,
            "understat_matched": [1] * n,
            "fixtures_json": ["{}"] * n,
            "minutes_pct_l5": [0.9] * n,
            "role_change_raw": [0.0] * n,
            "price_change": [0.0] * n,
            "team_attack_trend": [0.0] * n,
            "team_defence_trend": [0.0] * n,
        }
    )
    scored = score_feature_frame(df, weights, settings)
    am_high = scored.loc[(scored["pitch_attack"] >= 62) & (scored["npxgi90"] > 0.4), "npxgi90_s"].mean()
    am_low = scored.loc[(scored["pitch_attack"] >= 62) & (scored["npxgi90"] < 0.2), "npxgi90_s"].mean()
    assert am_high > am_low


def test_calibration_table_has_bins():
    pred = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 8)
    actual = pred + 1
    tab = calibration_table(pred, actual, n_bins=5)
    assert not tab.empty
    assert "pred_mean" in tab.columns
