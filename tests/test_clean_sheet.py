import pandas as pd

from fpl_model.scoring.clean_sheet import (
    ATTACK_EASE_EASY,
    ATTACK_EASE_HARD,
    ATTACK_FDR_CUTS,
    attack_fixture_ease,
    ease_band,
    fdr_from_cuts,
    fdr_like,
    fixture_p_cs,
    horizon_p_cs,
    isolate_event,
    pack_fixture,
    pack_match_for_team,
)


def test_own_defence_matters_not_just_opponent():
    leaky = fixture_p_cs(2.0, 1.3, home=True, league_xg=1.35)
    tight = fixture_p_cs(0.7, 1.3, home=True, league_xg=1.35)
    assert tight > leaky + 0.1


def test_expected_goals_for_drops_vs_elite_defence():
    from fpl_model.scoring.clean_sheet import expected_goals_for

    vs_avg = expected_goals_for(1.90, 1.35, home=False, league_xg=1.35)
    vs_ars = expected_goals_for(1.90, 0.76, home=False, league_xg=1.35)
    assert vs_ars < vs_avg * 0.70
    assert vs_ars < 1.2


def test_pack_fixture_includes_p_cs_and_own_xga():
    packed = pack_fixture(home=True, opp_attack=1.0, opp_defence_leak=1.4, own_xga=0.9, league_xg=1.35)
    assert 0.03 <= packed["p_cs"] <= 0.45
    assert packed["home"] == 1.0
    assert packed["own_xga"] == 0.9
    assert packed["own_xg"] == 1.35
    assert 5 <= packed["team_xg_context"] <= 100


def test_team_xg_context_rises_with_own_attack():
    weak = pack_fixture(
        home=True, opp_attack=1.2, opp_defence_leak=1.2, own_xga=1.2, league_xg=1.35, own_xg=0.8, position="MID"
    )
    strong = pack_fixture(
        home=True, opp_attack=1.2, opp_defence_leak=1.2, own_xga=1.2, league_xg=1.35, own_xg=1.9, position="MID"
    )
    assert strong["team_xg_context"] > weak["team_xg_context"] + 8


def test_horizon_p_cs_averages_matches_and_skips_blanks():
    # events 10 and 12 (blank 11); mean of two p_cs values
    fx = '{"10": [{"p_cs": 0.4}], "12": [{"p_cs": 0.2}]}'
    assert abs(horizon_p_cs(fx, horizon=3, fallback=0.1) - 0.3) < 1e-9
    assert horizon_p_cs("{}", 5, fallback=0.22) == 0.22


def test_pack_match_uses_venue_rates():
    lookup = pd.DataFrame(
        {
            "team_attack": [1.4, 1.1],
            "team_defence": [-1.1, -1.8],
            "team_attack_home": [1.5, 1.2],
            "team_attack_away": [1.0, 1.05],
            "team_defence_home": [-1.0, -1.6],
            "team_defence_away": [-1.2, -2.0],
        },
        index=[1, 2],
    )
    home = pack_match_for_team(home=True, team_id=1, opp_id=2, lookup=lookup, league_xg=1.35)
    assert abs(home["opp_attack"] - 1.05) < 1e-9
    assert abs(home["opp_defence_leak"] - 2.0) < 1e-9
    assert abs(home["own_xg"] - 1.5) < 1e-9
    assert abs(home["own_xga"] - 1.0) < 1e-9
    away = pack_match_for_team(home=False, team_id=1, opp_id=2, lookup=lookup, league_xg=1.35)
    assert abs(away["opp_attack"] - 1.2) < 1e-9
    assert abs(away["opp_defence_leak"] - 1.6) < 1e-9


def test_fdr_like_and_bands_are_league_wide():
    leaks = [1.0, 1.2, 1.4, 1.6, 2.0]
    assert fdr_like(2.0, leaks, higher_is_easier=True) == 1
    assert fdr_like(1.0, leaks, higher_is_easier=True) == 5
    lo, hi = 1.2, 1.6
    assert ease_band(2.0, lo, hi, higher_is_easier=True) == "easy"
    assert ease_band(1.0, lo, hi, higher_is_easier=True) == "hard"
    assert ease_band(1.4, lo, hi, higher_is_easier=True) == "mid"


def test_attack_ease_rewards_own_xg_and_absolute_bands():
    mid_leak = 1.35
    strong = attack_fixture_ease(mid_leak, 2.0, 1.55)
    weak = attack_fixture_ease(mid_leak, 1.1, 1.55)
    assert strong > weak + 0.2
    assert ease_band(strong, ATTACK_EASE_HARD, ATTACK_EASE_EASY) == "easy"
    assert ease_band(weak, ATTACK_EASE_HARD, ATTACK_EASE_EASY) in ("hard", "mid")
    assert fdr_from_cuts(strong, ATTACK_FDR_CUTS) <= 2
    assert fdr_from_cuts(weak, ATTACK_FDR_CUTS) >= 4


def test_residual_ha_tilt_is_small_when_rates_match():
    home = fixture_p_cs(1.35, 1.35, home=True, league_xg=1.35)
    away = fixture_p_cs(1.35, 1.35, home=False, league_xg=1.35)
    assert home > away
    assert (home - away) < 0.04


def test_isolate_event_does_not_promote_later_gw_after_blank():
    fx = '{"3": [{"p_cs": 0.4}], "4": [{"p_cs": 0.2}]}'
    only = isolate_event(fx, 2)
    assert '"2"' in only
    assert horizon_p_cs(only, horizon=1, fallback=0.1) == 0.1
    assert abs(horizon_p_cs(isolate_event(fx, 3), horizon=1, fallback=0.1) - 0.4) < 1e-9


def test_dixon_coles_rho_zero_is_poisson():
    from fpl_model.scoring.clean_sheet import LAMBDA_MIN, LAMBDA_MAX, PCS_MAX, PCS_MIN, dixon_coles_p_cs

    mu = 1.2
    pois = float(__import__("numpy").exp(-mu))
    got = dixon_coles_p_cs(mu, 1.4, rho=0.0)
    clipped = float(__import__("numpy").clip(pois, PCS_MIN, PCS_MAX))
    assert abs(got - clipped) < 1e-9
    with_rho = dixon_coles_p_cs(mu, 1.4, rho=0.10)
    assert got != with_rho
    assert LAMBDA_MIN <= mu <= LAMBDA_MAX


def test_pack_fixture_has_match_xg():
    packed = pack_fixture(home=True, opp_attack=1.0, opp_defence_leak=1.4, own_xga=0.9, league_xg=1.35, own_xg=1.8)
    assert packed["match_xg"] > packed["match_xga"]
    assert packed["p_cs"] == packed["p_cs"]

