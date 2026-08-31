from fpl_model.scoring.clean_sheet import fixture_p_cs, horizon_p_cs, pack_fixture


def test_own_defence_matters_not_just_opponent():
    leaky = fixture_p_cs(2.0, 1.3, home=True, league_xg=1.35)
    tight = fixture_p_cs(0.7, 1.3, home=True, league_xg=1.35)
    assert tight > leaky + 0.1


def test_pack_fixture_includes_p_cs_and_own_xga():
    packed = pack_fixture(home=True, opp_attack=1.0, opp_defence_leak=1.4, own_xga=0.9, league_xg=1.35)
    assert 0.03 <= packed["p_cs"] <= 0.45
    assert packed["home"] == 1.0
    assert packed["own_xga"] == 0.9


def test_horizon_p_cs_averages_matches_and_skips_blanks():
    # events 10 and 12 (blank 11); mean of two p_cs values
    fx = '{"10": [{"p_cs": 0.4}], "12": [{"p_cs": 0.2}]}'
    assert abs(horizon_p_cs(fx, horizon=3, fallback=0.1) - 0.3) < 1e-9
    assert horizon_p_cs("{}", 5, fallback=0.22) == 0.22
