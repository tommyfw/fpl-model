from fpl_model.squad.wildcard import advise_wildcard, clip_windows, _ft_catch_share
from tests.test_transfers import _p, _squad_and_pool


def test_clip_windows_caps_and_includes_max():
    assert clip_windows(13) == [1, 3, 5, 8, 13]
    assert clip_windows(6) == [1, 3, 5, 6]
    assert clip_windows(1) == [1]
    assert clip_windows(34) == [1, 3, 5, 8, 13, 20, 26, 34]


def test_persist_weight_decays():
    from fpl_model.squad.wildcard import persist_weight, window_confidence

    assert persist_weight(0) == 1.0
    assert persist_weight(8) < persist_weight(3) < 1.0
    assert persist_weight(20) < persist_weight(13)
    assert persist_weight(40) == 0.40
    assert window_confidence(20) == "shape only"


def test_ft_catch_share():
    assert _ft_catch_share(0, 1, 5) == 1.0
    assert abs(_ft_catch_share(8, 1, 1) - 0.125) < 1e-9
    assert _ft_catch_share(4, 1, 4) == 1.0


def test_wildcard_already_used():
    squad, pool = _squad_and_pool()
    out = advise_wildcard(
        squad,
        pool,
        windows=[1, 5, 8],
        chips_used=["wildcard"],
        as_of_gw=5,
    )
    assert out["ok"]
    assert out["verdict"] == "used"
    assert out["windows"] == []


def _legalize(players):
    for i, p in enumerate(players):
        p["team_id"] = (i % 8) + 1
        p["team"] = f"T{p['team_id']}"


def _with_weeks(players, weekly):
    for p in players:
        p["xpts_gws"] = [float(weekly)] * 8
        p["xpts_next"] = float(weekly)
        p["availability"] = 1.0
        p["minutes_risk"] = {"level": "ok", "labels": []}


def test_wildcard_wait_when_current_fifteen_is_the_template():
    squad, _pool = _squad_and_pool()
    _legalize(squad)
    _with_weeks(squad, 2.5)
    out = advise_wildcard(squad, list(squad), windows=[1, 3, 5, 8], bank=0, budget=100, as_of_gw=5)
    assert out["ok"]
    assert out["n_out"] == 0
    assert out["verdict"] in {"wait", "consider"}
    assert out["windows"]
    assert [w["gws"] for w in out["windows"]] == [1, 3, 5, 8]
    assert out["delays"]
    assert any(d.get("best") for d in out["delays"])


def test_wildcard_play_when_squad_is_dead_wood():
    squad, pool = _squad_and_pool()
    _legalize(squad)
    _legalize(pool)
    _with_weeks(squad, 0.2)
    _with_weeks(pool, 0.2)
    extras = [
        _p(110 + i, f"WC{i}", pos, 8 + i, 20.0, 40)
        for i, pos in enumerate(["GK", "GK", "DEF", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "MID", "FWD", "FWD", "FWD"])
    ]
    for i, p in enumerate(extras):
        p["team_id"] = 20 + (i % 8)
        p["price"] = 4.5 if p["position"] == "GK" else 5.0
        p["xpts_gws"] = [4.0] * 8
        p["xpts_next"] = 4.0
        p["availability"] = 1.0
    pool = list(pool) + extras
    for p in squad:
        p["availability"] = 0.2
        p["minutes_risk"] = {"level": "risk", "labels": ["injured"]}
    out = advise_wildcard(squad, pool, windows=[1, 5, 8], bank=20, budget=120, as_of_gw=5, free_transfers=1)
    assert out["ok"]
    assert out["n_out"] >= 4
    long_row = next(w for w in out["windows"] if w["gws"] == 8)
    assert long_row["gain"] > 8
    assert out["verdict"] in {"play", "consider"}
    assert len(out["ids"]) == 15


def test_wildcard_lock_forces_and_exclude_drops():
    from fpl_model.squad.transfers import rebuild_on_chip

    squad, pool = _squad_and_pool()
    _legalize(squad)
    _legalize(pool)
    must = _p(999, "MustHave", "MID", 99, 4.5, 0.05)
    must["availability"] = 1.0
    star = max((p for p in pool if p["position"] != "GK"), key=lambda p: float(p.get("xpts") or 0))
    out = rebuild_on_chip(
        squad,
        list(pool) + [must],
        chip="wildcard",
        bank=20,
        budget=120,
        lock=[999],
        exclude=[star["fpl_id"]],
    )
    assert out["ok"]
    ids = {p["fpl_id"] for p in out["squad"]}
    assert 999 in ids
    assert star["fpl_id"] not in ids

    _with_weeks(squad, 0.4)
    _with_weeks(pool, 0.4)
    must["xpts_gws"] = [0.05] * 8
    must["xpts_next"] = 0.05
    must["minutes_risk"] = {"level": "ok", "labels": []}
    advised = advise_wildcard(
        squad,
        list(pool) + [must],
        windows=[1, 5],
        bank=20,
        budget=120,
        lock=[999],
        exclude=[star["fpl_id"]],
        as_of_gw=5,
    )
    assert advised["ok"]
    assert 999 in set(advised["ids"])
    assert star["fpl_id"] not in set(advised["ids"])


def test_wildcard_max_value_caps_spend():
    from fpl_model.squad.transfers import chip_rebuild_budget, rebuild_on_chip

    squad, pool = _squad_and_pool()
    spent = sum(p["price"] for p in squad)
    cap, purse = chip_rebuild_budget(100, spent, 20, max_value=spent)
    assert cap == spent
    assert purse == spent + 20
    open_ = rebuild_on_chip(squad, pool, chip="wildcard", bank=20, budget=100)
    assert open_["ok"]
    ids_open = {p["fpl_id"] for p in open_["squad"]}
    assert 102 in ids_open
    tight = rebuild_on_chip(squad, pool, chip="wildcard", bank=20, budget=100, max_value=85)
    assert tight["ok"]
    assert tight["spent"] <= 85 + 1e-9
    assert tight["max_value"] == 85
    assert 102 not in {p["fpl_id"] for p in tight["squad"]}
    advised = advise_wildcard(
        squad,
        pool,
        windows=[1, 5],
        bank=20,
        budget=100,
        max_value=85,
        as_of_gw=5,
    )
    assert advised["ok"]
    assert advised["spent"] <= 85 + 1e-9
    assert advised["max_value"] == 85


def test_ft_aware_score_front_weights_hold_window():
    from fpl_model.squad.wildcard import ft_aware_score

    spike = _p(1, "Spike", "MID", 1, 6.5, 6.0)
    spike["xpts_gws"] = [6.0, 6.0, 6.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    steady = _p(2, "Steady", "MID", 2, 6.5, 2.4)
    steady["xpts_gws"] = [2.4] * 8
    assert ft_aware_score(spike) > ft_aware_score(steady)
    premium = dict(spike)
    premium["price"] = 12.0
    assert ft_aware_score(spike) > ft_aware_score(premium)


def test_blend_stints_keeps_cheap_three_gw_name():
    from fpl_model.squad.wildcard import blend_stints, _weekly_xi

    core, _ = _squad_and_pool()
    _legalize(core)
    _with_weeks(core, 2.0)
    hold = [dict(p) for p in core]
    spike = dict(hold[7])
    spike["fpl_id"] = 500
    spike["web_name"] = "Spike"
    spike["price"] = 6.5
    spike["xpts_gws"] = [6.0, 6.0, 6.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    spike["team_id"] = 30
    displaced = hold[7]
    hold[7] = spike
    seed, stints = blend_stints(core, hold, cap=200.0, hold=3)
    ids = {p["fpl_id"] for p in seed}
    assert 500 in ids
    assert displaced["fpl_id"] not in ids
    assert stints
    assert stints[0]["fpl_id"] == 500
    assert "FT" not in (stints[0].get("reason") or "")
    blended = _weekly_xi(seed, 3)
    static = _weekly_xi(core, 3)
    assert sum(blended) > sum(static)


def test_blend_stints_rejects_premium_stint():
    from fpl_model.squad.wildcard import blend_stints

    core, _ = _squad_and_pool()
    _legalize(core)
    _with_weeks(core, 2.0)
    hold = [dict(p) for p in core]
    haal = dict(hold[12])
    haal["fpl_id"] = 501
    haal["web_name"] = "Pricey"
    haal["price"] = 14.0
    haal["xpts_gws"] = [8.0, 8.0, 8.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    hold[12] = haal
    seed, stints = blend_stints(core, hold, cap=200.0, hold=3)
    assert 501 not in {p["fpl_id"] for p in seed}
    assert stints == []


def test_advise_wildcard_has_no_future_ft_plan():
    squad, pool = _squad_and_pool()
    _legalize(squad)
    _with_weeks(squad, 2.0)
    out = advise_wildcard(squad, list(squad), windows=[1, 3, 5, 8], bank=0, budget=100, as_of_gw=5)
    assert out["ok"]
    assert "ft_plan" not in out
    assert "stints" in out
