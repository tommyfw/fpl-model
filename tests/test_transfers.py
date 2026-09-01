from fpl_model.squad.transfers import suggest_transfers


def _p(fpl_id, name, pos, team_id, price, xpts, overall=50):
    return {
        "fpl_id": fpl_id,
        "web_name": name,
        "position": pos,
        "team_id": team_id,
        "team": f"T{team_id}",
        "price": price,
        "xpts": xpts,
        "overall": overall,
        "action": "HOLD",
        "status": "a",
    }


def _squad_and_pool():
    squad = (
        [_p(1, "GK1", "GK", 1, 4.5, 10), _p(2, "GK2", "GK", 2, 4.0, 8)]
        + [_p(10 + i, f"D{i}", "DEF", i, 4.5, 12 - i * 0.5) for i in range(1, 6)]
        + [_p(20 + i, f"M{i}", "MID", i, 6.0, 15 - i) for i in range(1, 6)]
        + [_p(30 + i, f"F{i}", "FWD", i, 7.0, 14 - i) for i in range(1, 4)]
    )
    pool = list(squad) + [
        _p(100, "UpgradeMid", "MID", 8, 7.0, 25, 80),
        _p(101, "UpgradeFwd", "FWD", 9, 8.0, 22, 78),
        _p(102, "TooPricey", "MID", 10, 12.0, 40, 90),
    ]
    return squad, pool


def test_suggests_upgrade_within_bank():
    squad, pool = _squad_and_pool()
    out = suggest_transfers(squad, pool, bank=2.0, free_transfers=1, objective="xpts")
    assert out["ok"]
    assert out["suggestions"]
    top = out["suggestions"][0]
    assert top["moves"][0]["in"]["web_name"] == "UpgradeMid"
    assert top["net_gain"] > 0


def test_blocks_unaffordable():
    squad, pool = _squad_and_pool()
    out = suggest_transfers(squad, pool, bank=0.0, free_transfers=1, objective="xpts")
    names = [s["moves"][0]["in"]["web_name"] for s in out["suggestions"] if s["transfers"] == 1]
    assert "TooPricey" not in names


def test_hit_needs_clear_edge():
    squad, pool = _squad_and_pool()
    # Tiny upgrade vs worst mid (10 xPts): gain 1 → net −3 after a hit
    pool = list(pool) + [_p(200, "TinyUp", "MID", 11, 6.5, 11.0, 60)]
    out = suggest_transfers(squad, pool, bank=1.0, free_transfers=0, want_transfers=1, objective="xpts")
    tiny = [s for s in out["suggestions"] if s["moves"][0]["in"]["web_name"] == "TinyUp"]
    assert tiny
    assert tiny[0]["hits"] == 1
    assert tiny[0]["net_gain"] < 0
    assert tiny[0]["hit_caution"]


def test_want_two_shows_hit_with_one_ft():
    squad, pool = _squad_and_pool()
    out = suggest_transfers(squad, pool, bank=3.0, free_transfers=1, want_transfers=2, objective="xpts")
    doubles = [s for s in out["suggestions"] if s["transfers"] == 2]
    assert doubles
    top_d = doubles[0]
    assert top_d["hits"] == 1
    names = {m["in"]["web_name"] for s in doubles for m in s["moves"]}
    assert "UpgradeMid" in names
    assert "UpgradeFwd" in names


def test_committed_manual_is_scored():
    squad, pool = _squad_and_pool()
    out = suggest_transfers(
        squad,
        pool,
        bank=2.0,
        free_transfers=1,
        want_transfers=2,
        objective="xpts",
        committed=[(25, 100)],
    )
    assert out["ok"]
    assert out["committed"]
    assert out["committed"]["moves"][0]["in"]["web_name"] == "UpgradeMid"
    assert out["committed"]["gain"] > 0
    assert out["committed"]["hits"] == 0
    extras = [m["in"]["web_name"] for s in out["suggestions"] for m in s["moves"]]
    assert "UpgradeMid" not in extras
    assert out["suggestions"]
    assert "combined" in out["suggestions"][0]
    assert out["suggestions"][0]["combined"]["transfers"] >= 2


def test_committed_can_overspend_bank():
    squad, pool = _squad_and_pool()
    out = suggest_transfers(
        squad,
        pool,
        bank=0.0,
        free_transfers=1,
        want_transfers=1,
        objective="xpts",
        committed=[(25, 100)],
        keep=[100],
    )
    assert out["ok"]
    assert out["committed"]["moves"][0]["in"]["web_name"] == "UpgradeMid"
    assert out["bank_now"] < 0
    assert out["restore"]["needed"]
    assert not out["restore"]["ok"]
    assert not out["suggestions"]


def test_suggests_restore_after_manual_overspend():
    squad, pool = _squad_and_pool()
    pool = list(pool) + [_p(203, "CheapMid", "MID", 14, 4.5, 8.0, 40)]
    out = suggest_transfers(
        squad,
        pool,
        bank=0.0,
        free_transfers=1,
        want_transfers=2,
        objective="xpts",
        committed=[(25, 100)],
    )
    assert out["ok"]
    assert out["bank_now"] < 0
    assert out["restore"]["needed"]
    assert out["restore"]["ok"]
    assert out["suggestions"]
    top = out["suggestions"][0]
    assert top["legal_after"]
    assert top["bank_after"] >= -1e-9
    assert top["spent_after"] <= out["budget"] + 1e-9


def test_restore_prefers_higher_xpts_not_cheapest():
    squad, pool = _squad_and_pool()
    pool = list(pool) + [
        _p(203, "CheapMid", "MID", 14, 4.0, 6.0, 30),
        _p(204, "Slater", "MID", 15, 5.0, 14.0, 70),
    ]
    out = suggest_transfers(
        squad,
        pool,
        bank=0.0,
        free_transfers=1,
        want_transfers=1,
        objective="xpts",
        committed=[(25, 100)],
    )
    assert out["restore"]["ok"]
    ins = [m["in"]["web_name"] for m in out["suggestions"][0]["moves"]]
    assert "Slater" in ins
    assert "CheapMid" not in ins


def test_restore_two_downgrades_while_premium_locked():
    squad, pool = _squad_and_pool()
    for p in squad:
        if p["fpl_id"] == 33:
            p["price"] = 14.0
            p["web_name"] = "Haaland"
        if p["fpl_id"] in (21, 22):
            p["price"] = 9.0
    pool = list(pool) + [
        _p(210, "FodderA", "MID", 15, 4.0, 5.0, 30),
        _p(211, "FodderB", "MID", 16, 4.0, 5.0, 30),
        _p(212, "Wilson", "MID", 17, 4.5, 13.0, 65),
        _p(213, "Slater", "MID", 18, 4.5, 12.5, 64),
    ]
    spent = sum(p["price"] for p in squad)
    out = suggest_transfers(
        squad,
        pool,
        bank=-9.0,
        budget=spent,
        free_transfers=1,
        want_transfers=2,
        objective="xpts",
        keep=[33],
    )
    assert out["ok"]
    assert out["restore"]["needed"]
    assert out["restore"]["ok"]
    names_out = {m["out"]["web_name"] for s in out["suggestions"] for m in s["moves"]}
    assert "Haaland" not in names_out
    top_two = next(s for s in out["suggestions"] if s["transfers"] == 2 and s["legal_after"])
    ins = {m["in"]["web_name"] for m in top_two["moves"]}
    assert "FodderA" not in ins and "FodderB" not in ins
    assert ins & {"Wilson", "Slater"}


def test_suggests_include_gain_next():
    squad, pool = _squad_and_pool()
    for p in pool:
        p["xpts_next"] = float(p["xpts"]) / 5.0
    out = suggest_transfers(squad, pool, bank=2.0, free_transfers=1, objective="xpts")
    assert out["ok"]
    assert "gain_next" in out["suggestions"][0]


def test_exclude_blocks_transfer_in():
    squad, pool = _squad_and_pool()
    out = suggest_transfers(squad, pool, bank=2.0, free_transfers=1, objective="xpts", exclude=[100])
    assert out["ok"]
    names = [m["in"]["web_name"] for s in out["suggestions"] for m in s["moves"]]
    assert "UpgradeMid" not in names


def test_overspend_prefers_moves_that_restore_cap():
    squad, pool = _squad_and_pool()
    # Swap a 7.0 FWD for a 14.0 Haaland so the 15 is ~£7 over a £82-style...
    # Actual test squad is ~£82; bump one FWD and set a tight cap.
    for p in squad:
        if p["fpl_id"] == 33:
            p["price"] = 14.0
            p["web_name"] = "Haaland"
            p["xpts"] = 22.0
    pool = list(pool) + [
        _p(201, "CheapFwd", "FWD", 12, 5.5, 8.0, 40),
        _p(202, "OkayFwd", "FWD", 13, 6.5, 11.0, 55),
    ]
    spent = sum(p["price"] for p in squad)
    cap = spent - 6.0
    out = suggest_transfers(squad, pool, bank=0.0, budget=cap, free_transfers=2, objective="xpts")
    assert out["ok"]
    assert out["overspend"] >= 6.0
    assert out["suggestions"]
    top = out["suggestions"][0]
    assert top["fixes_budget"]
    assert top["spent_after"] <= cap + 1e-9
    names_out = [m["out"]["web_name"] for m in top["moves"]]
    assert "Haaland" in names_out


def test_upgrade_still_blocked_when_it_breaks_cap():
    squad, pool = _squad_and_pool()
    spent = sum(p["price"] for p in squad)
    out = suggest_transfers(squad, pool, bank=20.0, budget=spent, free_transfers=1, objective="xpts")
    names = [s["moves"][0]["in"]["web_name"] for s in out["suggestions"] if s["transfers"] == 1]
    assert "TooPricey" not in names
    assert "UpgradeMid" not in names


def test_exclude_team_blocks_transfer_in():
    squad, pool = _squad_and_pool()
    out = suggest_transfers(
        squad, pool, bank=2.0, free_transfers=1, objective="xpts", exclude_teams=[8]
    )
    names = [m["in"]["web_name"] for s in out["suggestions"] for m in s["moves"]]
    assert "UpgradeMid" not in names


def test_keep_blocks_transfer_out():
    squad, pool = _squad_and_pool()
    worst_mid = next(p for p in squad if p["web_name"] == "M5")
    out = suggest_transfers(
        squad, pool, bank=2.0, free_transfers=1, objective="xpts", keep=[worst_mid["fpl_id"]]
    )
    names_out = [m["out"]["web_name"] for s in out["suggestions"] for m in s["moves"]]
    assert "M5" not in names_out
    names_in = [m["in"]["web_name"] for s in out["suggestions"] for m in s["moves"]]
    assert "UpgradeMid" in names_in


def test_keep_all_mids_blocks_mid_upgrades():
    squad, pool = _squad_and_pool()
    keep = [p["fpl_id"] for p in squad if p["position"] == "MID"]
    out = suggest_transfers(squad, pool, bank=2.0, free_transfers=1, objective="xpts", keep=keep)
    names_in = [m["in"]["web_name"] for s in out["suggestions"] for m in s["moves"]]
    assert "UpgradeMid" not in names_in
    assert "UpgradeFwd" in names_in


def test_suggestion_has_horizon_and_xi_after():
    squad, pool = _squad_and_pool()
    for p in squad + pool:
        p["xpts_gws"] = [float(p["xpts"]) / 5.0] * 5
    out = suggest_transfers(squad, pool, bank=2.0, free_transfers=1, objective="xpts")
    assert out["suggestions"]
    top = out["suggestions"][0]
    assert top["xi_after"]
    assert top["xi_after"]["formation"]
    assert "gain_horizon" in top
    assert "xi_gain" in top
