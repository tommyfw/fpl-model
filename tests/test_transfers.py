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
