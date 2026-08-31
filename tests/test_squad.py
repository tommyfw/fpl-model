from fpl_model.squad.optimize import build_squad, pick_xi


def _p(fid, name, pos, price, xpts, overall, team_id, status="a"):
    return {
        "fpl_id": fid,
        "web_name": name,
        "position": pos,
        "price": price,
        "xpts": xpts,
        "overall": overall,
        "team_id": team_id,
        "status": status,
    }


def _pool():
    """Expensive high-xPts outfielders plus two 4.0 GKs with ~0 xPts and two real GKs."""
    players = [
        _p(1, "Button", "GK", 4.0, 0.0, 40, 1),
        _p(2, "Bentley", "GK", 4.0, 0.2, 41, 2),
        _p(3, "Raya", "GK", 5.5, 12.0, 75, 3),
        _p(4, "Alisson", "GK", 5.5, 11.0, 74, 4),
        _p(5, "Pope", "GK", 5.0, 10.0, 70, 5),
    ]
    # 8 teams of outfielders so 3-per-club never blocks a legal 15
    fid = 10
    for team in range(10, 18):
        for i in range(3):
            players.append(_p(fid, f"DEF{fid}", "DEF", 4.5 + i, 4.0 + i, 55 + i, team))
            fid += 1
        for i in range(3):
            players.append(_p(fid, f"MID{fid}", "MID", 6.0 + i * 0.5, 8.0 + i * 2, 60 + i * 5, team))
            fid += 1
        players.append(_p(fid, f"FWD{fid}", "FWD", 7.0, 9.0, 62, team))
        fid += 1
        players.append(_p(fid, f"FWD{fid}", "FWD", 11.0, 18.0, 80, team))
        fid += 1
    return players


def test_xpts_xi_starts_a_real_keeper_not_the_4m_zeros():
    out = build_squad(_pool(), budget=100.0, objective="xpts")
    assert out["ok"]
    xi_gks = [p for p in out["xi"] if p["position"] == "GK"]
    assert len(xi_gks) == 1
    assert xi_gks[0]["web_name"] not in {"Button", "Bentley"}
    assert xi_gks[0]["xpts"] >= 10.0


def test_xi_score_beats_naive_star_then_cheap_gk_fill():
    out = build_squad(_pool(), budget=100.0, objective="xpts")
    xi_gks = [p for p in out["xi"] if p["position"] == "GK"]
    # A feasible squad can start Raya/Alisson/Pope; maximizing XI should not leave 0 xPts in goal
    assert sum(p["xpts"] for p in out["xi"]) > sum(p["xpts"] for p in out["bench"])
    assert xi_gks[0]["web_name"] in {"Raya", "Alisson", "Pope"}


def test_overall_objective_also_picks_a_playing_keeper():
    out = build_squad(_pool(), budget=100.0, objective="overall")
    assert out["ok"]
    xi_gk = next(p for p in out["xi"] if p["position"] == "GK")
    assert xi_gk["overall"] >= 70


def test_pick_xi_benches_blank_week_midfielder():
    squad = [
        _p(1, "GK1", "GK", 5.0, 4.0, 70, 1),
        _p(2, "GK2", "GK", 4.0, 1.0, 40, 2),
    ]
    for i in range(5):
        squad.append(_p(10 + i, f"DEF{i}", "DEF", 4.5, 3.0, 55, 10 + i))
    for i in range(4):
        squad.append(_p(20 + i, f"MID{i}", "MID", 6.0, 8.0, 70, 10 + i))
    squad.append(_p(29, "BlankMID", "MID", 6.0, 20.0, 80, 17))  # strong 5-GW, blank this GW
    for i in range(3):
        squad.append(_p(30 + i, f"FWD{i}", "FWD", 7.0, 6.0, 65, 10 + i))
    for p in squad:
        p["xpts_gw"] = 0.0 if p["web_name"] == "BlankMID" else 5.0
    pack = pick_xi(squad, objective="xpts_gw")
    xi_names = {p["web_name"] for p in pack["xi"]}
    assert "BlankMID" not in xi_names
    assert pack["formation"] is not None
    assert len(pack["xi"]) == 11

