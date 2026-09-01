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


def test_exclude_omits_player_from_built_squad():
    pool = _pool()
    first = build_squad(pool, budget=100.0, objective="xpts")
    assert first["ok"]
    ban = first["xi"][0]["fpl_id"]
    out = build_squad(pool, budget=100.0, objective="xpts", exclude=[ban])
    assert out["ok"]
    assert all(p["fpl_id"] != ban for p in out["squad"])
    assert ban in out["excluded"]


def test_exclude_team_skips_starters_but_can_fill_bench():
    pool = _pool()
    first = build_squad(pool, budget=100.0, objective="xpts")
    assert first["ok"]
    star = next(p for p in first["xi"] if p["position"] != "GK")
    tid = star["team_id"]
    cheap = _p(900, "Fodder", "DEF", 4.0, 0.05, 30, tid)
    pool = pool + [cheap]
    out = build_squad(pool, budget=100.0, objective="xpts", exclude_teams=[tid])
    assert out["ok"]
    xi_ids = {p["fpl_id"] for p in out["xi"]}
    assert star["fpl_id"] not in xi_ids
    bench_ids = {p["fpl_id"] for p in out["bench"]}
    squad_ids = {p["fpl_id"] for p in out["squad"]}
    same_team_xi = [p for p in out["xi"] if p.get("team_id") == tid]
    assert same_team_xi == []
    # leftover cheap DEF from the banned club may still sit on the bench
    if cheap["fpl_id"] in squad_ids:
        assert cheap["fpl_id"] in bench_ids or cheap["fpl_id"] not in xi_ids


def test_include_overrides_team_ban():
    pool = _pool()
    first = build_squad(pool, budget=100.0, objective="xpts")
    star = next(p for p in first["xi"] if p["position"] != "GK")
    out = build_squad(
        pool,
        budget=100.0,
        objective="xpts",
        exclude_teams=[star["team_id"]],
        include=[star["fpl_id"]],
    )
    assert out["ok"]
    assert any(p["fpl_id"] == star["fpl_id"] for p in out["squad"])


def test_lock_forces_low_scoring_player_into_squad():
    pool = _pool() + [_p(999, "MustHave", "MID", 12.0, 1.0, 30, 18)]
    free = build_squad(pool, budget=100.0, objective="xpts")
    assert free["ok"]
    assert all(p["fpl_id"] != 999 for p in free["squad"])
    out = build_squad(pool, budget=100.0, objective="xpts", lock=[999])
    assert out["ok"]
    assert any(p["fpl_id"] == 999 for p in out["squad"])
    assert 999 in out["locked"]
    # remaining budget still fills a legal 15 around the lock
    assert len(out["squad"]) == 15
    assert out["spent"] <= 100.0 + 1e-9


def test_lock_wins_over_exclude():
    pool = _pool()
    first = build_squad(pool, budget=100.0, objective="xpts")
    star = first["xi"][0]["fpl_id"]
    out = build_squad(pool, budget=100.0, objective="xpts", exclude=[star], lock=[star])
    assert out["ok"]
    assert any(p["fpl_id"] == star for p in out["squad"])


def test_lock_too_many_forwards_fails():
    pool = _pool()
    fwds = [p["fpl_id"] for p in pool if p["position"] == "FWD"][:4]
    out = build_squad(pool, budget=100.0, objective="xpts", lock=fwds)
    assert not out["ok"]
    assert out["error"]


def test_backup_keeper_is_the_4m_option():
    out = build_squad(_pool(), budget=100.0, objective="xpts")
    assert out["ok"]
    gks = [p for p in out["squad"] if p["position"] == "GK"]
    assert len(gks) == 2
    starter = next(p for p in out["xi"] if p["position"] == "GK")
    backup = next(p for p in out["bench"] if p["position"] == "GK")
    assert starter["web_name"] in {"Raya", "Alisson", "Pope"}
    assert backup["price"] <= 4.0 + 1e-9
    assert backup["web_name"] in {"Button", "Bentley"}


def test_backup_keeper_covers_starter_blank():
    pool = _pool()
    for p in pool:
        x = float(p["xpts"] or 0)
        p["xpts_gws"] = [round(x / 5.0, 3)] * 5
    raya = next(p for p in pool if p["web_name"] == "Raya")
    raya["xpts_gws"] = [0.0, 3.0, 3.0, 3.0, 3.0]
    for p in pool:
        if p["position"] == "GK" and p["web_name"] != "Raya":
            p["xpts_gws"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    cover = _p(801, "CoverGK", "GK", 4.5, 1.0, 45, 20)
    cover["xpts_gws"] = [6.0, 0.0, 0.0, 0.0, 0.0]
    pool.append(cover)
    out = build_squad(pool, budget=100.0, objective="xpts", lock=[3])
    assert out["ok"]
    assert next(p for p in out["xi"] if p["position"] == "GK")["web_name"] == "Raya"
    names = {p["web_name"] for p in out["squad"]}
    assert "CoverGK" in names
    assert "Alisson" not in names
    assert "Pope" not in names


def test_backup_keeper_covers_starter_blank():
    pool = _pool()
    for p in pool:
        x = float(p["xpts"] or 0)
        p["xpts_gws"] = [round(x / 5.0, 3)] * 5
    raya = next(p for p in pool if p["web_name"] == "Raya")
    raya["xpts_gws"] = [0.0, 3.0, 3.0, 3.0, 3.0]
    for p in pool:
        if p["position"] == "GK" and p["web_name"] != "Raya":
            p["xpts_gws"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    cover = _p(801, "CoverGK", "GK", 4.5, 1.0, 45, 20)
    cover["xpts_gws"] = [6.0, 0.0, 0.0, 0.0, 0.0]
    pool.append(cover)
    out = build_squad(pool, budget=100.0, objective="xpts", lock=[3])
    assert out["ok"]
    assert next(p for p in out["xi"] if p["position"] == "GK")["web_name"] == "Raya"
    names = {p["web_name"] for p in out["squad"]}
    assert "CoverGK" in names
    assert "Alisson" not in names
    assert "Pope" not in names


def test_lock_keeper_starts_and_does_not_buy_a_second_premium():
    out = build_squad(_pool(), budget=100.0, objective="xpts", lock=[3])  # Raya
    assert out["ok"]
    xi_gk = next(p for p in out["xi"] if p["position"] == "GK")
    assert xi_gk["web_name"] == "Raya"
    squad_gks = [p["web_name"] for p in out["squad"] if p["position"] == "GK"]
    assert "Alisson" not in squad_gks
    assert "Pope" not in squad_gks
    backup = next(p for p in out["bench"] if p["position"] == "GK")
    assert backup["price"] <= 4.0 + 1e-9


def test_combined_objective_builds_and_mixes_scores():
    pool = _pool()
    out = build_squad(pool, budget=100.0, objective="xpts")
    mix = build_squad(pool, budget=100.0, objective="combined")
    assert mix["ok"]
    assert mix["objective"] == "combined"
    assert mix["xi"]
    assert all(p.get("combined") is not None for p in mix["squad"])
    assert mix["xi_objective"] > 0
    # Combined captain still prefers playing xPts, not the 4.0 zeros
    gk = next(p for p in mix["xi"] if p["position"] == "GK")
    assert gk["web_name"] not in {"Button", "Bentley"}


def test_outfield_bench_prefers_fixture_cover_over_zeros():
    pool = _pool()
    for p in pool:
        x = float(p["xpts"] or 0)
        p["xpts_gws"] = [round(x / 5.0, 3)] * 5
    zero = _p(901, "ZeroFWD", "FWD", 4.5, 0.0, 30, 18)
    zero["xpts_gws"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    cover = _p(902, "CoverFWD", "FWD", 4.5, 0.4, 31, 19)
    cover["xpts_gws"] = [7.0, 0.0, 0.0, 0.0, 0.0]
    pool = pool + [zero, cover]
    stars = [p["fpl_id"] for p in pool if p["position"] == "FWD" and p["price"] >= 11][:2]
    out = build_squad(pool, budget=100.0, objective="xpts", lock=stars)
    assert out["ok"]
    names = {p["web_name"] for p in out["squad"]}
    assert "CoverFWD" in names
    assert "ZeroFWD" not in names


def test_trim_keeps_locked_player_outside_top():
    players = _pool()
    fid = 1000
    for team in range(20, 50):
        for i in range(3):
            players.append(_p(fid, f"DEF{fid}", "DEF", 5.0 + i * 0.1, 8.0 + i, 70, team))
            fid += 1
    players.append(_p(9999, "MustPlay", "DEF", 4.0, 0.1, 20, 50))
    out = build_squad(players, budget=100.0, objective="xpts", lock=[9999])
    assert out["ok"]
    assert 9999 in {p["fpl_id"] for p in out["squad"]}

