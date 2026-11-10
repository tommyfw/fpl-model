from fpl_model.squad.chips import apply_chip, freehit_indicator, recommend_chips, scan_chip_weeks
from fpl_model.squad.optimize import pick_xi


def _p(fid, name, pos, xpts_gw, team_id=1):
    return {
        "fpl_id": fid,
        "web_name": name,
        "position": pos,
        "team_id": team_id,
        "xpts_gw": xpts_gw,
        "xpts": xpts_gw * 5,
        "availability": 1.0,
    }


def _squad():
    return (
        [_p(1, "GK1", "GK", 4.0), _p(2, "GK2", "GK", 1.0)]
        + [_p(10 + i, f"D{i}", "DEF", 3.0, team_id=i) for i in range(1, 6)]
        + [_p(20 + i, f"M{i}", "MID", 5.0 - i * 0.2, team_id=i) for i in range(1, 6)]
        + [_p(30 + i, f"F{i}", "FWD", 6.0 - i * 0.1, team_id=i) for i in range(1, 4)]
    )


def test_apply_bench_boost_adds_bench_xpts():
    squad = _squad()
    base = pick_xi(squad, objective="xpts_gw")
    bb = apply_chip(squad, "bboost", objective="xpts_gw")
    assert bb["ok"]
    assert bb["include_bench"]
    bench_pts = sum(p["xpts_gw"] for p in base["bench"])
    assert abs(bb["xi_objective"] - (base["xi_objective"] + bench_pts)) < 1e-6


def test_apply_triple_captain_adds_one_more_captain_share():
    squad = _squad()
    base = pick_xi(squad, objective="xpts_gw")
    tc = apply_chip(squad, "3xc", objective="xpts_gw")
    assert tc["ok"]
    assert tc["triple"]
    extra = float(base["captain"]["xpts_gw"])
    assert abs(tc["xi_objective"] - (base["xi_objective"] + extra)) < 1e-6


def test_scan_chip_weeks_flags_bb_and_triple():
    squad = _squad()
    for p in squad:
        p["xpts_gws"] = [4.0, 2.0, 9.0 if p["web_name"] == "F1" else 3.0]
    scan = scan_chip_weeks(squad, as_of_gw=5)
    assert len(scan["weeks"]) == 3
    chips = {b["chip"] for b in scan["best"]}
    assert "bboost" in chips
    assert "3xc" in chips
    tc = next(b for b in scan["best"] if b["chip"] == "3xc")
    assert tc["event"] == 7


def test_recommend_3xc_from_scan_best_week():
    squad = _squad()
    for p in squad:
        p["xpts_gws"] = [4.0, 2.0, 9.0 if p["web_name"] == "F1" else 3.0]
        p["xpts_next"] = 4.0
    rec = recommend_chips(squad, as_of_gw=5)
    tc = next(c for c in rec if c["chip"] == "3xc")
    assert tc["event"] == 7


def test_freehit_indicator_scores_this_gw_upgrade():
    squad = _squad()
    for p in squad:
        p["price"] = 4.5
        p["xpts_next"] = 1.0
    pool = list(squad)
    fid = 200
    for i, pos in enumerate(["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3):
        pool.append(
            {
                "fpl_id": fid,
                "web_name": f"Star{fid}",
                "position": pos,
                "team_id": 20 + i,
                "price": 6.0,
                "xpts_next": 8.0,
                "xpts": 8.0,
                "availability": 1.0,
            }
        )
        fid += 1
    out = freehit_indicator(squad, pool, as_of_gw=9)
    assert out["ok"]
    assert out["xi_gain"] > 20
    assert out["score"] >= 8
    assert out["verdict"] == "strong"
    assert out["n_out"] >= 10


def test_freehit_indicator_skip_when_squad_already_good():
    squad = _squad()
    for p in squad:
        p["price"] = 6.0
        p["xpts_next"] = 7.0
    out = freehit_indicator(squad, squad, as_of_gw=9)
    assert out["ok"]
    assert out["xi_gain"] == 0
    assert out["score"] <= 2
    assert out["verdict"] == "skip"
