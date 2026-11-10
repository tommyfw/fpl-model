from __future__ import annotations

from typing import Any

from fpl_model.squad.optimize import _price, _gw_pts, _this_gw_pts, build_squad, pick_xi

CHIP_NAMES = ("wildcard", "freehit", "bboost", "3xc")


def _n_weeks(squad: list[dict[str, Any]]) -> int:
    for p in squad:
        xs = p.get("xpts_gws")
        if isinstance(xs, (list, tuple)) and xs:
            return len(xs)
    return 0


def recommend_chips(
    squad: list[dict[str, Any]],
    *,
    chips_used: list[str] | None = None,
    free_transfers: int = 1,
    as_of_gw: int | None = None,
) -> list[dict[str, Any]]:
    """Heuristic chip advice for a saved 15. Does not place the chip."""
    used = {str(c).lower() for c in (chips_used or [])}
    if len(squad) != 15:
        return []
    pack = pick_xi(squad, objective="xpts")
    xi = pack.get("xi") or []
    bench = pack.get("bench") or []
    cap = pack.get("captain")
    out: list[dict[str, Any]] = []

    injured = [p for p in squad if p.get("availability") is not None and float(p["availability"]) < 0.35]
    blanks = [p for p in xi if p.get("blank_next")]
    n = _n_weeks(squad)
    scan = scan_chip_weeks(squad, as_of_gw=as_of_gw, chips_used=list(used))
    blank_weeks = 0
    bb_best = 0.0
    for t in range(max(1, n)):
        pack_t = pick_xi(squad, objective=f"gw:{t}") if n else pack
        xi_t = pack_t.get("xi") or xi
        bench_t = pack_t.get("bench") or bench
        if n:
            weak = sum(1 for p in xi_t if _gw_pts(p, t) < 0.45)
            blank_weeks = max(blank_weeks, weak)
            bb_best = max(bb_best, sum(_gw_pts(p, t) for p in bench_t))
        else:
            bb_best = sum(float(p.get("xpts_next") or 0) for p in bench_t)
    bench_next = sum(float(p.get("xpts_next") or (float(p.get("xpts") or 0) / 5.0)) for p in bench)
    if bb_best <= 0:
        bb_best = bench_next
    xi_next = [float(p.get("xpts_next") or 0) for p in xi]
    cap_next = float((cap or {}).get("xpts_next") or 0)
    second = sorted(xi_next, reverse=True)[1] if len(xi_next) > 1 else 0.0

    if "wildcard" not in used and (len(injured) >= 3 or blank_weeks >= 3 or len(blanks) >= 3):
        out.append(
            {
                "chip": "wildcard",
                "use": True,
                "reason": f"{len(injured)} low-avail, {max(len(blanks), blank_weeks)} XI blanks in a GW",
            }
        )
    if "freehit" not in used and (len(blanks) >= 3 or blank_weeks >= 3):
        out.append(
            {
                "chip": "freehit",
                "use": True,
                "reason": f"{max(len(blanks), blank_weeks)} starters blank in a week — FH if the rest of the run is fine",
            }
        )
    if "bboost" not in used and bb_best >= 8.0:
        best_bb = next((r for r in scan.get("best") or [] if r.get("chip") == "bboost"), None)
        gw_bit = f" GW{best_bb['event']}" if best_bb and best_bb.get("event") else ""
        out.append(
            {
                "chip": "bboost",
                "use": True,
                "reason": f"best-week{gw_bit} bench ~{bb_best:.1f} xPts (BB bar ≈ 8)",
                "event": (best_bb or {}).get("event"),
            }
        )
    best_tc = next((r for r in scan.get("best") or [] if r.get("chip") == "3xc"), None)
    tc_row = None
    if best_tc is not None:
        weeks = scan.get("weeks") or []
        off = best_tc.get("offset")
        tc_row = next((w for w in weeks if w.get("offset") == off), None)
    tc_x = float((tc_row or {}).get("captain_xpts") or cap_next or 0)
    if "3xc" not in used and tc_x >= 8.0:
        name = (tc_row or {}).get("captain") or (cap or {}).get("web_name") or "captain"
        ev = float((tc_row or {}).get("triple_ev") or 3 * tc_x)
        c_ev = float((tc_row or {}).get("c_ev") or 2 * tc_x)
        gw_bit = f" GW{best_tc['event']}" if best_tc and best_tc.get("event") else " this GW"
        out.append(
            {
                "chip": "3xc",
                "use": True,
                "reason": f"{name} {tc_x:.1f}{gw_bit} (C EV ≈ {c_ev:.1f}, 3XC ≈ {ev:.1f})",
                "event": (best_tc or {}).get("event"),
            }
        )
    best_fh = next((r for r in scan.get("best") or [] if r.get("chip") == "freehit"), None)
    if "freehit" not in used and best_fh and not any(r.get("chip") == "freehit" for r in out):
        off = best_fh.get("offset")
        fh_week = next((w for w in (scan.get("weeks") or []) if w.get("offset") == off), None)
        if fh_week and int(fh_week.get("weak_starters") or 0) >= 3:
            out.append(
                {
                    "chip": "freehit",
                    "use": True,
                    "reason": best_fh.get("reason") or "weakest packed XI week",
                    "event": best_fh.get("event"),
                }
            )
    for row in out:
        row["as_of_gw"] = as_of_gw
        row["free_transfers"] = free_transfers
        row.setdefault("scan", scan)
    return out


def scan_chip_weeks(
    squad: list[dict[str, Any]],
    *,
    as_of_gw: int | None = None,
    chips_used: list[str] | None = None,
) -> dict[str, Any]:
    """Per-GW BB / 3XC / weak-XI (FH) scan from packed xPts."""
    used = {str(c).lower() for c in (chips_used or [])}
    n = _n_weeks(squad)
    weeks: list[dict[str, Any]] = []
    hz = max(1, n) if n else 1
    for t in range(hz):
        obj = f"gw:{t}" if n else "xpts_gw"
        pack = pick_xi(squad, objective=obj)
        cap = pack.get("captain")
        vice = pack.get("vice_captain")
        bench = pack.get("bench") or []
        cap_pts = _gw_pts(cap, t) if n and cap else (_this_gw_pts(cap, obj) if cap else 0.0)
        bench_pts = sum(_gw_pts(p, t) if n else _this_gw_pts(p, obj) for p in bench)
        xi_pts = float(pack.get("xi_objective") or 0)
        weak = 0
        if n:
            weak = sum(1 for p in (pack.get("xi") or []) if _gw_pts(p, t) < 0.45)
        weeks.append(
            {
                "offset": t,
                "event": (int(as_of_gw) + t) if as_of_gw is not None else None,
                "formation": pack.get("formation"),
                "captain": (cap or {}).get("web_name"),
                "vice": (vice or {}).get("web_name"),
                "captain_xpts": round(float(cap_pts), 2),
                "xi": round(xi_pts, 2),
                "bench": round(float(bench_pts), 2),
                "bb": round(xi_pts + float(bench_pts), 2),
                "c_ev": pack.get("captain_ev"),
                "triple_ev": pack.get("triple_ev"),
                "weak_starters": weak,
            }
        )
    best: list[dict[str, Any]] = []
    if weeks and "bboost" not in used:
        row = max(weeks, key=lambda r: r["bench"])
        best.append(
            {
                "chip": "bboost",
                "event": row.get("event"),
                "offset": row["offset"],
                "score": row["bench"],
                "reason": f"BB GW{row.get('event') or row['offset']} bench {row['bench']:.1f}",
            }
        )
    if weeks and "3xc" not in used:
        row = max(weeks, key=lambda r: r["captain_xpts"])
        best.append(
            {
                "chip": "3xc",
                "event": row.get("event"),
                "offset": row["offset"],
                "score": row["captain_xpts"],
                "reason": f"3XC GW{row.get('event') or row['offset']} {row.get('captain') or 'C'} {row['triple_ev']:.1f} EV",
            }
        )
    if weeks and "freehit" not in used:
        row = min(weeks, key=lambda r: r["xi"])
        if row["weak_starters"] >= 3 or (len(weeks) > 1 and row["xi"] + 4.0 < max(w["xi"] for w in weeks)):
            best.append(
                {
                    "chip": "freehit",
                    "event": row.get("event"),
                    "offset": row["offset"],
                    "score": row["xi"],
                    "reason": f"FH GW{row.get('event') or row['offset']} weakest XI {row['xi']:.1f} ({row['weak_starters']} blanks)",
                }
            )
    return {"weeks": weeks, "best": best}


def _this_week_xpts(p: dict[str, Any]) -> float:
    if p.get("xpts_next") is not None:
        return float(p.get("xpts_next") or 0)
    if p.get("xpts_gw") is not None:
        return float(p.get("xpts_gw") or 0)
    return float(p.get("xpts") or 0)


def _fh_verdict(score: int) -> str:
    if score >= 8:
        return "strong"
    if score >= 6:
        return "look"
    if score >= 4:
        return "marginal"
    return "skip"


def freehit_indicator(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    as_of_gw: int | None = None,
    bank: float = 0.0,
    budget: float = 100.0,
) -> dict[str, Any]:
    """This-GW FH rebuild vs your XI. Score 0–10 from XI xPts gain (captain included)."""
    if len(squad) != 15:
        return {"ok": False, "score": 0, "verdict": "skip", "reason": "need a full 15"}
    hold = [dict(p) for p in squad]
    for p in hold:
        p["xpts"] = _this_week_xpts(p)
    by_id: dict[int, dict[str, Any]] = {}
    for p in list(pool) + hold:
        q = dict(p)
        q["xpts"] = _this_week_xpts(q)
        by_id[int(q["fpl_id"])] = q
    work = list(by_id.values())
    spent = sum(_price(p) for p in hold)
    cap = round(max(float(budget), spent) + float(bank), 1)
    built = build_squad(work, budget=cap, objective="xpts", polish=False)
    if not built.get("ok"):
        before = pick_xi(hold, objective="xpts")
        current_xi = round(float(before.get("xi_objective") or 0), 2)
        return {
            "ok": True,
            "chip": "freehit",
            "event": as_of_gw,
            "score": 0,
            "verdict": "skip",
            "xi_gain": 0.0,
            "current_xi": current_xi,
            "fh_xi": current_xi,
            "n_out": 0,
            "weak_starters": 0,
            "reason": "could not beat your 15 this GW",
        }
    before = pick_xi(hold, objective="xpts")
    after = pick_xi(built.get("squad") or [], objective="xpts")
    current_xi = float(before.get("xi_objective") or 0)
    fh_xi = float(after.get("xi_objective") or 0)
    xi_gain = round(fh_xi - current_xi, 2)
    before_ids = {int(p["fpl_id"]) for p in squad}
    after_ids = {int(p["fpl_id"]) for p in (built.get("squad") or [])}
    n_out = len(before_ids - after_ids)
    weak = sum(1 for p in (before.get("xi") or []) if _this_week_xpts(p) < 0.45)
    raw = xi_gain / 1.25
    if weak >= 3:
        raw += 1.5
    if weak >= 5:
        raw += 1.0
    score = int(max(0, min(10, round(raw))))
    verdict = _fh_verdict(score)
    ins = [
        p.get("web_name")
        for p in (built.get("squad") or [])
        if int(p["fpl_id"]) not in before_ids
    ][:6]
    ins_bit = f" in: {', '.join(ins)}" if ins else ""
    reason = (
        f"FH XI {fh_xi:.1f} vs yours {current_xi:.1f} ({xi_gain:+.1f}), "
        f"{n_out} changes{ins_bit}"
    )
    return {
        "ok": True,
        "chip": "freehit",
        "event": as_of_gw,
        "score": score,
        "verdict": verdict,
        "xi_gain": xi_gain,
        "current_xi": round(current_xi, 2),
        "fh_xi": round(fh_xi, 2),
        "n_out": n_out,
        "weak_starters": weak,
        "formation": after.get("formation") or built.get("formation"),
        "captain": (after.get("captain") or {}).get("web_name"),
        "reason": reason,
    }


def apply_chip(
    squad: list[dict[str, Any]],
    chip: str,
    *,
    objective: str = "xpts_gw",
) -> dict[str, Any]:
    """Place bench boost or triple captain on a 15. WC/FH are squad rebuilds, not this."""
    name = str(chip or "").lower()
    pack = pick_xi(squad, objective=objective)
    out = {
        "ok": bool(pack.get("xi")),
        "chip": name,
        "formation": pack.get("formation"),
        "xi": pack.get("xi") or [],
        "bench": pack.get("bench") or [],
        "captain": pack.get("captain"),
        "xi_objective": float(pack.get("xi_objective") or 0),
        "include_bench": False,
        "triple": False,
    }
    if not out["ok"]:
        out["error"] = "could not pick an XI"
        return out
    if name == "bboost":
        bench_pts = sum(float(p.get(objective) or p.get("xpts_next") or p.get("xpts") or 0) for p in out["bench"])
        out["include_bench"] = True
        out["bench_xpts"] = round(bench_pts, 2)
        out["xi_objective"] = round(out["xi_objective"] + bench_pts, 2)
        out["reason"] = f"bench boost adds {bench_pts:.1f} xPts from the 4 on the pine"
        return out
    if name == "3xc":
        cap = out["captain"] or {}
        extra = float(cap.get(objective) or cap.get("xpts_next") or cap.get("xpts") or 0)
        out["triple"] = True
        out["xi_objective"] = round(out["xi_objective"] + extra, 2)
        out["reason"] = f"triple {(cap.get('web_name') or 'captain')} (+{extra:.1f} vs captain)"
        return out
    out["ok"] = False
    out["error"] = f"chip {name} is not placeable here (use wildcard/freehit rebuild)"
    return out


def ft_plan(
    *,
    free_transfers: int,
    best_single_net: float,
    best_double_gain: float,
    hit_min: float = 1.5,
) -> dict[str, Any]:
    """Bank a FT when this week's single is weak and a two-move week is clearly better."""
    ft = max(0, int(free_transfers))
    if ft >= 2:
        return {
            "action": "play",
            "reason": "already on 2+ FT",
            "best_single_net": round(best_single_net, 2),
            "two_ft_value": round(best_double_gain, 2),
        }
    if ft == 1 and best_single_net < hit_min and best_double_gain >= best_single_net + 2.5:
        return {
            "action": "bank",
            "reason": f"this-week single {best_single_net:.1f}; two-move week {best_double_gain:.1f} — save the FT",
            "best_single_net": round(best_single_net, 2),
            "two_ft_value": round(best_double_gain, 2),
        }
    return {
        "action": "play",
        "reason": "play the free transfer if net gain is clear",
        "best_single_net": round(best_single_net, 2),
        "two_ft_value": round(best_double_gain, 2),
    }
