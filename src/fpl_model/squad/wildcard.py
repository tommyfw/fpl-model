from __future__ import annotations

from collections import Counter
from typing import Any

from fpl_model.squad.optimize import SQUAD_NEED, _gw_pts, _price, build_squad, pick_xi
from fpl_model.squad.transfers import chip_rebuild_budget, rebuild_on_chip

# Hold window is what you must live with after a WC. Later weeks are partly
# replaceable with ordinary FTs, so they only softly pull the 15.
HOLD_H = 3
CORE_H = 8
MAX_STINTS = 2
STINT_MAX_PRICE = 8.0
MIN_STINT_GAIN = 1.2
TAIL_W_CHEAP = 0.30
TAIL_W_PREMIUM = 0.65
SPIKE_GW = 3.5
MAX_PER_TEAM = 3

# Rest-of-season sits in the last slot when fixtures go that far.
WC_WINDOW_TARGETS = (1, 3, 5, 8, 13, 20, 26, 38)
MAX_WC_HORIZON = 38
# Week t of the look-ahead keeps this fraction of modeled xPts (form / injuries drift).
PERSIST = 0.97


def persist_weight(offset: int) -> float:
    """Week 0 = 1.0; ~8 GW ≈ 0.78; 13 ≈ 0.67; 20 ≈ 0.54; 26 ≈ 0.45."""
    return float(max(0.40, PERSIST ** max(0, int(offset))))


def window_confidence(gws: int) -> str:
    h = int(gws)
    if h <= 3:
        return "high"
    if h <= 8:
        return "medium"
    if h <= 13:
        return "low"
    return "shape only"


def clip_windows(horizon_cap: int) -> list[int]:
    cap = max(1, int(horizon_cap))
    out = [h for h in WC_WINDOW_TARGETS if h < cap]
    out.append(cap)
    return out


def _weekly_xi(squad: list[dict[str, Any]], n: int) -> list[float]:
    n_have = 0
    for p in squad:
        xs = p.get("xpts_gws")
        if isinstance(xs, (list, tuple)):
            n_have = max(n_have, len(xs))
    hz = min(int(n), n_have or int(n))
    out: list[float] = []
    for t in range(hz):
        pack = pick_xi(squad, objective=f"gw:{t}")
        out.append(float(pack.get("xi_objective") or 0))
    return out


def _window_label(h: int, cap: int) -> str:
    if h == cap and cap > 13:
        return f"rest of season ({h} GW)"
    return f"next {h} GW" + ("s" if h != 1 else "")


def _delay_scan(
    cur_w: list[float],
    wc_w: list[float],
    *,
    n_out: int,
    free_transfers: int,
    as_of_gw: int,
    max_delay: int = 8,
) -> list[dict[str, Any]]:
    n = min(len(cur_w), len(wc_w))
    if n <= 0:
        return []
    rows = []
    last = min(int(max_delay), n - 1)
    for d in range(0, last + 1):
        adj = 0.0
        raw = 0.0
        for t in range(n):
            w = persist_weight(t)
            if t < d:
                share = _ft_catch_share(n_out, free_transfers, t + 1)
                pts = cur_w[t] + share * (wc_w[t] - cur_w[t])
            else:
                pts = wc_w[t]
            raw += pts
            adj += pts * w
        rows.append(
            {
                "delay": d,
                "play_event": int(as_of_gw) + d,
                "xi_raw": round(raw, 2),
                "xi_adj": round(adj, 2),
            }
        )
    best = max(r["xi_adj"] for r in rows)
    for r in rows:
        r["delta_vs_now"] = round(r["xi_adj"] - rows[0]["xi_adj"], 2)
        r["best"] = abs(r["xi_adj"] - best) < 0.05
    return rows


def _sum_gws(p: dict[str, Any], n: int) -> float:
    xs = p.get("xpts_gws")
    if not isinstance(xs, (list, tuple)) or not xs:
        return float(p.get("xpts_next") or p.get("xpts") or 0)
    return float(sum(float(x or 0) for x in list(xs)[: max(1, n)]))


def _gw_sum(p: dict[str, Any], start: int, end: int) -> float:
    return float(sum(_gw_pts(p, t) for t in range(max(0, start), max(start, end))))


def _n_gws_safe(players: list[dict[str, Any]]) -> int:
    n = 0
    for p in players:
        xs = p.get("xpts_gws")
        if isinstance(xs, (list, tuple)):
            n = max(n, len(xs))
    return n


def ft_aware_score(p: dict[str, Any], hold: int = HOLD_H, core: int = CORE_H) -> float:
    """Next `hold` GWs in full; later weeks down-weighted because FTs can patch them."""
    n = _n_gws_safe([p]) or core
    hold_n = max(1, min(int(hold), n))
    core_n = max(hold_n, min(int(core), n))
    prefix = _gw_sum(p, 0, hold_n)
    tail = _gw_sum(p, hold_n, core_n)
    spike = 0.0
    for t in range(hold_n):
        pts = _gw_pts(p, t)
        if pts >= SPIKE_GW:
            spike += 0.2 * (pts - 2.5)
    score = prefix + TAIL_W_CHEAP * tail + spike
    if _price(p) > STINT_MAX_PRICE:
        score += (TAIL_W_PREMIUM - TAIL_W_CHEAP) * tail
        pre_avg = prefix / hold_n
        tail_avg = tail / max(1, core_n - hold_n)
        if pre_avg >= max(1.15, tail_avg * 1.35):
            score -= 1.5
    return float(score)


def _is_stint_profile(p: dict[str, Any], hold: int = HOLD_H, core: int = CORE_H) -> bool:
    n = _n_gws_safe([p]) or core
    hold_n = max(1, min(int(hold), n))
    core_n = max(hold_n + 1, min(int(core), n))
    pre = _gw_sum(p, 0, hold_n) / hold_n
    tail_n = max(1, core_n - hold_n)
    tail = _gw_sum(p, hold_n, hold_n + tail_n) / tail_n
    spike = any(_gw_pts(p, t) >= SPIKE_GW for t in range(hold_n))
    return pre >= max(1.15, tail * 1.35) or (spike and pre >= tail * 1.2)


def _team_ok(squad: list[dict[str, Any]]) -> bool:
    counts: Counter = Counter()
    for p in squad:
        tid = p.get("team_id")
        if tid is None:
            continue
        counts[tid] += 1
        if counts[tid] > MAX_PER_TEAM:
            return False
    need = dict(SQUAD_NEED)
    for p in squad:
        pos = p.get("position")
        if pos not in need:
            return False
        need[pos] -= 1
    return len(squad) == 15 and all(v == 0 for v in need.values())


def _swap_player(squad: list[dict[str, Any]], out_id: int, inn: dict[str, Any]) -> list[dict[str, Any]] | None:
    nxt = []
    replaced = False
    for p in squad:
        if int(p["fpl_id"]) == int(out_id) and not replaced:
            nxt.append(inn)
            replaced = True
        else:
            nxt.append(p)
    if not replaced or len(nxt) != 15 or not _team_ok(nxt):
        return None
    return nxt


def blend_stints(
    core_squad: list[dict[str, Any]],
    hold_squad: list[dict[str, Any]],
    *,
    cap: float,
    lock_ids: set[int] | None = None,
    hold: int = HOLD_H,
    max_stints: int = MAX_STINTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Put up to two cheap 3GW fixture names into the 8GW core. No later FT itinerary."""
    if not core_squad or not hold_squad or hold < 2:
        return list(core_squad or []), []
    lock_ids = {int(x) for x in (lock_ids or [])}
    core_ids = {int(p["fpl_id"]) for p in core_squad}
    hold_ids = {int(p["fpl_id"]) for p in hold_squad}
    seed = list(core_squad)
    stints: list[dict[str, Any]] = []
    used_out: set[int] = set()
    ranked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for cand in hold_squad:
        if int(cand["fpl_id"]) in core_ids:
            continue
        if _price(cand) > STINT_MAX_PRICE + 1e-9:
            continue
        if not _is_stint_profile(cand, hold=hold):
            continue
        pos = cand.get("position")
        outs = [
            p
            for p in seed
            if p.get("position") == pos
            and int(p["fpl_id"]) not in lock_ids
            and int(p["fpl_id"]) not in used_out
        ]
        if not outs:
            continue
        outs.sort(
            key=lambda p: (
                int(p["fpl_id"]) in hold_ids,
                _gw_sum(p, 0, hold),
                -_price(p),
            )
        )
        displaced = outs[0]
        gain = _gw_sum(cand, 0, hold) - _gw_sum(displaced, 0, hold)
        ranked.append((gain, cand, displaced))
    ranked.sort(key=lambda row: -row[0])
    spent = sum(_price(p) for p in seed)
    for gain, cand, displaced in ranked:
        if len(stints) >= max_stints:
            break
        if gain < MIN_STINT_GAIN:
            continue
        if int(displaced["fpl_id"]) in used_out:
            continue
        trial_spent = spent - _price(displaced) + _price(cand)
        if trial_spent > cap + 1e-9:
            continue
        trial = _swap_player(seed, int(displaced["fpl_id"]), cand)
        if trial is None:
            continue
        seed = trial
        spent = trial_spent
        used_out.add(int(displaced["fpl_id"]))
        spike = any(_gw_pts(cand, t) >= SPIKE_GW for t in range(hold))
        stints.append(
            {
                "fpl_id": int(cand["fpl_id"]),
                "web_name": cand.get("web_name"),
                "position": cand.get("position"),
                "price": round(_price(cand), 1),
                "prefix_gain": round(gain, 2),
                "reason": "blank/DGW spike" if spike else f"{hold}GW fixtures",
            }
        )
    return seed, stints


def clip_stint_slots(
    squad: list[dict[str, Any]],
    core_squad: list[dict[str, Any]],
    *,
    cap: float,
    lock_ids: set[int] | None = None,
    hold: int = HOLD_H,
    max_stints: int = MAX_STINTS,
) -> list[dict[str, Any]]:
    """Drop extra / expensive stints back to the 8GW core so FTs stay optional."""
    if not squad or not core_squad:
        return list(squad or [])
    lock_ids = {int(x) for x in (lock_ids or [])}
    core_by = {int(p["fpl_id"]): p for p in core_squad}
    core_ids = set(core_by)
    seed = list(squad)
    stint_rows = [
        p
        for p in seed
        if int(p["fpl_id"]) not in core_ids
        and int(p["fpl_id"]) not in lock_ids
        and _is_stint_profile(p, hold=hold)
    ]

    def _restore(p: dict[str, Any]) -> bool:
        nonlocal seed
        pos = p.get("position")
        backups = [
            c
            for c in core_squad
            if c.get("position") == pos and int(c["fpl_id"]) not in {int(x["fpl_id"]) for x in seed}
        ]
        if not backups:
            return False
        inn = backups[0]
        spent = sum(_price(x) for x in seed) - _price(p) + _price(inn)
        if spent > cap + 1e-9:
            return False
        nxt = _swap_player(seed, int(p["fpl_id"]), inn)
        if nxt is None:
            return False
        seed = nxt
        return True

    for p in list(stint_rows):
        if _price(p) > STINT_MAX_PRICE + 1e-9:
            if _restore(p):
                stint_rows = [x for x in stint_rows if int(x["fpl_id"]) != int(p["fpl_id"])]

    stint_rows.sort(key=lambda p: _gw_sum(p, 0, hold))
    while len(stint_rows) > max_stints:
        weakest = stint_rows.pop(0)
        _restore(weakest)
    return seed


def _copy_pool_for_horizon(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    work = []
    for p in pool:
        q = dict(p)
        q["xpts"] = _sum_gws(p, n)
        xs = p.get("xpts_gws")
        if isinstance(xs, (list, tuple)) and xs:
            q["xpts_next"] = float(xs[0] or 0)
        work.append(q)
    return work


def _copy_pool_ft_aware(pool: list[dict[str, Any]], hold: int, core: int) -> list[dict[str, Any]]:
    work = []
    for p in pool:
        q = dict(p)
        q["xpts"] = ft_aware_score(p, hold, core)
        xs = p.get("xpts_gws")
        if isinstance(xs, (list, tuple)) and xs:
            q["xpts_next"] = float(xs[0] or 0)
        work.append(q)
    return work


def _slim(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "fpl_id": p["fpl_id"],
        "web_name": p.get("web_name"),
        "position": p.get("position"),
        "team": p.get("team"),
        "team_id": p.get("team_id"),
        "price": p.get("price"),
        "xpts": p.get("xpts"),
        "xpts_next": p.get("xpts_next"),
        "availability": p.get("availability"),
        "minutes_risk": p.get("minutes_risk"),
        "stint": bool(p.get("stint")),
    }


def _stress_weeks(squad: list[dict[str, Any]], n: int, as_of_gw: int) -> list[dict[str, Any]]:
    out = []
    n_have = 0
    for p in squad:
        xs = p.get("xpts_gws")
        if isinstance(xs, (list, tuple)):
            n_have = max(n_have, len(xs))
    hz = min(int(n), n_have or int(n))
    for t in range(hz):
        pack = pick_xi(squad, objective=f"gw:{t}")
        xi = pack.get("xi") or []
        weak = []
        for p in xi:
            xs = p.get("xpts_gws")
            val = float(xs[t] or 0) if isinstance(xs, (list, tuple)) and t < len(xs) else 0.0
            if val < 0.45:
                weak.append(p)
        injured = [
            p
            for p in squad
            if p.get("availability") is not None and float(p["availability"]) < 0.35
        ]
        if t == 0:
            injured_xi = [p for p in xi if p.get("availability") is not None and float(p["availability"]) < 0.55]
        else:
            injured_xi = []
        out.append(
            {
                "offset": t,
                "event": int(as_of_gw) + t,
                "blanks": len(weak),
                "blank_names": [p.get("web_name") for p in weak[:6]],
                "injured_xi": len(injured_xi) if t == 0 else 0,
                "injured_squad": len(injured) if t == 0 else 0,
                "xi_xpts": round(float(pack.get("xi_objective") or 0), 2),
            }
        )
    return out


def _must_sell(squad: list[dict[str, Any]], wc_ids: set[int], n: int) -> list[dict[str, Any]]:
    rows = []
    for p in squad:
        reasons: list[str] = []
        av = p.get("availability")
        if av is not None and float(av) < 0.40:
            reasons.append(f"avail {float(av):.0%}")
        risk = p.get("minutes_risk") or {}
        if risk.get("level") == "risk":
            reasons.append((risk.get("labels") or ["will not start"])[0])
        xs = p.get("xpts_gws")
        blanks = 0
        if isinstance(xs, (list, tuple)):
            blanks = sum(1 for x in list(xs)[:n] if float(x or 0) < 0.45)
        if blanks >= max(2, n // 3):
            reasons.append(f"{blanks} weak GWs")
        if int(p["fpl_id"]) not in wc_ids:
            reasons.append("not in WC 15")
        if not reasons:
            continue
        rows.append(
            {
                "fpl_id": p["fpl_id"],
                "web_name": p.get("web_name"),
                "position": p.get("position"),
                "reason": "; ".join(reasons),
                "xpts_horizon": round(_sum_gws(p, n), 2),
            }
        )
    rows.sort(key=lambda r: (0 if "avail" in r["reason"] or "not start" in r["reason"] else 1, -r["xpts_horizon"]))
    return rows


def _ft_catch_share(n_out: int, free_transfers: int, horizon: int) -> float:
    """Fraction of a full WC you can copy with 1 FT/week (no hits)."""
    if n_out <= 0:
        return 1.0
    fts = max(0, int(free_transfers)) + max(0, int(horizon) - 1)
    fts = min(fts, n_out)
    return float(fts) / float(n_out)


def _try_rebuild(
    squad: list[dict[str, Any]],
    work: list[dict[str, Any]],
    *,
    bank: float,
    budget: float,
    keep: Any,
    lock: Any,
    exclude: Any,
    exclude_teams: Any,
    include: Any,
    max_value: float | None,
) -> dict[str, Any]:
    built = rebuild_on_chip(
        squad,
        work,
        chip="wildcard",
        bank=bank,
        budget=budget,
        objective="xpts",
        keep=keep,
        lock=lock,
        exclude=exclude,
        exclude_teams=exclude_teams,
        include=include,
        max_value=max_value,
    )
    if built.get("ok"):
        return built
    force = {int(x) for x in (keep or [])} | {int(x) for x in (lock or [])}
    cap, _purse = chip_rebuild_budget(
        budget,
        sum(float(p.get("price") or 0) for p in squad),
        bank,
        max_value,
    )
    built = build_squad(
        work,
        budget=cap,
        objective="xpts",
        lock=force,
        exclude=exclude,
        exclude_teams=exclude_teams,
        include=include,
    )
    built["chip"] = "wildcard"
    return built


def advise_wildcard(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    windows: list[int],
    bank: float = 0.0,
    budget: float = 100.0,
    keep: set[int] | list[int] | None = None,
    lock: set[int] | list[int] | None = None,
    exclude: set[int] | list[int] | None = None,
    exclude_teams: set[int] | list[int] | None = None,
    include: set[int] | list[int] | None = None,
    max_value: float | None = None,
    chips_used: list[str] | None = None,
    free_transfers: int = 1,
    as_of_gw: int = 1,
) -> dict[str, Any]:
    """Compare current 15 vs a FT-aware wildcard 15. Does not change stored scores."""
    used = {str(c).lower().replace(" ", "") for c in (chips_used or [])}
    if "wildcard" in used or "wc" in used:
        return {
            "ok": True,
            "verdict": "used",
            "headline": "Wildcard already played this season (per FPL import).",
            "reasons": ["Import again after a new chip window if FPL has issued another WC."],
            "windows": [],
            "squad": [],
            "ids": [],
        }
    if len(squad) != 15:
        return {"ok": False, "error": f"need a full 15-man squad (currently {len(squad)})", "squad": []}
    hz_list = [int(h) for h in windows if int(h) >= 1]
    if not hz_list:
        return {"ok": False, "error": "no fixture look-ahead available", "squad": []}
    long_h = max(hz_list)
    short_h = 1
    hold_h = HOLD_H if long_h >= HOLD_H else max(1, long_h)
    core_h = CORE_H if long_h >= CORE_H else (5 if long_h >= 5 else long_h)
    mid_h = 5 if 5 in hz_list else (3 if 3 in hz_list else core_h)
    rebuild_hs: list[int] = []
    for h in (short_h, hold_h, mid_h, core_h, long_h):
        if h not in rebuild_hs:
            rebuild_hs.append(h)

    rebuilds: dict[int, dict[str, Any]] = {}
    for h in rebuild_hs:
        rebuilds[h] = _try_rebuild(
            _copy_pool_for_horizon(squad, h),
            _copy_pool_for_horizon(pool, h),
            bank=bank,
            budget=budget,
            keep=keep,
            lock=lock,
            exclude=exclude,
            exclude_teams=exclude_teams,
            include=include,
            max_value=max_value,
        )

    core_squad = (rebuilds.get(core_h) or {}).get("squad") or []
    hold_squad = (rebuilds.get(hold_h) or {}).get("squad") or core_squad
    short_squad = (rebuilds.get(short_h) or {}).get("squad") or hold_squad
    if not core_squad:
        return {
            "ok": False,
            "error": (rebuilds.get(core_h) or {}).get("error") or "could not build a wildcard 15",
            "squad": [],
        }

    force_ids = {int(x) for x in (keep or [])} | {int(x) for x in (lock or [])}
    cap, purse = chip_rebuild_budget(budget, sum(_price(p) for p in squad), bank, max_value)
    aware = _try_rebuild(
        _copy_pool_ft_aware(squad, hold_h, core_h),
        _copy_pool_ft_aware(pool, hold_h, core_h),
        bank=bank,
        budget=budget,
        keep=keep,
        lock=lock,
        exclude=exclude,
        exclude_teams=exclude_teams,
        include=include,
        max_value=max_value,
    )
    rec_squad = (aware.get("squad") if aware.get("ok") else None) or list(core_squad)
    rec_squad = clip_stint_slots(rec_squad, core_squad, cap=cap, lock_ids=force_ids, hold=hold_h)
    rec_squad, stints = blend_stints(
        rec_squad, hold_squad, cap=cap, lock_ids=force_ids, hold=hold_h
    )
    rec_squad = clip_stint_slots(rec_squad, core_squad, cap=cap, lock_ids=force_ids, hold=hold_h)
    stint_ids = {int(s["fpl_id"]) for s in stints}
    for p in rec_squad:
        p["stint"] = int(p["fpl_id"]) in stint_ids or (
            int(p["fpl_id"]) not in {int(c["fpl_id"]) for c in core_squad} and _is_stint_profile(p, hold=hold_h)
        )
        if p["stint"]:
            stint_ids.add(int(p["fpl_id"]))
    stints = [
        s
        if int(s["fpl_id"]) in {int(p["fpl_id"]) for p in rec_squad}
        else None
        for s in stints
    ]
    stints = [s for s in stints if s]
    extra_stints = [
        {
            "fpl_id": int(p["fpl_id"]),
            "web_name": p.get("web_name"),
            "position": p.get("position"),
            "price": round(_price(p), 1),
            "prefix_gain": round(_gw_sum(p, 0, hold_h), 2),
            "reason": f"{hold_h}GW fixtures",
        }
        for p in rec_squad
        if p.get("stint") and int(p["fpl_id"]) not in {int(s["fpl_id"]) for s in stints}
    ]
    stints = (stints + extra_stints)[:MAX_STINTS]

    xi_src = _copy_pool_for_horizon(rec_squad, hold_h)
    rec_xi = pick_xi(xi_src, objective="xpts")
    rec_by = {int(p["fpl_id"]): p for p in rec_squad}
    rec_xi_players = [rec_by.get(int(p["fpl_id"]), p) for p in (rec_xi.get("xi") or [])]
    rec_bench = [rec_by.get(int(p["fpl_id"]), p) for p in (rec_xi.get("bench") or [])]
    rec_spent = round(sum(_price(p) for p in rec_squad), 1)
    rec_built = {
        "squad": rec_squad,
        "xi": rec_xi_players,
        "bench": rec_bench,
        "captain": rec_by.get(int((rec_xi.get("captain") or {}).get("fpl_id") or 0)),
        "formation": rec_xi.get("formation"),
        "xi_objective": rec_xi.get("xi_objective"),
        "spent": rec_spent,
        "bank_after": round(max(purse, cap) - rec_spent, 1),
        "budget": cap,
        "max_value": round(float(max_value), 1) if max_value is not None else None,
        "chip": "wildcard",
        "ok": True,
    }

    long_squad = (rebuilds.get(long_h) or {}).get("squad") or core_squad
    long_ids = {int(p["fpl_id"]) for p in long_squad}
    core_ids = {int(p["fpl_id"]) for p in core_squad}
    short_ids = {int(p["fpl_id"]) for p in short_squad}
    current_ids = {int(p["fpl_id"]) for p in squad}
    rec_ids = {int(p["fpl_id"]) for p in rec_squad}
    n_out = len(current_ids - rec_ids)
    only_short = rec_ids - core_ids
    only_long = core_ids - rec_ids

    table = []
    cur_w = _weekly_xi(squad, long_h)
    wc_w = _weekly_xi(rec_squad, long_h)
    core_w = _weekly_xi(core_squad, long_h)
    n_w = min(len(cur_w), len(wc_w))
    for h in hz_list:
        hh = min(int(h), n_w)
        cur = round(sum(cur_w[:hh]), 2)
        wc = round(sum(wc_w[:hh]), 2)
        core_sum = round(sum(core_w[:hh]), 2) if core_w else wc
        share = _ft_catch_share(n_out, free_transfers, hh)
        ft_xi = round(cur + (core_sum - cur) * share, 2)
        gain = wc - cur
        gain_adj = sum((wc_w[t] - cur_w[t]) * persist_weight(t) for t in range(hh))
        table.append(
            {
                "gws": h,
                "label": _window_label(h, long_h),
                "rest_of_season": bool(h == long_h and long_h > 13),
                "shape_only": h > core_h,
                "current_xi": cur,
                "wc_xi": wc,
                "gain": round(gain, 2),
                "gain_adj": round(gain_adj, 2),
                "ft_path_xi": ft_xi,
                "ft_gap": round(wc - ft_xi, 2),
                "confidence": window_confidence(h),
                "n_out": n_out,
            }
        )

    by_h = {r["gws"]: r for r in table}
    w1 = by_h.get(1) or table[0]
    w_play = by_h.get(5) or by_h.get(3) or by_h.get(core_h) or table[0]
    w_core = by_h.get(core_h) or w_play
    delays = _delay_scan(cur_w, wc_w, n_out=n_out, free_transfers=free_transfers, as_of_gw=as_of_gw)
    best_delay = next((r for r in delays if r.get("best")), delays[0] if delays else None)
    stress = _stress_weeks(squad, long_h, as_of_gw)
    peak = max(stress, key=lambda r: (r["blanks"], -r["xi_xpts"])) if stress else None
    injured = [p for p in squad if p.get("availability") is not None and float(p["availability"]) < 0.35]
    must = _must_sell(squad, rec_ids, min(hold_h + 2, long_h))

    fh_only = short_ids - rec_ids
    reasons: list[str] = []
    wait_until = None
    verdict = "consider"
    if w1["gain"] >= 8.0 and w_play["gain"] < w1["gain"] + 4.0 and len(fh_only) >= 4:
        verdict = "lean_fh"
        reasons.append(
            f"This week’s WC jump is {w1['gain']:.1f} xPts but the {w_play['gws']}GW XI only adds {w_play['gain']:.1f} — a free hit fits a one-week spike better than burning the wildcard."
        )
    elif n_out <= 2 and w_play["gain"] < 8.0:
        verdict = "wait"
        reasons.append(
            f"A {w_play['gws']}GW rebuild only moves {n_out} player(s) for {w_play['gain']:.1f} XI xPts — ordinary transfers cover that."
        )
    elif (
        peak
        and peak["offset"] >= 3
        and w1["gain"] < 4.5
        and len(injured) < 3
        and peak["blanks"] >= 3
    ):
        verdict = "wait"
        wait_until = peak["event"]
        reasons.append(
            f"Your XI is fine now; the blank/weak cluster is GW{peak['event']} ({peak['blanks']} soft starters). Hold the wildcard until that week."
        )
    elif w_play["gain_adj"] >= 8.0 and n_out >= 4:
        verdict = "play"
        reasons.append(
            f"{w_play['gws']}GW WC XI is {w_play['gain']:.1f} ahead ({w_play['gain_adj']:.1f} after a persistence haircut) and needs {n_out} moves — FTs cannot copy that in time."
        )
    elif w_play["ft_gap"] >= 6.0 and n_out >= 3:
        verdict = "play"
        reasons.append(
            f"Even banking a FT each week only gets you to {w_play['ft_path_xi']:.1f} vs {w_play['wc_xi']:.1f} on the {w_play['gws']}GW XI."
        )
    elif w_play["gain"] >= 7.0 and len(injured) >= 3:
        verdict = "play"
        reasons.append(f"{len(injured)} low-availability players and a {w_play['gain']:.1f} xPts {w_play['gws']}GW gap — the squad is already broken.")
    elif w_core["gain_adj"] >= 10.0 and n_out >= 4:
        verdict = "play"
        reasons.append(
            f"The {core_h}GW core is {w_core['gain']:.1f} XI xPts ahead and needs {n_out} moves — waiting on FTs will not catch that."
        )
    elif w_play["gain"] < 5.0:
        verdict = "wait"
        reasons.append(f"{w_play['gws']}GW WC gain is only {w_play['gain']:.1f} XI xPts. Wait unless a blank cluster or injuries hit.")
    else:
        verdict = "consider"
        reasons.append(
            f"{w_play['gws']}GW gain {w_play['gain']:.1f} / this week {w1['gain']:.1f}. Not a slam dunk — check must-sells and whether you want this 15 now."
        )

    if len(injured) >= 3 and verdict == "wait":
        verdict = "consider"
        reasons.append(f"{len(injured)} injured/suspended names make waiting riskier.")

    if best_delay and best_delay["delay"] >= 2 and best_delay["delta_vs_now"] >= 1.5 and verdict != "lean_fh":
        verdict = "wait"
        wait_until = best_delay["play_event"]
        reasons.append(
            f"Best week to fire (persistence-weighted, FTs while you wait) is GW{wait_until} "
            f"— hold {best_delay['delay']} GW ({best_delay['delta_vs_now']:+.1f} vs playing now)."
        )
    elif best_delay and best_delay["delay"] == 0 and verdict == "wait" and n_out >= 5 and w_play["gain"] >= 6:
        reasons.append("Delay scan still prefers this week if you do play the wildcard on this run.")

    if stints and verdict != "lean_fh":
        bits = [f"{s.get('web_name')} ({s.get('reason')}, £{s.get('price')})" for s in stints]
        reasons.append(
            "This 15 is built for the next "
            f"{hold_h} GWs with a soft {core_h}GW core; later weeks can be patched with FTs. "
            "Fixture stints (max 2, ≤£"
            f"{STINT_MAX_PRICE:.0f}): " + "; ".join(bits) + "."
        )

    headlines = {
        "play": f"Wildcard now — {w_play['gws']}GW XI {w_play['gain']:+.1f} vs your current 15",
        "wait": f"Wait — FTs (or a later WC) look enough for the next {w_play['gws']} GWs",
        "lean_fh": "Don't burn the wildcard on this week alone — free hit (or a hit) fits better",
        "consider": f"Wildcard is a maybe — {w_play['gws']}GW XI {w_play['gain']:+.1f}",
    }

    rec_h = short_h if verdict == "lean_fh" else (5 if 5 in by_h else core_h)
    if verdict == "lean_fh":
        rec_squad = short_squad
        rec_ids = {int(p["fpl_id"]) for p in rec_squad}
        stints = []
        rec_spent = round(sum(_price(p) for p in rec_squad), 1)
        rec_built["squad"] = rec_squad
        rec_built["spent"] = rec_spent
        rec_built["bank_after"] = round(max(purse, cap) - rec_spent, 1)
        pack = pick_xi(_copy_pool_for_horizon(rec_squad, 1), objective="xpts")
        rec_built["xi"] = pack.get("xi") or []
        rec_built["bench"] = pack.get("bench") or []
        rec_built["captain"] = pack.get("captain")
        rec_built["formation"] = pack.get("formation")
        rec_built["xi_objective"] = pack.get("xi_objective")

    rising = []
    for p in rec_squad:
        pc = p.get("price_change")
        if pc is not None and float(pc) >= 0.2 and int(p["fpl_id"]) not in current_ids:
            rising.append(f"{p.get('web_name')} £{float(pc):+.1f}")
    if rising and verdict in {"play", "consider"}:
        reasons.append("Price: " + ", ".join(rising[:4]) + " already moving — WC now gets them at this cost.")

    return {
        "ok": True,
        "verdict": verdict,
        "headline": headlines[verdict],
        "reasons": reasons,
        "windows": table,
        "recommended_horizon": rec_h,
        "horizon_cap": long_h,
        "hold_gws": hold_h,
        "core_gws": core_h,
        "as_of_gw": as_of_gw,
        "n_out": n_out,
        "wait_until_gw": wait_until,
        "delays": delays,
        "best_delay": best_delay,
        "stress": stress,
        "must_sell": must[:10],
        "injured": [p.get("web_name") for p in injured],
        "core": [_slim(p) for p in rec_squad if int(p["fpl_id"]) in core_ids],
        "stints": stints,
        "only_short": [p.get("web_name") for p in rec_squad if int(p["fpl_id"]) in only_short][:8],
        "only_long": [p.get("web_name") for p in core_squad if int(p["fpl_id"]) in only_long][:8],
        "chip": "wildcard",
        "formation": rec_built.get("formation"),
        "xi_objective": rec_built.get("xi_objective"),
        "spent": rec_built.get("spent"),
        "bank_after": rec_built.get("bank_after"),
        "max_value": rec_built.get("max_value") if rec_built.get("max_value") is not None else (
            round(float(max_value), 1) if max_value is not None else None
        ),
        "gain": rec_built.get("gain"),
        "xi_gain": rec_built.get("xi_gain"),
        "squad": [_slim(p) for p in rec_squad],
        "ids": [int(p["fpl_id"]) for p in rec_squad],
        "xi": [_slim(p) for p in (rec_built.get("xi") or [])],
        "bench": [_slim(p) for p in (rec_built.get("bench") or [])],
        "captain": _slim(rec_built["captain"]) if rec_built.get("captain") else None,
        "rebuilds": {
            str(h): {
                "ids": [int(p["fpl_id"]) for p in (rebuilds[h].get("squad") or [])],
                "names": [p.get("web_name") for p in (rebuilds[h].get("squad") or [])],
                "spent": rebuilds[h].get("spent"),
                "formation": rebuilds[h].get("formation"),
            }
            for h in rebuild_hs
            if rebuilds.get(h, {}).get("ok")
        },
    }
