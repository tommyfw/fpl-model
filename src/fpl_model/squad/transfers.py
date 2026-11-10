from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

from fpl_model.scoring.clean_sheet import blank_events, parse_fixtures_json
from fpl_model.squad.chips import ft_plan, recommend_chips
from fpl_model.squad.optimize import SQUAD_NEED, _minutes_mult, _obj, _price, build_squad, pick_xi

# Hits need clear edge after the −4; early season is noisy.
HIT_MIN_NET = 1.5
WANT_MAX = 5
HORIZON_PLAN = 5
HORIZON_PERSIST = 0.97


def _money_legal(bank_after: float, spent_after: float, budget: float) -> bool:
    return bank_after >= -1e-9 and spent_after <= budget + 1e-9


def _money_broke(bank: float, spent: float, budget: float) -> bool:
    return bank < -1e-9 or spent > budget + 1e-9


def chip_rebuild_budget(
    budget: float,
    spent: float,
    bank: float,
    max_value: float | None = None,
) -> tuple[float, float]:
    """Optimizer spent cap and FPL purse (current 15 + ITB)."""
    spent_f = float(spent)
    bank_f = float(bank)
    purse = round(spent_f + bank_f, 1)
    cap = round(max(float(budget), spent_f) + bank_f, 1)
    if max_value is not None:
        cap = round(min(cap, float(max_value)), 1)
    return cap, purse


def suggest_transfers(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    bank: float = 0.0,
    budget: float = 100.0,
    free_transfers: int = 1,
    objective: str = "xpts",
    max_suggestions: int = 12,
    horizon_gws: int = 5,
    exclude: set[int] | list[int] | None = None,
    exclude_teams: set[int] | list[int] | None = None,
    include: set[int] | list[int] | None = None,
    keep: set[int] | list[int] | None = None,
    want_transfers: int | None = None,
    committed: list | None = None,
) -> dict[str, Any]:
    """Rank transfers for a saved 15: upgrades in budget, or sales that restore the cap."""
    if objective not in {"xpts", "overall", "combined"}:
        raise ValueError("objective must be 'xpts', 'overall', or 'combined'")
    ban = {int(x) for x in (exclude or [])}
    team_ban = {int(x) for x in (exclude_teams or [])}
    allow = {int(x) for x in (include or [])}
    keep_ids = {int(x) for x in (keep or [])}
    squad_ids = {p["fpl_id"] for p in squad}
    if len(squad) != 15 or len(squad_ids) != 15:
        return {
            "ok": False,
            "error": "need a full 15-man squad",
            "suggestions": [],
            "squad_objective": 0.0,
        }

    by_id = {p["fpl_id"]: p for p in pool}
    squad_players = [by_id[i] for i in squad_ids if i in by_id]
    if len(squad_players) != 15:
        return {"ok": False, "error": "squad players missing from score table", "suggestions": [], "squad_objective": 0.0}

    if objective == "combined":
        from fpl_model.squad.optimize import _attach_combined

        _attach_combined(list(pool))

    base_obj = sum(_obj(p, objective) for p in squad_players)
    base_next = sum(_next_obj(p, objective) for p in squad_players)
    spent = sum(_price(p) for p in squad_players)
    cap = float(budget)
    ft = max(0, int(free_transfers))
    want = max(1, min(WANT_MAX, int(want_transfers if want_transfers is not None else 2)))
    committed_pairs = _parse_pairs(committed)

    applied, err = _apply_committed(squad_players, by_id, committed_pairs, bank)
    if err:
        return {"ok": False, "error": err, "suggestions": [], "squad_objective": round(base_obj, 2)}

    squad_now, bank_now, spent_now, packed_committed = applied
    committed_block = (
        _committed_block(
            squad_players,
            squad_now,
            packed_committed,
            bank,
            bank_now,
            cap,
            ft,
            objective,
            horizon_gws,
        )
        if packed_committed
        else None
    )
    overspend = max(0.0, spent_now - cap)
    broke = _money_broke(bank_now, spent_now, cap)
    n_done = len(committed_pairs)
    leftover_ft = max(0, ft - n_done)
    leftover_want = want
    keep_now = {i for i in keep_ids if any(p["fpl_id"] == i for p in squad_now)}

    buy_pool = [
        p
        for p in pool
        if int(p["fpl_id"]) not in ban
        and (
            int(p["fpl_id"]) in allow
            or p.get("team_id") is None
            or int(p["team_id"]) not in team_ban
        )
    ]
    suggestions: list[dict[str, Any]] = []
    if broke:
        for n in range(1, leftover_want + 1):
            for s in _restore_transfers(
                squad_now, buy_pool, bank_now, cap, spent_now, objective, n=n, keep=keep_now
            ):
                suggestions.append(
                    _finalize_suggestion(
                        s,
                        hits=_hits(n, leftover_ft),
                        objective=objective,
                        horizon_gws=horizon_gws,
                        squad=squad_now,
                    )
                )
    else:
        if leftover_want >= 1:
            singles = _single_transfers(
                squad_now,
                buy_pool,
                bank_now,
                cap,
                spent_now,
                objective,
                keep=keep_now,
            )
            for s in singles:
                suggestions.append(
                    _finalize_suggestion(
                        s, hits=_hits(1, leftover_ft), objective=objective, horizon_gws=horizon_gws, squad=squad_now
                    )
                )
        if leftover_want >= 2:
            for n in range(2, leftover_want + 1):
                combos = _combo_transfers(
                    squad_now, buy_pool, bank_now, cap, spent_now, objective, n=n, keep=keep_now
                )
                for d in combos:
                    suggestions.append(
                        _finalize_suggestion(
                            d, hits=_hits(n, leftover_ft), objective=objective, horizon_gws=horizon_gws, squad=squad_now
                        )
                    )

    if broke:
        suggestions.sort(
            key=lambda x: (
                not x.get("legal_after"),
                -x["net_gain"],
                -x.get("xi_gain", 0),
                -x.get("gain_horizon", 0),
                -x["gain"],
                x["transfers"],
            )
        )
    elif overspend > 1e-9:
        suggestions.sort(
            key=lambda x: (
                not x.get("fixes_budget"),
                float(x.get("spent_after") or spent_now) if not x.get("fixes_budget") else 0.0,
                -x["net_gain"],
                -x.get("xi_gain", 0),
                -x.get("gain_horizon", 0),
                -x.get("gain_next", 0),
                x["transfers"],
            )
        )
    else:
        suggestions.sort(
            key=lambda x: (
                -x["net_gain"],
                -x.get("xi_gain", 0),
                -x.get("gain_horizon", 0),
                -x.get("gain_next", 0),
                -x["gain"],
                x["transfers"],
            )
        )
    seen: set[tuple] = set()
    uniq: list[dict[str, Any]] = []
    for s in suggestions:
        key = tuple(sorted((m["out"]["fpl_id"], m["in"]["fpl_id"]) for m in s["moves"]))
        if key in seen:
            continue
        if broke:
            if not s.get("legal_after"):
                continue
        else:
            repair = overspend > 1e-9 and (s.get("fixes_budget") or float(s.get("cost") or 0) < -0.05)
            if not repair and s["net_gain"] <= 0 and s["gain"] <= 0 and s.get("gain_next", 0) <= 0:
                continue
        seen.add(key)
        if committed_block:
            s["combined"] = _combine_plans(committed_block, s, ft, objective)
        uniq.append(s)
        if len(uniq) >= max_suggestions:
            break

    singles_net = [s["net_gain"] for s in uniq if s["transfers"] == 1] or [0.0]
    doubles_gain = [s["gain"] for s in uniq if s["transfers"] == 2] or [0.0]
    plan = ft_plan(
        free_transfers=ft,
        best_single_net=max(singles_net),
        best_double_gain=max(doubles_gain),
    )
    chips = recommend_chips(squad_now, chips_used=None, free_transfers=leftover_ft)
    restore = _restore_status(
        broke=broke,
        bank_now=bank_now,
        spent_now=spent_now,
        budget=cap,
        leftover_want=leftover_want,
        n_done=n_done,
        want=want,
        restored=any(s.get("legal_after") for s in uniq) if broke else True,
    )

    return {
        "ok": True,
        "objective": objective,
        "bank": round(float(bank), 1),
        "bank_now": round(float(bank_now), 1),
        "budget": round(cap, 1),
        "spent": round(spent, 1),
        "spent_now": round(spent_now, 1),
        "overspend": round(overspend, 1),
        "free_transfers": ft,
        "want_transfers": want,
        "transfers_left": leftover_want,
        "squad_objective": round(base_obj, 2),
        "squad_next_gw": round(base_next, 2),
        "squad_objective_now": round(sum(_obj(p, objective) for p in squad_now), 2),
        "plan": plan,
        "chips": chips,
        "keep": sorted(keep_now),
        "committed": committed_block,
        "restore": restore,
        "suggestions": uniq,
    }


def _price_rank_adj(buy: dict[str, Any], sell: dict[str, Any]) -> float:
    adj = 0.0
    bpc = buy.get("price_change")
    if bpc is not None:
        adj += 0.4 * float(bpc)
    spc = sell.get("price_change")
    if spc is not None:
        adj -= 0.25 * float(spc)
    return adj


def _buy_risk_haircut(buy: dict[str, Any]) -> float:
    cut = 0.0
    risk = buy.get("minutes_risk") if isinstance(buy.get("minutes_risk"), dict) else {}
    if risk.get("level") == "risk":
        cut += 3.0
    elif risk.get("level") == "watch":
        cut += 0.5
    av = buy.get("availability")
    if av is not None and float(av) < 0.55:
        cut += 1.5
    cut += max(0.0, 1.0 - _minutes_mult(buy)) * 1.5
    return cut


def _move_gain(buy: dict[str, Any], sell: dict[str, Any], objective: str) -> float:
    gain = _obj(buy, objective) - _obj(sell, objective)
    if objective in {"xpts", "combined", "xpts_next", "xpts_gw"}:
        gain += _price_rank_adj(buy, sell)
        gain -= _buy_risk_haircut(buy)
    return gain


def _next_obj(p: dict[str, Any], objective: str) -> float:
    if objective == "xpts":
        v = p.get("xpts_next")
        if v is not None:
            return float(v)
    return float(_obj(p, objective))


def _horizon_pts(p: dict[str, Any], n: int = HORIZON_PLAN) -> float:
    xs = p.get("xpts_gws")
    hz = max(1, int(n))
    if isinstance(xs, (list, tuple)) and xs:
        return float(
            sum(float(x or 0) * (HORIZON_PERSIST**t) for t, x in enumerate(list(xs)[:hz]))
        )
    nxt = p.get("xpts_next")
    if nxt is not None:
        return float(nxt) * sum(HORIZON_PERSIST**t for t in range(hz))
    return float(p.get("xpts") or 0)


def _trial_from_moves(squad: list[dict[str, Any]], moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trial = list(squad)
    for m in moves:
        sell_id = m["out"]["fpl_id"]
        buy = m["in"]
        trial = [buy if p["fpl_id"] == sell_id else p for p in trial]
    return trial


def _xi_pack(squad: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    pack = pick_xi(squad, objective=objective)
    cap = pack.get("captain") or {}
    return {
        "formation": pack.get("formation"),
        "xi_objective": pack.get("xi_objective"),
        "xi": [p.get("web_name") for p in (pack.get("xi") or [])],
        "captain": cap.get("web_name"),
    }


def _finalize_suggestion(
    s: dict[str, Any],
    *,
    hits: int,
    objective: str,
    horizon_gws: int,
    squad: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gain = float(s["gain"])
    gain_next = 0.0
    hz = max(1, min(HORIZON_PLAN, int(horizon_gws)))
    gain_horizon = 0.0
    notes: list[str] = []
    for m in s["moves"]:
        buy, sell = m["in"], m["out"]
        gain_next += _next_obj(buy, objective) - _next_obj(sell, objective)
        gain_horizon += _horizon_pts(buy, hz) - _horizon_pts(sell, hz)
        if buy.get("blank_next"):
            notes.append(f"{buy['web_name']} blank next")
        if sell.get("blank_next"):
            notes.append(f"{sell['web_name']} blank next (out)")
        b_av = buy.get("availability")
        if b_av is not None and float(b_av) < 0.55:
            notes.append(f"START {buy['web_name']} avail {float(b_av):.0%}")
        s_av = sell.get("availability")
        if s_av is not None and float(s_av) < 0.55:
            notes.append(f"START {sell['web_name']} avail {float(s_av):.0%} (out)")
        for who, tag in ((buy, ""), (sell, " (out)")):
            risk = who.get("minutes_risk") or {}
            if risk.get("level") == "risk":
                lab = (risk.get("labels") or ["will not start"])[0]
                notes.append(f"START {who['web_name']} {lab}{tag}")
            elif risk.get("level") == "watch":
                lab = (risk.get("labels") or ["minutes watch"])[0]
                notes.append(f"START {who['web_name']} {lab}{tag}")
        blanks = buy.get("blank_events") or []
        if blanks:
            notes.append(f"{buy['web_name']} blanks GW{','.join(str(b) for b in blanks)}")
        bpc = buy.get("price_change")
        if bpc is not None and abs(float(bpc)) >= 0.2:
            direction = "rising" if float(bpc) > 0 else "falling"
            notes.append(f"PRICE {buy['web_name']} {direction} £{abs(float(bpc)):.1f}")
        spc = sell.get("price_change")
        if spc is not None and abs(float(spc)) >= 0.2:
            direction = "rising" if float(spc) > 0 else "falling"
            notes.append(f"PRICE {sell['web_name']} {direction} £{abs(float(spc)):.1f} (out)")

    net = gain
    if hits and objective == "xpts":
        net -= 4.0 * hits
    hit_caution = bool(hits and objective == "xpts")
    if hit_caution and net < HIT_MIN_NET:
        notes = list(notes) + ["hit needs a clear edge"]
    notes.sort(key=lambda n: 0 if n.startswith("START ") else 1 if n.startswith("PRICE ") else 2)
    spent_after = s.get("spent_after")
    xi_after = None
    xi_gain = 0.0
    if squad:
        trial = _trial_from_moves(squad, s["moves"])
        before = pick_xi(squad, objective=objective)
        after = pick_xi(trial, objective=objective)
        xi_gain = float(after.get("xi_objective") or 0) - float(before.get("xi_objective") or 0)
        xi_after = _xi_pack(trial, objective)
    seen_n: set[str] = set()
    uniq_notes: list[str] = []
    for n in notes:
        if n not in seen_n:
            seen_n.add(n)
            uniq_notes.append(n)
    return {
        **s,
        "transfers": len(s["moves"]),
        "hits": hits,
        "net_gain": round(net, 2),
        "gain_next": round(gain_next, 2),
        "gain_horizon": round(gain_horizon, 2),
        "horizon_gws": hz,
        "xi_gain": round(xi_gain, 2),
        "xi_after": xi_after,
        "gain_xpts": round(
            sum(float(m["in"].get("xpts") or 0) - float(m["out"].get("xpts") or 0) for m in s["moves"]), 2
        ),
        "gain_overall": round(
            sum(float(m["in"].get("overall") or 0) - float(m["out"].get("overall") or 0) for m in s["moves"]), 2
        ),
        "hit_caution": hit_caution,
        "fixes_budget": bool(s.get("fixes_budget")),
        "spent_after": round(float(spent_after), 1) if spent_after is not None else None,
        "notes": uniq_notes,
    }


def _slim(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "fpl_id": p["fpl_id"],
        "web_name": p["web_name"],
        "position": p["position"],
        "team": p.get("team"),
        "team_id": p.get("team_id"),
        "price": p.get("price"),
        "xpts": p.get("xpts"),
        "xpts_next": p.get("xpts_next"),
        "xpts_gws": list(p.get("xpts_gws") or [])[:5] or None,
        "overall": p.get("overall"),
        "action": p.get("action"),
        "availability": p.get("availability"),
        "minutes_risk": p.get("minutes_risk"),
        "blank_next": bool(p.get("blank_next")),
        "blank_events": list(p.get("blank_events") or []),
        "price_change": p.get("price_change"),
    }


def _team_ok(squad: list[dict[str, Any]], buy: dict[str, Any], sell_id: int) -> bool:
    teams = Counter(p.get("team_id") for p in squad if p["fpl_id"] != sell_id and p.get("team_id") is not None)
    tid = buy.get("team_id")
    if tid is not None and teams[tid] >= 3:
        return False
    return True


def _pack_move(
    *,
    gain: float,
    cost: float,
    bank: float,
    spent_after: float,
    budget: float,
    spent: float,
    moves: list[dict[str, Any]],
) -> dict[str, Any]:
    was_over = spent > budget + 1e-9
    bank_after = round(bank - cost, 1)
    return {
        "gain": round(gain, 2),
        "cost": round(cost, 1),
        "bank_after": bank_after,
        "spent_after": round(spent_after, 1),
        "fixes_budget": was_over and spent_after <= budget + 1e-9,
        "legal_after": _money_legal(bank_after, spent_after, budget),
        "moves": moves,
    }


def _restore_status(
    *,
    broke: bool,
    bank_now: float,
    spent_now: float,
    budget: float,
    leftover_want: int,
    n_done: int,
    want: int,
    restored: bool,
) -> dict[str, Any]:
    over = max(0.0, spent_now - budget)
    bits = []
    if bank_now < -1e-9:
        bits.append(f"bank £{bank_now:.1f}")
    if over > 1e-9:
        bits.append(f"£{over:.1f} over the £{budget:.1f} cap")
    money = " and ".join(bits) if bits else "illegal money"
    if not broke:
        return {
            "needed": False,
            "ok": True,
            "transfers_left": leftover_want,
            "reason": "",
        }
    if leftover_want <= 0:
        reason = (
            f"Still {money} after {n_done} transfer{'s' if n_done != 1 else ''}. "
            f"Raise Transfers, unlock a player, or undo."
        )
        return {"needed": True, "ok": False, "transfers_left": 0, "reason": reason}
    if restored:
        return {
            "needed": True,
            "ok": True,
            "transfers_left": leftover_want,
            "reason": f"Restores {money} within {leftover_want} extra transfer{'s' if leftover_want != 1 else ''}.",
        }
    used = f" after {n_done} already made" if n_done else ""
    reason = (
        f"Cannot get back to £0 bank and within the £{budget:.1f} cap in "
        f"{leftover_want} extra transfer{'s' if leftover_want != 1 else ''}{used}. "
        f"Locked players are not sold. Raise Transfers, unlock someone, or undo."
    )
    return {"needed": True, "ok": False, "transfers_left": leftover_want, "reason": reason}


def _hits(n_moves: int, free_transfers: int) -> int:
    return max(0, int(n_moves) - max(0, int(free_transfers)))


def _parse_pairs(raw: list | None) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in raw or []:
        if isinstance(item, dict):
            out.append((int(item["out"]), int(item["in"])))
        else:
            out.append((int(item[0]), int(item[1])))
    return out


def _apply_committed(
    squad: list[dict[str, Any]],
    by_id: dict[int, dict[str, Any]],
    pairs: list[tuple[int, int]],
    bank: float,
) -> tuple[tuple[list, float, float, list] | None, str | None]:
    trial = list(squad)
    remaining_bank = float(bank)
    packed: list[dict[str, Any]] = []
    for out_id, in_id in pairs:
        sell = next((p for p in trial if int(p["fpl_id"]) == int(out_id)), None)
        buy = by_id.get(int(in_id))
        if sell is None:
            return None, f"player {out_id} is not in the squad"
        if buy is None:
            return None, f"player {in_id} is not in the score table"
        if buy["position"] != sell["position"]:
            return None, f"{buy.get('web_name')} is not the same position as {sell.get('web_name')}"
        if any(int(p["fpl_id"]) == int(buy["fpl_id"]) for p in trial):
            return None, f"{buy.get('web_name')} is already in the squad"
        cost = _price(buy) - _price(sell)
        if not _team_ok(trial, buy, sell["fpl_id"]):
            return None, f"{buy.get('web_name')} would break the 3-per-club cap"
        remaining_bank -= cost
        trial = [buy if p["fpl_id"] == sell["fpl_id"] else p for p in trial]
        packed.append({"out": _slim(sell), "in": _slim(buy)})
    if packed:
        counts = Counter(p["position"] for p in trial)
        if any(counts[pos] != n for pos, n in SQUAD_NEED.items()):
            return None, "those transfers do not leave a legal 15"
    spent_after = sum(_price(p) for p in trial)
    return (trial, remaining_bank, spent_after, packed), None


def _committed_block(
    squad: list[dict[str, Any]],
    trial: list[dict[str, Any]],
    packed: list[dict[str, Any]],
    bank: float,
    remaining_bank: float,
    budget: float,
    ft: int,
    objective: str,
    horizon_gws: int,
) -> dict[str, Any]:
    spent = sum(_price(p) for p in squad)
    spent_after = sum(_price(p) for p in trial)
    gain = sum(_obj(p, objective) for p in trial) - sum(_obj(p, objective) for p in squad)
    raw = _pack_move(
        gain=gain,
        cost=bank - remaining_bank,
        bank=bank,
        spent_after=spent_after,
        budget=budget,
        spent=spent,
        moves=packed,
    )
    block = _finalize_suggestion(
        raw, hits=_hits(len(packed), ft), objective=objective, horizon_gws=horizon_gws, squad=squad
    )
    block["squad_objective"] = round(sum(_obj(p, objective) for p in trial), 2)
    return block


def _combine_plans(committed: dict[str, Any], extra: dict[str, Any], ft: int, objective: str) -> dict[str, Any]:
    moves = list(committed.get("moves") or []) + list(extra.get("moves") or [])
    n = len(moves)
    hits = _hits(n, ft)
    gain = float(committed.get("gain") or 0) + float(extra.get("gain") or 0)
    gain_next = float(committed.get("gain_next") or 0) + float(extra.get("gain_next") or 0)
    gain_horizon = float(committed.get("gain_horizon") or 0) + float(extra.get("gain_horizon") or 0)
    xi_gain = float(committed.get("xi_gain") or 0) + float(extra.get("xi_gain") or 0)
    net = gain
    if hits and objective == "xpts":
        net -= 4.0 * hits
    return {
        "transfers": n,
        "hits": hits,
        "gain": round(gain, 2),
        "gain_next": round(gain_next, 2),
        "gain_horizon": round(gain_horizon, 2),
        "xi_gain": round(xi_gain, 2),
        "xi_after": extra.get("xi_after") or committed.get("xi_after"),
        "net_gain": round(net, 2),
        "gain_xpts": round(
            float(committed.get("gain_xpts") or 0) + float(extra.get("gain_xpts") or 0), 2
        ),
        "gain_overall": round(
            float(committed.get("gain_overall") or 0) + float(extra.get("gain_overall") or 0), 2
        ),
        "cost": round(float(committed.get("cost") or 0) + float(extra.get("cost") or 0), 1),
        "bank_after": extra.get("bank_after"),
        "spent_after": extra.get("spent_after"),
        "moves": moves,
    }


def _restore_options(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    keep: set[int],
    objective: str,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    """Same-position replacements for restore: keep high-objective options, plus cash-raising ones."""
    squad_ids = {p["fpl_id"] for p in squad}
    by_pos: dict[str, list] = {}
    for p in pool:
        if p["fpl_id"] in squad_ids:
            continue
        by_pos.setdefault(p["position"], []).append(p)
    opts: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for sell in squad:
        if int(sell["fpl_id"]) in keep:
            continue
        scored: list[tuple] = []
        for buy in by_pos.get(sell["position"]) or []:
            if not _team_ok(squad, buy, sell["fpl_id"]):
                continue
            cost = _price(buy) - _price(sell)
            gain = _move_gain(buy, sell, objective)
            scored.append((sell, buy, cost, gain))
        cheaper = [x for x in scored if x[2] < -1e-9]
        picked: dict[int, tuple] = {}
        for item in sorted(cheaper, key=lambda x: -x[3])[:25]:
            picked[item[1]["fpl_id"]] = item
        for item in sorted(cheaper, key=lambda x: x[2])[:10]:
            picked[item[1]["fpl_id"]] = item
        for item in sorted(scored, key=lambda x: -x[3])[:12]:
            if item[2] <= 0.6:
                picked[item[1]["fpl_id"]] = item
        opts.extend((s, b, c) for s, b, c, _ in picked.values())
    return opts


def _restore_transfers(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    bank: float,
    budget: float,
    spent: float,
    objective: str,
    n: int,
    keep: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Find n-transfer plans that restore money while maximizing the objective."""
    keep = {int(x) for x in (keep or [])}
    packed: list[dict[str, Any]] = []

    def consider(picks: list[tuple]) -> None:
        sells = [p[0] for p in picks]
        buys = [p[1] for p in picks]
        if len({p["fpl_id"] for p in sells}) < n or len({p["fpl_id"] for p in buys}) < n:
            return
        cost = sum(p[2] for p in picks)
        spent_after = spent + cost
        bank_after = bank - cost
        if not _money_legal(bank_after, spent_after, budget):
            return
        trial = list(squad)
        for sell, buy, _ in picks:
            if not _team_ok(trial, buy, sell["fpl_id"]):
                return
            trial = [buy if p["fpl_id"] == sell["fpl_id"] else p for p in trial]
        counts = Counter(p["position"] for p in trial)
        if any(counts[pos] != need for pos, need in SQUAD_NEED.items()):
            return
        gain = sum(_move_gain(b, s, objective) for s, b, _ in picks)
        packed.append(
            _pack_move(
                gain=gain,
                cost=cost,
                bank=bank,
                spent_after=spent_after,
                budget=budget,
                spent=spent,
                moves=[{"out": _slim(s), "in": _slim(b)} for s, b, _ in picks],
            )
        )

    if n == 1:
        squad_ids = {p["fpl_id"] for p in squad}
        by_pos: dict[str, list] = {}
        for p in pool:
            if p["fpl_id"] in squad_ids:
                continue
            by_pos.setdefault(p["position"], []).append(p)
        for sell in squad:
            if int(sell["fpl_id"]) in keep:
                continue
            for buy in by_pos.get(sell["position"]) or []:
                if not _team_ok(squad, buy, sell["fpl_id"]):
                    continue
                cost = _price(buy) - _price(sell)
                consider([(sell, buy, cost)])
    else:
        opts = _restore_options(squad, pool, keep, objective)
        if not opts:
            return []
        seed = opts if n == 2 else opts[:40]
        for combo in combinations(range(len(seed)), n):
            consider([seed[i] for i in combo])

    packed.sort(key=lambda x: (-x["gain"], -x.get("xi_gain", 0), x["cost"]))
    return packed[:40]


def _single_transfers(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    bank: float,
    budget: float,
    spent: float,
    objective: str,
    keep: set[int] | None = None,
    require_legal: bool = False,
) -> list[dict[str, Any]]:
    squad_ids = {p["fpl_id"] for p in squad}
    keep = {int(x) for x in (keep or [])}
    broke = _money_broke(bank, spent, budget)
    out: list[dict[str, Any]] = []
    for sell in squad:
        if int(sell["fpl_id"]) in keep:
            continue
        for buy in pool:
            if buy["fpl_id"] in squad_ids:
                continue
            if buy["position"] != sell["position"]:
                continue
            cost = _price(buy) - _price(sell)
            spent_after = spent + cost
            bank_after = bank - cost
            if broke:
                if bank_after < bank - 1e-9:
                    continue
                if spent_after > spent + 1e-9:
                    continue
                if require_legal and not _money_legal(bank_after, spent_after, budget):
                    continue
            else:
                if cost > bank + 1e-9:
                    continue
                if spent_after > budget + 1e-9:
                    continue
                if _move_gain(buy, sell, objective) <= 1e-9:
                    continue
            if not _team_ok(squad, buy, sell["fpl_id"]):
                continue
            gain = _move_gain(buy, sell, objective)
            out.append(
                _pack_move(
                    gain=gain,
                    cost=cost,
                    bank=bank,
                    spent_after=spent_after,
                    budget=budget,
                    spent=spent,
                    moves=[{"out": _slim(sell), "in": _slim(buy)}],
                )
            )
    if broke:
        out.sort(key=lambda x: (not x["legal_after"], x["spent_after"], -x["bank_after"], -x["gain"]))
        return out[:80]
    out.sort(key=lambda x: -x["gain"])
    return out[:40]


def _combo_transfers(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    bank: float,
    budget: float,
    spent: float,
    objective: str,
    n: int = 2,
    limit_each: int = 8,
    keep: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Combine n single-transfer ideas with joint bank / team checks."""
    if n < 2:
        return []
    broke = _money_broke(bank, spent, budget)
    if broke:
        singles = _single_transfers(
            squad, pool, bank, budget, spent, objective, keep=keep, require_legal=False
        )
    else:
        singles = _single_transfers(squad, pool, bank + 20.0, budget, spent, objective, keep=keep)
    by_out: dict[int, list] = {}
    for s in singles:
        oid = s["moves"][0]["out"]["fpl_id"]
        by_out.setdefault(oid, []).append(s)
        if len(by_out[oid]) > 3:
            by_out[oid] = by_out[oid][:3]
    cap_n = 12 if n >= 3 else limit_each * 3
    flat = [s for opts in by_out.values() for s in opts][:cap_n]
    out: list[dict[str, Any]] = []
    by_id = {p["fpl_id"]: p for p in squad}
    pool_by_id = {p["fpl_id"]: p for p in pool}
    for combo in combinations(range(len(flat)), n):
        picks = [flat[i] for i in combo]
        outs = [s["moves"][0]["out"]["fpl_id"] for s in picks]
        ins = [s["moves"][0]["in"]["fpl_id"] for s in picks]
        if len(set(outs) | set(ins)) < 2 * n:
            continue
        sells = [by_id[i] for i in outs]
        buys = [pool_by_id[i] for i in ins]
        if any(sells[i]["position"] != buys[i]["position"] for i in range(n)):
            continue
        cost = sum(_price(buys[i]) - _price(sells[i]) for i in range(n))
        spent_after = spent + cost
        bank_after = bank - cost
        if broke:
            if not _money_legal(bank_after, spent_after, budget):
                continue
        else:
            if cost > bank + 1e-9:
                continue
            if spent_after > budget + 1e-9:
                continue
        trial = [p for p in squad if p["fpl_id"] not in set(outs)] + buys
        counts = Counter(p["position"] for p in trial)
        if any(counts[pos] != need for pos, need in SQUAD_NEED.items()):
            continue
        teams = Counter(p.get("team_id") for p in trial if p.get("team_id") is not None)
        if any(v > 3 for v in teams.values()):
            continue
        gain = sum(_move_gain(buys[i], sells[i], objective) for i in range(n))
        out.append(
            _pack_move(
                gain=gain,
                cost=cost,
                bank=bank,
                spent_after=spent_after,
                budget=budget,
                spent=spent,
                moves=[{"out": _slim(sells[i]), "in": _slim(buys[i])} for i in range(n)],
            )
        )
    if broke:
        out.sort(key=lambda x: (not x["legal_after"], x["spent_after"], -x["bank_after"], -x["gain"]))
        return out[:40]
    out.sort(key=lambda x: -x["gain"])
    return out[:30]


def rebuild_on_chip(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    chip: str,
    bank: float = 0.0,
    budget: float = 100.0,
    objective: str = "xpts",
    keep: set[int] | list[int] | None = None,
    lock: set[int] | list[int] | None = None,
    exclude: set[int] | list[int] | None = None,
    exclude_teams: set[int] | list[int] | None = None,
    include: set[int] | list[int] | None = None,
    max_value: float | None = None,
) -> dict[str, Any]:
    """Full 15 rebuild for wildcard or free hit (one-week xPts)."""
    name = str(chip or "wildcard").lower().replace(" ", "")
    if name in {"fh", "free-hit"}:
        name = "freehit"
    if name not in {"wildcard", "freehit"}:
        return {"ok": False, "error": "chip must be wildcard or freehit", "squad": []}
    if len(squad) != 15:
        return {"ok": False, "error": "need a full 15-man squad", "squad": []}
    spent = sum(_price(p) for p in squad)
    cap, purse = chip_rebuild_budget(budget, spent, bank, max_value)
    work = [dict(p) for p in pool]
    obj = objective
    if name == "freehit":
        obj = "xpts"
        for p in work:
            nxt = p.get("xpts_next")
            if nxt is not None:
                p["xpts"] = float(nxt)
    force = {int(x) for x in (keep or [])} | {int(x) for x in (lock or [])}
    built = build_squad(
        work,
        budget=cap,
        objective=obj,
        lock=force,
        exclude=exclude,
        exclude_teams=exclude_teams,
        include=include,
    )
    if not built.get("ok"):
        built["chip"] = name
        return built
    after = built.get("squad") or []
    before_xi = pick_xi(squad, objective=obj)
    after_xi = pick_xi(after, objective=obj)
    built["chip"] = name
    built["bank_after"] = round(max(purse, cap) - float(built.get("spent") or 0), 1)
    built["from_spent"] = round(spent, 1)
    built["max_value"] = round(float(max_value), 1) if max_value is not None else None
    built["xi_gain"] = round(
        float(after_xi.get("xi_objective") or 0) - float(before_xi.get("xi_objective") or 0), 2
    )
    built["gain"] = round(sum(_obj(p, obj) for p in after) - sum(_obj(p, obj) for p in squad), 2)
    return built


def annotate_fixture_flags(players: list[dict[str, Any]], horizon_gws: int = 5) -> None:
    """Attach blank_next / blank_events from fixtures_json (mutates in place)."""
    for p in players:
        fx = p.get("fixtures_json") or "{}"
        blanks = blank_events(fx, horizon_gws)
        p["blank_events"] = blanks
        by = parse_fixtures_json(fx)
        if not by:
            p["blank_next"] = True
            continue
        first = min(int(k) for k in by.keys())
        p["blank_next"] = first in blanks or not by.get(str(first))
