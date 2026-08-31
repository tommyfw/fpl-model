from __future__ import annotations

from collections import Counter
from typing import Any

from fpl_model.squad.optimize import SQUAD_NEED, _obj, _price


def suggest_transfers(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    bank: float = 0.0,
    free_transfers: int = 1,
    objective: str = "xpts",
    max_suggestions: int = 12,
) -> dict[str, Any]:
    """Rank legal transfers that improve a saved 15-man squad."""
    if objective not in {"xpts", "overall"}:
        raise ValueError("objective must be 'xpts' or 'overall'")
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

    base_obj = sum(_obj(p, objective) for p in squad_players)
    suggestions: list[dict[str, Any]] = []

    # Single transfers (use 1 FT, or a -4 hit if no FT and objective is xpts)
    singles = _single_transfers(squad_players, pool, bank, objective)
    ft = max(0, int(free_transfers))
    for s in singles:
        hits = 0 if ft >= 1 else 1
        gain = s["gain"]
        if hits and objective == "xpts":
            gain -= 4.0
        suggestions.append({**s, "transfers": 1, "hits": hits, "net_gain": round(gain, 2)})

    # Two-transfer moves when FT >= 2 (or one hit for second transfer)
    if ft >= 1:
        doubles = _double_transfers(squad_players, pool, bank, objective, limit_each=8)
        for d in doubles:
            used = 2
            hits = max(0, used - ft)
            gain = d["gain"]
            if hits and objective == "xpts":
                gain -= 4.0 * hits
            suggestions.append({**d, "transfers": 2, "hits": hits, "net_gain": round(gain, 2)})

    suggestions.sort(key=lambda x: (-x["net_gain"], -x["gain"], x["transfers"]))
    # Prefer unique outs/ins in the top list
    seen: set[tuple] = set()
    uniq: list[dict[str, Any]] = []
    for s in suggestions:
        key = tuple(sorted((m["out"]["fpl_id"], m["in"]["fpl_id"]) for m in s["moves"]))
        if key in seen:
            continue
        if s["net_gain"] <= 0 and s["gain"] <= 0:
            continue
        seen.add(key)
        uniq.append(s)
        if len(uniq) >= max_suggestions:
            break

    return {
        "ok": True,
        "objective": objective,
        "bank": round(float(bank), 1),
        "free_transfers": ft,
        "squad_objective": round(base_obj, 2),
        "suggestions": uniq,
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
        "overall": p.get("overall"),
        "action": p.get("action"),
    }


def _team_ok(squad: list[dict[str, Any]], buy: dict[str, Any], sell_id: int) -> bool:
    teams = Counter(p.get("team_id") for p in squad if p["fpl_id"] != sell_id and p.get("team_id") is not None)
    tid = buy.get("team_id")
    if tid is not None and teams[tid] >= 3:
        return False
    return True


def _single_transfers(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    bank: float,
    objective: str,
) -> list[dict[str, Any]]:
    squad_ids = {p["fpl_id"] for p in squad}
    spent = sum(_price(p) for p in squad)
    out: list[dict[str, Any]] = []
    for sell in squad:
        for buy in pool:
            if buy["fpl_id"] in squad_ids:
                continue
            if buy["position"] != sell["position"]:
                continue
            if _obj(buy, objective) <= _obj(sell, objective) + 1e-9:
                continue
            cost = _price(buy) - _price(sell)
            if cost > bank + 1e-9:
                continue
            if not _team_ok(squad, buy, sell["fpl_id"]):
                continue
            gain = _obj(buy, objective) - _obj(sell, objective)
            out.append(
                {
                    "gain": round(gain, 2),
                    "cost": round(cost, 1),
                    "bank_after": round(bank - cost, 1),
                    "moves": [{"out": _slim(sell), "in": _slim(buy)}],
                }
            )
    out.sort(key=lambda x: -x["gain"])
    return out[:40]


def _double_transfers(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    bank: float,
    objective: str,
    limit_each: int = 8,
) -> list[dict[str, Any]]:
    """Pair two independent single-transfer ideas with combined budget/team checks."""
    singles = _single_transfers(squad, pool, bank + 20.0, objective)  # loose bank first
    # Keep diverse outs
    by_out: dict[int, list] = {}
    for s in singles:
        oid = s["moves"][0]["out"]["fpl_id"]
        by_out.setdefault(oid, []).append(s)
        if len(by_out[oid]) > 3:
            by_out[oid] = by_out[oid][:3]
    flat = [s for opts in by_out.values() for s in opts][: limit_each * 3]
    out: list[dict[str, Any]] = []
    by_id = {p["fpl_id"]: p for p in squad}
    pool_by_id = {p["fpl_id"]: p for p in pool}
    for i, a in enumerate(flat):
        a_out = a["moves"][0]["out"]["fpl_id"]
        a_in = a["moves"][0]["in"]["fpl_id"]
        for b in flat[i + 1 :]:
            b_out = b["moves"][0]["out"]["fpl_id"]
            b_in = b["moves"][0]["in"]["fpl_id"]
            if len({a_out, b_out, a_in, b_in}) < 4:
                continue
            sell_a, sell_b = by_id[a_out], by_id[b_out]
            buy_a, buy_b = pool_by_id[a_in], pool_by_id[b_in]
            if sell_a["position"] != buy_a["position"] or sell_b["position"] != buy_b["position"]:
                continue
            cost = (_price(buy_a) - _price(sell_a)) + (_price(buy_b) - _price(sell_b))
            if cost > bank + 1e-9:
                continue
            trial = [p for p in squad if p["fpl_id"] not in {a_out, b_out}] + [buy_a, buy_b]
            # Position counts
            counts = Counter(p["position"] for p in trial)
            if any(counts[pos] != n for pos, n in SQUAD_NEED.items()):
                continue
            teams = Counter(p.get("team_id") for p in trial if p.get("team_id") is not None)
            if any(v > 3 for v in teams.values()):
                continue
            gain = (_obj(buy_a, objective) - _obj(sell_a, objective)) + (
                _obj(buy_b, objective) - _obj(sell_b, objective)
            )
            out.append(
                {
                    "gain": round(gain, 2),
                    "cost": round(cost, 1),
                    "bank_after": round(bank - cost, 1),
                    "moves": [
                        {"out": _slim(sell_a), "in": _slim(buy_a)},
                        {"out": _slim(sell_b), "in": _slim(buy_b)},
                    ],
                }
            )
    out.sort(key=lambda x: -x["gain"])
    return out[:30]
