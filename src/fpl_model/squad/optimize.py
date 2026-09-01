from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

SQUAD_NEED = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
FORMATIONS = (
    (1, 3, 4, 3),
    (1, 3, 5, 2),
    (1, 4, 4, 2),
    (1, 4, 3, 3),
    (1, 4, 5, 1),
    (1, 5, 3, 2),
    (1, 5, 4, 1),
)
POS_ORDER = ("GK", "DEF", "MID", "FWD")
OBJECTIVES = ("xpts", "overall", "combined")


def _xi_start_key(p: dict[str, Any], objective: str, lock_ids: set[int] | None = None) -> tuple:
    """Locked players start on the stacked XI; weekly lineups may rotate them for blanks."""
    lock_ids = lock_ids or set()
    locked = int(p["fpl_id"]) in lock_ids
    force_start = locked and not str(objective).startswith("gw:")
    return (
        p.get("_xi_ok") is False,
        not force_start,
        -_obj(p, objective),
    )


def _mark_bench_only(p: dict[str, Any], xi_ok_ids: set[int] | None) -> dict[str, Any]:
    if xi_ok_ids is None or p["fpl_id"] in xi_ok_ids:
        return p
    q = dict(p)
    q["_xi_ok"] = False
    return q


def _obj(p: dict[str, Any], key: str) -> float:
    if key.startswith("gw:"):
        return _gw_pts(p, int(key[3:]))
    if key == "combined":
        v = p.get("combined")
        if v is not None:
            return float(v)
        return float(p.get("xpts") or 0) + 0.35 * float(p.get("overall") or 0)
    v = p.get(key)
    return float(v) if v is not None else 0.0


def _gw_pts(p: dict[str, Any], t: int) -> float:
    xs = p.get("xpts_gws")
    if isinstance(xs, (list, tuple)) and 0 <= t < len(xs):
        return float(xs[t] or 0)
    return 0.0


def _n_gws(players: Iterable[dict[str, Any]]) -> int:
    for p in players:
        xs = p.get("xpts_gws")
        if isinstance(xs, (list, tuple)) and xs:
            return len(xs)
    return 0


def _attach_combined(players: list[dict[str, Any]]) -> None:
    """Equal-weight min-max of xPts and Overall within this pool (0–100)."""
    if not players:
        return
    xs = [float(p.get("xpts") or 0) for p in players]
    os = [float(p.get("overall") or 0) for p in players]
    xmin, xmax = min(xs), max(xs)
    omin, omax = min(os), max(os)
    dx = xmax - xmin if xmax > xmin else 1.0
    do = omax - omin if omax > omin else 1.0
    for p in players:
        xn = (float(p.get("xpts") or 0) - xmin) / dx
        on = (float(p.get("overall") or 0) - omin) / do
        p["combined"] = round(50.0 * (xn + on), 3)


def _price(p: dict[str, Any]) -> float:
    v = p.get("price")
    return float(v) if v is not None else 0.0


def _eligible(players: Iterable[dict[str, Any]], *, allow_injured: bool = True) -> list[dict[str, Any]]:
    out = []
    for p in players:
        if p.get("position") not in SQUAD_NEED:
            continue
        if _price(p) <= 0:
            continue
        status = (p.get("status") or "a").lower()
        if status in {"u", "n"}:
            continue
        if not allow_injured and status in {"i", "s"}:
            continue
        out.append(p)
    return out


_POOL_TOP = {"GK": 20, "DEF": 40, "MID": 45, "FWD": 24}
_POOL_CHEAP = {"GK": 14, "DEF": 18, "MID": 16, "FWD": 12}


def trim_optimizer_pool(
    players: list[dict[str, Any]],
    *,
    objective: str,
    keep_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Keep stars, cheap cover, and must-include ids. Full 600+ pools make hill-climb slow."""
    keep_ids = {int(x) for x in (keep_ids or [])}
    if len(players) <= 140:
        return players
    keep: dict[int, dict[str, Any]] = {}
    for p in players:
        if int(p["fpl_id"]) in keep_ids:
            keep[int(p["fpl_id"])] = p
    for pos in POS_ORDER:
        rows = [p for p in players if p["position"] == pos]
        for p in sorted(rows, key=lambda x: (-_obj(x, objective), _price(x)))[: _POOL_TOP[pos]]:
            keep[int(p["fpl_id"])] = p
        for p in sorted(rows, key=_price)[: _POOL_CHEAP[pos]]:
            keep[int(p["fpl_id"])] = p
    return list(keep.values())


def _cheapest_fill_cost(
    pool: list[dict[str, Any]],
    need: dict[str, int],
    team_counts: Counter,
    exclude: set[int],
    max_per_team: int = 3,
) -> float | None:
    """Lower bound: cheapest players who can still fill remaining slots under team caps."""
    cost = 0.0
    local = Counter(team_counts)
    for pos in POS_ORDER:
        n = need[pos]
        if n <= 0:
            continue
        cands = sorted(
            (p for p in pool if p["position"] == pos and p["fpl_id"] not in exclude),
            key=_price,
        )
        picked = 0
        for p in cands:
            tid = p.get("team_id")
            if tid is not None and local[tid] >= max_per_team:
                continue
            cost += _price(p)
            if tid is not None:
                local[tid] += 1
            picked += 1
            if picked >= n:
                break
        if picked < n:
            return None
    return cost


def _can_add(
    p: dict[str, Any],
    need: dict[str, int],
    spent: float,
    budget: float,
    teams: Counter,
    pool: list[dict[str, Any]],
    exclude: set[int],
    max_per_team: int = 3,
) -> bool:
    pos = p["position"]
    if need[pos] <= 0:
        return False
    price = _price(p)
    if spent + price > budget + 1e-9:
        return False
    tid = p.get("team_id")
    if tid is not None and teams[tid] >= max_per_team:
        return False
    next_need = dict(need)
    next_need[pos] -= 1
    next_teams = Counter(teams)
    if tid is not None:
        next_teams[tid] += 1
    fill = _cheapest_fill_cost(pool, next_need, next_teams, exclude | {p["fpl_id"]}, max_per_team)
    if fill is None:
        return False
    return spent + price + fill <= budget + 1e-9


def _squad_legal(squad: list[dict[str, Any]], budget: float, max_per_team: int) -> bool:
    if len(squad) != 15:
        return False
    if len({p["fpl_id"] for p in squad}) != 15:
        return False
    if sum(_price(p) for p in squad) > budget + 1e-9:
        return False
    counts = Counter(p["position"] for p in squad)
    if any(counts[pos] != n for pos, n in SQUAD_NEED.items()):
        return False
    teams = Counter(p.get("team_id") for p in squad if p.get("team_id") is not None)
    return all(v <= max_per_team for v in teams.values())


_START_K = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
_BENCH_SLACK = 1.0
_GK_BACKUP_CAP = 5.0  # rotation keepers only; never a second 5.0+ starter


def _is_premium_gk(p: dict[str, Any]) -> bool:
    return p.get("position") == "GK" and _price(p) >= _GK_BACKUP_CAP - 1e-9


def _marginal_cover(p: dict[str, Any], squad: list[dict[str, Any]]) -> float:
    """How many weekly points p would add vs the current same-position group (blanks / weak GWs)."""
    xs = p.get("xpts_gws")
    if not isinstance(xs, (list, tuple)) or not xs:
        return 0.0
    pos = p["position"]
    k = _START_K.get(pos, 2)
    total = 0.0
    others = [q for q in squad if q["position"] == pos]
    for t, raw in enumerate(xs):
        v = float(raw or 0)
        week = sorted((_gw_pts(q, t) for q in others), reverse=True)
        floor = week[k - 1] if len(week) >= k else 0.0
        total += max(0.0, v - floor)
    return total


def _fill_cheapest(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    need: dict[str, int],
    spent: float,
    budget: float,
    teams: Counter,
    chosen: set[int],
    objective: str,
    max_per_team: int,
    xi_ok_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Fill leftover slots with cheap cover: rotation GK + outfield for weak fixture weeks."""

    def legal(pos: str) -> list[dict[str, Any]]:
        out = []
        for p in pool:
            if p["fpl_id"] in chosen or p["position"] != pos:
                continue
            if not _can_add(p, need, spent, budget, teams, pool, chosen, max_per_team):
                continue
            out.append(p)
        return out

    def add(p: dict[str, Any]) -> None:
        nonlocal spent
        squad.append(_mark_bench_only(p, xi_ok_ids))
        chosen.add(p["fpl_id"])
        spent += _price(p)
        need[p["position"]] -= 1
        tid = p.get("team_id")
        if tid is not None:
            teams[tid] += 1

    while sum(need.values()) > 0:
        if need["GK"] > 0:
            pos = "GK"
        else:
            pos = next((x for x in POS_ORDER if need[x] > 0), None)
            if pos is None:
                break
        cands = legal(pos)
        if not cands:
            break
        if _n_gws(cands) == 0:
            pick = min(cands, key=lambda x: (_price(x), -_obj(x, objective), x.get("web_name") or ""))
        else:
            floor = min(_price(x) for x in cands)
            band = [p for p in cands if _price(p) <= floor + _BENCH_SLACK + 1e-9]
            if pos == "GK":
                rot = [p for p in band if not _is_premium_gk(p)]
                band = rot or band
            pick = max(
                band,
                key=lambda x: (
                    _marginal_cover(x, squad),
                    sum(_gw_pts(x, t) for t in range(_n_gws([x]))),
                    -_price(x),
                    _obj(x, objective),
                ),
            )
        add(pick)
    return squad, spent


def _horizon_xi_score(squad: list[dict[str, Any]], lock_ids: set[int] | None = None) -> float:
    n = _n_gws(squad)
    if n <= 0:
        return 0.0
    return sum(
        float(pick_xi(squad, objective=f"gw:{t}", lock_ids=lock_ids)["xi_objective"]) for t in range(n)
    )


def _xi_and_squad_scores(
    squad: list[dict[str, Any]],
    objective: str,
    lock_ids: set[int] | None = None,
    cache: dict[tuple, tuple[float, float, float]] | None = None,
) -> tuple[float, float, float]:
    """Weekly lineups first (bench cover), then the stacked XI, then cheap backup GK."""
    lock_ids = lock_ids or set()
    key = (frozenset(int(p["fpl_id"]) for p in squad), objective, frozenset(lock_ids))
    if cache is not None and key in cache:
        return cache[key]
    pack = pick_xi(squad, objective=objective, lock_ids=lock_ids)
    xi_ids = {p["fpl_id"] for p in pack["xi"]}
    backup_gk = sum(_price(p) for p in squad if p["position"] == "GK" and p["fpl_id"] not in xi_ids)
    stacked = float(pack["xi_objective"])
    weekly = _horizon_xi_score(squad, lock_ids)
    premium_backup = backup_gk if backup_gk >= _GK_BACKUP_CAP - 1e-9 else 0.0
    if weekly > 0:
        out = (weekly, stacked, -premium_backup)
    else:
        out = (stacked, -backup_gk, sum(_obj(p, objective) for p in squad))
    if cache is not None:
        cache[key] = out
    return out


def _candidates_by_pos(pool: list[dict[str, Any]], objective: str) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pool:
        by[p["position"]].append(p)
    out: dict[str, list[dict[str, Any]]] = {}
    for pos, rows in by.items():
        top = sorted(rows, key=lambda p: (-_obj(p, objective), _price(p)))[:50]
        cheap = sorted(rows, key=lambda p: (_price(p), -_obj(p, objective)))[:12]
        cover = sorted(
            rows,
            key=lambda p: (-sum(_gw_pts(p, t) for t in range(_n_gws([p]))), _price(p)),
        )[:12]
        seen: set[int] = set()
        merged = []
        for p in top + cheap + cover:
            if p["fpl_id"] in seen:
                continue
            seen.add(p["fpl_id"])
            merged.append(p)
        out[pos] = merged
    return out


def _improve(
    squad: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    budget: float,
    objective: str,
    max_per_team: int,
    rounds: int = 5,
    lock_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Hill-climb the starting XI score (then squad total) with 1-for-1 and funded 2-for-2 swaps."""
    if rounds <= 0:
        return squad
    lock_ids = lock_ids or set()
    cands = _candidates_by_pos(pool, objective)
    cache: dict[tuple, tuple[float, float, float]] = {}
    squad = _improve_single(squad, cands, budget, objective, max_per_team, rounds, lock_ids, cache)
    squad = _improve_funded(squad, cands, budget, objective, max_per_team, lock_ids, cache)
    squad = _improve_single(squad, cands, budget, objective, max_per_team, rounds=3, lock_ids=lock_ids, cache=cache)
    return squad


def _improve_single(
    squad: list[dict[str, Any]],
    cands: dict[str, list[dict[str, Any]]],
    budget: float,
    objective: str,
    max_per_team: int,
    rounds: int,
    lock_ids: set[int] | None = None,
    cache: dict[tuple, tuple[float, float, float]] | None = None,
) -> list[dict[str, Any]]:
    lock_ids = lock_ids or set()
    for _ in range(rounds):
        best_key = _xi_and_squad_scores(squad, objective, lock_ids, cache)
        starter_ids = {p["fpl_id"] for p in pick_xi(squad, objective=objective, lock_ids=lock_ids)["xi"]}
        best_trial: list[dict[str, Any]] | None = None
        ids = {p["fpl_id"] for p in squad}
        for i, cur in enumerate(squad):
            if cur["fpl_id"] in lock_ids:
                continue
            bench_gk = cur["position"] == "GK" and cur["fpl_id"] not in starter_ids
            for cand in cands.get(cur["position"], []):
                if cand["fpl_id"] in ids:
                    continue
                if bench_gk and _is_premium_gk(cand):
                    continue
                trial = squad[:i] + [cand] + squad[i + 1 :]
                if not _squad_legal(trial, budget, max_per_team):
                    continue
                key = _xi_and_squad_scores(trial, objective, lock_ids, cache)
                if key > best_key:
                    best_key = key
                    best_trial = trial
        if best_trial is None:
            break
        squad = best_trial
    return squad


def _improve_funded(
    squad: list[dict[str, Any]],
    cands: dict[str, list[dict[str, Any]]],
    budget: float,
    objective: str,
    max_per_team: int,
    lock_ids: set[int] | None = None,
    cache: dict[tuple, tuple[float, float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Downgrade one player to fund an upgrade elsewhere — the greedy trap that left 4.0 GKs in the XI."""
    lock_ids = lock_ids or set()
    best = squad
    best_key = _xi_and_squad_scores(squad, objective, lock_ids, cache)
    starter_ids = {p["fpl_id"] for p in pick_xi(squad, objective=objective, lock_ids=lock_ids)["xi"]}
    ids = {p["fpl_id"] for p in squad}
    for i, donor in enumerate(squad):
        if donor["fpl_id"] in lock_ids:
            continue
        cheaper = [c for c in cands.get(donor["position"], []) if c["fpl_id"] not in ids and _price(c) + 0.05 < _price(donor)]
        cheaper = cheaper[:10]
        if not cheaper:
            continue
        for j, target in enumerate(squad):
            if i == j or target["fpl_id"] in lock_ids:
                continue
            bench_gk = target["position"] == "GK" and target["fpl_id"] not in starter_ids
            better = [
                c
                for c in cands.get(target["position"], [])
                if c["fpl_id"] not in ids
                and _obj(c, objective) > _obj(target, objective) + 1e-9
                and not (bench_gk and _is_premium_gk(c))
            ]
            better = better[:20]
            for d in cheaper:
                for t in better:
                    if d["fpl_id"] == t["fpl_id"]:
                        continue
                    trial = list(squad)
                    trial[i], trial[j] = d, t
                    if not _squad_legal(trial, budget, max_per_team):
                        continue
                    key = _xi_and_squad_scores(trial, objective, lock_ids, cache)
                    if key > best_key:
                        best_key = key
                        best = trial
    return best


def _build_for_formation(
    pool: list[dict[str, Any]],
    bench_pool: list[dict[str, Any]],
    budget: float,
    objective: str,
    form: tuple[int, int, int, int],
    max_per_team: int,
    polish: bool = True,
    xi_ok_ids: set[int] | None = None,
    locked: list[dict[str, Any]] | None = None,
    lock_ids: set[int] | None = None,
) -> list[dict[str, Any]] | None:
    """Fill starting-XI slots first (max objective), then cheapest remaining squad slots."""
    need_xi = dict(zip(POS_ORDER, form, strict=True))
    need = dict(SQUAD_NEED)
    squad: list[dict[str, Any]] = []
    spent = 0.0
    teams: Counter = Counter()
    chosen: set[int] = set()
    lock_ids = lock_ids or set()
    for p in locked or []:
        pos = p["position"]
        if need.get(pos, 0) <= 0:
            return None
        price = _price(p)
        if spent + price > budget + 1e-9:
            return None
        tid = p.get("team_id")
        if tid is not None and teams[tid] >= max_per_team:
            return None
        squad.append(p)
        chosen.add(p["fpl_id"])
        spent += price
        need[pos] -= 1
        if tid is not None:
            teams[tid] += 1
        if need_xi[pos] > 0:
            need_xi[pos] -= 1
    # Only one GK is picked as a starter; the second is rotation/cover fill.
    if need["GK"] < SQUAD_NEED["GK"]:
        need_xi["GK"] = 0

    ranked = sorted(pool, key=lambda p: (-_obj(p, objective), _price(p), p.get("web_name") or ""))
    for p in ranked:
        if p["fpl_id"] in chosen:
            continue
        if sum(need_xi.values()) == 0:
            break
        if need_xi[p["position"]] <= 0:
            continue
        if not _can_add(p, need, spent, budget, teams, bench_pool, chosen, max_per_team):
            continue
        squad.append(p)
        chosen.add(p["fpl_id"])
        spent += _price(p)
        need[p["position"]] -= 1
        need_xi[p["position"]] -= 1
        if p.get("team_id") is not None:
            teams[p["team_id"]] += 1

    squad, spent = _fill_cheapest(
        squad, bench_pool, need, spent, budget, teams, chosen, objective, max_per_team, xi_ok_ids=xi_ok_ids
    )
    if len(squad) != 15 or sum(need.values()) > 0:
        return None
    if not polish:
        return squad
    return _improve(squad, pool, budget, objective, max_per_team, lock_ids=lock_ids)


def build_squad(
    players: list[dict[str, Any]],
    *,
    budget: float = 100.0,
    objective: str = "xpts",
    max_per_team: int = 3,
    polish: bool = True,
    exclude: set[int] | list[int] | None = None,
    exclude_teams: set[int] | list[int] | None = None,
    include: set[int] | list[int] | None = None,
    lock: set[int] | list[int] | None = None,
) -> dict[str, Any]:
    """Build a 15-man FPL squad maximizing the starting-XI objective under FPL constraints."""
    if objective not in OBJECTIVES:
        raise ValueError("objective must be 'xpts', 'overall', or 'combined'")
    lock_ids = {int(x) for x in (lock or [])}
    ban = {int(x) for x in (exclude or [])} - lock_ids
    team_ban = {int(x) for x in (exclude_teams or [])}
    allow = {int(x) for x in (include or [])} | lock_ids

    def team_blocked(p: dict[str, Any]) -> bool:
        if int(p["fpl_id"]) in allow:
            return False
        tid = p.get("team_id")
        return tid is not None and int(tid) in team_ban

    pool_all = [p for p in players if int(p["fpl_id"]) not in ban]
    if objective == "combined":
        pool_all = [dict(p) for p in pool_all]
        _attach_combined(pool_all)
    pool_all = trim_optimizer_pool(pool_all, objective=objective, keep_ids=allow)
    xi_src = [p for p in pool_all if not team_blocked(p)]
    xi_pool = _eligible(xi_src, allow_injured=False)
    if len(xi_pool) < 11:
        xi_pool = _eligible(xi_src, allow_injured=True)
    bench_pool = _eligible(pool_all, allow_injured=True)
    xi_ok_ids = {p["fpl_id"] for p in xi_src} if team_ban else None
    by_id = {int(p["fpl_id"]): p for p in pool_all}
    locked_players: list[dict[str, Any]] = []
    lock_fail = False
    for fid in lock_ids:
        p = by_id.get(int(fid))
        if p is None:
            lock_fail = True
            break
        locked_players.append(p)
    locked_need = Counter(p["position"] for p in locked_players)
    if any(locked_need[pos] > SQUAD_NEED[pos] for pos in SQUAD_NEED):
        lock_fail = True

    best_squad: list[dict[str, Any]] | None = None
    best_pack: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    if not lock_fail:
        drafts: list[tuple[tuple[float, float, float], list[dict[str, Any]]]] = []
        for form in FORMATIONS:
            squad = _build_for_formation(
                xi_pool,
                bench_pool,
                budget,
                objective,
                form,
                max_per_team,
                polish=False,
                xi_ok_ids=xi_ok_ids,
                locked=locked_players,
                lock_ids=lock_ids,
            )
            if not squad:
                continue
            key = _xi_and_squad_scores(squad, objective, lock_ids)
            drafts.append((key, squad))
        drafts.sort(key=lambda x: x[0], reverse=True)
        if polish and drafts:
            n_pol = 1
            if len(drafts) > 1 and drafts[1][0][0] >= drafts[0][0][0] * 0.99:
                n_pol = 2
            for _, squad in drafts[:n_pol]:
                squad = _improve(squad, xi_pool, budget, objective, max_per_team, lock_ids=lock_ids)
                key = _xi_and_squad_scores(squad, objective, lock_ids)
                if best_key is None or key > best_key:
                    best_key = key
                    best_squad = squad
                    best_pack = pick_xi(squad, objective=objective, lock_ids=lock_ids)
        else:
            for key, squad in drafts:
                if best_key is None or key > best_key:
                    best_key = key
                    best_squad = squad
                    best_pack = pick_xi(squad, objective=objective, lock_ids=lock_ids)

    empty = {
        "ok": False,
        "budget": budget,
        "spent": 0.0,
        "remaining": budget,
        "objective": objective,
        "squad_objective": 0.0,
        "squad": [],
        "by_position": {pos: [] for pos in POS_ORDER},
        "xi": [],
        "bench": [],
        "formation": None,
        "xi_objective": 0.0,
        "captain": None,
        "excluded": sorted(ban),
        "excluded_teams": sorted(team_ban),
        "included": sorted(allow),
        "locked": sorted(lock_ids),
        "error": "could not keep locked players in a legal 15" if lock_ids else None,
    }
    if best_squad is None or best_pack is None:
        return empty

    spent = sum(_price(p) for p in best_squad)
    by_pos = {pos: [p for p in best_squad if p["position"] == pos] for pos in POS_ORDER}
    return {
        "ok": _squad_legal(best_squad, budget, max_per_team),
        "budget": budget,
        "spent": round(spent, 1),
        "remaining": round(budget - spent, 1),
        "objective": objective,
        "squad_objective": round(sum(_obj(p, objective) for p in best_squad), 2),
        "squad": best_squad,
        "by_position": by_pos,
        "excluded": sorted(ban),
        "excluded_teams": sorted(team_ban),
        "included": sorted(allow),
        "locked": sorted(lock_ids),
        **best_pack,
    }


def pick_xi(
    squad: list[dict[str, Any]],
    *,
    objective: str = "xpts",
    lock_ids: set[int] | list[int] | None = None,
) -> dict[str, Any]:
    """Choose the best legal starting XI + bench from a 15-man squad."""
    locked = {int(x) for x in (lock_ids or [])}
    by_pos = {
        pos: sorted(
            (p for p in squad if p["position"] == pos),
            key=lambda p: _xi_start_key(p, objective, locked),
        )
        for pos in POS_ORDER
    }
    best_xi: list[dict[str, Any]] | None = None
    best_form: tuple[int, int, int, int] | None = None
    best_key: tuple[int, float] | None = None
    for form in FORMATIONS:
        counts = dict(zip(POS_ORDER, form, strict=True))
        if any(len(by_pos[pos]) < n for pos, n in counts.items()):
            continue
        xi = []
        for pos, n in counts.items():
            xi.extend(by_pos[pos][:n])
        banned_n = sum(1 for p in xi if p.get("_xi_ok") is False)
        score = sum(_obj(p, objective) for p in xi)
        key = (-banned_n, score)
        if best_key is None or key > best_key:
            best_key = key
            best_xi = xi
            best_form = form
    if best_xi is None:
        return {"xi": [], "bench": squad, "formation": None, "xi_objective": 0.0, "captain": None}
    xi_ids = {p["fpl_id"] for p in best_xi}
    bench = [p for p in squad if p["fpl_id"] not in xi_ids]
    bench_gk = [p for p in bench if p["position"] == "GK"]
    bench_out = sorted(
        (p for p in bench if p["position"] != "GK"),
        key=lambda p: (
            -(float(p["availability"]) if p.get("availability") is not None else 1.0)
            * float(p.get("xpts_next") or _obj(p, objective) or 0),
            -_obj(p, objective),
        ),
    )
    bench = bench_out + bench_gk
    captain = _pick_captain(best_xi, objective)
    xi_score = sum(_obj(p, objective) for p in best_xi)
    if _points_objective(objective) and captain is not None:
        xi_score += _obj(captain, objective)
    return {
        "xi": best_xi,
        "bench": bench,
        "formation": f"{best_form[1]}-{best_form[2]}-{best_form[3]}",
        "xi_objective": round(xi_score, 2),
        "captain": captain,
    }


def _points_objective(objective: str) -> bool:
    return objective in {"xpts", "xpts_next", "xpts_gw"} or str(objective).startswith("gw:")


def _pick_captain(xi: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    """Captain on this-GW xPts when available (not the 5-GW hold)."""
    eligible = [p for p in xi if p.get("_xi_ok") is not False] or xi
    if objective in {"xpts_gw", "xpts_next"}:
        return max(eligible, key=lambda p: _obj(p, objective))
    if objective == "combined":
        return max(eligible, key=lambda p: float(p.get("xpts_next") or p.get("xpts") or 0))
    if any(p.get("xpts_next") is not None for p in eligible):
        return max(eligible, key=lambda p: float(p.get("xpts_next") or _obj(p, objective)))
    return max(eligible, key=lambda p: _obj(p, objective))
