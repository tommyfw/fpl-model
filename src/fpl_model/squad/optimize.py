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


def _obj(p: dict[str, Any], key: str) -> float:
    v = p.get(key)
    return float(v) if v is not None else 0.0


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
) -> tuple[list[dict[str, Any]], float]:
    """Fill leftover squad slots with the cheapest legal players (prefer higher obj at the same price)."""
    ordered = sorted(pool, key=lambda x: (_price(x), -_obj(x, objective), x.get("web_name") or ""))
    for p in ordered:
        if sum(need.values()) == 0:
            break
        if p["fpl_id"] in chosen:
            continue
        if need[p["position"]] <= 0:
            continue
        if spent + _price(p) > budget + 1e-9:
            continue
        tid = p.get("team_id")
        if tid is not None and teams[tid] >= max_per_team:
            continue
        squad.append(p)
        chosen.add(p["fpl_id"])
        spent += _price(p)
        need[p["position"]] -= 1
        if tid is not None:
            teams[tid] += 1
    return squad, spent


def _xi_and_squad_scores(squad: list[dict[str, Any]], objective: str) -> tuple[float, float]:
    pack = pick_xi(squad, objective=objective)
    return float(pack["xi_objective"]), sum(_obj(p, objective) for p in squad)


def _candidates_by_pos(pool: list[dict[str, Any]], objective: str) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pool:
        by[p["position"]].append(p)
    out: dict[str, list[dict[str, Any]]] = {}
    for pos, rows in by.items():
        top = sorted(rows, key=lambda p: (-_obj(p, objective), _price(p)))[:50]
        cheap = sorted(rows, key=lambda p: (_price(p), -_obj(p, objective)))[:12]
        seen: set[int] = set()
        merged = []
        for p in top + cheap:
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
    rounds: int = 8,
) -> list[dict[str, Any]]:
    """Hill-climb the starting XI score (then squad total) with 1-for-1 and funded 2-for-2 swaps."""
    if rounds <= 0:
        return squad
    cands = _candidates_by_pos(pool, objective)
    squad = _improve_single(squad, cands, budget, objective, max_per_team, rounds)
    squad = _improve_funded(squad, cands, budget, objective, max_per_team)
    squad = _improve_single(squad, cands, budget, objective, max_per_team, rounds=3)
    return squad


def _improve_single(
    squad: list[dict[str, Any]],
    cands: dict[str, list[dict[str, Any]]],
    budget: float,
    objective: str,
    max_per_team: int,
    rounds: int,
) -> list[dict[str, Any]]:
    for _ in range(rounds):
        base_xi, base_sq = _xi_and_squad_scores(squad, objective)
        best_trial: list[dict[str, Any]] | None = None
        best_key = (base_xi, base_sq)
        ids = {p["fpl_id"] for p in squad}
        for i, cur in enumerate(squad):
            for cand in cands.get(cur["position"], []):
                if cand["fpl_id"] in ids:
                    continue
                trial = squad[:i] + [cand] + squad[i + 1 :]
                if not _squad_legal(trial, budget, max_per_team):
                    continue
                key = _xi_and_squad_scores(trial, objective)
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
) -> list[dict[str, Any]]:
    """Downgrade one player to fund an upgrade elsewhere — the greedy trap that left 4.0 GKs in the XI."""
    base_xi, base_sq = _xi_and_squad_scores(squad, objective)
    best = squad
    best_key = (base_xi, base_sq)
    ids = {p["fpl_id"] for p in squad}
    n = len(squad)
    for i, donor in enumerate(squad):
        cheaper = [c for c in cands.get(donor["position"], []) if c["fpl_id"] not in ids and _price(c) + 0.05 < _price(donor)]
        cheaper = cheaper[:10]
        if not cheaper:
            continue
        for j, target in enumerate(squad):
            if i == j:
                continue
            better = [
                c
                for c in cands.get(target["position"], [])
                if c["fpl_id"] not in ids and _obj(c, objective) > _obj(target, objective) + 1e-9
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
                    key = _xi_and_squad_scores(trial, objective)
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
) -> list[dict[str, Any]] | None:
    """Fill starting-XI slots first (max objective), then cheapest remaining squad slots."""
    need_xi = dict(zip(POS_ORDER, form, strict=True))
    need = dict(SQUAD_NEED)
    squad: list[dict[str, Any]] = []
    spent = 0.0
    teams: Counter = Counter()
    chosen: set[int] = set()
    ranked = sorted(pool, key=lambda p: (-_obj(p, objective), _price(p), p.get("web_name") or ""))
    for p in ranked:
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

    squad, spent = _fill_cheapest(squad, bench_pool, need, spent, budget, teams, chosen, objective, max_per_team)
    if len(squad) != 15 or sum(need.values()) > 0:
        return None
    if not polish:
        return squad
    return _improve(squad, pool, budget, objective, max_per_team)


def build_squad(
    players: list[dict[str, Any]],
    *,
    budget: float = 100.0,
    objective: str = "xpts",
    max_per_team: int = 3,
    polish: bool = True,
) -> dict[str, Any]:
    """Build a 15-man FPL squad maximizing the starting-XI objective under FPL constraints."""
    if objective not in {"xpts", "overall"}:
        raise ValueError("objective must be 'xpts' or 'overall'")
    xi_pool = _eligible(players, allow_injured=False)
    bench_pool = _eligible(players, allow_injured=True)
    if len(xi_pool) < 11:
        xi_pool = bench_pool

    best_squad: list[dict[str, Any]] | None = None
    best_pack: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for form in FORMATIONS:
        squad = _build_for_formation(xi_pool, bench_pool, budget, objective, form, max_per_team, polish=polish)
        if not squad:
            continue
        pack = pick_xi(squad, objective=objective)
        spent = sum(_price(p) for p in squad)
        key = (float(pack["xi_objective"]), sum(_obj(p, objective) for p in squad), -spent)
        if best_key is None or key > best_key:
            best_key = key
            best_squad = squad
            best_pack = pack

    if best_squad is None or best_pack is None:
        return {
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
        }

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
        **best_pack,
    }


def pick_xi(squad: list[dict[str, Any]], *, objective: str = "xpts") -> dict[str, Any]:
    """Choose the best legal starting XI + bench from a 15-man squad."""
    by_pos = {
        pos: sorted((p for p in squad if p["position"] == pos), key=lambda p: -_obj(p, objective)) for pos in POS_ORDER
    }
    best_xi: list[dict[str, Any]] | None = None
    best_form: tuple[int, int, int, int] | None = None
    best_score = -1.0
    for form in FORMATIONS:
        counts = dict(zip(POS_ORDER, form, strict=True))
        if any(len(by_pos[pos]) < n for pos, n in counts.items()):
            continue
        xi = []
        for pos, n in counts.items():
            xi.extend(by_pos[pos][:n])
        score = sum(_obj(p, objective) for p in xi)
        if score > best_score:
            best_score = score
            best_xi = xi
            best_form = form
    if best_xi is None:
        return {"xi": [], "bench": squad, "formation": None, "xi_objective": 0.0, "captain": None}
    xi_ids = {p["fpl_id"] for p in best_xi}
    bench = [p for p in squad if p["fpl_id"] not in xi_ids]
    bench_gk = [p for p in bench if p["position"] == "GK"]
    bench_out = sorted((p for p in bench if p["position"] != "GK"), key=lambda p: -_obj(p, objective))
    bench = bench_out + bench_gk
    captain = max(best_xi, key=lambda p: _obj(p, objective))
    return {
        "xi": best_xi,
        "bench": bench,
        "formation": f"{best_form[1]}-{best_form[2]}-{best_form[3]}",
        "xi_objective": round(best_score, 2),
        "captain": captain,
    }
