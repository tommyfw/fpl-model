from __future__ import annotations

from typing import Any

from fpl_model.squad.optimize import pick_xi, _gw_pts

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
        out.append(
            {
                "chip": "bboost",
                "use": True,
                "reason": f"best-week bench ~{bb_best:.1f} xPts (BB bar ≈ 8)",
            }
        )
    if "3xc" not in used and cap_next >= 8.0 and cap_next >= second + 1.5:
        name = (cap or {}).get("web_name") or "captain"
        out.append(
            {
                "chip": "3xc",
                "use": True,
                "reason": f"{name} {cap_next:.1f} vs next {second:.1f} this GW (C EV ≈ {2 * cap_next:.1f})",
            }
        )
    for row in out:
        row["as_of_gw"] = as_of_gw
        row["free_transfers"] = free_transfers
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
