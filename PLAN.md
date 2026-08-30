# FPL Suitability Model — Reference Plan

## Goal
Score every player for FPL using **only data known before the gameweek deadline**. Output an explainable player card (Overall + Upside) and backtest predicted GW \(T \ldots T+4\) vs actual points.

**As-of rule:** at GW \(T\), use GW \(1 \ldots T-1\) plus fixtures known before GW \(T\). Never use GW \(T\) results, post-match `xP`, or season totals that include \(T\).

---

## Player card
```
Player Quality  Q     Underlying U / FPL Output F / Role R / Team T
Minutes         M
Fixtures        O
Value           V
Overall         0.50Q + 0.20M + 0.20O + 0.10V
Upside          Beta (50 = neutral)
Action          STRONG BUY / BUY / HOLD / AVOID
```

**STRONG BUY:** Overall ≥ 80 **and** (Upside ≥ 55 **or** Minutes ≥ 80 and Fixtures ≥ 70).  
Upside < 50 is not a buy signal.

---

## Scale (every leaf)
1. Rate (per 90 or per start).
2. Minutes floor + shrink toward **position mean**.
3. Compare **within position** (optionally CB vs FB, CDM vs AM).
4. Map to **0–100** (percentile / clipped z).

Promoted teams: prior → league average → observed team xG after ~6–8 GW.

---

## [Alpha] Overall

### Q — Player Quality (50%)
\(Q = 0.45U + 0.20F + 0.20R + 0.15T\)

#### U — Underlying (45% of Q)

Do **not** use nested xGI + xG + xA. Prefer **npxG / npxA / npxGI**. Do **not** treat xGA and xGC as two different team stats.

**MID**

| Metric | Wt |
|---|---|
| npxGI/90 | 25% |
| npxG/90 | 10% |
| npxA/90 | 10% |
| Shots in box/90 | 10% |
| Key passes/90 | 10% |
| DefCon hit rate \(P(\text{CBIT} \ge 12 \mid \text{start})\) | 15% |
| xGChain or xGBuildup/90 | 10% |
| Chance quality (big chances **or** shot-xG concentration, not both with shots in box) | 10% |

**FWD:** same, **no DefCon**; that 15% → npxGI (**npxGI 40%**).

**DEF**

| Metric | Wt |
|---|---|
| P(clean sheet) | 35% |
| Team defence (npxGA etc.) | 25% |
| DefCon \(P(\text{CBIT} \ge 10 \mid \text{start})\) | 10% |
| npxGI/90 | 10% |
| Shots in box/90 | 5% |
| Key passes/90 | 5% |
| Chance quality | 10% |

G/A is **not** in U (lives in F).

**GK** (own card — no MID/FWD U, no DefCon)

| Metric | Wt |
|---|---|
| Team defence / P(CS) | 40% |
| Saves/90 vs SoT faced | 25% |
| Shot-stopping (xGC vs GA) | 10% |
| FPL pts/90, bonus, BPS | 15% |
| Penalty exposure | 10% |

GK Minutes ≈ “are they #1?”

#### F — Actual FPL output (20% of Q)
Recency: **0.6 last 5 + 0.3 last 10 + 0.1 season**.

| Metric | Wt |
|---|---|
| FPL points/90 | 40% |
| Goals/90 | 15% |
| Assists/90 | 15% |
| Bonus/90 | 15% |
| BPS/90 | 15% |

DEF/MID from 2025/26: DefCon points sit inside points/90.  
**Never use vaastav `xP` as a pre-match feature.**

#### R — Role (20% of Q)

| Leaf | Wt | Source |
|---|---|---|
| Position on pitch | 25% | Understat match position mix |
| Attacking involvement | 25% | box touches, xGChain, Threat |
| Set pieces | 15% | FPL corners / DFK order |
| Penalties | 20% | FPL `penalties_order` |
| Advanced/central | 10% | AM/FW vs DM/CB minute share |
| Tactical importance | 5% | start rate when available |

#### T — Team strength (15% of Q)
**One xG model** (Understat primary). As-of \(T\) only. Home/away split after ~8–10 GW.

- **Attack:** npxG/90, goals/90 (shrunk), shots in box/90, xGChain, territorial if available  
- **Defence:** npxGA/90, GA/90 (shrunk), SoT against/90  

| Pos | Attack | Defence |
|---|---|---|
| FWD | 0.9 | 0.1 |
| MID | 0.8 | 0.2 |
| DEF | 0.2 | 0.8 |
| GK | 0.1 | 0.9 |

---

### M — Minutes / reliability (20%)
\(M = 0.40\,\text{start%} + 0.25\,\text{minutes%} + 0.20\,\text{rotation} + 0.15\,\text{availability}\)

| Input | Definition |
|---|---|
| Start % | Starts / team matches played |
| Minutes % | Minutes / (90 × team matches played) |
| Recent starts | last 3 / 5 / 10 |
| Minutes form | last-5 vs season (**not** points form) |
| Availability | FPL `status` + `chance_of_playing_*` |
| Rotation | start rate when the competing player also plays |

---

### O — Fixture / opportunity (20%)
Horizon (Blank = 0, **DGW = sum of both fixtures**):

\[
O = 0.40\,T + 0.25(T{+}1) + 0.15(T{+}2) + 0.10(T{+}3) + 0.10(T{+}4)
\]

Per fixture (player’s view):

| | Opp def (leaky=good) | Opp att (strong=bad CS) | H/A | Team npxG vs type | P(CS) |
|---|---|---|---|---|---|
| FWD | 55% | 5% | 15% | 20% | 5% |
| MID | 40% | 15% | 15% | 20% | 10% |
| DEF/GK | 15% | 40% | 15% | 10% | 20% |

FPL FDR is a weak extra, not a substitute for team strength.

---

### V — Value (10%)

| Metric | Wt |
|---|---|
| Forward xPts / £1m over the same 5-GW window, **current** price | 50% |
| Q / £1m | 30% |
| Price vs position median (premium inverted) | 20% |

Do **not** use season points / current price.

---

## [Beta] Upside (50 = neutral)
Signed 0–100. **Not** a points forecast.

| Factor | + Upside | − Downside |
|---|---|---|
| Finishing | xG − goals | goals − xG |
| Creation | xA − assists | assists overperformance |
| Minutes | last-5 min% > season | minutes falling |
| Team | last-8 npxG/npxGA improving | collapsing |
| Role | new pens/corners, more AM min | lost set pieces |
| Price | falling toward value | already spiked |
| Ownership | low own × high Overall | template + mediocre Overall |

Low ownership is a **rank** lever, not a points lever.

---

## Data
| Source | Use |
|---|---|
| vaastav `merged_gw`, `players_raw`, `fixtures`, `teams` | Points, minutes, price, ownership, pens/corners, GW xG (~2022/23+), DefCon **2025/26+** |
| vaastav Understat + `id_dict.csv` | npxG/npxA, shots, KP, xGChain/Buildup, positions; shot coords → shots in box |
| FPL API (live) | Status, prices, fixtures, `element-summary` |
| FBref (later) | CBIT before 2025/26, PPDA/tilt; cache/rate-limit |

**Identity table:** `fpl_id` ↔ `understat_id` ↔ `fbref_id` / `opta_code`.

**Era flag:** DefCon/points rules change in 2025/26 — do not mix eras as one scoring system.  
Drop `xP` (or lag it). Future fixture **scores** stay null until played.

---

## Database
Raw facts ≠ scores. SQLite + DuckDB for analytics.

- `dim_player`, `dim_team`, `map_player_ids`
- `fact_fpl_player_gw`, `fact_understat_player_match`, `fact_team_match`, `fact_fixture`
- `feat_player_asof` — features at deadline of GW \(T\) (**max round = \(T-1\)**)
- `score_player_asof` — Q, U, F, R, T, M, O, V, Overall, Upside

Leakage tests: no `round >= T`; upcoming fixtures have null results.

Weights live in `config/weights.yaml` (not hardcoded).

---

## Backtest
For each usable season, \(T = 8 \ldots 34\):

1. Features/scores as-of \(T\).
2. Predict GW \(T \ldots T+4\) (Overall rank + xPts).
3. Compare to actual FPL points.

**By position:** Spearman/Kendall, top-10/20 overlap, MAE/RMSE of xPts, £100m XI vs template.

If rank correlation is weak: **recalibrate YAML weights**, do not add more correlated leaves.

---

## Build order
0. Repo, YAML weights, pytest  
1. Ingest vaastav + Understat map  
2. `feat_player_asof` + leakage tests  
3. MID/FWD scores + player-card JSON  
4. Walk-forward backtest  
5. DEF, GK, Role/pens, DefCon  
6. FBref, live API, Upside  
7. UI / transfers (last)

**First milestone:** ingest → as-of features → MID/FWD Overall → Spearman on next 5 GWs.
