# fpl-model

Explainable Fantasy Premier League suitability scores with leakage-safe backtests.

**Spec:** see [PLAN.md](PLAN.md) — scoring tree, metrics, data, and build order.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Pipeline

```bash
# Historical / calibration (keep 2025-26; do not mix GWs into 2026-27)
fpl-model ingest --season 2025-26 --source vaastav
fpl-model backtest --season 2025-26

# Live 2026-27: FPL API + Understat, then score the next 5 GWs
fpl-model update --skip-fbref
fpl-model card --season 2026-27 --as-of 2 --name Haaland
```

FBref is behind Cloudflare. Automated fetch usually fails; the parser works if you save the two league pages from a browser into `data/raw/fbref/2026-27/stats.html` and `defense.html`, then:

```bash
pip install -e ".[fbref]"
playwright install chromium   # optional auto-fetch attempt
fpl-model fbref --season 2026-27
```

As-of GW `T` uses only GW `< T` plus fixtures known before that deadline.
