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
fpl-model init-db
fpl-model ingest --season 2025-26
fpl-model features --season 2025-26 --as-of 10
fpl-model score --season 2025-26 --as-of 10
fpl-model card --season 2025-26 --as-of 10 --name Salah
fpl-model backtest --season 2025-26
```

As-of GW `T` uses only GW `< T` plus fixtures known before that deadline.
