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
fpl-model ingest --season 2024-25
fpl-model features --season 2024-25 --as-of 10
fpl-model score --season 2024-25 --as-of 10
fpl-model card --season 2024-25 --as-of 10 --name Salah
fpl-model backtest --season 2024-25
```

As-of GW `T` uses only GW `< T` plus fixtures known before that deadline.
