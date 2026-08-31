Weight-tune lab (isolated)
==========================

This folder is a sandbox. It does NOT:

- write to data/fpl.sqlite (live scores / features stay put)
- change config/weights.yaml
- run as `fpl-model score` or `fpl-model update`

It only reads the main SQLite file (read-only) and writes pickle caches +
CSV/YAML under this directory.


1. One-time setup (own venv)
-----------------------------
cd /Users/thomas/FPL-Model/experiments/weight_tune
./setup.sh
source .venv/bin/activate


2. Need historical GWs in the main DB
---------------------------------------
If 2024-25 / 2025-26 are not ingested yet, do this in the MAIN project venv
(not this lab):

  cd /Users/thomas/FPL-Model
  source .venv/bin/activate
  fpl-model ingest --season 2024-25 --source vaastav
  fpl-model ingest --season 2025-26 --source vaastav


3. Run the grid
---------------
cd /Users/thomas/FPL-Model/experiments/weight_tune
source .venv/bin/activate
python run.py

First pass builds feature caches as cache/{season}_gw{T}.pkl (slow).
Later runs only re-score those caches (fast). Edit grids.yaml and rerun.

Outputs (this folder only):

  results/grid_rows.csv      per GW / position / mix
  results/summary.csv         mean Spearman + top-20 by season
  results/holdout.txt        best mix on 2024-25, checked on 2025-26
  results/suggested.yaml     copy-paste snippet — you apply it to
                             config/weights.yaml yourself if you want


4. What it tunes
---------------
Factored search (not Overall × Quality × xPts):

- Overall Q/M/O/V × Quality U/F/R/T, with xPts left as YAML
- xPts recipes (start_l5, fixture tilt, cs_scale, bonus, residual, FWD scale)
  with Overall/Quality left as YAML

Named mixes are **nudges on live** `config/weights.yaml`, not fixed Q/M/O/V
blocks. `more_m` means +8pp minutes on whatever production currently uses
(and on each `overall_by_pos` row). `yaml_as_is` is always the live file.

Not the 15 MID underlying leaves.

Final mix pick (suggested.yaml / holdout): 0.7 Spearman + 0.3 Top-20, averaged
over positions, then 0.35 × 2024-25 + 0.65 × 2025-26. Edit ranking: in
grids.yaml to change those weights. Per-season combined tables in the UI use
the same score, without the season blend.


5. Lab UI (separate from live)
--------------------------------
Live cards stay on port 8766. This UI is amber-themed on 8777 and only
re-scores lab caches / mixes in memory.

  source .venv/bin/activate
  python serve_ui.py

Then open http://127.0.0.1:8777

Pick a mix, season, and as-of GW. Grid metrics needs results/ from run.py.
