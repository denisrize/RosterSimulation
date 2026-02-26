# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Roster Advisor is a single Python CLI package for professional road cycling roster optimization using XGBoost and TrueSkill. No web server, database, or Docker is needed. All interaction is via the `roster-advisor` CLI.

### Running the application

See `README.md` for CLI usage. Key commands:

- `roster-advisor list teams/races/schemes/status` — discovery
- `roster-advisor run --team "..." --race "..." --scheme equal_weight` — run simulation
- `roster-advisor recommend --csv results.csv --output recs.csv` — analyze results
- `roster-advisor download-data` — fetch Zenodo datasets (needed before first simulation)

### Non-obvious caveats

- **PATH**: The `roster-advisor` CLI installs to `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is set before invoking it.
- **Zenodo datasets**: The large CSV files (~1 GB total) in `roster_advisor/data/datasets/` are required for simulations and are `.gitignore`d. Run `roster-advisor download-data` after install. The `time_lag_trueskill_team.csv` file is currently unavailable on the configured Zenodo record; use `equal_weight` or `position` schemes instead.
- **No automated tests**: The repository has no test suite or test framework configured.
- **No linter configured**: There is no linting configuration (e.g., ruff, flake8, mypy) in the repo.
- **DtypeWarning on CSV load**: pandas emits `DtypeWarning` for mixed-type columns in `rider_features.csv` and TrueSkill CSVs — this is expected and harmless.
- **Simulation runtime**: Even a small simulation (10 riders, roster-size 4 = 210 combos) takes ~2.5 minutes. Full default simulations (16 riders, roster-size 8 = 12870 combos) can take much longer.
