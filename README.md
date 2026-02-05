# Roster Advisor

**A recommendation engine for data-driven roster optimization in professional road cycling.**

## About

Roster Advisor is a recommendation system that leverages machine learning and simulation to generate actionable roster recommendations for professional cycling teams. The engine evaluates thousands of roster combinations and provides confidence-scored recommendations for optimal leader-helper pairings.

**Key Capabilities:**
- **Simulate roster combinations** for specific races and evaluate predicted performance
- **Identify optimal leader-helper pairings** based on race profile and terrain
- **Analyze simulation results** to extract leader and helper recommendations with confidence scores
- **Evaluate individual rider potential** with supporting cast recommendations

## Table of Contents

- [Quick Start](#quick-start)
- [Data Setup](#data-setup)
- [Package Structure](#package-structure)
- [CLI Usage](#cli-usage)
- [Recommendation Analysis](#recommendation-analysis)
- [Configuration](#configuration)
- [Output Examples](#output-examples)
- [Installation](#installation)

## Quick Start

### Running a Simulation

```bash
# Basic simulation
roster-advisor run --team "Israel - Premier Tech" --race "Giro d'Italia" --scheme time_lag

# With additional options
roster-advisor run \
  --team "Israel - Premier Tech" \
  --race "Giro d'Italia" \
  --scheme time_lag \
  --num-cyclists 18 \
  --roster-size 8 \
  --year 2026 \
  --output-dir results
```

### Analyzing Results

```bash
# Team-based analysis (identify emergent leaders)
roster-advisor recommend --csv results/simulation.csv --output recommendations.csv

# Individual rider analysis (evaluate each rider's potential)
roster-advisor recommend --csv results/simulation.csv --output individual.csv --individual
```

### Discovery Commands

```bash
roster-advisor list status    # Check data availability
roster-advisor list teams     # List available teams
roster-advisor list races     # List available races
roster-advisor list schemes   # List available schemes
roster-advisor info "Giro d'Italia"  # Get race information
```

## Data Setup

The package requires CSV datasets that are too large for GitHub (>100MB). These files are hosted externally on Zenodo and must be downloaded before running simulations.

### Automatic Download

```bash
# Download all required datasets from Zenodo
roster-advisor download-data

# Check data status without downloading
roster-advisor download-data --status

# Force re-download existing files
roster-advisor download-data --force
```

### Manual Download

If automatic download fails, download the datasets manually from:
- **Zenodo Record:** https://zenodo.org/records/17225472

Place the following files in `roster_advisor/data/datasets/`:
- `rider_features.csv` - Historical rider features
- `trueskill_leader.csv` - Leader TrueSkill ratings (common)
- `equal_weight_trueskill_team.csv` - Team ratings (equal_weight scheme)
- `position_trueskill_team.csv` - Team ratings (position scheme)
- `time_lag_trueskill_team.csv` - Team ratings (time_lag scheme)

### Verify Installation

```bash
# Check all data files are present
roster-advisor download-data --status

# Or use the list command
roster-advisor list status
```

### Programmatic Usage

```python
from roster_advisor import create_config, RosterOptimizer
from dataclasses import asdict

# Create config
config = create_config(
    team="Israel - Premier Tech",
    race="Giro d'Italia",
    scheme="time_lag",
    riders_pool=[4, 4, 4, 4],  # 16 riders total
    roster_size=8,
)

# Run simulation
optimizer = RosterOptimizer(
    model_path=config.paths.model_path,
    rider_features_path=config.paths.rider_features_path,
    trueskill_leader_path=config.paths.trueskill_leader_path,
    trueskill_team_path=config.paths.trueskill_team_path,
    ...
)

optimizer.simulate_best_rosters(
    team_name=config.run.team_name,
    race_name=config.run.race_name,
    race_context=asdict(config.run.race_context),
    ...
)
```

## Package Structure

```
roster_advisor/
├── cli.py                    # Command-line interface
├── __init__.py               # Package exports
│
├── engine/                   # Core recommendation engine
│   ├── optimizer.py          # Roster optimization and simulation
│   ├── features.py           # Feature aggregation for rosters
│   └── reference_race.py     # Reference race selection
│
├── analysis/                 # Results analysis module
│   ├── analyze.py            # Leader/helper recommendation analysis
│   └── README.md             # Detailed analysis documentation
│
├── utils/                    # Utility modules
│   ├── config.py             # Configuration management
│   ├── data_registry.py      # Data file registry and validation
│   └── types.py              # Type definitions
│
├── models/                   # ML model utilities
│   └── xgb_wrapper.py        # XGBoost model wrapper
│
├── io/                       # Data I/O
│   └── loaders.py            # Data loading utilities
│
└── data/                     # Internal data (models, datasets, catalogs)
    ├── models/               # Pre-trained XGBoost models
    ├── datasets/             # Rider features and TrueSkill ratings
    ├── races/                # Race catalog
    └── teams/                # Team catalog
```

## CLI Usage

### Simulation Command

```bash
roster-advisor run [OPTIONS]

Required:
  --team, -t          Team name (e.g., "Israel - Premier Tech")
  --race, -r          Race name (e.g., "Giro d'Italia")

Optional:
  --scheme, -s        Weighting scheme: time_lag, equal_weight, position (default: time_lag)
  --num-cyclists, -n  Total cyclists in selection pool (default: 16)
  --riders-pool       Custom riders per category: "N1,N2,N3,N4"
  --roster-size       Number of cyclists per roster (default: 8)
  --year, -y          Season year (default: current year)
  --time-horizon      Days before race for feature cutoff
  --output-dir, -o    Output directory (default: results)
  --exclude-riders    Comma-separated riders to exclude
  --include-riders    Comma-separated riders to include
  --uncertainty-penalty, -k  Rating penalty k in μ - kσ (default: 3.0)
```

### Recommendation Command

```bash
roster-advisor recommend [OPTIONS]

Required:
  --csv               Path to simulation results CSV
  --output            Output file path (.json or .csv)

Optional:
  --individual        Analyze each rider's personal best (instead of team-based)
  --top_leaders       Number of top leaders to show (team mode)
  --top_helpers       Number of top helpers per rider (individual mode, default: 7)
```

### Download Data Command

```bash
roster-advisor download-data [OPTIONS]

Optional:
  --status            Only check data status without downloading
  --force             Re-download files even if they already exist
```

## Recommendation Analysis

The analysis module provides two complementary approaches:

### 1. Team-Based Analysis (Default)

Identifies **emergent leaders** based on who achieved the team's best rank in each combination.

```bash
roster-advisor recommend --csv results/simulation.csv --output team_recs.csv
```

**Output includes:**
- Leaders grouped by rank buckets (1-5, 6-10, etc.)
- Recommended helpers with confidence scores
- Helper statistics (occurrence, avg rank, best team contribution)

### 2. Individual Analysis (`--individual`)

Evaluates each rider's **personal best potential** regardless of teammates.

```bash
roster-advisor recommend --csv results/simulation.csv --output individual.csv --individual
```

**Output includes:**
- Each rider's best achievable rank
- Confidence (how often they achieve their best rank group)
- Recommended helpers for their peak performance

See `analysis/README.md` for detailed documentation on the analysis logic.

## Configuration

### User Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `team` | Team name (must exist in team catalog) | Required |
| `race` | Race name (must exist in race catalog) | Required |
| `scheme` | Weighting scheme: `time_lag`, `equal_weight`, `position` | `time_lag` |
| `riders_pool` | Riders per category [leader, gc_leader, helper, gc_helper] | `[4,4,4,4]` |
| `roster_size` | Number of cyclists per roster | 8 |
| `year` | Season year | Current year |
| `time_horizon` | Days before race for feature cutoff | None |
| `output_dir` | Output directory for results | `results` |
| `exclude_riders` | Riders to exclude from pool | None |
| `include_riders` | Riders to include in pool | None |
| `uncertainty_penalty` | k in rating = μ - kσ | 3.0 |

### Programmatic Configuration

```python
from roster_advisor import create_config, UserConfig, ConfigBuilder

# Option 1: Use create_config convenience function
config = create_config(
    team="Israel - Premier Tech",
    race="Giro d'Italia",
    scheme="time_lag",
)

# Option 2: Build config manually
user_config = UserConfig(
    team_name="Israel - Premier Tech",
    race_name="Giro d'Italia",
    scheme="time_lag",
    riders_pool=[5, 4, 4, 3],
    roster_size=8,
)
builder = ConfigBuilder()
config = builder.build(user_config)
```

## Output Examples

### Leader Recommendations

| name | rank_group | occurrences | avg_rank | recommended_helpers |
|------|------------|-------------|----------|---------------------|
| ACKERMANN Pascal | Rnk(1,5) | 241 | 5.27 | VAN TRICHT (82.2%), BLACKMORE (67.6%) |
| STRONG Corbin | Rnk(1,5) | 20 | 6.63 | VAN TRICHT (100%), BLACKMORE (95%) |

**Interpretation:** ACKERMANN emerges as leader with top-5 finish in 241 combinations. VAN TRICHT appears in 82.2% of those combinations, indicating strong synergy.

### Helper Recommendations

| name | total_occurrences | avg_personal_rank | helper_best_team_rank | confidence |
|------|-------------------|-------------------|----------------------|------------|
| ACKERMANN Pascal | 330 | 5.27 | 6 | 66.67% |
| BLACKMORE Joseph | 330 | 23.75 | 5 | 66.67% |

**Interpretation:** BLACKMORE consistently appears in successful rosters and contributes to achieving top-5 team results.

## Installation

```bash
# Navigate to package directory
cd roster_advisor

# Install in development mode
pip install -e .

# Download required datasets from Zenodo
roster-advisor download-data

# Verify installation
roster-advisor --help
roster-advisor list status
```

## Dependencies

- `pandas >= 1.3.0`
- `numpy >= 1.21.0, < 2.0.0`
- `xgboost >= 1.5.0`
- `tqdm >= 4.62.0`
- `trueskill`
