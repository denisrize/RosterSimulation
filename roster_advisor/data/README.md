# Internal Data Directory

This directory contains all the internal data required for the simulation package to work with simplified user parameters.

## Directory Structure

```
data/
├── README.md                              # This file
├── models/                                # Pre-trained XGBoost models
│   ├── time_lag/
│   │   └── model.json                     # Model trained with 
│   ├── equal_weight/
│   │   └── model.json                     # Model trained with 
│   └── position/
│       └── model.json                     # Model trained with 
├── datasets/                              # Feature and rating datasets
│   ├── rider_features.csv                 # Historical rider features 
│   ├── trueskill_leader.csv               # Leader TrueSkill ratings 
│   ├── time_lag_trueskill_team.csv        # Team ratings for time_lag 
│   ├── equal_weight_trueskill_team.csv    # Team ratings for equal weight
│   └── position_trueskill_team.csv        # Team ratings for position scheme
├── races/
│   └── race_catalog.json                  # Valid races with context
├── teams/
│   └── team_catalog.json                  # Valid team names
└── config/
    └── feature_columns.json               # Feature column definitions
```

**Note:** Only the team/helper TrueSkill files are scheme-specific because helper ratings are computed differently based on the weighting method. Leader ratings remain the same across schemes. You only need to add team files for the schemes you plan to use.

## Required Files

### 1. Models (`models/`)

Place your trained XGBoost models in the appropriate scheme folder:

- `models/time_lag/model.json` - For time_lag weighting scheme
- `models/equal_weight/model.json` - For equal_weight weighting scheme  
- `models/position/model.json` - For position weighting scheme

Each model file should be a valid XGBoost JSON model exported via `model.save_model("model.json")`.

**Note:** At least one model is required. Only schemes with models will be available.

### 2. Datasets (`datasets/`)

#### `rider_features.csv` (Common)

Historical rider features dataset. Required columns:
- `rider` - Rider name (string)
- `team` - Team name at time of race (string)
- `date` - Date of race (datetime, format: YYYY-MM-DD)
- `year` - Season year (int)
- `race` - Race name (string)
- `cluster` - Race terrain profile (string)
- `classification` - Race classification (string: "WT", "Pro", "1", "2")
- Plus all feature columns used by the model

#### `trueskill_leader.csv` (Common)

Leader TrueSkill ratings. These are the same regardless of scheme.

Required columns:
- `rider` - Rider name (string)
- `date` - Date of rating update (datetime)
- `race` - Race name (string)
- `team` - Team name (string)
- `cluster` - Race cluster (string)
- `classification` - Race classification (string)
- `race_cluster_leader_mu` - Leader skill mean for race cluster (float)
- `race_cluster_leader_sigma` - Leader skill uncertainty (float)
- `gc_leader_mu` - GC leader skill mean (float)
- `gc_leader_sigma` - GC leader skill uncertainty (float)

#### `{scheme}_trueskill_team.csv` (Scheme-specific)

Teammate/helper TrueSkill ratings computed with the specific weighting scheme.

**Naming convention:** `time_lag_trueskill_team.csv`, `equal_weight_trueskill_team.csv`, `position_trueskill_team.csv`

Required columns:
- `rider` - Rider name (string)
- `date` - Date of rating update (datetime)
- `race` - Race name (string)
- `team` - Team name (string)
- `cluster` - Race cluster (string)
- `classification` - Race classification (string)
- `race_cluster_teammate_mu` - Teammate skill mean for race cluster (float)
- `race_cluster_teammate_sigma` - Teammate skill uncertainty (float)
- `gc_teammate_mu` - GC teammate skill mean (float)
- `gc_teammate_sigma` - GC teammate skill uncertainty (float)

**Important:** Only team/helper TrueSkill files are scheme-specific. You only need to add team files for the schemes you plan to use. For example, if you only use `time_lag`, you need `time_lag_trueskill_team.csv`.

### 3. Race Catalog (`races/race_catalog.json`)

JSON object mapping race names to their context. Example:

```json
{
  "Giro d'Italia": {
    "cluster": "Hills, uphill finish",
    "classification": "WT",
    "typical_date": "05-01",
    "distance": 180.0,
    "verticalMeters": 3500.0
  },
  "Tour de France": {
    "cluster": "Hills, uphill finish",
    "classification": "WT",
    "typical_date": "07-01",
    "distance": 180.0,
    "verticalMeters": 3800.0
  }
}
```

**Fields:**
- `cluster` - Race terrain profile (required)
  - Valid values: "Flat", "Hills, flat finish", "Hills, uphill finish", "Mountains, flat finish", "Mountains, uphill finish", "Time Trial"
- `classification` - Race tier (required)
  - Valid values: "WT", "Pro", "1", "2"
- `typical_date` - Default date in MM-DD format (required)
- `distance` - Typical stage/race distance in km (optional)
- `verticalMeters` - Typical elevation gain (optional)

### 4. Team Catalog (`teams/team_catalog.json`)

JSON array of valid team names. Example:

```json
[
  "Israel - Premier Tech",
  "UAE Team Emirates",
  "Visma - Lease a Bike",
  "INEOS Grenadiers"
]
```

**Important:** Team names must exactly match those in the datasets.

### 5. Feature Columns (`config/feature_columns.json`)

JSON array of feature column names expected by the model. Example:

```json
[
  "race_cluster_leader_mu",
  "race_cluster_leader_sigma",
  "gc_leader_mu",
  "gc_leader_sigma",
  "roster_helper_1_mu_race_cluster",
  "cluster_Flat",
  "cluster_Hills, flat finish",
  "StageRace",
  "race_class_ord"
]
```

This must match exactly what the model was trained on.

## Verification

Run the following to check your data setup:

```bash
roster-sim list status
```

This will show which files are present and which are missing.

## Usage

Once all files are in place:

```bash
# List available resources
roster-sim list teams
roster-sim list races
roster-sim list schemes

# Run a simulation
roster-sim run --team "Israel - Premier Tech" --race "Giro d'Italia" --scheme time_lag
```

Or programmatically:

```python
from simulation_pkg import create_config, RosterOptimizer

config = create_config(
    team="Israel - Premier Tech",
    race="Giro d'Italia",
    scheme="time_lag",
    num_cyclists=18,
    roster_size=8,
)

optimizer = RosterOptimizer.from_config(config)
```
