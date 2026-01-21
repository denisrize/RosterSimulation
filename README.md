## Roster Simulation Package

**A simulation engine for data-driven roster optimization in professional road cycling.**

## About

This package provides a practical simulation engine that leverages the VeloRost-Ex ranking model to generate actionable roster recommendations for upcoming race seasons. Built on top of the [VeloRost-Ex framework](https://github.com/denisrize/VeloRost-Ex), which uses Bayesian dual-skill modeling to predict race outcomes, this engine enables team managers to:

- **Simulate thousands of roster combinations** for specific races
- **Identify optimal leader-helper pairings** based on race profile and terrain
- **Receive confidence-scored recommendations** for strategic team selection
- **Plan rosters across an entire season** using predicted performance metrics

The engine takes trained models from the VeloRost-Ex pipeline and applies them to real-world roster planning scenarios, transforming predictive rankings into practical team management decisions.

**Key Links:**
- 🔗 **Training Framework**: [VeloRost-Ex](https://github.com/denisrize/VeloRost-Ex) - Model training and ranking pipeline
- 📊 **Dataset**: [VeloRost-Ex-Data](https://github.com/denisrize/VeloRost-Ex-Data) - Raw race results (2017-2023)

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
  - [Simulation Process](#simulation-process)
  - [Recommendation Logic](#recommendation-logic)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Output Examples](#output-examples)
- [Installation](#installation)

## Overview

The simulation package helps team managers answer critical questions:
- **Who should be the team leader** for a specific race?
- **Which helpers** work best with each potential leader?
- **What roster combination** maximizes the team's chances of success?

Instead of relying on intuition, the package:
1. Simulates all possible roster combinations from your rider pool
2. Predicts each combination's performance using the trained VeloRost-Ex model
3. Analyzes patterns to identify optimal leader-helper pairings
4. Generates ranked recommendations with confidence scores

## How It Works

### Simulation Process

The simulation engine (`optimizer.py`) follows these steps:

#### 1. **Rider Pool Selection**
Selects the top N riders (default: 18) from the team based on:
- **Profile-specific leader skills** (race cluster µ)
- **GC leader skills** (all-around capability)
- **Helper skills** (teammate µ for race profile and GC)

This creates a balanced pool representing:
- ~6 riders strong in the race's specific terrain (e.g., climbers for mountain stages)
- ~6 riders with strong GC capabilities (versatile all-rounders)
- ~6 riders with high helper skills (tactical support specialists)

#### 2. **Roster Combination Generation**
Generates all possible roster combinations:
```
C(N, roster_size) = C(18, 8) = 43,758 combinations
```

Each combination represents a unique 8-rider roster from the 18-rider pool.

#### 3. **Reference Race Selection**
Identifies a past race with similar characteristics:
- **Same race profile** (e.g., "Hills, flat finish")
- **Similar competition level** (e.g., WorldTour)
- **Recent race** (from training data)

The reference race provides the competitive field (all other teams' riders) that your simulated roster will compete against.

**Important Heuristic:** The simulation uses the following approach to create realistic race scenarios:
- **Roster Composition**: Uses the same rival teams and riders who participated in the reference race (typically from the previous season)
- **Updated Features**: Applies the most current rider features and TrueSkill ratings available at simulation time
- **Result**: A realistic competitive field with updated performance metrics

This ensures both the target team and rival participants reflect their most recent form and capabilities, while maintaining realistic roster compositions based on historical race participation patterns.

#### 4. **Performance Prediction**
For each roster combination:
1. **Constructs team features** by aggregating leader and helper TrueSkill ratings (using most recent available data)
2. **Merges with reference race field** (same riders from the reference race, but with updated features reflecting their current form)
3. **Updates all rider features** to the simulation date, ensuring both the target team and rivals have current performance metrics
4. **Predicts rankings** using the trained XGBoost model
5. **Extracts team metrics**:
   - `best_rank`: Best finishing position by any team rider
   - `best_rider`: Name of the rider achieving best_rank
   - `top_10_count`: Number of team riders finishing in top 10
   - `mean_rank`: Average finishing position across the roster

#### 5. **Optimization**
Tracks the top 10 unique roster combinations by prioritizing:
1. **Best rank** (primary criterion - minimize)
2. **Top-10 count** (secondary criterion - maximize)

Results are continuously saved to CSV for incremental analysis.

### Recommendation Logic

The recommendation analyzer (`analyze.py`) processes simulation results to extract actionable insights:

#### Leader Recommendations

**Identification Process:**
1. **Counts leader occurrences**: How many times each rider was the `best_rider` in their roster
2. **Groups by performance bracket**: Organizes results into rank groups (1-5, 6-10, 11-15, etc.)
3. **Finds helper co-occurrence**: For each leader, identifies which helpers most frequently appeared in the same successful rosters

**Metrics Computed:**
- `leader_best_rank`: Best finishing position achieved as team leader
- `avg_personal_rank`: Average predicted rank across all appearances
- `leader_occurrences`: Number of combinations where this rider led the team
- `total_occurrences`: Total appearances in any role
- `recommended_helpers`: Top 7 helpers with occurrence counts and confidence scores

**Confidence Calculation:**
```python
confidence = (helper_occurrences_with_leader / total_leader_occurrences) × 100
```

Example: If a helper appeared in 45 out of 50 combinations where a specific leader performed best, their confidence is 90%.

#### Helper Recommendations

**Ranking Process:**
1. **Total occurrences**: How many successful roster combinations included this rider
2. **Average personal rank**: Mean predicted finishing position
3. **Best team rank contribution**: Best team result when this rider was a helper
4. **Confidence**: Percentage of top combinations including this rider

**Key Insight:** High-confidence helpers appear in many successful rosters, regardless of who the leader is. These are versatile tactical assets.

## CLI Usage

```bash
# Run simulation for a specific race
roster-sim simulate --config path/to/sim_config.json

# Analyze a single simulation result
roster-sim recommend --csv path/to/top10_progress.csv --output out/recs --top_leaders 3

# Batch analyze multiple simulation results
roster-sim recommend-batch --input_dir results/roster_sims --output_dir out/recs --top_leaders 3
```

## Configuration

```json
{
  "paths": {
    "model_path": "models/model.json",
    "hyperparams_path": "models/hyperparams.json",
    "rider_features_path": "data_sets/rider_features/rider_features.csv",
    "trueskill_leader_path": "data_sets/leader_power/leader.csv",
    "trueskill_team_path": "data_sets/team_power/team.csv",
    "feature_columns_path": "configs/feature_columns.json",
    "clusters": ["Flat", "Hills, flat finish", "Hills, uphill finish",
      "Mountains, flat finish", "Mountains, uphill finish", "Time Trial"],
    "leader_feature_columns": ["race_cluster_leader_mu", "race_cluster_leader_sigma", "gc_leader_mu", "gc_leader_sigma"],
    "teammate_feature_columns": ["race_cluster_teammate_mu", "race_cluster_teammate_sigma", "gc_teammate_mu", "gc_teammate_sigma"]
  },
  "run": {
    "team_name": "Israel - Premier Tech",
    "race_name": "Giro d'Italia",
    "year": 2026,
    "level": "rider",
    "scheme": "time_lag",
    "pool_size": 18,
    "roster_size": 8,
    "top_k": 10,
    "output_dir": "results/roster_sims",
    "race_context": {
      "cluster": "Hills, uphill finish",
      "classification": "WT",
      "date": "2026-05-01",
      "distance": 180.0,
      "verticalMeters": 3000.0
    }
  }
}
```

**Key Parameters:**
- `pool_size`: Number of riders to consider (default: 18)
- `roster_size`: Number of riders per roster (default: 8)
- `top_k`: Number of top combinations to save
- `race_context.cluster`: Race terrain profile
- `race_context.classification`: Competition tier (WT, Pro, 1, 2)
- `race_context.date`: Race date for feature extraction cutoff

## Output Examples

### Leader Recommendations

Example output from **E3 Saxo Classic** (Hills, flat finish):

| name | leader_best_rank | avg_personal_rank | leader_occurrences | total_occurrences | recommended_helpers |
|------|------------------|-------------------|-------------------|-------------------|---------------------|
| **GIRMAY Biniam** | 2 | 3.04 | 5,874 | 5,874 | ACKERMANN Pascal (99%), STRONG Corbin (98%), PICKRELL Riley (97%) |
| **ACKERMANN Pascal** | 4 | 5.67 | 2,823 | 8,987 | STRONG Corbin (95%), PICKRELL Riley (93%), BLACKMORE Joseph (89%) |
| **STRONG Corbin** | 4 | 6.77 | 290 | 3,568 | ACKERMANN Pascal (94%), PICKRELL Riley (91%), HOFSTETTER Hugo (87%) |

**Interpretation:**
- **GIRMAY Biniam** is the strongest leader candidate:
  - Expected to finish **2nd place** in the best scenario
  - Led the team in **5,874 out of ~9,000** evaluated combinations (65%)
  - Works best with **ACKERMANN Pascal** (appeared together in 99% of successful rosters)
  
- **ACKERMANN Pascal** is a secondary option:
  - Can achieve **4th place** as leader
  - Also appears frequently as a helper (8,987 total appearances)
  - Dual-role capability makes him valuable for roster flexibility

### Helper Recommendations

Example helper rankings for the same race:

| name | total_occurrences | avg_personal_rank | helper_best_team_rank | confidence |
|------|-------------------|-------------------|----------------------|------------|
| **ASKEY Lewis** | 4,771 | 31.95 | 2 | 53.09% |
| **BLACKMORE Joseph** | 4,406 | 18.54 | 2 | 49.03% |
| **BENNETT George** | 4,392 | 101.13 | 2 | 48.87% |
| **PICKRELL Riley** | 3,559 | 9.61 | 2 | 39.60% |
| **HOFSTETTER Hugo** | 2,857 | 14.50 | 2 | 31.79% |

**Interpretation:**
- **ASKEY Lewis** appears in **53%** of top combinations:
  - Contributed to rosters achieving the **best team rank of 2nd place**
  - Average personal finish: **~32nd place** (tactical helper role)
  - High confidence indicates consistent value regardless of leader choice

- **PICKRELL Riley** shows dual capability:
  - Average personal rank of **9.61** suggests podium potential
  - Can serve as either primary helper or backup leader
  - Lower confidence (39.6%) indicates more selective pairing

**Helper Categories Identified:**
1. **Tactical Specialists** (e.g., ASKEY Lewis): High occurrence, moderate personal rank - pure support role
2. **Dual-Threat Riders** (e.g., PICKRELL Riley): High occurrence, strong personal rank - can lead or support
3. **Depth Options** (e.g., BENNETT George): Frequent appearance despite poor personal rank - reliable tactical asset

### Simulation Output Files

Each simulation produces:

1. **`{team}_{race}_{profile}_top10_progress.csv`**: Top 10 roster combinations with detailed metrics
2. **`{team}_{race}_{profile}_leaders.csv`**: Leader recommendations with helper pairings
3. **`{team}_{race}_{profile}_helpers.csv`**: Helper rankings by occurrence and confidence
4. **`{team}_{race}_{profile}_README.txt`**: Column descriptions and metadata

## Installation

```bash
# Install the package in development mode
cd simulation_pkg
pip install -e .

# Verify installation
roster-sim --help
```

## Dependencies

- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `xgboost >= 1.5.0`
- `tqdm >= 4.62.0`

## Related Packages

This simulation package works in conjunction with:
- **[VeloRost-Ex](https://github.com/denisrize/VeloRost-Ex)**: Main training framework for the ranking model
- **[VeloRost-Ex-Data](https://github.com/denisrize/VeloRost-Ex-Data)**: Raw race results dataset (2017-2023)
