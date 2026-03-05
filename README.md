# Roster Advisor

**A recommendation engine for data-driven roster optimization in professional road cycling.**

## About

Roster Advisor is a simulation tool that leverages machine learning to generate actionable roster recommendations for professional cycling teams. The engine evaluates thousands of roster combinations and provides confidence-scored recommendations for optimal leader-helper pairings.

**Key Capabilities:**
- **Simulate roster combinations** for specific races and evaluate predicted performance
- **Identify optimal leader-helper pairings** based on race profile and terrain
- **Analyze simulation results** to extract leader and helper recommendations with confidence scores
- **Evaluate individual rider potential** with supporting cast recommendations

## Table of Contents

- [Typical Workflow](#typical-workflow)
- [Data Setup](#data-setup)
- [Project Structure](#project-structure)
- [CLI Usage](#cli-usage)
- [Rider Pool Selection](#rider-pool-selection)
- [Recommendation Analysis](#recommendation-analysis)
- [Programmatic Usage](#programmatic-usage)
- [Dependencies](#dependencies)

## Typical Workflow

The system is designed around a sequential workflow. Each step builds on the previous one — you start by setting up data, explore what's available, run a simulation, and then analyze the results from two complementary angles. Below is the recommended order of operations with example commands; replace the team, race, and parameters with your own.

### Step 1: Install and Download Data

```bash
cd simulation_pkg
pip install -r requirements.txt

# Download all required datasets from Zenodo
python main.py download-data
```

### Step 2: Explore Available Resources

Before configuring a simulation, check which teams, races, and weighting schemes are available in the data, and verify that all required files are present.

```bash
python main.py list status    # Check data file availability
python main.py list teams     # List all teams in the catalog
python main.py list races     # List all races in the catalog
python main.py list schemes   # List schemes that have trained models
```

### Step 3: Inspect Target Race

Once you've chosen a race, inspect its profile to understand the terrain and race context — this helps you decide how to configure the rider pool.

```bash
python main.py info "Milano-Sanremo"
```

### Step 4: Run Simulation

Run the simulation for your chosen team and race. You control the rider pool size and distribution (see [Rider Pool Selection](#rider-pool-selection) for details). Use `--exclude-riders` to remove riders who left the team or are unavailable, and `--include-riders` to add new signings.

```bash
# Simple: even distribution across all four rating categories
python main.py run \
  --team "Israel - Premier Tech" \
  --race "E3 Saxo Classic" \
  --scheme equal_weight \
  --year 2025 \
  --num-cyclists 15 \
  --roster-size 8 \
  --output-dir results/simulation_results/ \
  --exclude-riders "GEE Derek,RICCITELLO Matthew,WOODS Michael,FUGLSANG Jakob,CLARKE Simon" \
  --include-riders "GIRMAY Biniam,PINARELLO Alessandro"

# Advanced: explicit distribution weighted toward race-cluster leaders
python main.py run \
  --team "Israel - Premier Tech" \
  --race "Giro d'Italia | Stage 1" \
  --scheme equal_weight \
  --year 2025 \
  --riders-pool "5,4,4,3" \
  --roster-size 8 \
  --output-dir results/simulation_results/ \
  --exclude-riders "GEE Derek,RICCITELLO Matthew,WOODS Michael,FUGLSANG Jakob,CLARKE Simon" \
  --uncertainty-penalty 3.0
```

The simulation outputs a CSV file containing all roster combinations with predicted ranks for every rider in each combination.

### Step 5: Generate Individual Rider Recommendations

Analyze the simulation results to see each rider's personal best achievable rank and the helpers that support their peak performance. This answers: *"What is each rider's ceiling, and who should surround them?"*

```bash
python main.py recommend \
  --csv results/simulation_results/Milano-Sanremo/Israel_-_Premier_Tech_Milano-Sanremo_Hills,\ flat\ finish_progress.csv \
  --output results/recommendations/Milano-Sanremo/IPT_recommendation.csv \
  --individual
```

### Step 6: Generate Top Ranks Recommendations

Analyze the same simulation results from a team perspective — identify which rider-roster configurations produce the best overall team ranks. Use `--top_ranks` to focus on the top N leader-rank entries. This answers: *"Which rosters unlock the best team results, and how do different helpers change a rider's performance?"*

```bash
python main.py recommend \
  --csv results/simulation_results/Milano-Sanremo/Israel_-_Premier_Tech_Milano-Sanremo_Hills,\ flat\ finish_progress.csv \
  --output results/recommendations/Milano-Sanremo/IPT_recommendation.json \
  --top_ranks 5
```

Both recommendation modes are described in detail under [Recommendation Analysis](#recommendation-analysis).

## Data Setup

The program requires CSV datasets that are too large for GitHub (>100MB). These files are hosted externally on Zenodo and must be downloaded before running simulations.

### Automatic Download

```bash
# Download all required datasets from Zenodo
python main.py download-data

# Check data status without downloading
python main.py download-data --status

# Force re-download existing files
python main.py download-data --force
```

### Manual Download

If automatic download fails, download the datasets manually from:
- **Zenodo Record:** https://zenodo.org/records/18493146

Place the following files in `roster_advisor/data/datasets/`:
- `rider_features.csv` - Historical rider features
- `trueskill_leader.csv` - Leader TrueSkill ratings (common)
- `equal_weight_trueskill_team.csv` - Team ratings (equal_weight scheme)
- `position_trueskill_team.csv` - Team ratings (position scheme)
- `time_lag_trueskill_team.csv` - Team ratings (time_lag scheme)

## Project Structure

```
simulation_pkg/
├── main.py                       # Entry point (python main.py <command>)
├── requirements.txt              # Python dependencies
│
└── roster_advisor/
    ├── cli.py                    # Command-line interface
    ├── __init__.py               # Module exports
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
python main.py run [OPTIONS]

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
python main.py recommend [OPTIONS]

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
python main.py download-data [OPTIONS]

Optional:
  --status            Only check data status without downloading
  --force             Re-download files even if they already exist
```

## Rider Pool Selection

In most cases you don't want to evaluate every possible rider on the team roster — the combinatorial explosion makes that impractical, and many riders are clearly unsuitable for a given race profile. Instead, the simulation narrows the team roster down to a **rider pool** of candidates before generating roster combinations. There are two ways to control who enters this pool.

### How It Works

Each rider on the team has TrueSkill ratings across four role categories:

| Category | Column prefix | Description |
|----------|---------------|-------------|
| `race_cluster_leader` | `race_cluster_leader_mu/sigma` | Leadership ability in races of the same cluster (e.g., hilly classics, flat sprints) |
| `gc_leader` | `gc_leader_mu/sigma` | Leadership ability in general classification (GC) stage races |
| `race_cluster_teammate` | `race_cluster_teammate_mu/sigma` | Helper/domestique value in races of the same cluster |
| `gc_teammate` | `gc_teammate_mu/sigma` | Helper/domestique value in GC stage races |

The selection algorithm picks the **top-rated riders** from each category using a conservative rating formula: `rating = μ - k × σ`, where `k` is the uncertainty penalty (default 3.0). Riders are selected greedily — once a rider is picked from one category, they won't be duplicated from another, so the pool contains only unique riders.

### Option 1: `--num-cyclists` (simple, even distribution)

When you only specify the total pool size, the budget is split **evenly** across the four categories. Any remainder from integer division is distributed to the first categories (leaders first).

```bash
python main.py run --team "Israel - Premier Tech" --race "E3 Saxo Classic" \
  --num-cyclists 16 --roster-size 8
```

With `--num-cyclists 16`, the distribution becomes `[4, 4, 4, 4]` — 4 riders from each category. With `--num-cyclists 15`, it becomes `[4, 4, 4, 3]` — the first three categories get 4 riders and the last (gc_teammate) gets 3.

### Option 2: `--riders-pool` (explicit distribution)

For finer control, you can specify exactly how many riders to pick from each category as a comma-separated list of four values: `race_cluster_leader, gc_leader, race_cluster_teammate, gc_teammate`.

```bash
python main.py run --team "Israel - Premier Tech" --race "Milano-Sanremo" \
  --riders-pool "5,4,4,3" --roster-size 8
```

This selects 5 race-cluster leaders, 4 GC leaders, 4 race-cluster helpers, and 3 GC helpers = 16 unique riders total. Weighting the distribution toward race-cluster leaders makes sense for a one-day classic like Milano-Sanremo, where race-specific leadership matters more than GC ability.

### Fine-tuning with `--exclude-riders` and `--include-riders`

You can further refine the pool by excluding riders who left the team or are unavailable, and including riders who may be new signings not yet reflected in the team roster data:

```bash
python main.py run --team "Israel - Premier Tech" --race "E3 Saxo Classic" \
  --num-cyclists 15 --roster-size 8 \
  --exclude-riders "WOODS Michael,FUGLSANG Jakob,CLARKE Simon" \
  --include-riders "GIRMAY Biniam,PINARELLO Alessandro"
```

Excluded riders are removed from the candidate list before selection. Included riders are added to the candidate list (if they exist in the dataset from any team) so they can be selected by the rating-based algorithm.

## Recommendation Analysis

The `recommend` command provides two complementary analysis modes that answer different questions about your simulation results. Both operate on the same simulation CSV but slice the data differently.

### 1. Individual Rider Analysis (`--individual`)

**Question answered:** *For each rider in the pool, what is their personal best achievable rank and which helpers should surround them?*

This mode evaluates **every rider** independently. For each rider it:
1. Finds their **personal best rank** across all roster combinations they appeared in
2. Identifies the rank group bucket (e.g., `Rnk(6,10)`) that best rank falls into
3. Collects all combinations where the rider achieved a rank within that bucket
4. Calculates **confidence** — the percentage of the rider's total appearances where they reached that best bucket
5. Ranks **recommended helpers** by how often they co-occurred in those best-bucket combinations

Each rider appears **exactly once** in the output, at their single best rank group.

```bash
python main.py recommend --csv results/simulation.csv --output individual.csv --individual
```

**Example output** (Israel - Premier Tech, Milano-Sanremo):

| name | best_rank | best_rank_group | occurrences_in_best_group | total_occurrences | confidence | avg_personal_rank | top helpers (confidence) |
|------|-----------|-----------------|---------------------------|-------------------|------------|-------------------|--------------------------|
| GIRMAY Biniam | 6 | Rnk(6,10) | 10600 | 10600 | 100.0% | 7.63 | ACKERMANN (100%), BENNETT (75.5%), BLACKMORE (49.7%) |
| ACKERMANN Pascal | 8 | Rnk(6,10) | 4828 | 10600 | 45.55% | 11.15 | BENNETT (71.9%), BLACKMORE (63.1%), BOIVIN (54.3%) |
| STRONG Corbin | 10 | Rnk(6,10) | 1044 | 10600 | 9.85% | 12.8 | ACKERMANN (100%), BENNETT (73.8%), BLACKMORE (59.2%) |
| PICKRELL Riley | 13 | Rnk(11,15) | 85 | 10600 | 0.87% | 19.02 | ACKERMANN (100%), RAISBERG (100%), BENNETT (76.5%) |

**Interpretation:** GIRMAY Biniam's personal best rank is 6. In every combination he appeared in (100% confidence) he achieved a rank within the Rnk(6,10) bucket. ACKERMANN Pascal appeared in all 10,600 combinations but only reached the Rnk(6,10) bucket in 45.55% of them — the remaining appearances placed him in weaker rank groups, meaning the supporting cast matters significantly for his performance.

### 2. Top Ranks Analysis (default)

**Question answered:** *Which rider-roster configurations produce the best team ranks, and how do different rosters change a rider's performance?*

This mode looks at team-level results: in each combination, the rider who achieved the **best rank** is treated as the leader. Results are grouped by **(rider, rank bucket)** pairs, which means the **same rider can appear multiple times** at different rank buckets when different rosters push them to different performance levels.

```bash
# Show all leader-rank entries
python main.py recommend --csv results/simulation.csv --output recs.json

# Show only the top 5 leader-rank entries
python main.py recommend --csv results/simulation.csv --output recs.json --top_ranks 5
```

**Example output** (Israel - Premier Tech, Milano-Sanremo, `--top_ranks 5`):

| name | rank_group | occurrences_in_group | recommended helpers (confidence) |
|------|------------|----------------------|----------------------------------|
| ACKERMANN Pascal | Rnk(6,10) | 4753 | BENNETT (72.0%), BLACKMORE (63.2%), BOIVIN (54.1%) |
| GIRMAY Biniam | Rnk(6,10) | 3978 | ACKERMANN (100%), BENNETT (75.5%), BLACKMORE (49.7%) |
| STRONG Corbin | Rnk(6,10) | 385 | ACKERMANN (100%), BENNETT (73.8%), BLACKMORE (59.2%) |
| **ACKERMANN Pascal** | **Rnk(11,15)** | **1366** | **BENNETT (89.3%), HOULE (57.3%), HOFSTETTER (54.0%)** |
| STRONG Corbin | Rnk(11,15) | 116 | ACKERMANN (100%), BLACKMORE (74.1%), STEWART (70.7%) |

Notice that **ACKERMANN Pascal appears twice**: once at Rnk(6,10) with 4,753 combinations and once at Rnk(11,15) with 1,366 combinations. This reveals that different rosters lead to different outcomes for the same rider. In his Rnk(6,10) appearances, BOIVIN Guillaume is a key helper (54.1%), but when he drops to Rnk(11,15), HOULE Hugo and HOFSTETTER Hugo become more prominent — suggesting that the absence of certain helpers is what causes the performance drop.

This mode also produces a **helpers summary** that ranks all riders by how frequently they appear across successful team combinations, along with their average personal rank and the best team rank they contributed to.

### Comparing the Two Modes

| Aspect | Individual (`--individual`) | Top Ranks (default) |
|--------|----------------------------|---------------------|
| Unit of analysis | Each rider's personal rank | Team's best rank per combination |
| Rider duplicates | Each rider appears once | Same rider can appear at multiple rank groups |
| Key insight | "What is each rider's ceiling?" | "Which rosters unlock the best team results?" |
| Helper meaning | "Who helps this rider reach their personal best?" | "Who is present when this rider leads the team at rank X?" |
| Best for | Scouting individual potential | Roster construction and lineup decisions |

See `analysis/README.md` for detailed documentation on the analysis logic.

## Programmatic Usage

The package can also be used as a Python library:

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

## Dependencies

- `pandas >= 1.3.0`
- `numpy >= 1.21.0, < 2.0.0`
- `xgboost >= 1.5.0`
- `tqdm >= 4.62.0`
- `trueskill`
