# Analysis Module

This module analyzes roster simulation results to extract actionable recommendations for team roster construction. It provides two complementary analysis approaches for leader and helper identification.

## Overview

After running a roster simulation that evaluates all possible roster combinations, this module processes the results to identify:
1. **Team-based recommendations**: Who should lead based on team's best achievable result
2. **Individual-based recommendations**: Each rider's personal best potential and optimal supporting cast

---

## 1. Team-Based Analysis (default mode)

**Command:**
```bash
roster-advisor recommend --csv <simulation.csv> --output <output.csv>
```

### Logic

This analysis identifies leaders based on **who achieved the team's best rank** in each roster combination. The key concept is the **emergent leader** — the rider who naturally becomes the team's top finisher.

#### Calculation Steps:

1. **Parse each combination**: For each roster combination, identify:
   - `best_rider`: The rider with the lowest (best) predicted rank
   - `best_rank`: That rider's predicted finishing position

2. **Aggregate rider statistics**:
   - `total_occurrences`: How many combinations include this rider
   - `leader_occurrences`: How many times this rider was the `best_rider`
   - `personal_ranks`: All predicted finishing positions for this rider
   - `helper_best_team_rank`: Best team result when this rider is NOT the leader (or second-best rank when they ARE the leader)

3. **Group leaders by rank buckets**: Leaders are organized into rank groups (1-5, 6-10, 11-15, etc.)

4. **Calculate helper co-occurrence**: For each leader in each rank group, count how often each other rider appeared in the same combination

5. **Compute confidence scores**:
   ```
   helper_confidence = (helper_occurrences / total_combos_in_rank_group) × 100
   ```

### Output Fields

**Leaders CSV:**
| Field | Description |
|-------|-------------|
| `name` | Rider name |
| `rank_group` | Rank bucket (e.g., "Rnk(1,5)") |
| `occurrences_in_group` | Times this rider was leader in this rank bucket |
| `total_occurrences` | Total combinations including this rider |
| `avg_personal_rank` | Average finishing position across all combinations |
| `recommended_helpers` | Top 7 helpers with confidence scores |
| `example_combo_id` | Reference combination ID |
| `example_roster_with_ranks` | Full roster with predicted ranks |

**Helpers CSV:**
| Field | Description |
|-------|-------------|
| `name` | Rider name |
| `total_occurrences` | Times rider appeared in any combination |
| `avg_personal_rank` | Average personal finishing position |
| `helper_best_team_rank` | Best team result when acting as helper |
| `confidence` | Percentage of combinations including this rider |

### Example

Given simulation with 495 combinations for an 8-rider roster from 12 candidates:

```
Leaders Output:
ACKERMANN Pascal | Rnk(1,5) | 241 occurrences | 300 total occurrences | Helpers: VAN TRICHT (82.2%), BLACKMORE (67.6%)
STRONG Corbin    | Rnk(1,5) | 20 occurrences  | Helpers: VAN TRICHT (100%), BLACKMORE (95%)
```

**Interpretation**: ACKERMANN emerges as leader with a top-5 finish in 241 of 300 when he was evaluated in combination out of 495 combinations. When he achieves this, VAN TRICHT is in the roster 82.2% of the time — indicating a strong synergy.

---

## 2. Individual Leader Analysis (`--individual` flag)

**Command:**
```bash
roster-advisor recommend --csv <simulation.csv> --output <output.csv> --individual
```

### Logic

This analysis evaluates **each rider's personal best performance** regardless of whether they were the team's best finisher. It answers: "What is this rider's ceiling, and who should support them to reach it?"

#### Calculation Steps:

1. **Collect personal ranks**: For each rider, gather their predicted finishing position in every combination they appeared in

2. **Find personal best**: Identify each rider's minimum (best) predicted rank across all combinations

3. **Group by best rank bucket**: Find all combinations where the rider achieved their best rank group

4. **Calculate occurrence confidence**:
   ```
   confidence = (combos_in_best_rank_group / total_occurrences) × 100
   ```

5. **Identify optimal helpers**: Count helper co-occurrence in combinations where rider achieved their best rank group

### Output Fields

| Field | Description |
|-------|-------------|
| `name` | Rider name |
| `best_rank` | Personal best predicted finishing position |
| `best_rank_group` | Rank bucket for best rank (e.g., "Rnk(1,5)") |
| `occurrences_in_best_group` | Times rider achieved this rank group |
| `total_occurrences` | Total combinations including this rider |
| `confidence` | Percentage of combos achieving best rank group |
| `avg_personal_rank` | Average finishing position across all combinations |
| `recommended_helpers` | Top helpers when rider achieved best rank |
| `example_combo_id` | Reference combination ID |
| `example_roster_with_ranks` | Full roster with predicted ranks |

### Example

```
Individual Leaders Output:
ACKERMANN Pascal | Best: 5  | Rnk(1,5) | 241/330 (73.0%) | Helpers: VAN TRICHT (82.2%)
STRONG Corbin    | Best: 5  | Rnk(1,5) | 20/330 (6.1%)   | Helpers: VAN TRICHT (100%)
BLACKMORE Joseph | Best: 18 | Rnk(16,20) | 10/330 (3.0%) | Helpers: BOIVIN (100%)
```

**Interpretation**: ACKERMANN can achieve rank 5 in 73% of his combinations — a reliable leader option. STRONG can also achieve rank 5, but only in 6.1% of combinations — higher variance. BLACKMORE's ceiling is rank 18.

---

## Key Differences Between Analyses

| Aspect | Team-Based (`recommend`) | Individual (`recommend-individual`) |
|--------|--------------------------|-------------------------------------|
| Leader definition | Best finisher in the roster | Each rider individually |
| Metric focus | Team's best achievable result | Rider's personal ceiling |
| Use case | "Who should we designate as leader?" | "What can each rider achieve?" |
| Confidence meaning | How often rider beats teammates | How often rider reaches personal best |

---

## Practical Usage

### Scenario 1: Roster Construction
Use **team-based analysis** (default) to identify your primary leader candidates and their optimal supporting cast.
```bash
roster-advisor recommend --csv results/simulation.csv --output team_recommendations.csv
```

### Scenario 2: Rider Evaluation
Use **individual analysis** (`--individual`) to understand each rider's potential ceiling and which teammates help them perform best.
```bash
roster-advisor recommend --csv results/simulation.csv --output individual.csv --individual
```

### Scenario 3: Combined Strategy
1. Run individual analysis to identify riders with the best personal ceilings
2. Run team-based analysis to see which of those riders most consistently emerges as the team's best option
3. Cross-reference helper recommendations to find teammates that appear in both analyses

```bash
# Both analyses on same simulation
roster-advisor recommend --csv results/simulation.csv --output team.csv
roster-advisor recommend --csv results/simulation.csv --output individual.csv --individual
```

---

## Helper Confidence Interpretation

- **>80%**: Very strong synergy — this helper almost always appears in successful combinations
- **60-80%**: Good synergy — reliable supporting option
- **40-60%**: Moderate synergy — useful but not critical
- **<40%**: Weak synergy — may be interchangeable with other helpers

---

## Technical Notes

### Rank Groups
Ranks are bucketed in groups of 5: (1-5), (6-10), (11-15), etc. This grouping:
- Reduces noise from minor rank variations
- Aligns with typical race significance thresholds (top-5, top-10)
- Provides meaningful aggregation for statistical confidence

### Helper Best Team Rank
For the team-based analysis, `helper_best_team_rank` is calculated as:
- **When rider is helper**: The leader's rank (team's best result)
- **When rider is leader**: The second-best teammate's rank (simulating the rider as helper to their strongest teammate)

This ensures every combination contributes to each rider's helper statistics.
