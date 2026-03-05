"""
Analyze roster simulation results and extract recommendations.
"""

from __future__ import annotations

import os
import json
from collections import Counter
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd


# =============================================================================
# Shared Helper Functions
# =============================================================================

def get_rank_group(rank: Optional[float], bucket_size: int = 5) -> Optional[str]:
    """Convert a rank to a rank group bucket string.
    
    Args:
        rank: The rank value (can be None)
        bucket_size: Size of each bucket (default: 5)
        
    Returns:
        String like "Rnk(1,5)" or None if rank is None
    """
    if rank is None:
        return None
    group_start = ((int(rank) - 1) // bucket_size) * bucket_size + 1
    group_end = group_start + bucket_size - 1
    return f"Rnk({group_start},{group_end})"


def parse_rank_group_start(rank_group: Optional[str]) -> int:
    """Extract the starting rank from a rank group string for sorting.
    
    Args:
        rank_group: String like "Rnk(1,5)" or None
        
    Returns:
        Starting rank as integer, or 999 if parsing fails
    """
    if not rank_group or not isinstance(rank_group, str):
        return 999
    try:
        start_str = rank_group.split("(")[1].split(",")[0]
        return int(start_str)
    except (IndexError, ValueError):
        return 999


def parse_simulation_rows(df: pd.DataFrame) -> Tuple[int, List[Dict[str, Any]]]:
    """Parse simulation CSV rows into structured data.
    
    Args:
        df: DataFrame loaded from simulation CSV
        
    Returns:
        Tuple of (n_riders per roster, list of parsed row dicts)
    """
    rider_cols = [col for col in df.columns if col.startswith("rider_")]
    n_riders = len(rider_cols)
    
    parsed_rows = []
    for _, row in df.iterrows():
        roster_riders = []
        rider_ranks = {}
        
        for i in range(1, n_riders + 1):
            rider_col = f"rider_{i}"
            rank_col = f"race_rank_{i}"
            if rider_col in row and pd.notna(row[rider_col]):
                rider_name = row[rider_col]
                roster_riders.append(rider_name)
                if rank_col in row and pd.notna(row[rank_col]):
                    rider_ranks[rider_name] = row[rank_col]
        
        parsed_rows.append({
            "combo_id": row["combo_id"],
            "best_rider": row["best_rider"],
            "best_rank": row["best_rank"],
            "roster_riders": roster_riders,
            "rider_ranks": rider_ranks,
        })
    
    return n_riders, parsed_rows


def calculate_helper_confidence(
    combos: List[Dict[str, Any]], 
    leader_name: str, 
    top_n: int = 7
) -> List[Tuple[str, int]]:
    """Calculate helper co-occurrence counts for a leader's combinations.
    
    Args:
        combos: List of combination dicts with "roster" or "roster_riders" key
        leader_name: Name of the leader to exclude from helpers
        top_n: Number of top helpers to return
        
    Returns:
        List of (helper_name, count) tuples sorted by count descending
    """
    helper_counter = Counter()
    for combo in combos:
        roster = combo.get("roster") or combo.get("roster_riders", [])
        helpers = [r for r in roster if r != leader_name]
        helper_counter.update(helpers)
    return helper_counter.most_common(top_n)


def format_helpers_with_confidence(
    helpers: List[Tuple[str, int]], 
    total_combos: int
) -> List[Dict[str, Any]]:
    """Format helper tuples into dicts with confidence percentage.
    
    Args:
        helpers: List of (helper_name, count) tuples
        total_combos: Total number of combinations for confidence calculation
        
    Returns:
        List of dicts with name, occurrences, confidence
    """
    return [
        {
            "name": helper,
            "occurrences": count,
            "confidence": round(count / total_combos * 100, 1) if total_combos > 0 else 0,
        }
        for helper, count in helpers
    ]


def save_recommendations(
    recommendations: Dict[str, Any],
    output_file: str,
    leaders_key: str = "leaders",
    helpers_key: str = "helpers"
) -> None:
    """Save recommendations to JSON or CSV files.
    
    Args:
        recommendations: Dict containing metadata and results
        output_file: Output path (.json or .csv)
        leaders_key: Key for leaders data in recommendations
        helpers_key: Key for helpers data in recommendations (can be None)
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    if output_file.endswith(".json"):
        with open(output_file, "w") as f:
            json.dump(recommendations, f, indent=2)
    elif output_file.endswith(".csv"):
        base_name = output_file.replace(".csv", "")
        
        if leaders_key and leaders_key in recommendations:
            leaders_df = pd.DataFrame(recommendations[leaders_key])
            if "example_roster_with_ranks" in leaders_df.columns:
                leaders_df["example_roster_with_ranks"] = leaders_df["example_roster_with_ranks"].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else "{}"
                )
            leaders_df.to_csv(f"{base_name}_leaders.csv", index=False)
        
        if helpers_key and helpers_key in recommendations:
            helpers_df = pd.DataFrame(recommendations[helpers_key])
            helpers_df.to_csv(f"{base_name}_helpers.csv", index=False)


# =============================================================================
# Main Analysis Functions
# =============================================================================

def analyze_single_file(csv_file: str, output_file: Optional[str], top_ranks: Optional[int] = None):
    """Analyze simulation results based on team's best rider (emergent leader).
    
    This function identifies leaders based on who achieved the best rank in each
    roster combination. It calculates statistics for each rider as both a potential
    leader (when they had the best rank) and as a helper (contributing to team success).
    
    Args:
        csv_file: Path to simulation results CSV
        output_file: Output path for recommendations (.json or .csv)
        top_ranks: Limit number of ranks achieved by riders in the rider pool (None = all)
        
    Returns:
        Dict containing metadata, leaders, and helpers recommendations
    """
    print(f"\n{'='*100}")
    print(f"Processing: {csv_file}")
    print(f"{'='*100}")

    is_time_trial = "Time Trial" in csv_file or "TT" in os.path.basename(csv_file)
    if is_time_trial:
        print("Time Trial detected - individual race format")

    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} combinations")

    n_riders, parsed_rows = parse_simulation_rows(df)
    print(f"✓ Roster size: {n_riders} riders per combination")

    rider_stats = {}

    for row_data in parsed_rows:
        best_rider = row_data["best_rider"]
        best_rank = row_data["best_rank"]
        roster_riders = row_data["roster_riders"]
        rider_ranks = row_data["rider_ranks"]

        for rider_name in roster_riders:
            if rider_name not in rider_stats:
                rider_stats[rider_name] = {
                    "total_occurrences": 0,
                    "leader_occurrences": 0,
                    "leader_best_rank": None,
                    "helper_best_team_rank": None,
                    "leader_example_rows": [],
                    "personal_ranks": [],
                }

            rider_stats[rider_name]["total_occurrences"] += 1
            if rider_name in rider_ranks:
                rider_stats[rider_name]["personal_ranks"].append(rider_ranks[rider_name])

            # Determine the team rank to use for helper_best_team_rank
            if rider_name != best_rider:
                # Rider is a helper - use the leader's rank (best_rank)
                team_rank_for_helper = best_rank
            else:
                # Rider is the leader - use the second-best rank (next best teammate)
                other_ranks = [r for name, r in rider_ranks.items() if name != rider_name]
                team_rank_for_helper = min(other_ranks) if other_ranks else None

            if team_rank_for_helper is not None:
                if rider_stats[rider_name]["helper_best_team_rank"] is None:
                    rider_stats[rider_name]["helper_best_team_rank"] = team_rank_for_helper
                else:
                    rider_stats[rider_name]["helper_best_team_rank"] = min(
                        rider_stats[rider_name]["helper_best_team_rank"], team_rank_for_helper
                    )

            if rider_name == best_rider:
                rider_stats[rider_name]["leader_occurrences"] += 1
                if rider_stats[rider_name]["leader_best_rank"] is None:
                    rider_stats[rider_name]["leader_best_rank"] = best_rank
                else:
                    rider_stats[rider_name]["leader_best_rank"] = min(
                        rider_stats[rider_name]["leader_best_rank"], best_rank
                    )
                rider_stats[rider_name]["leader_example_rows"].append({
                    "rank": best_rank,
                    "roster": roster_riders.copy(),
                    "roster_with_ranks": {r: rider_ranks.get(r, None) for r in roster_riders},
                    "combo_id": row_data["combo_id"],
                })

    # Build leader entries grouped by rank bucket
    leader_entries = []
    for name, stats in rider_stats.items():
        if stats["leader_occurrences"] == 0:
            continue
        rank_bucket_combos = {}
        for ex in stats["leader_example_rows"]:
            rank_group = get_rank_group(ex["rank"])
            rank_bucket_combos.setdefault(rank_group, []).append(ex)

        for rank_group, combos in rank_bucket_combos.items():
            top_helpers = calculate_helper_confidence(combos, name)
            leader_entries.append({
                "name": name,
                "rank_group": rank_group,
                "occurrences_in_group": len(combos),
                "total_occurrences": stats["total_occurrences"],
                "avg_personal_rank": sum(stats["personal_ranks"]) / len(stats["personal_ranks"]) if stats["personal_ranks"] else None,
                "helpers": top_helpers,
                "example_combos": combos,
            })

    leader_entries.sort(key=lambda x: (parse_rank_group_start(x["rank_group"]), -x["occurrences_in_group"]))
    
    # Limit leaders if specified
    if top_ranks is None:
        top_leaders_list = leader_entries
    else:
        top_leaders_list = leader_entries[:top_ranks]

    recommendations = {
        "metadata": {
            "csv_file": csv_file,
            "total_combinations": len(df),
            "total_unique_riders": len(rider_stats),
            "top_leaders_count": len(top_leaders_list),
        },
        "leaders": [],
        "helpers": [],
    }

    for entry in top_leaders_list:
        leader_info = {
            "name": entry["name"],
            "rank_group": entry["rank_group"],
            "occurrences_in_group": entry["occurrences_in_group"],
            "total_occurrences": entry["total_occurrences"],
            "avg_personal_rank": round(entry["avg_personal_rank"], 2) if entry["avg_personal_rank"] else None,
        }
        if not is_time_trial and entry["helpers"]:
            combos_count = len(entry["example_combos"])
            leader_info["recommended_helpers"] = format_helpers_with_confidence(entry["helpers"], combos_count)
            if entry["example_combos"]:
                example = entry["example_combos"][0]
                leader_info["example_combo_id"] = example["combo_id"]
                leader_info["example_roster_with_ranks"] = example.get("roster_with_ranks", {})
        recommendations["leaders"].append(leader_info)

    helpers = [(name, stats) for name, stats in rider_stats.items()]
    helpers.sort(key=lambda x: -x[1]["total_occurrences"])
    for name, stats in helpers:
        avg_personal_rank = sum(stats["personal_ranks"]) / len(stats["personal_ranks"]) if stats["personal_ranks"] else None
        helper_info = {
            "name": name,
            "total_occurrences": stats["total_occurrences"],
            "avg_personal_rank": round(avg_personal_rank, 2) if avg_personal_rank else None,
            "helper_best_team_rank": stats["helper_best_team_rank"],
            "confidence": round(stats["total_occurrences"] / len(df) * 100, 2),
        }
        recommendations["helpers"].append(helper_info)

    if output_file:
        save_recommendations(recommendations, output_file)

    return recommendations


def analyze_individual_leaders(csv_file: str, output_file: Optional[str], top_helpers: int = 7):
    """Analyze each rider's personal best performance as a leader.
    
    Unlike analyze_single_file which looks at who was the team's best finisher,
    this function evaluates each rider individually based on their own personal
    best rank across all combinations they appeared in.
    
    For each rider:
    - Finds their best achieved rank (personal, not relative to teammates)
    - Groups combinations where they achieved ranks in that bucket
    - Calculates occurrence/confidence of achieving that rank
    - Identifies recommended helpers who appeared most often in those combinations
    
    Args:
        csv_file: Path to simulation results CSV
        output_file: Output path for recommendations (.json or .csv)
        top_helpers: Number of top helpers to include per rider (default: 7)
        
    Returns:
        Dict containing metadata and individual leader analysis
    """
    print(f"\n{'='*100}")
    print(f"Individual Leader Analysis: {csv_file}")
    print(f"{'='*100}")

    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} combinations")

    n_riders, parsed_rows = parse_simulation_rows(df)
    print(f"✓ Roster size: {n_riders} riders per combination")

    # Collect all combinations for each rider with their personal rank
    rider_combos: Dict[str, List[Dict[str, Any]]] = {}
    
    for row_data in parsed_rows:
        roster_riders = row_data["roster_riders"]
        rider_ranks = row_data["rider_ranks"]
        
        for rider_name in roster_riders:
            if rider_name not in rider_combos:
                rider_combos[rider_name] = []
            
            personal_rank = rider_ranks.get(rider_name)
            if personal_rank is not None:
                rider_combos[rider_name].append({
                    "combo_id": row_data["combo_id"],
                    "personal_rank": personal_rank,
                    "roster_riders": roster_riders.copy(),
                    "roster_with_ranks": {r: rider_ranks.get(r, None) for r in roster_riders},
                })

    print(f"✓ Found {len(rider_combos)} unique riders in pool")

    # Analyze each rider
    individual_leaders = []
    
    for rider_name, combos in rider_combos.items():
        if not combos:
            continue
            
        total_occurrences = len(combos)
        all_ranks = [c["personal_rank"] for c in combos]
        best_rank = min(all_ranks)
        avg_rank = sum(all_ranks) / len(all_ranks)
        best_rank_group = get_rank_group(best_rank)
        
        # Get combinations where rider achieved their best rank group
        best_group_combos = [
            c for c in combos 
            if get_rank_group(c["personal_rank"]) == best_rank_group
        ]
        
        occurrences_in_best_group = len(best_group_combos)
        confidence = round(occurrences_in_best_group / total_occurrences * 100, 2)
        
        # Calculate recommended helpers for best rank group
        top_helpers_list = calculate_helper_confidence(best_group_combos, rider_name, top_helpers)
        
        leader_entry = {
            "name": rider_name,
            "best_rank": int(best_rank),
            "best_rank_group": best_rank_group,
            "occurrences_in_best_group": occurrences_in_best_group,
            "total_occurrences": total_occurrences,
            "confidence": confidence,
            "avg_personal_rank": round(avg_rank, 2),
            "recommended_helpers": format_helpers_with_confidence(top_helpers_list, occurrences_in_best_group),
        }
        
        # Add example combination
        if best_group_combos:
            example = best_group_combos[0]
            leader_entry["example_combo_id"] = example["combo_id"]
            leader_entry["example_roster_with_ranks"] = example["roster_with_ranks"]
        
        individual_leaders.append(leader_entry)
    
    # Sort by best rank (ascending), then by confidence (descending)
    individual_leaders.sort(key=lambda x: (x["best_rank"], -x["confidence"]))
    
    recommendations = {
        "metadata": {
            "csv_file": csv_file,
            "total_combinations": len(df),
            "total_unique_riders": len(rider_combos),
            "analysis_type": "individual_leader",
            "description": "Each rider evaluated by their personal best rank across all combinations",
        },
        "individual_leaders": individual_leaders,
    }
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        if output_file.endswith(".json"):
            with open(output_file, "w") as f:
                json.dump(recommendations, f, indent=2)
        elif output_file.endswith(".csv"):
            leaders_df = pd.DataFrame(individual_leaders)
            if "example_roster_with_ranks" in leaders_df.columns:
                leaders_df["example_roster_with_ranks"] = leaders_df["example_roster_with_ranks"].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else "{}"
                )
            if "recommended_helpers" in leaders_df.columns:
                leaders_df["recommended_helpers"] = leaders_df["recommended_helpers"].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else "[]"
                )
            leaders_df.to_csv(output_file, index=False)
    
    print(f"✓ Analyzed {len(individual_leaders)} riders as individual leaders")
    
    return recommendations
