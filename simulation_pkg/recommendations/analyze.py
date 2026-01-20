"""
Analyze roster simulation results and extract recommendations.
"""

from __future__ import annotations

import os
import json
from collections import Counter
from typing import Optional

import pandas as pd


def analyze_single_file(csv_file: str, output_file: Optional[str], top_leaders: int = 3):
    print(f"\n{'='*100}")
    print(f"Processing: {csv_file}")
    print(f"{'='*100}")

    is_time_trial = "Time Trial" in csv_file or "TT" in os.path.basename(csv_file)
    if is_time_trial:
        print("Time Trial detected - individual race format")

    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} combinations")

    rider_cols = [col for col in df.columns if col.startswith("rider_")]
    n_riders = len(rider_cols)
    print(f"✓ Roster size: {n_riders} riders per combination")

    rider_stats = {}

    for _, row in df.iterrows():
        best_rider = row["best_rider"]
        best_rank = row["best_rank"]

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

            if rider_stats[rider_name]["helper_best_team_rank"] is None:
                rider_stats[rider_name]["helper_best_team_rank"] = best_rank
            else:
                rider_stats[rider_name]["helper_best_team_rank"] = min(
                    rider_stats[rider_name]["helper_best_team_rank"], best_rank
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
                    "combo_id": row["combo_id"],
                })

    def get_rank_group(rank):
        if rank is None:
            return None
        group_start = ((rank - 1) // 5) * 5 + 1
        group_end = group_start + 4
        return f"Rnk({group_start},{group_end})"

    def parse_rank_group_start(rank_group):
        if not rank_group or not isinstance(rank_group, str):
            return 999
        try:
            start_str = rank_group.split("(")[1].split(",")[0]
            return int(start_str)
        except (IndexError, ValueError):
            return 999

    leader_entries = []
    for name, stats in rider_stats.items():
        if stats["leader_occurrences"] == 0:
            continue
        rank_bucket_combos = {}
        for ex in stats["leader_example_rows"]:
            rank_group = get_rank_group(ex["rank"])
            rank_bucket_combos.setdefault(rank_group, []).append(ex)

        for rank_group, combos in rank_bucket_combos.items():
            helper_counter = Counter()
            for combo in combos:
                helpers = [r for r in combo["roster"] if r != name]
                helper_counter.update(helpers)
            top_helpers = helper_counter.most_common(7)
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
    top_leaders_list = leader_entries[:top_leaders]

    recommendations = {
        "metadata": {
            "csv_file": csv_file,
            "total_combinations": len(df),
            "total_unique_riders": len(rider_stats),
            "top_leaders_count": top_leaders,
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
            leader_info["recommended_helpers"] = [
                {
                    "name": helper,
                    "occurrences": count,
                    "confidence": round(count / combos_count * 100, 1) if combos_count > 0 else 0,
                }
                for helper, count in entry["helpers"]
            ]
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
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        if output_file.endswith(".json"):
            import json
            with open(output_file, "w") as f:
                json.dump(recommendations, f, indent=2)
        elif output_file.endswith(".csv"):
            base_name = output_file.replace(".csv", "")
            leaders_df = pd.DataFrame(recommendations["leaders"])
            if "example_roster_with_ranks" in leaders_df.columns:
                leaders_df["example_roster_with_ranks"] = leaders_df["example_roster_with_ranks"].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else "{}"
                )
            leaders_df.to_csv(f"{base_name}_leaders.csv", index=False)

            helpers_df = pd.DataFrame(recommendations["helpers"])
            helpers_df.to_csv(f"{base_name}_helpers.csv", index=False)

    return recommendations
