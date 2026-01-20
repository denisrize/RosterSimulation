"""
Batch processing for roster simulation recommendations.
"""

from __future__ import annotations

import os
import glob
import re
from typing import List

from .analyze import analyze_single_file


def extract_race_info(filename: str) -> str:
    basename = os.path.basename(filename)
    basename = basename.replace("_top10_progress.csv", "")
    parts = basename.split("_")
    race_start_idx = 0

    if "-" in parts:
        dash_idx = parts.index("-")
        potential_start = dash_idx + 2
        if potential_start + 1 < len(parts) and len(parts[potential_start]) < 7:
            potential_start += 1
        race_start_idx = potential_start if potential_start < len(parts) else 0

    if race_start_idx == 0 or race_start_idx >= len(parts):
        race_patterns = [
            "Tour de France", "Tour_de_France",
            "Giro d", "Giro_d",
            "Vuelta",
            "Milano-Sanremo", "Milano",
            "Santos Tour",
            "E3 Saxo", "E3_Saxo",
            "Paris-", "Paris_",
            "Liège", "Liege",
            "Strade",
        ]
        text = basename
        for pattern in race_patterns:
            if pattern in text:
                idx = text.find(pattern)
                race_start_idx = text[:idx].count("_")
                break

    if race_start_idx == 0:
        race_start_idx = min(4, max(0, len(parts) - 3))

    race_terrain = "_".join(parts[race_start_idx:])
    race_terrain = race_terrain.replace(",", "")
    race_terrain = race_terrain.replace(" ", "_")
    race_terrain = race_terrain.replace("'", "")
    race_terrain = re.sub(r"_+", "_", race_terrain)
    race_terrain = race_terrain.strip("_")
    return race_terrain


def process_all_simulations(input_dir: str, output_dir: str, top_leaders: int = 3) -> List[dict]:
    pattern = os.path.join(input_dir, "*_top10_progress.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No CSV files found matching pattern: {pattern}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        race_terrain = extract_race_info(filename)
        output_base = os.path.join(output_dir, race_terrain)

        try:
            recommendations = analyze_single_file(
                csv_file=csv_file,
                output_file=f"{output_base}.csv",
                top_leaders=top_leaders,
            )
            results.append({
                "input_file": filename,
                "race_terrain": race_terrain,
                "status": "success",
                "num_combinations": recommendations["metadata"]["total_combinations"],
                "num_riders": recommendations["metadata"]["total_unique_riders"],
                "leaders": [l["name"] for l in recommendations["leaders"]],
            })
        except Exception as exc:
            results.append({
                "input_file": filename,
                "race_terrain": race_terrain,
                "status": "failed",
                "error": str(exc),
            })

    summary_file = os.path.join(output_dir, "BATCH_PROCESSING_SUMMARY.txt")
    with open(summary_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("BATCH PROCESSING SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Total files processed: {len(results)}\n")
        f.write(f"Successful: {sum(1 for r in results if r['status'] == 'success')}\n")
        f.write(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}\n\n")

        for result in results:
            f.write(f"Input:  {result['input_file']}\n")
            f.write(f"Output: {result['race_terrain']}\n")
            f.write(f"Status: {result['status'].upper()}\n")
            if result["status"] == "success":
                f.write(f"  - Unique riders: {result['num_riders']}\n")
                f.write(f"  - Combinations analyzed: {result['num_combinations']}\n")
                f.write(f"  - Top leaders: {', '.join(result['leaders'])}\n")
            else:
                f.write(f"  - Error: {result.get('error', 'Unknown error')}\n")
            f.write("\n")

    return results
