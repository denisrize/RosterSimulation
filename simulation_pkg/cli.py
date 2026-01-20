from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from .config import load_config
from .io.loaders import load_feature_columns
from .recommendations.analyze import analyze_single_file
from .recommendations.batch import process_all_simulations
from .simulation.optimizer import RosterOptimizer


def _run_simulation(config_path: str) -> None:
    config = load_config(config_path)
    feature_columns = load_feature_columns(
        config.paths.feature_columns,
        config.paths.feature_columns_path,
    )

    optimizer = RosterOptimizer(
        model_path=config.paths.model_path,
        hyperparams_path=config.paths.hyperparams_path,
        rider_features_path=config.paths.rider_features_path,
        trueskill_leader_path=config.paths.trueskill_leader_path,
        trueskill_team_path=config.paths.trueskill_team_path,
        feature_columns=feature_columns,
        leader_feature_columns=config.paths.leader_feature_columns,
        teammate_feature_columns=config.paths.teammate_feature_columns,
        clusters=config.paths.clusters,
        race_class="all",
        scheme=config.run.scheme,
        year=config.run.year,
        level=config.run.level,
        time_gap=config.run.time_gap,
    )

    race_context = asdict(config.run.race_context)
    optimizer.simulate_best_rosters(
        team_name=config.run.team_name,
        race_name=config.run.race_name,
        race_context=race_context,
        N=config.run.pool_size,
        n_riders_per_roster=config.run.roster_size,
        save_path=config.run.output_dir,
        top_k=config.run.top_k,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Roster simulation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="Run roster simulation")
    simulate_parser.add_argument("--config", required=True, help="Path to JSON/YAML config file")

    rec_parser = subparsers.add_parser("recommend", help="Analyze a single simulation CSV")
    rec_parser.add_argument("--csv", required=True, help="Path to *_top10_progress.csv")
    rec_parser.add_argument("--output", required=True, help="Output file path (.json or .csv base)")
    rec_parser.add_argument("--top_leaders", type=int, default=3, help="Number of top leaders")

    batch_parser = subparsers.add_parser("recommend-batch", help="Analyze all simulation CSVs in a directory")
    batch_parser.add_argument("--input_dir", required=True, help="Directory with *_top10_progress.csv files")
    batch_parser.add_argument("--output_dir", required=True, help="Directory for recommendation outputs")
    batch_parser.add_argument("--top_leaders", type=int, default=3, help="Number of top leaders")

    args = parser.parse_args()

    if args.command == "simulate":
        _run_simulation(args.config)
    elif args.command == "recommend":
        analyze_single_file(args.csv, args.output, top_leaders=args.top_leaders)
    elif args.command == "recommend-batch":
        process_all_simulations(args.input_dir, args.output_dir, top_leaders=args.top_leaders)


if __name__ == "__main__":
    main()
