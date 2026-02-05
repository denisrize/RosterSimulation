from __future__ import annotations

import argparse
import sys
from dataclasses import asdict

from .io.loaders import load_feature_columns
from .analysis.analyze import analyze_single_file, analyze_individual_leaders
from .engine.optimizer import RosterOptimizer
from .utils.download import download_data, print_data_status, check_data_status


def _print_error(message: str) -> None:
    """Print error message to stderr with formatting."""
    print(f"\n❌ Error: {message}\n", file=sys.stderr)


def _print_success(message: str) -> None:
    """Print success message."""
    print(f"\n✓ {message}\n")


def _run_simulation(args) -> None:
    """Run simulation using parameters with internal data retrieval."""
    from .utils.config import (
        create_config,
        ConfigurationError,
    )
    from .utils.data_registry import (
        RaceNotFoundError,
        TeamNotFoundError,
        SchemeNotFoundError,
        DataNotFoundError,
    )

    try:
        # Parse rider lists from comma-separated strings
        exclude_riders = None
        if args.exclude_riders:
            exclude_riders = [r.strip() for r in args.exclude_riders.split(",")]
        
        include_riders = None
        if args.include_riders:
            include_riders = [r.strip() for r in args.include_riders.split(",")]
        
        # Parse riders_pool if provided, otherwise use num_cyclists to create even distribution
        if args.riders_pool:
            riders_pool = [int(x.strip()) for x in args.riders_pool.split(",")]
            if len(riders_pool) != 4:
                _print_error("--riders-pool must have exactly 4 comma-separated values")
                sys.exit(1)
        else:
            # Convert num_cyclists to even distribution across 4 categories
            n_per_cat = args.num_cyclists // 4
            remainder = args.num_cyclists % 4
            riders_pool = [n_per_cat] * 4
            for i in range(remainder):
                riders_pool[i] += 1
        
        # Build configuration from parameters
        config = create_config(
            team=args.team,
            race=args.race,
            scheme=args.scheme,
            riders_pool=riders_pool,
            roster_size=args.roster_size,
            year=args.year,
            time_horizon=args.time_horizon,
            output_dir=args.output_dir,
            exclude_riders=exclude_riders,
            include_riders=include_riders,
            uncertainty_penalty=args.uncertainty_penalty,
        )
        
        # Feature columns are optional - they will be extracted from the model automatically
        feature_columns = None
        try:
            feature_columns = load_feature_columns(
                config.paths.feature_columns,
                config.paths.feature_columns_path,
            )
        except (ValueError, FileNotFoundError):
            # Feature columns will be extracted from the model instead
            pass
        
        # Create optimizer (feature columns will be auto-extracted from model if not provided)
        optimizer = RosterOptimizer(
            model_path=config.paths.model_path,
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
        
        # Run simulation
        race_context = asdict(config.run.race_context)
        optimizer.simulate_best_rosters(
            team_name=config.run.team_name,
            race_name=config.run.race_name,
            race_context=race_context,
            riders_pool=config.run.riders_pool,
            n_riders_per_roster=config.run.roster_size,
            save_path=config.run.output_dir,
            exclude_riders=config.run.exclude_riders,
            include_riders=config.run.include_riders,
            uncertainty_penalty=config.run.uncertainty_penalty,
        )
        
    except ConfigurationError as e:
        _print_error(str(e))
        sys.exit(1)
    except (RaceNotFoundError, TeamNotFoundError, SchemeNotFoundError) as e:
        _print_error(str(e))
        sys.exit(1)
    except DataNotFoundError as e:
        _print_error(f"Data not found:\n{e}\n\nPlease ensure all required data files are present in the internal data directory.")
        sys.exit(1)


def _list_resources(args) -> None:
    """List available resources (teams, races, schemes)."""
    from .utils.config import (
        list_available_teams,
        list_available_races,
        list_available_schemes,
        check_system_status,
    )
    from .utils.data_registry import DataNotFoundError
    
    try:
        if args.resource == "teams":
            teams = list_available_teams()
            if teams:
                print("\nAvailable Teams:")
                print("-" * 40)
                for team in sorted(teams):
                    print(f"  • {team}")
                print(f"\nTotal: {len(teams)} teams")
            else:
                print("\nNo teams found. Please add team_catalog.json to the data directory.")
                
        elif args.resource == "races":
            races = list_available_races()
            if races:
                print("\nAvailable Races:")
                print("-" * 40)
                for race in sorted(races):
                    print(f"  • {race}")
                print(f"\nTotal: {len(races)} races")
            else:
                print("\nNo races found. Please add race_catalog.json to the data directory.")
                
        elif args.resource == "schemes":
            schemes = list_available_schemes()
            if schemes:
                print("\nAvailable Schemes:")
                print("-" * 40)
                for scheme in schemes:
                    print(f"  • {scheme}")
            else:
                print("\nNo schemes with models found. Please add model files to the data directory.")
                
        elif args.resource == "all" or args.resource == "status":
            status = check_system_status()
            print("\nSystem Status:")
            print("=" * 50)
            print(f"Data directory exists: {'✓' if status['data_directory_exists'] else '✗'}")
            print(f"\nAvailable schemes (with models): {', '.join(status['available_schemes']) or 'None'}")
            
            print(f"\nCommon Datasets:")
            for dataset, exists in status['datasets']['common'].items():
                print(f"  {'✓' if exists else '✗'} {dataset}")
            
            print(f"\nScheme-Specific Datasets:")
            for scheme, datasets in status['datasets']['by_scheme'].items():
                print(f"  {scheme}:")
                for dataset, exists in datasets.items():
                    print(f"    {'✓' if exists else '✗'} {dataset}")
            
            print(f"\nCatalogs:")
            print(f"  {'✓' if status['race_catalog'] else '✗'} race_catalog.json")
            print(f"  {'✓' if status['team_catalog'] else '✗'} team_catalog.json")
            print(f"  {'✓' if status['feature_columns'] else '○'} feature_columns.json (optional - extracted from model)")
            
    except DataNotFoundError as e:
        _print_error(str(e))
        sys.exit(1)


def _show_race_info(args) -> None:
    """Show information about a specific race."""
    from .utils.config import get_race_info
    from .utils.data_registry import RaceNotFoundError
    
    try:
        info = get_race_info(args.race)
        print(f"\nRace: {args.race}")
        print("-" * 40)
        for key, value in info.items():
            print(f"  {key}: {value}")
    except RaceNotFoundError as e:
        _print_error(str(e))
        sys.exit(1)


def _show_model_features(args) -> None:
    """Show or export the feature names from a model."""
    from .models.xgb_wrapper import load_xgb_model, get_model_feature_names
    from .utils.data_registry import get_registry, SchemeNotFoundError
    import json
    
    try:
        # Get model path - either from scheme or direct path
        if args.model_path:
            model_path = args.model_path
        elif args.scheme:
            registry = get_registry()
            model_path = registry.get_model_path(args.scheme)
        else:
            _print_error("Either --scheme or --model-path is required")
            sys.exit(1)
        
        print(f"\nLoading model from: {model_path}")
        model = load_xgb_model(model_path)
        feature_names = get_model_feature_names(model)
        
        if feature_names is None:
            _print_error("Could not extract feature names from model.\nThe model may have been trained without feature names.")
            sys.exit(1)
        
        print(f"\nModel expects {len(feature_names)} features (in this exact order):\n")
        print("-" * 60)
        for i, name in enumerate(feature_names, 1):
            print(f"  {i:3d}. {name}")
        print("-" * 60)
        
        # Export to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(feature_names, f, indent=2)
            print(f"\n✓ Feature names exported to: {args.output}")
            print(f"  You can copy this to data/config/feature_columns.json")
        
    except SchemeNotFoundError as e:
        _print_error(str(e))
        sys.exit(1)
    except FileNotFoundError as e:
        _print_error(str(e))
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roster Advisor - Cycling team roster recommendation engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation
  roster-advisor run --team "Israel - Premier Tech" --race "Giro d'Italia" --scheme time_lag

  # Run with all options
  roster-advisor run -t "Israel - Premier Tech" -r "Giro d'Italia" -s time_lag \\
    --num-cyclists 18 --roster-size 8 --year 2026 --output-dir results

  # List available teams/races/schemes
  roster-advisor list teams
  roster-advisor list races
  roster-advisor list schemes
  roster-advisor list status

  # Get race information
  roster-advisor info "Giro d'Italia"

  # Analyze simulation results (team-based leaders)
  roster-advisor recommend --csv results/simulation.csv --output recommendations.csv

  # Analyze individual rider performance
  roster-advisor recommend --csv results/simulation.csv --output individual.csv --individual

  # Download required datasets from Zenodo
  roster-advisor download-data
  roster-advisor download-data --status  # Check data status only
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------------------------------------------------------------
    # 'run' command - Main simulation command
    # -------------------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run roster simulation",
        description="Run roster simulation. System retrieves data internally based on parameters.",
    )
    run_parser.add_argument(
        "--team", "-t",
        required=True,
        help="Team name (e.g., 'Israel - Premier Tech')",
    )
    run_parser.add_argument(
        "--race", "-r",
        required=True,
        help="Race name (e.g., 'Giro d\\'Italia')",
    )
    run_parser.add_argument(
        "--scheme", "-s",
        default="time_lag",
        choices=["time_lag", "equal_weight", "position"],
        help="Weighting scheme (default: time_lag)",
    )
    run_parser.add_argument(
        "--num-cyclists", "-n",
        type=int,
        default=16,
        dest="num_cyclists",
        help="Total cyclists in selection pool, distributed evenly across categories (default: 16)",
    )
    run_parser.add_argument(
        "--riders-pool",
        type=str,
        default=None,
        dest="riders_pool",
        help="Custom riders per category: 'N1,N2,N3,N4' for race_cluster_leader,gc_leader,"
             "race_cluster_teammate,gc_teammate (overrides --num-cyclists)",
    )
    run_parser.add_argument(
        "--roster-size",
        type=int,
        default=8,
        dest="roster_size",
        help="Number of cyclists per roster (default: 8)",
    )
    run_parser.add_argument(
        "--year", "-y",
        type=int,
        default=None,
        help="Season year (default: current year)",
    )
    run_parser.add_argument(
        "--time-horizon",
        type=int,
        default=None,
        dest="time_horizon",
        help="Days before race for feature cutoff (optional)",
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        default="results",
        dest="output_dir",
        help="Output directory for results (default: results)",
    )
    run_parser.add_argument(
        "--exclude-riders",
        default=None,
        dest="exclude_riders",
        help="Comma-separated list of riders to exclude (e.g., 'WOODS Michael,FUGLSANG Jakob')",
    )
    run_parser.add_argument(
        "--include-riders",
        default=None,
        dest="include_riders",
        help="Comma-separated list of riders to include (e.g., 'NEW Signing,ANOTHER Rider')",
    )
    run_parser.add_argument(
        "--uncertainty-penalty", "-k",
        type=float,
        default=3.0,
        dest="uncertainty_penalty",
        help="Penalty for uncertainty in rating: rating = mu - k*sigma (default: 3.0)",
    )

    # -------------------------------------------------------------------------
    # 'list' command
    # -------------------------------------------------------------------------
    list_parser = subparsers.add_parser(
        "list",
        help="List available resources (teams, races, schemes)",
    )
    list_parser.add_argument(
        "resource",
        choices=["teams", "races", "schemes", "status", "all"],
        help="Resource type to list",
    )

    # -------------------------------------------------------------------------
    # 'info' command
    # -------------------------------------------------------------------------
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a race",
    )
    info_parser.add_argument(
        "race",
        help="Race name to get information for",
    )

    # -------------------------------------------------------------------------
    # 'model-features' command
    # -------------------------------------------------------------------------
    features_parser = subparsers.add_parser(
        "model-features",
        help="Show feature names expected by a model",
        description="Extract and display the feature names a model was trained with. "
                    "Use this to ensure your feature_columns.json matches the model.",
    )
    features_group = features_parser.add_mutually_exclusive_group(required=True)
    features_group.add_argument(
        "--scheme", "-s",
        choices=["time_lag", "equal_weight", "position"],
        help="Load model for this scheme from internal data",
    )
    features_group.add_argument(
        "--model-path", "-m",
        dest="model_path",
        help="Direct path to model.json file",
    )
    features_parser.add_argument(
        "--output", "-o",
        help="Export feature names to JSON file (can be used as feature_columns.json)",
    )

    # -------------------------------------------------------------------------
    # 'recommend' command
    # -------------------------------------------------------------------------
    rec_parser = subparsers.add_parser(
        "recommend",
        help="Analyze simulation CSV for leader/helper recommendations",
        description="Analyze simulation results to identify optimal leaders and helpers. "
                    "Default mode analyzes team-based emergent leaders. Use --individual "
                    "to analyze each rider's personal best performance.",
    )
    rec_parser.add_argument(
        "--csv",
        required=True,
        help="Path to simulation results CSV",
    )
    rec_parser.add_argument(
        "--output",
        required=True,
        help="Output file path (.json or .csv)",
    )
    rec_parser.add_argument(
        "--individual",
        action="store_true",
        help="Analyze each rider's personal best rank instead of team-based leaders",
    )
    rec_parser.add_argument(
        "--top_leaders",
        type=int,
        default=None,
        help="Number of top leaders to show in team mode (default: all)",
    )
    rec_parser.add_argument(
        "--top_helpers",
        type=int,
        default=7,
        help="Number of top helpers per rider in individual mode (default: 7)",
    )

    # -------------------------------------------------------------------------
    # 'download-data' command
    # -------------------------------------------------------------------------
    download_parser = subparsers.add_parser(
        "download-data",
        help="Download required datasets from Zenodo",
        description="Download required CSV datasets from Zenodo external hosting. "
                    "These files are too large for GitHub and must be downloaded separately.",
    )
    download_parser.add_argument(
        "--status",
        action="store_true",
        help="Only check data status without downloading",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Command dispatch
    # -------------------------------------------------------------------------
    if args.command == "run":
        _run_simulation(args)
    elif args.command == "list":
        _list_resources(args)
    elif args.command == "info":
        _show_race_info(args)
    elif args.command == "model-features":
        _show_model_features(args)
    elif args.command == "recommend":
        if args.individual:
            analyze_individual_leaders(args.csv, args.output, top_helpers=args.top_helpers)
        else:
            analyze_single_file(args.csv, args.output, top_leaders=args.top_leaders)
    elif args.command == "download-data":
        if args.status:
            print_data_status()
        else:
            success = download_data(force=args.force, verbose=True)
            sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
