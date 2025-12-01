#!/usr/bin/env python3
"""Visualize sensor distributions for occupancy detection.

This script visualizes the learned distributions for numeric sensors,
showing both the raw data histograms and the fitted Gaussian distributions.

Usage:
    python scripts/visualize_distributions.py <area_name> <entity_id> [--db-path PATH] [--days DAYS] [--bins BINS]

Example:
    python scripts/visualize_distributions.py "Living Room" sensor.temperature --days 30
"""

import argparse
from datetime import UTC, datetime, timedelta
import math
from pathlib import Path
import sys

# ruff: noqa: T201, E402, PLC0415
# Add the project root to the path before other imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from custom_components.area_occupancy.const import DB_NAME
from custom_components.area_occupancy.db.schema import (
    Correlations,
    NumericSamples,
    OccupiedIntervalsCache,
)
from custom_components.area_occupancy.db.utils import ensure_timezone_aware


def gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Calculate Gaussian probability density function.

    Args:
        x: Array of values to evaluate
        mean: Mean of the distribution
        std: Standard deviation of the distribution

    Returns:
        Array of probability densities
    """
    if std <= 0:
        return np.zeros_like(x)
    exponent = -0.5 * ((x - mean) / std) ** 2
    return (1.0 / (std * math.sqrt(2 * math.pi))) * np.exp(exponent)


def is_timestamp_occupied(
    ts: datetime, intervals: list[tuple[datetime, datetime]]
) -> bool:
    """Check if a timestamp falls within any occupied interval.

    Args:
        ts: Timestamp to check
        intervals: List of (start, end) tuples for occupied intervals

    Returns:
        True if timestamp is within an occupied interval
    """
    ts_aware = ensure_timezone_aware(ts)
    return any(start <= ts_aware <= end for start, end in intervals)


def visualize_sensor_distribution(
    db_path: Path,
    entry_id: str,
    area_name: str,
    entity_id: str,
    analysis_period_days: int = 30,
    bins: int = 50,
    output_file: Path | None = None,
) -> None:
    """Visualize the distribution of sensor values for occupied vs unoccupied states.

    Args:
        db_path: Path to the database file
        entry_id: Config entry ID
        area_name: Area name
        entity_id: Entity ID to visualize
        analysis_period_days: Number of days to analyze
        bins: Number of histogram bins
        output_file: Path to save the visualization to
    """
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        sys.exit(1)

    # Create database engine
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        period_end = datetime.now(UTC).replace(tzinfo=None)
        period_start = period_end - timedelta(days=analysis_period_days)
        period_start_utc = ensure_timezone_aware(period_start)
        period_end_utc = ensure_timezone_aware(period_end)

        # Get raw samples
        samples = (
            session.query(NumericSamples)
            .filter(
                NumericSamples.entry_id == entry_id,
                NumericSamples.area_name == area_name,
                NumericSamples.entity_id == entity_id,
                NumericSamples.timestamp >= period_start_utc,
                NumericSamples.timestamp <= period_end_utc,
            )
            .order_by(NumericSamples.timestamp)
            .all()
        )

        if not samples:
            print(
                f"No samples found for {entity_id} in area {area_name} "
                f"between {period_start_utc} and {period_end_utc}"
            )
            return

        # Get occupied intervals
        occupied_intervals_raw = (
            session.query(OccupiedIntervalsCache)
            .filter(
                OccupiedIntervalsCache.entry_id == entry_id,
                OccupiedIntervalsCache.area_name == area_name,
                OccupiedIntervalsCache.start_time <= period_end_utc,
                OccupiedIntervalsCache.end_time >= period_start_utc,
            )
            .all()
        )

        occupied_intervals = [
            (
                ensure_timezone_aware(interval.start_time),
                ensure_timezone_aware(interval.end_time),
            )
            for interval in occupied_intervals_raw
        ]

        # Separate samples by occupancy
        occupied_values = []
        unoccupied_values = []

        for sample in samples:
            try:
                value = float(sample.value)
            except (ValueError, TypeError):
                continue

            sample_ts = ensure_timezone_aware(sample.timestamp)
            is_occupied = is_timestamp_occupied(sample_ts, occupied_intervals)

            if is_occupied:
                occupied_values.append(value)
            else:
                unoccupied_values.append(value)

        if not occupied_values:
            print(f"Error: No occupied samples found for {entity_id}")
            return
        if not unoccupied_values:
            print(f"Error: No unoccupied samples found for {entity_id}")
            return

        # Get learned Gaussian parameters
        correlation = (
            session.query(Correlations)
            .filter_by(
                entry_id=entry_id,
                area_name=area_name,
                entity_id=entity_id,
            )
            .order_by(Correlations.calculation_date.desc())
            .first()
        )

        # Create figure with subplots
        _, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Histogram with Gaussian overlays
        ax1 = axes[0]

        # Determine value range for plotting
        all_values = occupied_values + unoccupied_values
        value_min = min(all_values)
        value_max = max(all_values)
        value_range = value_max - value_min
        plot_min = value_min - 0.1 * value_range
        plot_max = value_max + 0.1 * value_range

        # Create histogram bins
        bin_edges = np.linspace(plot_min, plot_max, bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]

        # Plot histograms
        ax1.hist(
            occupied_values,
            bins=bin_edges,
            alpha=0.6,
            label=f"Occupied (n={len(occupied_values)})",
            color="red",
            density=False,
        )

        ax1.hist(
            unoccupied_values,
            bins=bin_edges,
            alpha=0.6,
            label=f"Unoccupied (n={len(unoccupied_values)})",
            color="blue",
            density=False,
        )

        # Overlay Gaussian distributions if available
        if correlation and correlation.mean_value_when_occupied is not None:
            # Create smooth curve for Gaussian
            x_smooth = np.linspace(plot_min, plot_max, 1000)

            # Occupied Gaussian
            mean_occ = correlation.mean_value_when_occupied
            std_occ = correlation.std_dev_when_occupied or 1.0
            gauss_occ = gaussian_pdf(x_smooth, mean_occ, std_occ)
            # Scale to match histogram (multiply by sample count and bin width)
            gauss_occ_scaled = gauss_occ * len(occupied_values) * bin_width
            ax1.plot(
                x_smooth,
                gauss_occ_scaled,
                "r--",
                linewidth=2,
                label=f"Gaussian (μ={mean_occ:.2f}, σ={std_occ:.2f})",
            )

            # Unoccupied Gaussian
            mean_unocc = correlation.mean_value_when_unoccupied
            std_unocc = correlation.std_dev_when_unoccupied or 1.0
            gauss_unocc = gaussian_pdf(x_smooth, mean_unocc, std_unocc)
            gauss_unocc_scaled = gauss_unocc * len(unoccupied_values) * bin_width
            ax1.plot(
                x_smooth,
                gauss_unocc_scaled,
                "b--",
                linewidth=2,
                label=f"Gaussian (μ={mean_unocc:.2f}, σ={std_unocc:.2f})",
            )

        ax1.set_xlabel("Sensor Value")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Sensor Distribution: {entity_id}\nArea: {area_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Probability density functions (normalized)
        ax2 = axes[1]

        # Plot normalized histograms
        ax2.hist(
            occupied_values,
            bins=bin_edges,
            alpha=0.6,
            label="Occupied (normalized)",
            color="red",
            density=True,
        )

        ax2.hist(
            unoccupied_values,
            bins=bin_edges,
            alpha=0.6,
            label="Unoccupied (normalized)",
            color="blue",
            density=True,
        )

        # Overlay Gaussian PDFs
        if correlation and correlation.mean_value_when_occupied is not None:
            x_smooth = np.linspace(plot_min, plot_max, 1000)

            mean_occ = correlation.mean_value_when_occupied
            std_occ = correlation.std_dev_when_occupied or 1.0
            gauss_occ = gaussian_pdf(x_smooth, mean_occ, std_occ)
            ax2.plot(x_smooth, gauss_occ, "r--", linewidth=2, label="Occupied PDF")

            mean_unocc = correlation.mean_value_when_unoccupied
            std_unocc = correlation.std_dev_when_unoccupied or 1.0
            gauss_unocc = gaussian_pdf(x_smooth, mean_unocc, std_unocc)
            ax2.plot(x_smooth, gauss_unocc, "b--", linewidth=2, label="Unoccupied PDF")

        ax2.set_xlabel("Sensor Value")
        ax2.set_ylabel("Probability Density")
        ax2.set_title("Probability Density Functions (PDFs)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Print statistics
        print(f"\nStatistics for {entity_id} in {area_name}:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Occupied samples: {len(occupied_values)}")
        print(f"  Unoccupied samples: {len(unoccupied_values)}")
        print(
            f"\n  Occupied - Mean: {np.mean(occupied_values):.2f}, "
            f"Std: {np.std(occupied_values):.2f}"
        )
        print(
            f"  Unoccupied - Mean: {np.mean(unoccupied_values):.2f}, "
            f"Std: {np.std(unoccupied_values):.2f}"
        )

        if correlation:
            print("\n  Learned Parameters:")
            print(f"    Correlation: {correlation.correlation_coefficient:.3f}")
            print(f"    Correlation Type: {correlation.correlation_type}")
            if correlation.confidence:
                print(f"    Confidence: {correlation.confidence:.3f}")
            print(
                f"    Occupied - μ={correlation.mean_value_when_occupied:.2f}, "
                f"σ={correlation.std_dev_when_occupied:.2f}"
            )
            print(
                f"    Unoccupied - μ={correlation.mean_value_when_unoccupied:.2f}, "
                f"σ={correlation.std_dev_when_unoccupied:.2f}"
            )
        else:
            print("\n  No learned correlation parameters found in database.")

        plt.tight_layout()

        if output_file:
            # Save to file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"\n  Visualization saved to: {output_file}")
            plt.close()
        else:
            # Display interactively
            plt.show()

    finally:
        session.close()
        engine.dispose()


def find_entry_id(session: Any, area_name: str) -> str | None:
    """Find the entry_id for a given area name.

    Args:
        session: Database session
        area_name: Area name to search for

    Returns:
        Entry ID if found, None otherwise
    """
    from custom_components.area_occupancy.db.schema import Areas

    area = session.query(Areas).filter_by(area_name=area_name).first()
    return area.entry_id if area else None


def list_available_entities(db_path: Path, area_name: str | None = None) -> None:
    """List all available entities in the database.

    Args:
        db_path: Path to the database file
        area_name: Optional area name to filter by
    """
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        sys.exit(1)

    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        from custom_components.area_occupancy.db.schema import Correlations

        query = session.query(Correlations)
        if area_name:
            query = query.filter_by(area_name=area_name)

        correlations = query.order_by(
            Correlations.area_name, Correlations.entity_id
        ).all()

        if not correlations:
            print("No correlations found in database.")
            if area_name:
                print(f"  (filtered by area: {area_name})")
            return

        print("\nAvailable entities with correlations:")
        print(f"{'Area Name':<20} {'Entity ID':<40} {'Correlation':<12} {'Type':<20}")
        print("-" * 92)

        for corr in correlations:
            corr_type = corr.correlation_type or "none"
            corr_val = (
                f"{corr.correlation_coefficient:.3f}"
                if corr.correlation_coefficient
                else "N/A"
            )
            print(
                f"{corr.area_name:<20} {corr.entity_id:<40} {corr_val:<12} {corr_type:<20}"
            )

    finally:
        session.close()
        engine.dispose()


def main() -> None:
    """Main entry point for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize sensor distributions for occupancy detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a specific sensor
  python scripts/visualize_distributions.py "Living Room" sensor.temperature

  # Visualize with custom database path
  python scripts/visualize_distributions.py "Living Room" sensor.temperature \\
      --db-path /path/to/area_occupancy.db

  # Save visualization to file instead of displaying
  python scripts/visualize_distributions.py "Living Room" sensor.temperature \\
      --output plot.png

  # List all available entities
  python scripts/visualize_distributions.py --list

  # List entities for a specific area
  python scripts/visualize_distributions.py --list --area "Living Room"
        """,
    )

    parser.add_argument(
        "area_name",
        nargs="?",
        help="Area name (required unless --list is used)",
    )
    parser.add_argument(
        "entity_id",
        nargs="?",
        help="Entity ID to visualize (required unless --list is used)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        help=f"Path to database file (default: config/.storage/{DB_NAME})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to analyze (default: 30)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save visualization to file instead of displaying (e.g., --output plot.png)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available entities with correlations",
    )
    parser.add_argument(
        "--area",
        help="Area name filter for --list option",
    )

    args = parser.parse_args()

    # Determine database path
    if args.db_path:
        db_path = args.db_path
    else:
        # Default to config/.storage/area_occupancy.db
        db_path = project_root / "config" / ".storage" / DB_NAME

    # Handle list mode
    if args.list:
        list_available_entities(db_path, args.area)
        return

    # Validate required arguments
    if not args.area_name or not args.entity_id:
        parser.error("area_name and entity_id are required (or use --list)")

    # Find entry_id from database
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        entry_id = find_entry_id(session, args.area_name)
        if not entry_id:
            print(
                f"Error: Area '{args.area_name}' not found in database. "
                "Use --list to see available areas."
            )
            sys.exit(1)
    finally:
        session.close()
        engine.dispose()

    # Visualize
    visualize_sensor_distribution(
        db_path=db_path,
        entry_id=entry_id,
        area_name=args.area_name,
        entity_id=args.entity_id,
        analysis_period_days=args.days,
        bins=args.bins,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
