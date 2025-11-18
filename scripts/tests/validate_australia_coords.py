#!/usr/bin/env python3
"""
Utility script to sanity-check Australian samples used for CBM training.

Tasks:
    * Load the PanoramaCBMDataset filtered to Australia and requiring coordinates
    * Export the raw lat/lon/concept rows to data/export/<country>_coords.csv
    * Scatter plot longitude vs latitude to ensure coverage looks sane
    * Print split statistics to confirm stratified splits remain representative
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Dict

import matplotlib.pyplot as plt

from src.dataset import PanoramaCBMDataset, create_splits_stratified


def summarize_coords(samples: Iterable[Dict]) -> Dict:
    """Compute simple statistics for a list of samples."""
    samples = list(samples)
    if not samples:
        return {
            "count": 0,
            "lat_min": None,
            "lat_max": None,
            "lng_min": None,
            "lng_max": None,
            "lat_mean": None,
            "lng_mean": None,
        }

    lats = [float(s["lat"]) for s in samples]
    lngs = [float(s["lng"]) for s in samples]
    return {
        "count": len(samples),
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lng_min": min(lngs),
        "lng_max": max(lngs),
        "lat_mean": sum(lats) / len(lats),
        "lng_mean": sum(lngs) / len(lngs),
    }


def print_split_stats(name: str, samples: List[Dict]) -> None:
    stats = summarize_coords(samples)
    concept_counts = Counter(s["meta_name"] for s in samples)
    unique_concepts = len(concept_counts)
    print(
        f"{name}: {stats['count']} samples | "
        f"lat[{stats['lat_min']:.3f}, {stats['lat_max']:.3f}] "
        f"lng[{stats['lng_min']:.3f}, {stats['lng_max']:.3f}] "
        f"avg lat={stats['lat_mean']:.3f} lng={stats['lng_mean']:.3f} "
        f"| {unique_concepts} concepts"
    )


def export_csv(samples: List[Dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pano_id", "meta_name", "country", "lat", "lng"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "pano_id": sample["pano_id"],
                    "meta_name": sample["meta_name"],
                    "country": sample["country"],
                    "lat": sample["lat"],
                    "lng": sample["lng"],
                }
            )
    print(f"Wrote {len(samples)} rows to {output_csv}")


def plot_scatter(samples: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lats = [float(s["lat"]) for s in samples]
    lngs = [float(s["lng"]) for s in samples]
    concepts = [s["meta_name"] for s in samples]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(lngs, lats, c=range(len(samples)), cmap="viridis", s=15)
    plt.colorbar(scatter, label="Sample index")
    plt.title("Australian CBM Samples (lon vs lat)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved scatter plot to {output_path}")


def validate(country: str, out_dir: Path, seed: int) -> None:
    dataset = PanoramaCBMDataset(
        country=country,
        require_coordinates=True,
        max_samples=None,
        encoder_model=None,
    )
    samples = dataset.samples
    if len(samples) == 0:
        raise RuntimeError(f"No samples found for country={country}")

    export_csv(samples, out_dir / f"{country.lower()}_coords.csv")
    plot_scatter(samples, out_dir / f"{country.lower()}_coords.png")

    train, val, test = create_splits_stratified(samples, seed=seed)
    print_split_stats("Train", train)
    print_split_stats("Val", val)
    print_split_stats("Test", test)

    lat_out_of_range = [s for s in samples if not (-90.0 <= float(s["lat"]) <= 0.0)]
    if lat_out_of_range:
        print(
            f"Warning: {len(lat_out_of_range)} samples have lat outside southern hemisphere "
            "(expected for Australia)."
        )

    lng_out_of_range = [s for s in samples if not (100.0 <= float(s["lng"]) <= 180.0)]
    if lng_out_of_range:
        print(
            f"Warning: {len(lng_out_of_range)} samples have lng outside [100, 180]. "
            "Verify metadata for these entries."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Australian coordinate labels.")
    parser.add_argument(
        "--country", default="Australia", help="Country filter to validate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/export"),
        help="Directory to store CSV and scatter plot.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    args = parser.parse_args()

    validate(args.country, args.output_dir, args.seed)


if __name__ == "__main__":
    main()

