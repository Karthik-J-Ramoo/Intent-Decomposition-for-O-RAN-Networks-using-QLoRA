from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oran_dataset_generator import (  # noqa: E402
    generate_dataset,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic O-RAN intent decomposition dataset.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2000,
        help="Number of samples to generate (default: 2000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.json",
        help="Path to output JSON file (default: dataset.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N samples (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    dataset = generate_dataset(total_samples=args.size, progress_every=args.progress_every)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(dataset, file_handle, indent=2)

    scenario_counts = Counter(sample["scenario_type"] for sample in dataset)

    print(f"[done] generated {len(dataset)} samples")
    print(f"[done] dataset written to: {output_path}")
    print("[distribution] scenario counts:")
    for scenario_name in sorted(scenario_counts.keys()):
        print(f"  - {scenario_name}: {scenario_counts[scenario_name]}")


if __name__ == "__main__":
    main()
