"""Command-line entry points for data preparation and experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import assign_patient_splits, build_feature_table
from .modeling import run_experiment


def _build_features(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features = build_feature_table(
        Path(args.mimic_dir),
        observation_hours=args.hours,
        chunksize=args.chunksize,
    )
    features = assign_patient_splits(features, random_state=args.seed)
    features.to_parquet(output, index=False)
    print(f"Wrote {len(features):,} patient rows and {len(features.columns):,} columns to {output}")
    print(features.groupby("split")["mortality"].agg(["count", "mean"]).to_string())


def _train(args: argparse.Namespace) -> None:
    metrics = run_experiment(
        feature_path=Path(args.features),
        output_dir=Path(args.output_dir),
        alpha=args.alpha,
        min_retained_fraction=args.min_retained,
        random_state=args.seed,
    )
    print(json.dumps(metrics, indent=2, allow_nan=True))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uncertainty-aware MIMIC-IV mortality prediction"
    )
    subparsers = parser.add_subparsers(required=True)

    build = subparsers.add_parser("build-features", help="extract first-window MIMIC features")
    build.add_argument("--mimic-dir", default="mimic-iv-3.1")
    build.add_argument("--output", default="data/processed/mimic_features.parquet")
    build.add_argument("--hours", type=int, default=24)
    build.add_argument("--chunksize", type=int, default=2_000_000)
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(handler=_build_features)

    train = subparsers.add_parser("train", help="train and evaluate all model variants")
    train.add_argument("--features", default="data/processed/mimic_features.parquet")
    train.add_argument("--output-dir", default="outputs")
    train.add_argument("--alpha", type=float, default=0.1)
    train.add_argument("--min-retained", type=float, default=0.7)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(handler=_train)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
