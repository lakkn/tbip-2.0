#!/usr/bin/env python3
"""
Run full topic initialization pipeline.

Stages:
  1. Run Poisson factorization across multiple seeds.
  2. Run MALLET LDA.
  3. Scale MALLET outputs using PF runs.

Example:
  python -m init.init_topics \
    --data-name congress_114 \
    --chamber senate \
    --data-root data/congress_114 \
    --mallet-path /path/to/mallet \
    --num-topics 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from init.run_poisson_factorization import run as run_poisson_factorization
from init.run_mallet import run as run_mallet
from init.scale_mallet_with_pf import run as run_scale_mallet_with_pf


DEFAULT_SEEDS = [22, 194, 164, 98, 210, 10, 128, 42, 105, 101]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def build_dataset_name(data_name: str, chamber: str) -> str:
    """
    This must match the folder name expected by setup/poisson_factorization.py.

    If your PF script expects:
        data/floor_speeches_cong_114/clean_removing_procedural

    then pass:
        --pf-data-name floor_speeches_cong_114

    This helper is only a fallback.
    """
    if chamber:
        return f"{data_name}_{chamber}"
    return data_name


def run(
    data_root: Path,
    chamber: str = "house",
    num_topics: int = 50,
    seeds: list[int] | None = None,
    mallet_path: Path | None = None,
    pf_data_name: str | None = None,
    clean_data: str = "clean_removing_procedural",
    pf_script: Path = Path("setup/poisson_factorization.py"),
    lda_script: Path = Path("init/gensim_runner/lda.py"),
    scale_script: Path = Path("setup/scale_mallet_output_using_poisson_factorization_runs.py"),
    output_prefix: str = "pf-fits-removed-procedural-speeches",
    mallet_output_name: str = "mallet_fits_removed_procedural_speeches",
    optimize_interval: int = 10,
    workers: int = 8,
    python_executable: str = sys.executable,
    skip_pf: bool = False,
    skip_mallet: bool = False,
    skip_scale: bool = False,
    dry_run: bool = False,
    max_pf_workers: int = 2,
) -> None:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    if mallet_path is None and not skip_mallet:
        raise SystemExit("--mallet-path is required unless --skip-mallet is used")

    chamber_root = data_root / "processed" / chamber
    clean_dir = chamber_root / clean_data
    mallet_dir = chamber_root / mallet_output_name

    if not chamber_root.exists():
        raise SystemExit(f"Chamber processed directory not found: {chamber_root}")

    if not clean_dir.exists():
        raise SystemExit(f"Clean data directory not found: {clean_dir}")

    required_prepare_outputs = [
        clean_dir / "counts.npz",
        clean_dir / "vocabulary.json",
    ]

    for path in required_prepare_outputs:
        if not path.exists():
            raise SystemExit(f"Required prepare output missing: {path}")

    # This is the name expected by the legacy PF script.
    # For old layout, this may need to be something like:
    #   floor_speeches_cong_114
    # not:
    #   congress_114
    pf_data = pf_data_name or data_root.name

    logging.info("=== Init topics pipeline ===")
    logging.info("Data root: %s", data_root)
    logging.info("Chamber: %s", chamber)
    logging.info("PF data name: %s", pf_data)
    logging.info("Clean data dir name: %s", clean_data)
    logging.info("Number of topics: %d", num_topics)
    logging.info("Seeds: %s", seeds)

    if not skip_pf:
        logging.info("Step 1/3: Running Poisson factorization")
        run_poisson_factorization(
            data=pf_data,
            clean_data=str(Path("processed") / chamber / clean_data),
            num_topics=num_topics,
            seeds=seeds,
            output_prefix=output_prefix,
            pf_script=pf_script,
            python_executable=python_executable,
            dry_run=dry_run,
            max_workers=max_pf_workers,
        )
    else:
        logging.info("Skipping Poisson factorization")

    if not skip_mallet:
        logging.info("Step 2/3: Running MALLET LDA")
        run_mallet(
            input_dir=clean_dir,
            output_dir=mallet_dir,
            mallet_path=mallet_path,
            lda_script=lda_script,
            train_path="counts.npz",
            eval_path="counts.npz",
            vocab_path="vocabulary.json",
            num_topics=num_topics,
            optimize_interval=optimize_interval,
            workers=workers,
            python_executable=python_executable,
            dry_run=dry_run,
        )
    else:
        logging.info("Skipping MALLET")

    if not skip_scale:
        logging.info("Step 3/3: Scaling MALLET outputs with PF runs")
        glob_pattern = f"{output_prefix}-k{num_topics}-seed*"

        run_scale_mallet_with_pf(
            base_dir=chamber_root,
            input_mallet_dir=mallet_dir,
            glob_pattern=glob_pattern,
            scale_script=scale_script,
            beta_fname="beta.npy",
            theta_fname="doctopics.txt",
            python_executable=python_executable,
            dry_run=dry_run,
        )
    else:
        logging.info("Skipping scaling")

    logging.info("Init topics pipeline complete")
    logging.info("Final initialized files should be in: %s", mallet_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PF + MALLET + scaling topic initialization pipeline."
    )

    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Prepared dataset root, e.g. data/congress_114.",
    )
    parser.add_argument(
        "--chamber",
        choices=["house", "senate"],
        default="house",
        help="Target chamber. Default: house.",
    )
    parser.add_argument(
        "--pf-data-name",
        default=None,
        help=(
            "Dataset name passed to legacy poisson_factorization.py. "
            "If omitted, uses data_root.name. "
            "Use this if the legacy script expects names like floor_speeches_cong_114."
        ),
    )
    parser.add_argument(
        "--clean-data",
        default="clean_removing_procedural",
        help="Clean data directory name.",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=50,
        help="Number of topics.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="PF random seeds.",
    )
    parser.add_argument(
        "--mallet-path",
        type=Path,
        default=None,
        help="Path to MALLET executable.",
    )
    parser.add_argument(
        "--pf-script",
        type=Path,
        default=Path("setup/poisson_factorization.py"),
        help="Path to poisson_factorization.py.",
    )
    parser.add_argument(
        "--lda-script",
        type=Path,
        default=Path("init/gensim_runner/lda.py"),
        help="Path to vendored lda.py runner.",
    )
    parser.add_argument(
        "--scale-script",
        type=Path,
        default=Path("setup/scale_mallet_output_using_poisson_factorization_runs.py"),
        help="Path to MALLET scaling script.",
    )
    parser.add_argument(
        "--output-prefix",
        default="pf-fits-removed-procedural-speeches",
        help="PF output prefix.",
    )
    parser.add_argument(
        "--mallet-output-name",
        default="mallet_fits_removed_procedural_speeches",
        help="MALLET output directory name.",
    )
    parser.add_argument(
        "--optimize-interval",
        type=int,
        default=10,
        help="MALLET optimize interval.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of MALLET workers.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable for subprocess stages.",
    )
    parser.add_argument(
        "--max-pf-workers",
        type=int,
        default=2,
        help="Number of Poisson factorization seeds to run in parallel.",
    )
    parser.add_argument("--skip-pf", action="store_true")
    parser.add_argument("--skip-mallet", action="store_true")
    parser.add_argument("--skip-scale", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    run(
        data_root=args.data_root,
        chamber=args.chamber,
        num_topics=args.num_topics,
        seeds=args.seeds,
        mallet_path=args.mallet_path,
        pf_data_name=args.pf_data_name,
        clean_data=args.clean_data,
        pf_script=args.pf_script,
        lda_script=args.lda_script,
        scale_script=args.scale_script,
        output_prefix=args.output_prefix,
        mallet_output_name=args.mallet_output_name,
        optimize_interval=args.optimize_interval,
        workers=args.workers,
        python_executable=args.python_executable,
        skip_pf=args.skip_pf,
        skip_mallet=args.skip_mallet,
        skip_scale=args.skip_scale,
        dry_run=args.dry_run,
        max_pf_workers=args.max_pf_workers,
    )


if __name__ == "__main__":
    main()