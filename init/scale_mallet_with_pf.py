#!/usr/bin/env python3
"""
Scale MALLET LDA outputs using Poisson factorization runs.

This wraps the old setup script:

    setup/scale_mallet_output_using_poisson_factorization_runs.py

Expected PF run folders:
    data/congress_114/processed/senate/
        pf-fits-removed-procedural-speeches-k50-seed22/
        pf-fits-removed-procedural-speeches-k50-seed194/
        ...

Expected MALLET folder:
    data/congress_114/processed/senate/mallet_fits_removed_procedural_speeches/
        beta.npy
        doctopics.txt

Outputs written inside MALLET folder:
    beta_scaled.npy
    theta_scaled.npy
    topic_word.npy
    doc_topic.npy
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def find_pf_runs(base_dir: Path, glob_pattern: str) -> list[Path]:
    runs = sorted([p for p in base_dir.glob(glob_pattern) if p.is_dir()])
    return runs


def validate_pf_runs(pf_runs: list[Path]) -> None:
    if not pf_runs:
        raise SystemExit("No Poisson factorization run directories found.")

    required_files = [
        "document_shape.npy",
        "document_rate.npy",
        "topic_shape.npy",
        "topic_rate.npy",
    ]

    missing = []
    for run_dir in pf_runs:
        for fname in required_files:
            fpath = run_dir / fname
            if not fpath.exists():
                missing.append(str(fpath))

    if missing:
        raise SystemExit(
            "Some Poisson factorization run files are missing:\n"
            + "\n".join(missing)
        )


def validate_mallet_dir(
    input_mallet_dir: Path,
    beta_fname: str,
    theta_fname: str,
) -> None:
    if not input_mallet_dir.exists():
        raise SystemExit(f"MALLET directory not found: {input_mallet_dir}")

    beta_path = input_mallet_dir / beta_fname
    theta_path = input_mallet_dir / theta_fname

    missing = []
    if not beta_path.exists():
        missing.append(str(beta_path))
    if not theta_path.exists():
        missing.append(str(theta_path))

    if missing:
        raise SystemExit(
            "Missing required MALLET output files:\n" + "\n".join(missing)
        )


def validate_scaled_outputs(input_mallet_dir: Path) -> None:
    expected = [
        "beta_scaled.npy",
        "theta_scaled.npy",
        "topic_word.npy",
        "doc_topic.npy",
    ]

    missing = [
        str(input_mallet_dir / fname)
        for fname in expected
        if not (input_mallet_dir / fname).exists()
    ]

    if missing:
        raise SystemExit(
            "Scaling script finished, but expected outputs are missing:\n"
            + "\n".join(missing)
        )


def run(
    base_dir: Path,
    input_mallet_dir: Path,
    glob_pattern: str = "pf-fits-removed-procedural-speeches-k50-seed*",
    scale_script: Path = Path("setup/scale_mallet_output_using_poisson_factorization_runs.py"),
    beta_fname: str = "beta.npy",
    theta_fname: str = "doctopics.txt",
    python_executable: str = sys.executable,
    dry_run: bool = False,
) -> None:
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    if not scale_script.exists():
        raise SystemExit(f"Scaling script not found: {scale_script}")

    pf_runs = find_pf_runs(base_dir, glob_pattern)

    logging.info("Found %d PF run directories", len(pf_runs))
    for run_dir in pf_runs:
        logging.info("PF run: %s", run_dir)

    validate_pf_runs(pf_runs)
    validate_mallet_dir(
        input_mallet_dir=input_mallet_dir,
        beta_fname=beta_fname,
        theta_fname=theta_fname,
    )

    cmd = [
        python_executable,
        str(scale_script),
        "--base_dir",
        str(base_dir),
        "--glob_pattern",
        glob_pattern,
        "--input_mallet_dir",
        str(input_mallet_dir),
        "--beta_fname",
        beta_fname,
        "--theta_fname",
        theta_fname,
    ]

    logging.info("Running MALLET scaling")
    logging.info("Command: %s", " ".join(cmd))

    if dry_run:
        return

    subprocess.run(cmd, check=True)

    validate_scaled_outputs(input_mallet_dir)

    logging.info("Finished MALLET scaling")
    logging.info("Scaled outputs written to: %s", input_mallet_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale MALLET outputs using Poisson factorization runs."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        type=Path,
        help="Directory containing PF run subdirectories.",
    )
    parser.add_argument(
        "--input-mallet-dir",
        required=True,
        type=Path,
        help="Directory containing MALLET beta/theta outputs.",
    )
    parser.add_argument(
        "--glob-pattern",
        default="pf-fits-removed-procedural-speeches-k50-seed*",
        help="Glob pattern for PF run directories.",
    )
    parser.add_argument(
        "--scale-script",
        type=Path,
        default=Path("setup/scale_mallet_output_using_poisson_factorization_runs.py"),
        help="Path to legacy scaling script.",
    )
    parser.add_argument(
        "--beta-fname",
        default="beta.npy",
        help="MALLET beta filename.",
    )
    parser.add_argument(
        "--theta-fname",
        default="doctopics.txt",
        help="MALLET doctopics filename.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable to use.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without running it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    run(
        base_dir=args.base_dir,
        input_mallet_dir=args.input_mallet_dir,
        glob_pattern=args.glob_pattern,
        scale_script=args.scale_script,
        beta_fname=args.beta_fname,
        theta_fname=args.theta_fname,
        python_executable=args.python_executable,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()