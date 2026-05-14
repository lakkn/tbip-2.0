#!/usr/bin/env python3
"""
Run Poisson factorization topic model fits for multiple random seeds.

This replaces the old shell scripts like:
    floor_speeches_114_p1.sh
    floor_speeches_114_p2.sh
    floor_speeches_114_p3.sh

Example:
    python -m init.run_poisson_factorization \
      --data floor_speeches_cong_114 \
      --clean-data clean_removing_procedural \
      --num-topics 50 \
      --seeds 22 194 164 98 210 10 128 42 105 101
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_SEEDS = [22, 194, 164, 98, 210, 10, 128, 42, 105, 101]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def build_output_name(prefix: str, num_topics: int, seed: int) -> str:
    return f"{prefix}-k{num_topics}-seed{seed}"


def build_command(
    python_executable: str,
    pf_script: Path,
    data: str,
    clean_data: str,
    num_topics: int,
    output: str,
    seed: int,
) -> list[str]:
    return [
        python_executable,
        str(pf_script),
        f"--data={data}",
        f"--clean_data={clean_data}",
        f"--num_topics={num_topics}",
        f"--output={output}",
        f"--seed={seed}",
    ]


def run_one_seed(
    python_executable: str,
    pf_script: Path,
    data: str,
    clean_data: str,
    num_topics: int,
    output_prefix: str,
    seed: int,
    dry_run: bool = False,
) -> None:
    output = build_output_name(output_prefix, num_topics, seed)

    cmd = build_command(
        python_executable=python_executable,
        pf_script=pf_script,
        data=data,
        clean_data=clean_data,
        num_topics=num_topics,
        output=output,
        seed=seed,
    )

    logging.info("Running seed %s", seed)
    logging.info("Command: %s", " ".join(cmd))

    if dry_run:
        return

    subprocess.run(cmd, check=True)


def run(
    data: str,
    clean_data: str = "clean_removing_procedural",
    num_topics: int = 50,
    seeds: list[int] | None = None,
    output_prefix: str = "pf-fits-removed-procedural-speeches",
    pf_script: Path = Path("setup/poisson_factorization.py"),
    python_executable: str = sys.executable,
    dry_run: bool = False,
    max_workers: int = 2,
) -> None:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    if not pf_script.exists():
        raise SystemExit(f"Poisson factorization script not found: {pf_script}")

    logging.info("Starting Poisson factorization runs")
    logging.info("Data: %s", data)
    logging.info("Clean data dir: %s", clean_data)
    logging.info("Number of topics: %d", num_topics)
    logging.info("Seeds: %s", seeds)
    logging.info("Max parallel workers: %d", max_workers)

    if dry_run:
        for seed in seeds:
            run_one_seed(
                python_executable=python_executable,
                pf_script=pf_script,
                data=data,
                clean_data=clean_data,
                num_topics=num_topics,
                output_prefix=output_prefix,
                seed=seed,
                dry_run=True,
            )
        return

    failures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seed = {
            executor.submit(
                run_one_seed,
                python_executable,
                pf_script,
                data,
                clean_data,
                num_topics,
                output_prefix,
                seed,
                False,
            ): seed
            for seed in seeds
        }

        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                future.result()
                logging.info("Finished seed %s", seed)
            except Exception as exc:
                logging.error("Seed %s failed: %s", seed, exc)
                failures.append((seed, exc))

    if failures:
        failed_seeds = [str(seed) for seed, _ in failures]
        raise SystemExit(f"Poisson factorization failed for seeds: {', '.join(failed_seeds)}")

    logging.info("Finished all Poisson factorization runs")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Poisson factorization for multiple seeds."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset name passed to setup/poisson_factorization.py, e.g. floor_speeches_cong_114.",
    )
    parser.add_argument(
        "--clean-data",
        default="clean_removing_procedural",
        help="Clean data directory name passed as --clean_data.",
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
        help="Random seeds to run.",
    )
    parser.add_argument(
        "--output-prefix",
        default="pf-fits-removed-procedural-speeches",
        help="Output folder prefix before -k<num_topics>-seed<seed>.",
    )
    parser.add_argument(
        "--pf-script",
        type=Path,
        default=Path("setup/poisson_factorization.py"),
        help="Path to poisson_factorization.py.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable to use for each PF run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of PF seeds to run in parallel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    run(
        data=args.data,
        clean_data=args.clean_data,
        num_topics=args.num_topics,
        seeds=args.seeds,
        output_prefix=args.output_prefix,
        pf_script=args.pf_script,
        python_executable=args.python_executable,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()