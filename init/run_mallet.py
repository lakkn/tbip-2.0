#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def run(
    input_dir: Path,
    output_dir: Path,
    mallet_path: Path,
    lda_script: Path = Path("init/gensim_runner/lda.py"),
    train_path: str = "counts.npz",
    eval_path: str = "counts.npz",
    vocab_path: str = "vocabulary.json",
    num_topics: int = 50,
    optimize_interval: int = 10,
    workers: int = 8,
    python_executable: str = sys.executable,
    dry_run: bool = False,
) -> None:
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    if not lda_script.exists():
        raise SystemExit(f"LDA runner not found: {lda_script}")

    if not mallet_path.exists():
        raise SystemExit(f"MALLET binary not found: {mallet_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        input_dir / train_path,
        input_dir / eval_path,
        input_dir / vocab_path,
    ]

    for fpath in required_files:
        if not fpath.exists():
            raise SystemExit(f"Required MALLET input file not found: {fpath}")

    cmd = [
        python_executable,
        str(lda_script),
        "--mallet_path",
        str(mallet_path),
        "--input_dir",
        str(input_dir),
        "--model",
        "mallet",
        "--output_dir",
        str(output_dir),
        "--train_path",
        train_path,
        "--eval_path",
        eval_path,
        "--vocab_path",
        vocab_path,
        "--num_topics",
        str(num_topics),
        "--optimize_interval",
        str(optimize_interval),
        "--workers",
        str(workers),
    ]

    logging.info("Running MALLET LDA")
    logging.info("Command: %s", " ".join(cmd))

    if dry_run:
        return

    subprocess.run(cmd, check=True)

    expected_outputs = [
        output_dir / "beta.npy",
        output_dir / "train.theta.npy",
        output_dir / "topics.txt",
        output_dir / "metrics.json",
    ]

    missing_outputs = [path for path in expected_outputs if not path.exists()]
    if missing_outputs:
        missing = "\n".join(str(path) for path in missing_outputs)
        raise SystemExit(f"MALLET finished, but expected outputs are missing:\n{missing}")

    logging.info("MALLET LDA finished successfully")
    logging.info("Output directory: %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MALLET LDA using vendored gensim runner.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mallet-path", required=True, type=Path)
    parser.add_argument("--lda-script", type=Path, default=Path("init/gensim_runner/lda.py"))
    parser.add_argument("--train-path", default="counts.npz")
    parser.add_argument("--eval-path", default="counts.npz")
    parser.add_argument("--vocab-path", default="vocabulary.json")
    parser.add_argument("--num-topics", type=int, default=50)
    parser.add_argument("--optimize-interval", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mallet_path=args.mallet_path,
        lda_script=args.lda_script,
        train_path=args.train_path,
        eval_path=args.eval_path,
        vocab_path=args.vocab_path,
        num_topics=args.num_topics,
        optimize_interval=args.optimize_interval,
        workers=args.workers,
        python_executable=args.python_executable,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()