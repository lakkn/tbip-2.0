#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def read_vocabulary(input_fpath: Path, drop_empty: bool = True) -> list[str]:
    if not input_fpath.exists():
        raise SystemExit(f"Vocabulary file not found: {input_fpath}")
    vocab = input_fpath.read_text(encoding="utf-8").splitlines()
    return [t.strip() for t in vocab if t.strip()] if drop_empty else vocab


def validate_vocabulary(vocab: list[str], fail_on_duplicates: bool = True) -> None:
    if not vocab:
        raise SystemExit("Vocabulary is empty.")
    duplicates = len(vocab) - len(set(vocab))
    if duplicates > 0 and fail_on_duplicates:
        raise SystemExit(f"Vocabulary contains {duplicates} duplicate token(s).")


def build_vocab_mapping(vocab: list[str]) -> dict[str, int]:
    return {token: idx for idx, token in enumerate(vocab)}


def write_vocab_json(vocab_mapping: dict[str, int], output_fpath: Path, indent: int = 2) -> None:
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_fpath.write_text(json.dumps(vocab_mapping, indent=indent, ensure_ascii=False), encoding="utf-8")


def run(
    input_fpath: Path,
    output_fpath: Path,
    allow_duplicates: bool = False,
    keep_empty: bool = False,
    indent: int = 2,
) -> None:
    vocab = read_vocabulary(input_fpath=input_fpath, drop_empty=not keep_empty)
    validate_vocabulary(vocab=vocab, fail_on_duplicates=not allow_duplicates)
    mapping = build_vocab_mapping(vocab)
    write_vocab_json(mapping, output_fpath, indent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert vocabulary.txt to vocabulary.json.")
    parser.add_argument("--input-fpath", required=True, type=Path)
    parser.add_argument("--output-fpath", required=True, type=Path)
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--keep-empty", action="store_true")
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    run(
        input_fpath=args.input_fpath,
        output_fpath=args.output_fpath,
        allow_duplicates=args.allow_duplicates,
        keep_empty=args.keep_empty,
        indent=args.indent,
    )


if __name__ == "__main__":
    main()