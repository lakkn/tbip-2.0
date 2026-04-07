#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "Speaker_Bioguide_ID",
    "Speaker_Name",
    "Text",
    "Date",
    "Legislative Body",
}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def normalize_body(value: object) -> str:
    return "" if value is None else str(value).strip().lower()


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required column(s): {', '.join(sorted(missing))}")


def split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    body = df["Legislative Body"].map(normalize_body)
    return df[body.eq("house")].copy(), df[body.eq("senate")].copy()


def write_split_csvs(house_df: pd.DataFrame, senate_df: pd.DataFrame, house_out: Path, senate_out: Path) -> None:
    house_out.parent.mkdir(parents=True, exist_ok=True)
    senate_out.parent.mkdir(parents=True, exist_ok=True)
    house_df.to_csv(house_out, index=False)
    senate_df.to_csv(senate_out, index=False)
    logging.info("Wrote %d House rows to %s", len(house_df), house_out)
    logging.info("Wrote %d Senate rows to %s", len(senate_df), senate_out)


def stream_split_csv(input_csv: Path, house_out: Path, senate_out: Path, chunksize: int) -> None:
    if house_out.exists():
        house_out.unlink()
    if senate_out.exists():
        senate_out.unlink()

    first_house = True
    first_senate = True
    total_rows = total_house = total_senate = 0

    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        validate_columns(chunk)
        house_chunk, senate_chunk = split_dataframe(chunk)

        if not house_chunk.empty:
            house_chunk.to_csv(house_out, index=False, mode="a", header=first_house)
            first_house = False
            total_house += len(house_chunk)

        if not senate_chunk.empty:
            senate_chunk.to_csv(senate_out, index=False, mode="a", header=first_senate)
            first_senate = False
            total_senate += len(senate_chunk)

        total_rows += len(chunk)

    if first_house:
        pd.DataFrame(columns=list(REQUIRED_COLUMNS)).to_csv(house_out, index=False)
    if first_senate:
        pd.DataFrame(columns=list(REQUIRED_COLUMNS)).to_csv(senate_out, index=False)

    logging.info("Total input rows: %d", total_rows)
    logging.info("Total House rows: %d", total_house)
    logging.info("Total Senate rows: %d", total_senate)


def run(
    input_csv: Path,
    out_dir: Path | None = None,
    house_name: str = "speeches_house.csv",
    senate_name: str = "speeches_senate.csv",
    chunksize: int = 0,
) -> tuple[Path, Path]:
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    out_dir = out_dir or input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    house_out = out_dir / house_name
    senate_out = out_dir / senate_name

    if chunksize > 0:
        stream_split_csv(input_csv, house_out, senate_out, chunksize)
        return house_out, senate_out

    df = pd.read_csv(input_csv)
    validate_columns(df)
    house_df, senate_df = split_dataframe(df)
    write_split_csvs(house_df, senate_df, house_out, senate_out)
    return house_out, senate_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a combined speeches CSV into House and Senate CSV files.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--house-name", default="speeches_house.csv")
    parser.add_argument("--senate-name", default="speeches_senate.csv")
    parser.add_argument("--chunksize", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    run(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        house_name=args.house_name,
        senate_name=args.senate_name,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()