#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd


MONTH_TO_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

OUTPUT_COLUMNS = [
    "Speaker_Bioguide_ID",
    "Speaker_Name",
    "Text",
    "Date",
    "Legislative Body",
]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def safe_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() != "none" else None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_month(month: Any) -> int | None:
    if month is None:
        return None
    if isinstance(month, int):
        return month if 1 <= month <= 12 else None
    s = str(month).strip().lower()
    if s.isdigit():
        n = int(s)
        return n if 1 <= n <= 12 else None
    return MONTH_TO_NUM.get(s)


def header_to_date(header: dict[str, Any]) -> str | None:
    try:
        year = int(header.get("year"))
        day = int(header.get("day"))
    except (TypeError, ValueError):
        return None
    month = parse_month(header.get("month"))
    if month is None:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def load_json(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON object must be a dict")
    return data


def extract_rows_from_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    header = doc.get("header", {})
    if not isinstance(header, dict):
        header = {}

    date = header_to_date(header)
    chamber = safe_str(header.get("chamber"))

    content = doc.get("content", [])
    if not isinstance(content, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if safe_str(item.get("kind")) != "speech":
            continue

        text = safe_str(item.get("text"))
        if not text:
            continue

        rows.append(
            {
                "Speaker_Bioguide_ID": safe_str(item.get("speaker_bioguide")),
                "Speaker_Name": safe_str(item.get("speaker")),
                "Text": normalize_whitespace(text),
                "Date": date,
                "Legislative Body": chamber,
            }
        )
    return rows


def collect_json_files(root_dir: Path, pattern: str) -> list[Path]:
    return [fp for fp in sorted(root_dir.rglob(pattern)) if fp.is_file()]


def build_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if df.empty:
        return df
    df["Text"] = df["Text"].fillna("").astype(str).map(normalize_whitespace)
    return df


def write_failure_log(failed_files: list[str], output_csv: Path) -> None:
    if failed_files:
        failure_path = output_csv.with_suffix(".failed.txt")
        failure_path.write_text("\n".join(failed_files), encoding="utf-8")
        logging.warning("Wrote failure log: %s", failure_path)


def run(
    root_dir: Path,
    output_csv: Path,
    pattern: str = "*.json",
    limit: int = 0,
    drop_missing_text: bool = False,
) -> None:
    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"Input root directory does not exist or is not a directory: {root_dir}")

    json_files = collect_json_files(root_dir, pattern)
    if limit > 0:
        json_files = json_files[:limit]
    if not json_files:
        raise SystemExit(f"No files matched pattern '{pattern}' under {root_dir}")

    logging.info("Found %d JSON files", len(json_files))

    all_rows: list[dict[str, Any]] = []
    failed_files: list[str] = []

    for i, json_path in enumerate(json_files, start=1):
        try:
            doc = load_json(json_path)
            all_rows.extend(extract_rows_from_document(doc))
            if i % 1000 == 0:
                logging.info("Processed %d/%d files", i, len(json_files))
        except Exception as exc:
            failed_files.append(f"{json_path}: {exc}")

    df = build_dataframe(all_rows)
    if drop_missing_text and not df.empty:
        before = len(df)
        df = df[df["Text"].str.len() > 0].copy()
        logging.info("Dropped %d rows with empty text", before - len(df))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logging.info("Wrote %d rows to %s", len(df), output_csv)

    write_failure_log(failed_files, output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge nested Congressional Record JSON files into one speeches CSV.")
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--drop-missing-text", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    run(
        root_dir=args.root_dir,
        output_csv=args.output_csv,
        pattern=args.pattern,
        limit=args.limit,
        drop_missing_text=args.drop_missing_text,
    )


if __name__ == "__main__":
    main()