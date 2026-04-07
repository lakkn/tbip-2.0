#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from congress_pipeline.preprocess.filter_procedural import run as run_filter_procedural
from congress_pipeline.preprocess.preprocess_post_procedural import run as run_preprocess_post_procedural
from congress_pipeline.preprocess.preprocess_speeches import run as run_preprocess_speeches
from congress_pipeline.preprocess.process_raw_json_to_csv import run as run_process_raw_json_to_csv
from congress_pipeline.preprocess.split_house_senate import run as run_split_house_senate
from congress_pipeline.preprocess.vocab_txt_to_json import run as run_vocab_txt_to_json


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def chamber_list(chamber: str) -> list[str]:
    return ["house", "senate"] if chamber == "both" else [chamber]


def run_chamber_pipeline(
    chamber: str,
    raw_root: Path,
    interim_root: Path,
    processed_root: Path,
    base_stopwords: Path,
    procedural_model: Path,
    house_csv_name: str,
    senate_csv_name: str,
    min_speeches_per_speaker: int,
    min_df: float,
    max_df: float,
    min_authors_per_word: int,
    ngram_max: int,
    token_pattern: str,
    short_speech_max_chars: int,
    incidence_threshold: float,
) -> None:
    chamber_input_csv = raw_root / (house_csv_name if chamber == "house" else senate_csv_name)
    chamber_interim_dir = interim_root / chamber
    chamber_processed_dir = processed_root / chamber

    clean_dir = chamber_interim_dir / "clean"
    clean_removing_procedural_dir = chamber_processed_dir / "clean_removing_procedural"

    finalized_csv = chamber_interim_dir / "finalized_speeches.csv"
    filtered_raw_documents = clean_dir / "raw_documents_without_procedural.txt"
    procedural_stopwords = clean_dir / "procedural_stopwords.txt"
    output_finalized_csv = chamber_processed_dir / "finalized_speeches_removed_procedural.csv"

    logging.info("=== Running prepare pipeline for %s ===", chamber)

    run_preprocess_speeches(
        input_csv=chamber_input_csv,
        stopwords_fpath=base_stopwords,
        output_dir=clean_dir,
        finalized_csv=finalized_csv,
        min_speeches_per_speaker=min_speeches_per_speaker,
        min_df=min_df,
        max_df=max_df,
        min_authors_per_word=min_authors_per_word,
        ngram_max=ngram_max,
        token_pattern=token_pattern,
    )

    run_filter_procedural(
        input_fpath=clean_dir / "raw_documents.txt",
        output_fpath=filtered_raw_documents,
        model_fpath=procedural_model,
        stopwords_out=procedural_stopwords,
        short_speech_max_chars=short_speech_max_chars,
        max_ngram=ngram_max,
        incidence_threshold=incidence_threshold,
    )

    run_preprocess_post_procedural(
        original_raw_documents=clean_dir / "raw_documents.txt",
        filtered_raw_documents=filtered_raw_documents,
        finalized_csv=finalized_csv,
        base_stopwords=base_stopwords,
        procedural_stopwords=procedural_stopwords,
        output_dir=clean_removing_procedural_dir,
        output_finalized_csv=output_finalized_csv,
        min_df=min_df,
        max_df=max_df,
        min_authors_per_word=min_authors_per_word,
        ngram_max=ngram_max,
        token_pattern=token_pattern,
    )

    run_vocab_txt_to_json(
        input_fpath=clean_removing_procedural_dir / "vocabulary.txt",
        output_fpath=clean_removing_procedural_dir / "vocabulary.json",
    )

    logging.info("=== Finished prepare pipeline for %s ===", chamber)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full prepare pipeline for congressional floor speeches.")
    parser.add_argument("--scraped-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--base-stopwords", required=True, type=Path)
    parser.add_argument("--procedural-model", required=True, type=Path)
    parser.add_argument("--chamber", choices=["house", "senate", "both"], default="house")
    parser.add_argument("--combined-csv-name", default="speeches_combined.csv")
    parser.add_argument("--house-csv-name", default="speeches_house.csv")
    parser.add_argument("--senate-csv-name", default="speeches_senate.csv")
    parser.add_argument("--min-speeches-per-speaker", type=int, default=25)
    parser.add_argument("--min-df", type=float, default=0.001)
    parser.add_argument("--max-df", type=float, default=0.75)
    parser.add_argument("--min-authors-per-word", type=int, default=50)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--token-pattern", default=r"[a-zA-Z]+")
    parser.add_argument("--json-pattern", default="*.json")
    parser.add_argument("--json-limit", type=int, default=0)
    parser.add_argument("--split-chunksize", type=int, default=0)
    parser.add_argument("--short-speech-max-chars", type=int, default=400)
    parser.add_argument("--incidence-threshold", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    raw_root = args.output_root / "raw"
    interim_root = args.output_root / "interim"
    processed_root = args.output_root / "processed"

    raw_root.mkdir(parents=True, exist_ok=True)
    interim_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    combined_csv = raw_root / args.combined_csv_name

    logging.info("Step 1/6: Processing raw JSON into combined CSV")
    run_process_raw_json_to_csv(
        root_dir=args.scraped_root,
        output_csv=combined_csv,
        pattern=args.json_pattern,
        limit=args.json_limit,
    )

    logging.info("Step 2/6: Splitting combined CSV into House and Senate")
    run_split_house_senate(
        input_csv=combined_csv,
        out_dir=raw_root,
        house_name=args.house_csv_name,
        senate_name=args.senate_csv_name,
        chunksize=args.split_chunksize,
    )

    for chamber in chamber_list(args.chamber):
        run_chamber_pipeline(
            chamber=chamber,
            raw_root=raw_root,
            interim_root=interim_root,
            processed_root=processed_root,
            base_stopwords=args.base_stopwords,
            procedural_model=args.procedural_model,
            house_csv_name=args.house_csv_name,
            senate_csv_name=args.senate_csv_name,
            min_speeches_per_speaker=args.min_speeches_per_speaker,
            min_df=args.min_df,
            max_df=args.max_df,
            min_authors_per_word=args.min_authors_per_word,
            ngram_max=args.ngram_max,
            token_pattern=args.token_pattern,
            short_speech_max_chars=args.short_speech_max_chars,
            incidence_threshold=args.incidence_threshold,
        )

    logging.info("Prepare pipeline complete")


if __name__ == "__main__":
    main()