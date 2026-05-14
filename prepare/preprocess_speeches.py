#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


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


def validate_input_csv(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(sorted(missing))}")


def load_stopwords(stopwords_fpath: Path) -> list[str]:
    if not stopwords_fpath.exists():
        raise SystemExit(f"Stopwords file not found: {stopwords_fpath}")
    return [line.strip() for line in stopwords_fpath.read_text(encoding="utf-8").splitlines() if line.strip()]


def remove_low_activity_speakers(raw_data: pd.DataFrame, min_speeches_per_speaker: int) -> pd.DataFrame:
    speaker_counts = raw_data["Speaker_Bioguide_ID"].value_counts(dropna=False)
    keep = set(speaker_counts[speaker_counts >= min_speeches_per_speaker].index.tolist())
    return raw_data[raw_data["Speaker_Bioguide_ID"].isin(keep)].copy()


def build_author_indices(speakers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cleaned_speakers = [
        str(speaker).strip()
        if pd.notna(speaker) and str(speaker).strip()
        else "UNKNOWN_SPEAKER"
        for speaker in speakers
    ]

    speaker_to_id = {
        speaker: idx
        for idx, speaker in enumerate(sorted(set(cleaned_speakers)))
    }

    author_indices = np.array(
        [speaker_to_id[speaker] for speaker in cleaned_speakers],
        dtype=np.int32,
    )

    author_map = np.array(list(speaker_to_id.keys()))
    return author_indices, author_map


def vectorize_speeches(speeches: list[str], stopwords: list[str], min_df: float, max_df: float, ngram_max: int, token_pattern: str):
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        stop_words=stopwords,
        ngram_range=(1, ngram_max),
        token_pattern=token_pattern,
    )
    counts = vectorizer.fit_transform(speeches)
    vocabulary = np.array([term for term, _ in sorted(vectorizer.vocabulary_.items(), key=lambda kv: kv[1])])
    return counts, vocabulary


def build_author_to_indices(author_indices: np.ndarray, author_map: np.ndarray) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for author_idx, author in enumerate(tqdm(author_map, desc="Building author index map")):
        out[str(author)] = np.where(author_indices == author_idx)[0].tolist()
    return out


def get_per_author_counts(counts, author_to_inds: dict[str, list[int]]) -> np.ndarray:
    return np.concatenate([np.array(np.sum(counts[inds], axis=0)) for inds in author_to_inds.values()], axis=0)


def compute_acceptable_words(counts, vocabulary: np.ndarray, author_indices: np.ndarray, author_map: np.ndarray, min_authors_per_word: int) -> list[int]:
    author_to_inds = build_author_to_indices(author_indices, author_map)
    counts_per_author = get_per_author_counts(counts, author_to_inds)
    return [i for i in range(len(vocabulary)) if np.count_nonzero(counts_per_author[:, i]) >= min_authors_per_word]


def rebuild_counts_with_filtered_vocabulary(speeches: list[str], vocabulary_subset: np.ndarray, ngram_max: int):
    vectorizer = CountVectorizer(ngram_range=(1, ngram_max), vocabulary=vocabulary_subset)
    counts = vectorizer.fit_transform(speeches)
    vocabulary = np.array([term for term, _ in sorted(vectorizer.vocabulary_.items(), key=lambda kv: kv[1])])
    return counts, vocabulary


def build_ngram_overlap_maps(vocabulary: np.ndarray):
    token_lengths = np.array([len(word.split(" ")) for word in vocabulary])
    n_gram_indices = np.where(token_lengths > 1)[0]
    vocab_lookup = {word: idx for idx, word in enumerate(vocabulary)}

    n_gram_to_unigrams: dict[int, list[int]] = {}
    n_grams_to_bigrams: dict[int, list[int]] = {}

    for n_gram_index in n_gram_indices:
        split_tokens = vocabulary[n_gram_index].split(" ")
        n_gram_to_unigrams[n_gram_index] = [vocab_lookup[u] for u in split_tokens if u in vocab_lookup]
        if len(split_tokens) > 2:
            n_grams_to_bigrams[n_gram_index] = [
                vocab_lookup[bigram]
                for i in range(len(split_tokens) - 1)
                for bigram in [" ".join(split_tokens[i : i + 2])]
                if bigram in vocab_lookup
            ]

    return n_gram_indices, n_gram_to_unigrams, n_grams_to_bigrams


def deoverlap_ngrams(counts, vocabulary: np.ndarray):
    n_gram_indices, n_gram_to_unigrams, n_grams_to_bigrams = build_ngram_overlap_maps(vocabulary)
    for i in tqdm(range(counts.shape[0]), desc="De-overlapping n-grams"):
        n_grams_in_doc = np.where(counts[i, n_gram_indices].toarray() > 0)[0]
        sub_n_grams = n_gram_indices[n_grams_in_doc]

        for n_gram in sub_n_grams:
            if n_gram_to_unigrams[n_gram]:
                counts[i, n_gram_to_unigrams[n_gram]] = sparse.csr_matrix(
                    counts[i, n_gram_to_unigrams[n_gram]].toarray() - counts[i, n_gram]
                )
            if n_gram in n_grams_to_bigrams and n_grams_to_bigrams[n_gram]:
                counts[i, n_grams_to_bigrams[n_gram]] = sparse.csr_matrix(
                    counts[i, n_grams_to_bigrams[n_gram]].toarray() - counts[i, n_gram]
                )
    counts[counts < 0] = 0
    return counts


def filter_empty_documents(counts, author_indices: np.ndarray):
    existing = [i for i in tqdm(range(counts.shape[0]), desc="Dropping empty speeches") if counts[i].sum() > 0]
    return counts[existing], author_indices[existing], existing


def normalize_raw_documents(speeches: list[str], keep_indices: list[int]) -> list[str]:
    return [speeches[i].replace("\n", " ").replace("\r", " ") for i in keep_indices]


def save_outputs(output_dir: Path, finalized_csv: Path, counts, author_indices: np.ndarray, vocabulary: np.ndarray, author_map: np.ndarray, raw_documents: list[str], raw_data: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    finalized_csv.parent.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(output_dir / "counts.npz", counts.astype(np.float32))
    np.save(output_dir / "author_indices.npy", author_indices)
    np.savetxt(output_dir / "vocabulary.txt", vocabulary, fmt="%s")
    np.savetxt(output_dir / "author_map.txt", author_map, fmt="%s")

    with (output_dir / "raw_documents.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(raw_documents))

    raw_data.to_csv(finalized_csv, index=False)


def run(
    input_csv: Path,
    stopwords_fpath: Path,
    output_dir: Path,
    finalized_csv: Path,
    min_speeches_per_speaker: int = 25,
    min_df: float = 0.001,
    max_df: float = 0.75,
    min_authors_per_word: int = 50,
    ngram_max: int = 3,
    token_pattern: str = r"[a-zA-Z]+",
) -> None:
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    raw_data = pd.read_csv(input_csv)
    validate_input_csv(raw_data)
    raw_data = remove_low_activity_speakers(raw_data, min_speeches_per_speaker)
    # some speaker bioguide id are missing
    raw_data["Speaker_Bioguide_ID"] = raw_data["Speaker_Bioguide_ID"].fillna("UNKNOWN_SPEAKER").astype(str)
    raw_data["Text"] = raw_data["Text"].fillna("").astype(str)
    stopwords = load_stopwords(stopwords_fpath)

    speakers = list(raw_data["Speaker_Bioguide_ID"])
    speeches = list(raw_data["Text"])

    author_indices, author_map = build_author_indices(speakers)
    counts, vocabulary = vectorize_speeches(speeches, stopwords, min_df, max_df, ngram_max, token_pattern)

    acceptable_words = compute_acceptable_words(counts, vocabulary, author_indices, author_map, min_authors_per_word)
    counts, vocabulary = rebuild_counts_with_filtered_vocabulary(speeches, vocabulary[acceptable_words], ngram_max)

    counts = deoverlap_ngrams(counts, vocabulary)
    counts, author_indices, existing_speeches = filter_empty_documents(counts, author_indices)

    raw_documents = normalize_raw_documents(speeches, existing_speeches)
    raw_data = raw_data.iloc[existing_speeches].copy()

    save_outputs(output_dir, finalized_csv, counts, author_indices, vocabulary, author_map, raw_documents, raw_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initial preprocessing for congressional speeches.")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--stopwords-fpath", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--finalized-csv", required=True, type=Path)
    parser.add_argument("--min-speeches-per-speaker", type=int, default=25)
    parser.add_argument("--min-df", type=float, default=0.001)
    parser.add_argument("--max-df", type=float, default=0.75)
    parser.add_argument("--min-authors-per-word", type=int, default=50)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--token-pattern", default=r"[a-zA-Z]+")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    run(
        input_csv=args.input_csv,
        stopwords_fpath=args.stopwords_fpath,
        output_dir=args.output_dir,
        finalized_csv=args.finalized_csv,
        min_speeches_per_speaker=args.min_speeches_per_speaker,
        min_df=args.min_df,
        max_df=args.max_df,
        min_authors_per_word=args.min_authors_per_word,
        ngram_max=args.ngram_max,
        token_pattern=args.token_pattern,
    )


if __name__ == "__main__":
    main()