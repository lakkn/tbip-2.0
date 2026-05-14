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


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def validate_csv(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(sorted(missing))}")


def load_stopwords(*paths: Path) -> list[str]:
    words: list[str] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Stopwords file not found: {path}")
        words.extend(path.read_text(encoding="utf-8").splitlines())

    out: list[str] = []
    seen = set()
    for word in words:
        word = word.strip()
        if word and word not in seen:
            out.append(word)
            seen.add(word)
    return out


def find_kept_indices(original_docs: list[str], filtered_docs: list[str]) -> list[int]:
    kept_indices: list[int] = []
    j = 0
    for i, doc in enumerate(original_docs):
        if j >= len(filtered_docs):
            break
        if doc == filtered_docs[j]:
            kept_indices.append(i)
            j += 1
    if j != len(filtered_docs):
        raise SystemExit("Filtered file does not appear to be an ordered subset of original raw_documents.txt")
    return kept_indices


def build_author_indices(speakers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    speaker_to_id = {speaker: idx for idx, speaker in enumerate(sorted(set(speakers)))}
    return np.array([speaker_to_id[s] for s in speakers], dtype=np.int32), np.array(list(speaker_to_id.keys()))


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


def save_outputs(output_dir: Path, output_finalized_csv: Path, counts, author_indices: np.ndarray, vocabulary: np.ndarray, author_map: np.ndarray, raw_documents: list[str], finalized_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_finalized_csv.parent.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(output_dir / "counts.npz", counts.astype(np.float32))
    np.save(output_dir / "author_indices.npy", author_indices)
    np.savetxt(output_dir / "vocabulary.txt", vocabulary, fmt="%s")
    np.savetxt(output_dir / "author_map.txt", author_map, fmt="%s")

    with (output_dir / "raw_documents.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(raw_documents))

    finalized_df.to_csv(output_finalized_csv, index=False)


def run(
    original_raw_documents: Path,
    filtered_raw_documents: Path,
    finalized_csv: Path,
    base_stopwords: Path,
    procedural_stopwords: Path,
    output_dir: Path,
    output_finalized_csv: Path,
    min_df: float = 0.001,
    max_df: float = 0.75,
    min_authors_per_word: int = 50,
    ngram_max: int = 3,
    token_pattern: str = r"[a-zA-Z]+",
) -> None:
    original_docs = read_lines(original_raw_documents)
    filtered_docs = read_lines(filtered_raw_documents)

    finalized_df = pd.read_csv(finalized_csv)
    validate_csv(finalized_df)

    if len(finalized_df) != len(original_docs):
        raise SystemExit("Length mismatch between finalized CSV and original raw_documents.txt")

    kept_indices = find_kept_indices(original_docs, filtered_docs)
    filtered_df = finalized_df.iloc[kept_indices].copy()
    # account for NAN speaker_bioguide_id
    filtered_df["Speaker_Bioguide_ID"] = filtered_df["Speaker_Bioguide_ID"].fillna("UNKNOWN_SPEAKER").astype(str)
    filtered_df["Text"] = filtered_df["Text"].fillna("").astype(str)
    filtered_speeches = [original_docs[i] for i in kept_indices]

    stopwords = load_stopwords(base_stopwords, procedural_stopwords)
    speakers = list(filtered_df["Speaker_Bioguide_ID"])
    author_indices, author_map = build_author_indices(speakers)

    counts, vocabulary = vectorize_speeches(filtered_speeches, stopwords, min_df, max_df, ngram_max, token_pattern)
    acceptable_words = compute_acceptable_words(counts, vocabulary, author_indices, author_map, min_authors_per_word)
    counts, vocabulary = rebuild_counts_with_filtered_vocabulary(filtered_speeches, vocabulary[acceptable_words], ngram_max)

    counts = deoverlap_ngrams(counts, vocabulary)
    counts, author_indices, existing_speeches = filter_empty_documents(counts, author_indices)

    filtered_df = filtered_df.iloc[existing_speeches].copy()
    filtered_docs_final = [filtered_speeches[i].replace("\n", " ").replace("\r", " ") for i in existing_speeches]

    save_outputs(output_dir, output_finalized_csv, counts, author_indices, vocabulary, author_map, filtered_docs_final, filtered_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild model artifacts after procedural speech removal.")
    parser.add_argument("--original-raw-documents", required=True, type=Path)
    parser.add_argument("--filtered-raw-documents", required=True, type=Path)
    parser.add_argument("--finalized-csv", required=True, type=Path)
    parser.add_argument("--base-stopwords", required=True, type=Path)
    parser.add_argument("--procedural-stopwords", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-finalized-csv", required=True, type=Path)
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
        original_raw_documents=args.original_raw_documents,
        filtered_raw_documents=args.filtered_raw_documents,
        finalized_csv=args.finalized_csv,
        base_stopwords=args.base_stopwords,
        procedural_stopwords=args.procedural_stopwords,
        output_dir=args.output_dir,
        output_finalized_csv=args.output_finalized_csv,
        min_df=args.min_df,
        max_df=args.max_df,
        min_authors_per_word=args.min_authors_per_word,
        ngram_max=args.ngram_max,
        token_pattern=args.token_pattern,
    )


if __name__ == "__main__":
    main()