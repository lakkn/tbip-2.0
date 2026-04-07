#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import string
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


REPLACE_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def simplify_text(text: str) -> str:
    text = REPLACE_PUNCT.sub("", text)
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def generate_ngrams(tokens: list[str], max_n: int = 3) -> list[str]:
    out: list[str] = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            out.append("_".join(tokens[i : i + n]))
    return out


def read_documents(input_fpath: Path) -> list[str]:
    if not input_fpath.exists():
        raise SystemExit(f"Input file not found: {input_fpath}")
    return input_fpath.read_text(encoding="utf-8").splitlines()


def build_candidate_speeches(raw_lines: list[str]) -> list[tuple[int, str]]:
    speeches: list[tuple[int, str]] = []
    for idx, raw in enumerate(raw_lines):
        simplified = simplify_text(raw)
        if len(simplified) > 15 and len(re.findall(r"\s+", simplified)) > 1:
            speeches.append((idx, simplified))
    return speeches


def select_shortish_speeches(speeches: list[tuple[int, str]], max_chars: int = 400) -> list[tuple[int, str]]:
    return [(idx, speech) for idx, speech in speeches if len(speech) < max_chars]


def load_procedural_model(model_fpath: Path) -> tuple[list[str], np.ndarray, float]:
    if not model_fpath.exists():
        raise SystemExit(f"Procedural model file not found: {model_fpath}")

    lines = [line for line in model_fpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"Procedural model file is malformed or empty: {model_fpath}")

    terms: list[str] = []
    weights: list[float] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            terms.append(parts[0])
            weights.append(float(parts[1]))

    if not terms:
        raise SystemExit(f"No terms loaded from model file: {model_fpath}")

    bias_weight = weights[0]
    vocabulary_terms = terms[1:]
    vocabulary_weights = np.array(weights[1:], dtype=float)
    return vocabulary_terms, vocabulary_weights, float(bias_weight)


def tokenize_shortish_speeches(shortish_speeches: list[tuple[int, str]], max_n: int = 3) -> list[tuple[int, list[str]]]:
    return [(idx, generate_ngrams(speech.split(), max_n=max_n)) for idx, speech in shortish_speeches]


def build_document_term_matrix(tokenized_speeches: list[tuple[int, list[str]]], vocabulary_terms: list[str]):
    vectorizer = CountVectorizer(
        preprocessor=lambda x: x,
        analyzer=lambda x: x,
        vocabulary={term: i for i, term in enumerate(vocabulary_terms)},
        binary=True,
    )
    dtm = vectorizer.fit_transform(tokens for _, tokens in tokenized_speeches)
    return dtm, vectorizer


def write_filtered_documents(raw_lines: list[str], procedural_indices: set[int], output_fpath: Path) -> None:
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as f:
        for idx, raw in enumerate(raw_lines):
            if idx not in procedural_indices:
                f.write(raw + "\n")


def compute_procedural_stopwords(dtm, predicted_procedural: np.ndarray, vectorizer: CountVectorizer, incidence_threshold: float = 0.1) -> list[str]:
    if predicted_procedural.sum() == 0:
        return []
    incidence = dtm[predicted_procedural == 1].mean(axis=0).A1
    inv_vocab = {idx: term for term, idx in vectorizer.vocabulary_.items()}
    sorted_indices = np.argsort(-incidence)
    return [inv_vocab[idx] for idx in sorted_indices if incidence[idx] > incidence_threshold]


def write_stopwords(stopwords: list[str], stopwords_out: Path) -> None:
    stopwords_out.parent.mkdir(parents=True, exist_ok=True)
    stopwords_out.write_text("\n".join(stopwords), encoding="utf-8")


def run(
    input_fpath: Path,
    output_fpath: Path,
    model_fpath: Path = Path("model.nontest.tsv"),
    stopwords_out: Path = Path("procedural_stopwords.txt"),
    short_speech_max_chars: int = 400,
    max_ngram: int = 3,
    incidence_threshold: float = 0.1,
) -> None:
    raw_lines = read_documents(input_fpath)
    speeches = build_candidate_speeches(raw_lines)
    shortish_speeches = select_shortish_speeches(speeches, max_chars=short_speech_max_chars)

    logging.info("Loaded %d total lines", len(raw_lines))
    logging.info("Retained %d candidate speeches", len(speeches))
    logging.info("Selected %d short speeches for classification", len(shortish_speeches))

    if not shortish_speeches:
        write_filtered_documents(raw_lines, set(), output_fpath)
        write_stopwords([], stopwords_out)
        return

    vocabulary_terms, vocabulary_weights, bias_weight = load_procedural_model(model_fpath)
    tokenized_speeches = tokenize_shortish_speeches(shortish_speeches, max_n=max_ngram)
    dtm, vectorizer = build_document_term_matrix(tokenized_speeches, vocabulary_terms)

    model = LogisticRegression(penalty="l1")
    model.intercept_ = np.array([bias_weight])
    model.classes_ = np.array([0, 1])
    model.coef_ = vocabulary_weights[None, :]
    predicted_procedural = model.predict(dtm)

    procedural_indices = {
        idx for i, (idx, _) in enumerate(shortish_speeches) if predicted_procedural[i] == 1
    }

    logging.info("Identified %d procedural speeches", len(procedural_indices))
    write_filtered_documents(raw_lines, procedural_indices, output_fpath)

    stopwords = compute_procedural_stopwords(
        dtm=dtm,
        predicted_procedural=predicted_procedural,
        vectorizer=vectorizer,
        incidence_threshold=incidence_threshold,
    )
    write_stopwords(stopwords, stopwords_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove likely procedural speeches from raw_documents.txt.")
    parser.add_argument("--input-fpath", required=True, type=Path)
    parser.add_argument("--output-fpath", required=True, type=Path)
    parser.add_argument("--model-fpath", type=Path, default=Path("model.nontest.tsv"))
    parser.add_argument("--stopwords-out", type=Path, default=Path("procedural_stopwords.txt"))
    parser.add_argument("--short-speech-max-chars", type=int, default=400)
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument("--incidence-threshold", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    run(
        input_fpath=args.input_fpath,
        output_fpath=args.output_fpath,
        model_fpath=args.model_fpath,
        stopwords_out=args.stopwords_out,
        short_speech_max_chars=args.short_speech_max_chars,
        max_ngram=args.max_ngram,
        incidence_threshold=args.incidence_threshold,
    )


if __name__ == "__main__":
    main()