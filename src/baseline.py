"""TF-IDF cosine-similarity baseline for the Marcus Aurelius chatbot.

Team-written logic: prompts, retrieval assembly, lexical metrics, latency
measurement, and chat loop. TF-IDF and cosine similarity are provided by
scikit-learn, an external machine-learning library.
"""

import argparse
import json
import re
import time
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    BASELINE_RESULTS_PATH,
    HISTORICAL_FIGURE,
    TEST_SET_PATH,
    TRAIN_PATH,
    set_seed,
)


def rouge1_f1(reference: str, candidate: str) -> float:
    """Calculate a simple word-level ROUGE-1 F1 score (team-written)."""

    reference_tokens = re.findall(r"[^\W\d_]+", reference.lower(),
                                  flags=re.UNICODE)
    candidate_tokens = re.findall(r"[^\W\d_]+", candidate.lower(),
                                  flags=re.UNICODE)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    overlap = sum(
        (reference_counts & candidate_counts).values()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


class TfidfRetrievalBaseline:
    """Deterministic retrieval baseline built on external scikit-learn tools."""

    def __init__(self, passages: list[str]) -> None:
        self.passages = passages
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.passages_matrix = self.vectorizer.fit_transform(passages)

    def retrieve(self, query: str) -> tuple[str, float]:
        query_matrix = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_matrix, self.passages_matrix)[0]
        best_index = int(similarities.argmax())
        return self.passages[best_index], float(similarities[best_index])

    def answer(self, query: str) -> tuple[str, float]:
        passage, score = self.retrieve(query)
        return (
            f"As Marcus Aurelius might reflect: {passage}",
            score,
        )


def load_model() -> TfidfRetrievalBaseline:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_PATH}. Run python src/preprocess.py --download "
            "before starting the baseline."
        )
    frame = pd.read_csv(TRAIN_PATH)
    passages = frame["text"].dropna().astype(str).tolist()
    if not passages:
        raise ValueError("Training split contains no passages.")
    return TfidfRetrievalBaseline(passages)


def evaluate(model: TfidfRetrievalBaseline) -> pd.DataFrame:
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"Missing {TEST_SET_PATH}.")

    test_items = json.loads(
        TEST_SET_PATH.read_text(encoding="utf-8")
    )
    rows = []
    for item in test_items:
        prompt = item["prompt"]
        reference = item["reference_answer"]
        started = time.perf_counter()
        response, retrieval_similarity = model.answer(prompt)
        latency_ms = (time.perf_counter() - started) * 1000
        rows.append({
            "prompt": prompt,
            "category": item["category"],
            "response": response,
            "reference_answer": reference,
            "latency_ms": round(latency_ms, 3),
            "retrieval_similarity": round(retrieval_similarity, 4),
            "rouge1_f1": round(
                rouge1_f1(reference, response), 4
            ),
        })
    return pd.DataFrame(rows)


def run_chat(model: TfidfRetrievalBaseline) -> None:
    print(f"Chatting as {HISTORICAL_FIGURE}. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if prompt.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not prompt:
            continue
        response, similarity = model.answer(prompt)
        print(f"{HISTORICAL_FIGURE}: {response}")
        print(f"[retrieval similarity: {similarity:.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval baseline.")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive query session after model loading.",
    )
    args = parser.parse_args()

    set_seed()
    model = load_model()

    if args.chat:
        run_chat(model)
        return

    results = evaluate(model)
    BASELINE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(BASELINE_RESULTS_PATH, index=False, encoding="utf-8")
    print(results[[
        "prompt",
        "latency_ms",
        "retrieval_similarity",
        "rouge1_f1",
    ]].to_string(index=False))
    print("\nAggregate metrics")
    print(f"Average latency_ms: {results['latency_ms'].mean():.3f}")
    print(
        "Average retrieval_similarity: "
        f"{results['retrieval_similarity'].mean():.4f}"
    )
    print(f"Average ROUGE-1 F1: {results['rouge1_f1'].mean():.4f}")
    print(f"Saved {BASELINE_RESULTS_PATH}")


if __name__ == "__main__":
    main()
