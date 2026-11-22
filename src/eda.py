"""Generate summary statistics and publication-ready corpus charts.

Team-written aggregation and plotting flow. Pandas, NumPy, Matplotlib,
and Seaborn are external libraries.
"""

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import FIGURES_DIR, PROCESSED_DATA_DIR, set_seed


SPLIT_FILES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}


def load_splits() -> dict[str, pd.DataFrame]:
    """Load all processed splits."""

    splits = {}
    for name, filename in SPLIT_FILES.items():
        path = PROCESSED_DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python src/preprocess.py --download"
            )
        splits[name] = pd.read_csv(path)
    return splits


def tokenize(text: str) -> list[str]:
    """Return lowercase word tokens while retaining Unicode letters."""

    return re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)?", str(text).lower(),
                      flags=re.UNICODE)


def build_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build sample counts, length statistics, and vocabulary size."""

    rows = []
    for name, frame in splits.items():
        texts = frame["text"].fillna("").astype(str)
        token_lists = [tokenize(text) for text in texts]
        word_counts = [len(tokens) for tokens in token_lists]
        vocabulary = {token for tokens in token_lists for token in tokens}
        rows.append({
            "split": name,
            "samples": len(frame),
            "avg_words": round(sum(word_counts) / len(word_counts), 2),
            "avg_characters": round(
                texts.str.len().mean(), 2
            ),
            "vocabulary_size": len(vocabulary),
        })
    return pd.DataFrame(rows)


def save_length_chart(splits: dict[str, pd.DataFrame], output: Path) -> None:
    """Save a text-length distribution chart."""

    frames = []
    for name, frame in splits.items():
        plotted = frame[["text"]].copy()
        plotted["split"] = name
        plotted["word_count"] = plotted["text"].fillna("").astype(str).apply(
            lambda text: len(tokenize(text))
        )
        frames.append(plotted)

    combined = pd.concat(frames, ignore_index=True)
    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(9, 5.5))
    sns.histplot(
        data=combined,
        x="word_count",
        hue="split",
        bins=30,
        kde=True,
        ax=axis,
    )
    axis.set_title("Text Length Distribution Across Splits")
    axis.set_xlabel("Word Count")
    axis.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_frequency_chart(splits: dict[str, pd.DataFrame], output: Path) -> None:
    """Save the 15 most frequent non-stopword terms across all splits."""

    counts = Counter()
    for frame in splits.values():
        for text in frame["text"].fillna("").astype(str):
            counts.update(
                token for token in tokenize(text)
                if token not in ENGLISH_STOP_WORDS and len(token) > 2
            )

    most_common = counts.most_common(15)
    if not most_common:
        return

    terms, frequencies = zip(*most_common)
    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(frequencies), y=list(terms), hue=list(terms),
                legend=False, palette="crest", ax=axis)
    axis.set_title("Top 15 Most Frequent Non-Stopword Terms")
    axis.set_xlabel("Frequency")
    axis.set_ylabel("Term")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed()
    splits = load_splits()
    summary = build_summary(splits)
    print("Corpus Summary")
    print(summary.to_string(index=False))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_length_chart(splits, FIGURES_DIR / "text_length_distribution.png")
    save_frequency_chart(splits, FIGURES_DIR / "top_terms.png")
    print("Saved figures/text_length_distribution.png")
    print("Saved figures/top_terms.png")


if __name__ == "__main__":
    main()
