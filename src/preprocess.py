"""Download or load, clean, and deterministically split the Marcus Aurelius corpus.

Team-written logic: file acquisition, cleaning, paragraph construction, and
splitting. Pandas and scikit-learn are external libraries used for data I/O
and deterministic train/validation/test partitioning.
"""

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_TEXT_PATH,
    SEED,
    set_seed,
)


GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/2680/pg2680.txt"


def download_dataset(url: str, destination: Path) -> None:
    """Download a public-domain text, with a clear failure message."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Dataset URL must use HTTP or HTTPS.")

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_text(response.text, encoding="utf-8")
    print(f"Downloaded dataset to {destination}")


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving Unicode and historical punctuation."""

    text = text.replace("\ufeff", "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n ", "\n", text)
    text = re.sub(r"\s+\Z", "", text)
    return text.strip()


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove common Project Gutenberg header and footer markers."""

    start = re.search(
        r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if start:
        text = text[start.end():]

    end = re.search(
        r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if end:
        text = text[:end.start()]
    return text


def load_raw_text(path: Path) -> str:
    """Load .txt or .csv raw input and return cleaned text."""

    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {path}. Run with --download or place "
            "a UTF-8 text file in data/raw/."
        )

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        candidate = next(
            (column for column in ("text", "quote", "passage")
             if column in frame.columns),
            None,
        )
        if candidate is None:
            raise ValueError("CSV must contain a 'text', 'quote', or 'passage' column.")
        return "\n\n".join(frame[candidate].dropna().astype(str).tolist())

    return path.read_text(encoding="utf-8")


def paragraph_records(text: str) -> list[dict[str, str]]:
    """Create one-record-per-paragraph data suitable for retrieval."""

    cleaned = clean_text(strip_gutenberg_boilerplate(text))
    paragraphs = re.split(r"\n\s*\n", cleaned)
    records = []
    for index, paragraph in enumerate(paragraphs, start=1):
        paragraph = clean_text(paragraph)
        if len(paragraph.split()) >= 4:
            records.append({"id": f"passage_{index}", "text": paragraph})
    return records


def split_and_save(records: list[dict[str, str]]) -> tuple[int, int, int]:
    """Create deterministic 80/10/10 splits and save them as CSV."""

    if len(records) < 10:
        raise ValueError(
            f"Need at least 10 text passages; found {len(records)}."
        )

    frame = pd.DataFrame(records)
    train, remainder = train_test_split(
        frame,
        train_size=0.80,
        random_state=SEED,
        shuffle=True,
    )
    val, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False, encoding="utf-8")
    val.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False, encoding="utf-8")
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False, encoding="utf-8")
    return len(train), len(val), len(test)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess the historical figure text corpus."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the configured Project Gutenberg text if raw data is absent.",
    )
    args = parser.parse_args()

    set_seed()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_dataset(GUTENBERG_URL, RAW_TEXT_PATH)

    text = load_raw_text(RAW_TEXT_PATH)
    records = paragraph_records(text)
    train_count, val_count, test_count = split_and_save(records)

    print(
        f"Saved {len(records)} passages to data/processed/ "
        f"(train={train_count}, val={val_count}, test={test_count})."
    )


if __name__ == "__main__":
    main()
