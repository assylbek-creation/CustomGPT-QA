"""Download and clean an Ancient History corpus from Wikipedia.

Usage:
    python -m src.data --download
    python -m src.data --download --out data/raw --limit 30
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import wikipediaapi
from tqdm import tqdm


ARTICLES: list[str] = [
    # Greek
    "Ancient Greece", "Classical Athens", "Sparta", "Peloponnesian War",
    "Alexander the Great", "Hellenistic period", "Plato", "Aristotle",
    "Socrates", "Greco-Persian Wars",
    # Roman
    "Ancient Rome", "Roman Republic", "Roman Empire", "Julius Caesar",
    "Augustus", "Punic Wars", "Roman Senate", "Pompey",
    "Fall of the Western Roman Empire", "Pax Romana",
    # Egyptian
    "Ancient Egypt", "Pharaoh", "Cleopatra", "Tutankhamun",
    "Pyramid of Giza", "Hieroglyphs",
    # Mesopotamian
    "Mesopotamia", "Sumer", "Babylon", "Hammurabi",
    "Assyria", "Code of Hammurabi",
]


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    return _SLUG_RE.sub("_", title.lower()).strip("_")


def clean_text(text: str) -> str:
    """Strip Wikipedia section headers and collapse whitespace."""
    text = re.sub(r"={2,}.*?={2,}", "", text)        # == Section == markers
    text = re.sub(r"\[\d+\]", "", text)              # [1] citation refs
    text = re.sub(r"\n{3,}", "\n\n", text)           # collapse blank-line runs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def download(out_dir: Path, titles: list[str]) -> int:
    """Download each title, write data/raw/<slug>.txt, return chars written."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="customgpt-qa/0.1 (educational; contact via github)",
        language="en",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    total_chars = 0

    for title in tqdm(titles, desc="Downloading"):
        page = wiki.page(title)
        if not page.exists():
            tqdm.write(f"  ! missing: {title}")
            continue
        text = clean_text(page.text)
        if len(text) < 500:
            tqdm.write(f"  ! too short ({len(text)} chars): {title}")
            continue
        (out_dir / f"{slugify(title)}.txt").write_text(text, encoding="utf-8")
        total_chars += len(text)

    return total_chars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", default="data/raw", type=Path)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of articles (debug)")
    args = parser.parse_args()

    if not args.download:
        parser.error("nothing to do — pass --download")

    titles = ARTICLES if args.limit is None else ARTICLES[: args.limit]
    chars = download(args.out, titles)
    files = len(list(args.out.glob("*.txt")))
    print(f"\nWrote {files} files, {chars:,} chars to {args.out}")


if __name__ == "__main__":
    main()
