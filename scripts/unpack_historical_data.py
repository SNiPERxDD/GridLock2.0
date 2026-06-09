"""Unpack the compressed historical dataset for external inspection."""

from __future__ import annotations

import lzma
import shutil
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = RELEASE_ROOT / "data"
SOURCE_PATH = DATA_DIR / "historical_training.csv.xz"
TARGET_PATH = DATA_DIR / "historical_training.csv"


def main() -> None:
    """Extract the compressed historical file into plain CSV form."""
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing compressed dataset: {SOURCE_PATH}")

    with lzma.open(SOURCE_PATH, "rb") as src, TARGET_PATH.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)

    print(f"Extracted                : {TARGET_PATH.name}")


if __name__ == "__main__":
    main()
