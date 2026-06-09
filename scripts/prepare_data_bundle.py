"""Stage the GridLock 2.0 release datasets into the local data directory."""

from __future__ import annotations

import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:  # pragma: no cover
    kagglehub = None


RELEASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RELEASE_ROOT.parent
DATA_DIR = RELEASE_ROOT / "data"
HISTORICAL_XZ = DATA_DIR / "historical_training.csv.xz"


def _copy_if_missing(source: Path, target: Path) -> None:
    """Copy a file only when the target is absent."""
    if source.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _compress_historical(source: Path) -> None:
    """Compress the historical file into xz format when needed."""
    if HISTORICAL_XZ.exists():
        return
    import lzma

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, lzma.open(HISTORICAL_XZ, "wb", preset=9) as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def _locate_kaggle_root() -> Path | None:
    """Locate the KaggleHub dataset root when available."""
    if kagglehub is None:
        return None
    try:
        return Path(kagglehub.dataset_download("kweklydia5/grabtrafficdata"))
    except Exception:
        return None


def main() -> None:
    """Populate the release data directory from local files or KaggleHub."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _copy_if_missing(PROJECT_ROOT / "dataset" / "train.csv", DATA_DIR / "competition_train.csv")
    _copy_if_missing(PROJECT_ROOT / "dataset" / "test.csv", DATA_DIR / "competition_test.csv")
    _copy_if_missing(PROJECT_ROOT / "submissions" / "final_ground_truth.csv", DATA_DIR / "evaluation_ground_truth.csv")

    historical_source = PROJECT_ROOT / "training.csv"
    if historical_source.exists():
        _compress_historical(historical_source)
    else:
        kaggle_root = _locate_kaggle_root()
        if kaggle_root is not None:
            candidate = kaggle_root / "training.csv"
            if candidate.exists():
                _compress_historical(candidate)

    print(f"Data directory            : {DATA_DIR}")
    for path in sorted(DATA_DIR.glob("*")):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
