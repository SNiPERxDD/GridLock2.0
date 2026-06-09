"""Build the curated GridLock 2.0 release archive."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = RELEASE_ROOT / "gridlock_release_bundle.zip"
TAR_PATH = RELEASE_ROOT / "gridlock_release_bundle.tar.gz"

RELEASE_FILES = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "gridlock_release_pipeline.py",
    "notebooks/gridlock_release_pipeline.ipynb",
    "notebooks/gridlock_eda_companion.ipynb",
    "scripts/prepare_data_bundle.py",
    "scripts/unpack_historical_data.py",
    "scripts/build_release_archive.py",
    "data/competition_train.csv",
    "data/competition_test.csv",
    "data/evaluation_ground_truth.csv",
    "data/historical_training.csv.xz",
    "artifacts/gridlock_release_model.pkl",
    "artifacts/submission.csv",
    "artifacts/validation_summary.json",
]


def existing_files() -> list[Path]:
    """Return the existing curated release files."""
    files: list[Path] = []
    for relative in RELEASE_FILES:
        path = RELEASE_ROOT / relative
        if path.exists():
            files.append(path)
    return files


def build_zip(files: list[Path]) -> None:
    """Write the zip archive."""
    with zipfile.ZipFile(ZIP_PATH, mode="w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, arcname=path.relative_to(RELEASE_ROOT.parent))


def build_tar(files: list[Path]) -> None:
    """Write the tar.gz archive."""
    with tarfile.open(TAR_PATH, mode="w:gz") as handle:
        for path in files:
            handle.add(path, arcname=path.relative_to(RELEASE_ROOT.parent))


def main() -> None:
    """Build both release archives and list their contents."""
    files = existing_files()
    build_zip(files)
    build_tar(files)
    print(f"Files archived           : {len(files)}")
    for path in files:
        print(f" - {path.relative_to(RELEASE_ROOT.parent)}")
    print(f"Zip archive              : {ZIP_PATH.name}")
    print(f"Tar archive              : {TAR_PATH.name}")


if __name__ == "__main__":
    main()
