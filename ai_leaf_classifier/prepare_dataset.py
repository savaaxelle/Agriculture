from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import zipfile

from config import ARCHIVE_PATH, RAW_DATA_DIR

EXPECTED_TRAIN_PER_CLASS = 1000
EXPECTED_TEST_PER_CLASS = 100
HEALTHY_CLASS = "Tomato___healthy"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def extract_archive() -> None:
    """Extract archive.zip into raw_dataset/ if the tomato dataset is not present."""
    train_dir = RAW_DATA_DIR / "train"
    val_dir = RAW_DATA_DIR / "val"

    if train_dir.exists() and val_dir.exists():
        print("[INFO] Dataset is already extracted. Skipping extraction.")
        return

    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {ARCHIVE_PATH.name}. Put archive.zip in the project root."
        )

    extract_root = RAW_DATA_DIR.parent
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracting {ARCHIVE_PATH.name} ...")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
        archive.extractall(extract_root)

    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(
            "Archive was extracted, but raw_dataset/tomato/train and val were not found."
        )

    print("[DONE] Dataset extracted to raw_dataset/tomato/")


def class_counts(split_dir: Path) -> Counter:
    counts = Counter()
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        counts[class_dir.name] = sum(1 for p in class_dir.iterdir() if is_image(p))
    return counts


def validate_dataset() -> None:
    train_counts = class_counts(RAW_DATA_DIR / "train")
    test_counts = class_counts(RAW_DATA_DIR / "val")

    if HEALTHY_CLASS not in train_counts:
        raise RuntimeError(f"Required class '{HEALTHY_CLASS}' was not found.")

    if len(train_counts) != 10 or len(test_counts) != 10:
        raise RuntimeError(
            f"Expected 10 original tomato classes, found "
            f"{len(train_counts)} train and {len(test_counts)} val classes."
        )

    print("\n=== ARCHIVE DATASET CHECK ===")
    for class_name in sorted(train_counts):
        binary = "healthy" if class_name == HEALTHY_CLASS else "sick"
        print(
            f"{class_name:48s} | "
            f"train={train_counts[class_name]:4d} | "
            f"val={test_counts[class_name]:3d} | -> {binary}"
        )

    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    print(f"\nOriginal train images: {total_train}")
    print(f"Original val images  : {total_test}")

    if any(v != EXPECTED_TRAIN_PER_CLASS for v in train_counts.values()):
        print("[WARN] Some training classes do not contain 1,000 images.")
    if any(v != EXPECTED_TEST_PER_CLASS for v in test_counts.values()):
        print("[WARN] Some original validation classes do not contain 100 images.")

    print("\nBinary target:")
    print("  Tomato___healthy -> healthy")
    print("  all 9 disease classes -> sick")
    print("\n[READY] Dataset can be used by train.py.")


def main() -> None:
    extract_archive()
    validate_dataset()


if __name__ == "__main__":
    main()
