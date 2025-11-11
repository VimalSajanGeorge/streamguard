"""
Generate Small Deterministic Dataset for Unit Testing

Purpose:
    Create small, reproducible dataset samples for fast unit tests.
    Avoids loading full ~21,854 sample dataset during development.

Outputs:
    - data/sample/train_small.jsonl (100 samples)
    - data/sample/valid_small.jsonl (20 samples)
    - data/sample/test_small.jsonl (20 samples)

Usage:
    python scripts/generate_small_dataset.py
    python scripts/generate_small_dataset.py --samples 50  # Custom size

Features:
    - Deterministic sampling (seed=42)
    - Balanced class distribution
    - Preserves data format
    - Fast generation (<30 seconds)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import random
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num} in {path}: {e}")
    return records


def save_jsonl(records: List[Dict], path: Path) -> None:
    """Save list of dicts to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def sample_balanced(
    records: List[Dict],
    n_samples: int,
    label_key: str = "target",
    seed: int = 42
) -> List[Dict]:
    """
    Sample records with balanced class distribution.

    Args:
        records: List of data records
        n_samples: Total number of samples to extract
        label_key: Key containing the label (default: "target")
        seed: Random seed for reproducibility

    Returns:
        List of sampled records with balanced classes
    """
    random.seed(seed)

    # Separate by class
    class_0 = [r for r in records if r.get(label_key) == 0]
    class_1 = [r for r in records if r.get(label_key) == 1]

    logger.info(f"Original distribution: Class 0={len(class_0)}, Class 1={len(class_1)}")

    # Sample equally from each class
    n_per_class = n_samples // 2

    sampled_0 = random.sample(class_0, min(n_per_class, len(class_0)))
    sampled_1 = random.sample(class_1, min(n_per_class, len(class_1)))

    # Combine and shuffle
    sampled = sampled_0 + sampled_1
    random.shuffle(sampled)

    logger.info(f"Sampled distribution: Class 0={len(sampled_0)}, Class 1={len(sampled_1)}")

    return sampled[:n_samples]


def generate_small_dataset(
    source_dir: Path = Path("data/processed/codexglue"),
    output_dir: Path = Path("data/sample"),
    train_samples: int = 100,
    val_samples: int = 20,
    test_samples: int = 20,
    seed: int = 42
) -> None:
    """
    Generate small dataset samples for unit testing.

    Args:
        source_dir: Directory containing full dataset JSONL files
        output_dir: Directory to write small samples
        train_samples: Number of training samples (default: 100)
        val_samples: Number of validation samples (default: 20)
        test_samples: Number of test samples (default: 20)
        seed: Random seed for reproducibility
    """
    logger.info("="*60)
    logger.info("GENERATING SMALL DATASET FOR UNIT TESTS")
    logger.info("="*60)

    # Check if source files exist
    train_path = source_dir / "train.jsonl"
    val_path = source_dir / "valid.jsonl"
    test_path = source_dir / "test.jsonl"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate training sample
    logger.info(f"\n[1/3] Generating train_small.jsonl ({train_samples} samples)...")
    train_records = load_jsonl(train_path)
    logger.info(f"Loaded {len(train_records)} training records")
    train_small = sample_balanced(train_records, train_samples, seed=seed)
    save_jsonl(train_small, output_dir / "train_small.jsonl")
    logger.info(f"✓ Saved {len(train_small)} samples to {output_dir / 'train_small.jsonl'}")

    # Generate validation sample
    logger.info(f"\n[2/3] Generating valid_small.jsonl ({val_samples} samples)...")
    val_records = load_jsonl(val_path)
    logger.info(f"Loaded {len(val_records)} validation records")
    val_small = sample_balanced(val_records, val_samples, seed=seed+1)
    save_jsonl(val_small, output_dir / "valid_small.jsonl")
    logger.info(f"✓ Saved {len(val_small)} samples to {output_dir / 'valid_small.jsonl'}")

    # Generate test sample
    logger.info(f"\n[3/3] Generating test_small.jsonl ({test_samples} samples)...")
    test_records = load_jsonl(test_path)
    logger.info(f"Loaded {len(test_records)} test records")
    test_small = sample_balanced(test_records, test_samples, seed=seed+2)
    save_jsonl(test_small, output_dir / "test_small.jsonl")
    logger.info(f"✓ Saved {len(test_small)} samples to {output_dir / 'test_small.jsonl'}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Train: {len(train_small)} samples")
    logger.info(f"Valid: {len(val_small)} samples")
    logger.info(f"Test: {len(test_small)} samples")
    logger.info(f"Total: {len(train_small) + len(val_small) + len(test_small)} samples")
    logger.info("\nUsage in unit tests:")
    logger.info("  train_data = 'data/sample/train_small.jsonl'")
    logger.info("  val_data = 'data/sample/valid_small.jsonl'")


def main():
    parser = argparse.ArgumentParser(
        description="Generate small deterministic dataset for unit testing"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed/codexglue"),
        help="Directory containing full dataset JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sample"),
        help="Directory to write small samples"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=100,
        help="Number of training samples (default: 100)"
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=20,
        help="Number of validation samples (default: 20)"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=20,
        help="Number of test samples (default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    try:
        generate_small_dataset(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            seed=args.seed
        )
    except Exception as e:
        logger.error(f"Failed to generate small dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
