"""
Merge Public and Collected Datasets

Combines data from multiple sources into a unified training dataset:
- Public datasets (Devign, CodeXGLUE)
- Collector datasets (Synthetic, OSV, ExploitDB)

Usage:
    python merge_datasets.py --output data/training/merged_dataset.jsonl
    python merge_datasets.py --balance --test-split 0.2
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import random


class DatasetMerger:
    """Merge multiple vulnerability datasets into a unified format."""

    def __init__(self):
        self.samples = []
        self.stats = defaultdict(int)

    def load_jsonl(self, file_path: Path, source_label: str = None) -> int:
        """Load samples from a JSONL file."""
        if not file_path.exists():
            print(f"[!] File not found: {file_path}")
            return 0

        count = 0
        print(f"\nLoading: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())

                        # Add source label if provided
                        if source_label:
                            if 'metadata' not in sample:
                                sample['metadata'] = {}
                            sample['metadata']['dataset_source'] = source_label

                        self.samples.append(sample)
                        count += 1

                        # Track statistics
                        severity = sample.get('severity', 'UNKNOWN')
                        self.stats[f'severity_{severity}'] += 1
                        self.stats['total_samples'] += 1

                    except json.JSONDecodeError as e:
                        continue

            print(f"  [+] Loaded {count:,} samples")
            return count

        except Exception as e:
            print(f"  [!] Error loading {file_path}: {e}")
            return 0

    def balance_dataset(self, max_per_category: int = None) -> None:
        """Balance vulnerable vs safe samples."""
        print("\nBalancing dataset...")

        vulnerable = [s for s in self.samples if s.get('severity') not in ['SAFE', 'UNKNOWN']]
        safe = [s for s in self.samples if s.get('severity') == 'SAFE']

        print(f"  Before balancing:")
        print(f"    Vulnerable: {len(vulnerable):,}")
        print(f"    Safe: {len(safe):,}")

        # Balance to the smaller set
        target_count = min(len(vulnerable), len(safe))

        if max_per_category:
            target_count = min(target_count, max_per_category)

        vulnerable = random.sample(vulnerable, min(len(vulnerable), target_count))
        safe = random.sample(safe, min(len(safe), target_count))

        self.samples = vulnerable + safe
        random.shuffle(self.samples)

        print(f"  After balancing:")
        print(f"    Vulnerable: {len(vulnerable):,}")
        print(f"    Safe: {len(safe):,}")
        print(f"    Total: {len(self.samples):,}")

    def create_train_test_split(self, test_ratio: float = 0.2) -> tuple:
        """Split dataset into train and test sets."""
        random.shuffle(self.samples)

        split_idx = int(len(self.samples) * (1 - test_ratio))

        train_samples = self.samples[:split_idx]
        test_samples = self.samples[split_idx:]

        print(f"\nDataset split:")
        print(f"  Train: {len(train_samples):,} samples ({100*(1-test_ratio):.0f}%)")
        print(f"  Test: {len(test_samples):,} samples ({100*test_ratio:.0f}%)")

        return train_samples, test_samples

    def save_dataset(self, samples: List[Dict], output_file: Path) -> None:
        """Save samples to JSONL file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"\n[+] Saved {len(samples):,} samples to: {output_file}")

    def print_statistics(self) -> None:
        """Print dataset statistics."""
        print("\n" + "="*70)
        print("Dataset Statistics")
        print("="*70)

        # Count by source
        sources = defaultdict(int)
        severities = defaultdict(int)
        ecosystems = defaultdict(int)

        for sample in self.samples:
            source = sample.get('source', 'unknown')
            severity = sample.get('severity', 'UNKNOWN')
            ecosystem = sample.get('ecosystem', 'unknown')

            sources[source] += 1
            severities[severity] += 1
            ecosystems[ecosystem] += 1

        print(f"\nTotal Samples: {len(self.samples):,}\n")

        print("By Source:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source:20} {count:>8,} ({100*count/len(self.samples):>5.1f}%)")

        print("\nBy Severity:")
        for severity, count in sorted(severities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {severity:20} {count:>8,} ({100*count/len(self.samples):>5.1f}%)")

        print("\nBy Ecosystem:")
        for ecosystem, count in sorted(ecosystems.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {ecosystem:20} {count:>8,} ({100*count/len(self.samples):>5.1f}%)")

        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge vulnerability datasets from multiple sources"
    )

    parser.add_argument(
        '--public',
        nargs='+',
        help='Public dataset files (e.g., devign_processed.jsonl)'
    )

    parser.add_argument(
        '--collectors',
        nargs='+',
        help='Collector dataset files (e.g., synthetic_data.jsonl)'
    )

    parser.add_argument(
        '--output',
        default='data/training/merged_dataset.jsonl',
        help='Output file path (default: data/training/merged_dataset.jsonl)'
    )

    parser.add_argument(
        '--balance',
        action='store_true',
        help='Balance vulnerable vs safe samples'
    )

    parser.add_argument(
        '--max-per-category',
        type=int,
        help='Maximum samples per category when balancing'
    )

    parser.add_argument(
        '--test-split',
        type=float,
        default=0.0,
        help='Test set ratio (default: 0.0, no split)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("\n" + "="*70)
    print("StreamGuard - Dataset Merger")
    print("="*70)

    merger = DatasetMerger()

    # Load public datasets
    if args.public:
        print("\n" + "="*70)
        print("Loading Public Datasets")
        print("="*70)

        for file_path in args.public:
            path = Path(file_path)
            source_name = path.parent.name  # e.g., "devign", "codexglue"
            merger.load_jsonl(path, source_label=f"public_{source_name}")

    # Load collector datasets
    if args.collectors:
        print("\n" + "="*70)
        print("Loading Collector Datasets")
        print("="*70)

        for file_path in args.collectors:
            path = Path(file_path)
            source_name = path.parent.name  # e.g., "synthetic", "osv"
            merger.load_jsonl(path, source_label=f"collector_{source_name}")

    if not merger.samples:
        print("\n[!] No samples loaded. Check your input files.")
        return

    # Balance if requested
    if args.balance:
        merger.balance_dataset(max_per_category=args.max_per_category)

    # Create train/test split if requested
    if args.test_split > 0:
        train_samples, test_samples = merger.create_train_test_split(args.test_split)

        # Save train set
        output_path = Path(args.output)
        train_path = output_path.parent / f"{output_path.stem}_train.jsonl"
        merger.save_dataset(train_samples, train_path)

        # Save test set
        test_path = output_path.parent / f"{output_path.stem}_test.jsonl"
        merger.save_dataset(test_samples, test_path)

        # Update merger.samples to train samples for statistics
        merger.samples = train_samples + test_samples

    else:
        # Save all samples
        merger.save_dataset(merger.samples, Path(args.output))

    # Print statistics
    merger.print_statistics()

    print("[+] Merge complete!\n")


if __name__ == "__main__":
    main()
