"""Example usage of Enhanced Repository Miner.

This script demonstrates how to use the EnhancedRepoMiner to collect
security vulnerability samples from open-source repositories.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from training.scripts.collection.repo_miner_enhanced import EnhancedRepoMiner


def example_basic_usage():
    """Example of basic usage."""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)

    # Initialize miner with default settings
    miner = EnhancedRepoMiner(
        output_dir="data/raw/opensource",
        cache_enabled=True
    )

    # Collect samples
    print("Starting collection...")
    samples = miner.collect()

    # Save samples
    miner.save_samples_to_file(samples)

    # Print statistics
    stats = miner.get_stats()
    print(f"\nCollection complete!")
    print(f"Total samples: {len(samples)}")
    print(f"Errors: {stats['errors_count']}")


def example_custom_output():
    """Example of using custom output directory."""
    print("\n" + "="*60)
    print("Example 2: Custom Output Directory")
    print("="*60)

    # Initialize with custom output directory
    miner = EnhancedRepoMiner(
        output_dir="custom/output/path",
        cache_enabled=True
    )

    # Rest is the same as basic usage
    samples = miner.collect()
    miner.save_samples_to_file(samples, filename="custom_samples.jsonl")

    print(f"\nSaved {len(samples)} samples to custom/output/path/custom_samples.jsonl")


def example_analyze_single_repo():
    """Example of mining a single repository."""
    print("\n" + "="*60)
    print("Example 3: Single Repository Mining")
    print("="*60)

    miner = EnhancedRepoMiner(
        output_dir="data/raw/opensource",
        cache_enabled=True
    )

    # Mine just one repository
    repo_name = "pallets/flask"
    config = {"language": "python", "target": 100}  # Small target for testing

    print(f"Mining {repo_name}...")
    samples = miner.mine_repository(repo_name, config)

    print(f"\nCollected {len(samples)} samples from {repo_name}")

    # Print first sample as example
    if samples:
        print("\nExample sample:")
        sample = samples[0]
        print(f"Repository: {sample['repository']}")
        print(f"Vulnerability: {sample['vulnerability_type']}")
        print(f"File: {sample['file_path']}")
        print(f"Commit: {sample['commit_sha'][:8]}")
        print(f"\nVulnerable code (first 200 chars):")
        print(sample['vulnerable_code'][:200])
        print(f"\nFixed code (first 200 chars):")
        print(sample['fixed_code'][:200])


def example_statistics():
    """Example of analyzing collected statistics."""
    print("\n" + "="*60)
    print("Example 4: Statistics Analysis")
    print("="*60)

    miner = EnhancedRepoMiner(
        output_dir="data/raw/opensource",
        cache_enabled=True
    )

    # Collect samples
    samples = miner.collect()

    # Get statistics
    stats = miner.get_stats()

    print("\n--- Collection Statistics ---")
    print(f"Total samples: {len(samples)}")
    print(f"Total repositories: {len(miner.REPOSITORIES)}")
    print(f"Errors encountered: {stats['errors_count']}")

    # Count by language
    python_samples = sum(1 for s in samples if s.get('language') == 'python')
    js_samples = sum(1 for s in samples if s.get('language') == 'javascript')

    print(f"\n--- Language Distribution ---")
    print(f"Python samples: {python_samples}")
    print(f"JavaScript samples: {js_samples}")

    # Count by vulnerability type
    vuln_counts = {}
    for sample in samples:
        vuln = sample.get('vulnerability_type', 'unknown')
        vuln_counts[vuln] = vuln_counts.get(vuln, 0) + 1

    print(f"\n--- Top 5 Vulnerability Types ---")
    sorted_vulns = sorted(vuln_counts.items(), key=lambda x: x[1], reverse=True)
    for vuln, count in sorted_vulns[:5]:
        print(f"{vuln}: {count} samples")

    # Count by repository
    repo_counts = {}
    for sample in samples:
        repo = sample.get('repository', 'unknown')
        repo_counts[repo] = repo_counts.get(repo, 0) + 1

    print(f"\n--- Samples per Repository ---")
    for repo, count in sorted(repo_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{repo}: {count} samples")


def example_filtering():
    """Example of filtering collected samples."""
    print("\n" + "="*60)
    print("Example 5: Filtering Samples")
    print("="*60)

    miner = EnhancedRepoMiner(
        output_dir="data/raw/opensource",
        cache_enabled=True
    )

    # Collect samples
    samples = miner.collect()

    # Filter for specific vulnerability types
    sql_injection_samples = [
        s for s in samples
        if s.get('vulnerability_type') == 'sql_injection'
    ]

    print(f"\nTotal samples: {len(samples)}")
    print(f"SQL injection samples: {len(sql_injection_samples)}")

    # Filter for specific languages
    python_samples = [
        s for s in samples
        if s.get('language') == 'python'
    ]

    print(f"Python samples: {len(python_samples)}")

    # Filter for specific repositories
    django_samples = [
        s for s in samples
        if s.get('repository') == 'django/django'
    ]

    print(f"Django samples: {len(django_samples)}")

    # Save filtered samples
    if sql_injection_samples:
        miner.save_samples(sql_injection_samples, "sql_injection_only.jsonl")
        print(f"\nSaved SQL injection samples to sql_injection_only.jsonl")


def example_quality_check():
    """Example of checking sample quality."""
    print("\n" + "="*60)
    print("Example 6: Quality Checking")
    print("="*60)

    miner = EnhancedRepoMiner(
        output_dir="data/raw/opensource",
        cache_enabled=True
    )

    # Collect samples
    samples = miner.collect()

    # Check code lengths
    vulnerable_lengths = [len(s['vulnerable_code']) for s in samples]
    fixed_lengths = [len(s['fixed_code']) for s in samples]

    print(f"\nVulnerable code length statistics:")
    print(f"  Min: {min(vulnerable_lengths)} chars")
    print(f"  Max: {max(vulnerable_lengths)} chars")
    print(f"  Avg: {sum(vulnerable_lengths) // len(vulnerable_lengths)} chars")

    print(f"\nFixed code length statistics:")
    print(f"  Min: {min(fixed_lengths)} chars")
    print(f"  Max: {max(fixed_lengths)} chars")
    print(f"  Avg: {sum(fixed_lengths) // len(fixed_lengths)} chars")

    # Check for samples with commit messages
    with_messages = sum(1 for s in samples if s.get('commit_message'))
    print(f"\nSamples with commit messages: {with_messages}/{len(samples)}")

    # Check for samples with dates
    with_dates = sum(1 for s in samples if s.get('committed_date'))
    print(f"Samples with dates: {with_dates}/{len(samples)}")


def main():
    """Run all examples."""
    print("Enhanced Repository Miner - Usage Examples\n")

    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Output", example_custom_output),
        ("Single Repository", example_analyze_single_repo),
        ("Statistics", example_statistics),
        ("Filtering", example_filtering),
        ("Quality Check", example_quality_check),
    ]

    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nNote: These are demonstration examples.")
    print("To actually run mining, use: python repo_miner_enhanced.py")
    print("\nFor testing, run: python test_repo_miner.py")


if __name__ == "__main__":
    main()
