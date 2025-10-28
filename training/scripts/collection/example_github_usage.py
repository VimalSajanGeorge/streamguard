#!/usr/bin/env python3
"""
Example usage of the GitHub Advisory Collector.

This script demonstrates how to use the GitHubAdvisoryCollectorEnhanced
in different scenarios.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from github_advisory_collector_enhanced import GitHubAdvisoryCollectorEnhanced


def example_basic_usage():
    """Example 1: Basic usage - collect all advisories."""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)

    # Initialize collector
    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    # Collect samples (default: 10,000)
    samples = collector.collect()

    print(f"\nCollected {len(samples)} samples")
    print(f"Output: data/raw/github/github_advisories.jsonl")


def example_small_collection():
    """Example 2: Small collection for testing."""
    print("="*60)
    print("Example 2: Small Collection (100 samples)")
    print("="*60)

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    # Collect only 100 samples for quick testing
    samples = collector.collect_all_advisories(target_samples=100)

    print(f"\nCollected {len(samples)} samples")

    # Show some statistics
    stats = collector.get_stats()
    print(f"\nStatistics:")
    print(f"  Samples collected: {stats['samples_collected']}")
    print(f"  Errors: {stats['errors_count']}")


def example_ecosystem_specific():
    """Example 3: Collect for specific ecosystem."""
    print("="*60)
    print("Example 3: Ecosystem-Specific (Python/PIP only)")
    print("="*60)

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    # Collect Python vulnerabilities across all severities
    python_samples = []

    for severity in ["HIGH", "CRITICAL"]:
        samples = collector.collect_by_ecosystem_severity(
            ecosystem="PIP",
            severity=severity,
            max_samples=500
        )
        python_samples.extend(samples)

    # Save to separate file
    output_file = collector.save_samples(
        python_samples,
        "github_advisories_python.jsonl"
    )

    print(f"\nCollected {len(python_samples)} Python vulnerability samples")
    print(f"Output: {output_file}")


def example_high_severity_only():
    """Example 4: Collect only high-severity vulnerabilities."""
    print("="*60)
    print("Example 4: High-Severity Only")
    print("="*60)

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    high_severity_samples = []

    # Collect high and critical severity across all ecosystems
    for ecosystem in ["PIP", "NPM", "MAVEN", "GO"]:
        for severity in ["HIGH", "CRITICAL"]:
            samples = collector.collect_by_ecosystem_severity(
                ecosystem=ecosystem,
                severity=severity,
                max_samples=250
            )
            high_severity_samples.extend(samples)

    output_file = collector.save_samples(
        high_severity_samples,
        "github_advisories_high_severity.jsonl"
    )

    print(f"\nCollected {len(high_severity_samples)} high-severity samples")
    print(f"Output: {output_file}")


def example_with_filtering():
    """Example 5: Collect with custom filtering."""
    print("="*60)
    print("Example 5: Custom Filtering (SQL Injection only)")
    print("="*60)

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    # Collect samples
    all_samples = collector.collect_all_advisories(target_samples=1000)

    # Filter for SQL injection only
    sql_injection_samples = [
        sample for sample in all_samples
        if "sql" in sample["description"].lower()
        or sample["metadata"]["vulnerability_type"] == "sql_injection"
    ]

    # Save filtered samples
    output_file = collector.save_samples(
        sql_injection_samples,
        "github_advisories_sql_injection.jsonl"
    )

    print(f"\nCollected {len(all_samples)} total samples")
    print(f"Filtered to {len(sql_injection_samples)} SQL injection samples")
    print(f"Output: {output_file}")


def example_code_extraction():
    """Example 6: Extract code for specific package."""
    print("="*60)
    print("Example 6: Code Extraction for Specific Package")
    print("="*60)

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    # Extract code for a specific package
    vulnerable_code, fixed_code = collector.extract_code_with_diff(
        package_name="django",
        ecosystem="PIP",
        vulnerable_range="< 3.2.5",
        patched_version="3.2.5",
        references=[
            "https://github.com/django/django",
            "https://github.com/django/django/security/advisories"
        ]
    )

    if vulnerable_code and fixed_code:
        print("\nSuccessfully extracted code!")
        print("\nVulnerable code snippet:")
        print(vulnerable_code[:200] + "..." if len(vulnerable_code) > 200 else vulnerable_code)
        print("\nFixed code snippet:")
        print(fixed_code[:200] + "..." if len(fixed_code) > 200 else fixed_code)
    else:
        print("\nCould not extract code (may not be available)")


def example_with_statistics():
    """Example 7: Collect and display detailed statistics."""
    print("="*60)
    print("Example 7: Collection with Statistics")
    print("="*60)

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    # Collect samples
    samples = collector.collect_all_advisories(target_samples=500)

    # Get statistics
    stats = collector.get_stats()

    print("\n" + "="*60)
    print("COLLECTION STATISTICS")
    print("="*60)
    print(f"Samples collected: {stats['samples_collected']}")
    print(f"Errors encountered: {stats['errors_count']}")

    # Additional stats from collector
    print(f"\nTotal advisories processed: {collector.stats['total_advisories']}")
    print(f"Successful code extractions: {collector.stats['successful_extractions']}")
    print(f"Failed code extractions: {collector.stats['failed_extractions']}")

    success_rate = (
        collector.stats['successful_extractions'] /
        collector.stats['total_advisories'] * 100
        if collector.stats['total_advisories'] > 0 else 0
    )
    print(f"Code extraction success rate: {success_rate:.1f}%")

    print("\nBy Ecosystem:")
    for ecosystem, count in sorted(collector.stats['by_ecosystem'].items()):
        print(f"  {ecosystem}: {count}")

    print("\nBy Severity:")
    for severity, count in sorted(collector.stats['by_severity'].items()):
        print(f"  {severity}: {count}")


def main():
    """Run examples based on command-line argument."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GitHub Advisory Collector Examples"
    )
    parser.add_argument(
        "example",
        type=int,
        choices=range(1, 8),
        help="Example number to run (1-7)"
    )

    args = parser.parse_args()

    # Check for GitHub token
    if not os.getenv("GITHUB_TOKEN"):
        print("ERROR: GITHUB_TOKEN environment variable is not set.")
        print("Please set your GitHub Personal Access Token:")
        print("  export GITHUB_TOKEN='your_token_here'")
        sys.exit(1)

    # Run selected example
    examples = {
        1: example_basic_usage,
        2: example_small_collection,
        3: example_ecosystem_specific,
        4: example_high_severity_only,
        5: example_with_filtering,
        6: example_code_extraction,
        7: example_with_statistics
    }

    try:
        examples[args.example]()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nExample failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - show all examples
        print("GitHub Advisory Collector - Usage Examples")
        print("="*60)
        print("\nAvailable examples:")
        print("  1. Basic usage - collect all advisories")
        print("  2. Small collection (100 samples for testing)")
        print("  3. Ecosystem-specific (Python/PIP only)")
        print("  4. High-severity only")
        print("  5. Custom filtering (SQL injection only)")
        print("  6. Code extraction for specific package")
        print("  7. Collection with detailed statistics")
        print("\nUsage:")
        print("  python example_github_usage.py <example_number>")
        print("\nExample:")
        print("  python example_github_usage.py 2")
    else:
        main()
