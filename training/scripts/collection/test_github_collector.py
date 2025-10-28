#!/usr/bin/env python3
"""
Test script for GitHub Advisory Collector.
This will do a small test collection to identify issues.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from github_advisory_collector_enhanced import GitHubAdvisoryCollectorEnhanced


def main():
    """Run a small test collection."""
    print("="*70)
    print("GitHub Advisory Collector - DEBUG TEST")
    print("="*70)

    # Check for GitHub token
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("\nERROR: GITHUB_TOKEN not found!")
        print("Please set GITHUB_TOKEN in .env file")
        return 1

    print(f"\nGitHub token found: {github_token[:10]}...")

    # Initialize collector
    print("\nInitializing collector...")
    try:
        collector = GitHubAdvisoryCollectorEnhanced(
            output_dir="data/raw/github",
            cache_enabled=True
        )
        print("[OK] Collector initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize collector: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test small collection
    print("\n" + "="*70)
    print("STARTING SMALL TEST COLLECTION")
    print("="*70)
    print("\nCollecting 50 samples from PIP/HIGH to test...")
    print("This should take ~2 minutes")
    print()

    try:
        # Collect from one ecosystem/severity combination
        samples = collector.collect_by_ecosystem_severity(
            ecosystem="PIP",
            severity="HIGH",
            max_samples=50
        )

        print("\n" + "="*70)
        print("TEST COLLECTION RESULTS")
        print("="*70)
        print(f"\nSamples collected: {len(samples)}")

        if len(samples) > 0:
            print(f"\nFirst sample:")
            print(f"  Advisory ID: {samples[0].get('advisory_id')}")
            print(f"  Description: {samples[0].get('description', '')[:100]}...")
            print(f"  Ecosystem: {samples[0].get('ecosystem')}")
            print(f"  Severity: {samples[0].get('severity')}")

            # Save test results
            output_file = collector.save_samples(samples, "github_advisories_test.jsonl")
            print(f"\n[OK] Test samples saved to: {output_file}")

            # Check file
            if output_file.exists():
                file_size = output_file.stat().st_size
                print(f"[OK] File exists, size: {file_size} bytes")
            else:
                print(f"[ERROR] File not found at {output_file}")
        else:
            print("\n[ERROR] NO SAMPLES COLLECTED!")
            print("\nPossible issues:")
            print("  1. GraphQL query might be failing")
            print("  2. No data matches the filter criteria")
            print("  3. API authentication issue")
            print("  4. Rate limiting")

        # Print statistics
        stats = collector.get_stats()
        print(f"\nCollection statistics:")
        print(f"  Total samples collected: {stats['samples_collected']}")
        print(f"  Errors: {stats['errors_count']}")

        if stats['errors_count'] > 0:
            print("\nRecent errors:")
            for error in stats['errors'][-5:]:
                print(f"  - {error.get('error', 'Unknown error')}")

        return 0 if len(samples) > 0 else 1

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
