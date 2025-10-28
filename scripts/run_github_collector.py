#!/usr/bin/env python3
"""
Quick start script for GitHub Advisory Collector.

This script provides a simple interface to run the GitHub Advisory Collector
with common configurations.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.scripts.collection.github_advisory_collector_enhanced import (
    GitHubAdvisoryCollectorEnhanced
)


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("Checking prerequisites...")

    # Check GitHub token
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN environment variable is not set.")
        print("\nPlease set your GitHub Personal Access Token:")
        print("  Linux/Mac: export GITHUB_TOKEN='your_token_here'")
        print("  Windows:   set GITHUB_TOKEN=your_token_here")
        print("\nSee docs/github_advisory_collector_guide.md for details.")
        sys.exit(1)

    print(f"  GitHub Token: Found ({'*' * 8}{token[-4:]})")

    # Check output directory
    output_dir = Path("data/raw/github")
    if not output_dir.exists():
        print(f"  Output directory: Creating {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"  Output directory: {output_dir} (exists)")

    print("Prerequisites check: PASSED\n")


def quick_test():
    """Run a quick test collection (100 samples)."""
    print("="*60)
    print("QUICK TEST - Collecting 100 samples")
    print("="*60)

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    try:
        samples = collector.collect_all_advisories(target_samples=100)
        print(f"\nTest successful! Collected {len(samples)} samples.")
        print(f"Output: data/raw/github/github_advisories.jsonl")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()


def small_collection():
    """Run a small collection (1,000 samples)."""
    print("="*60)
    print("SMALL COLLECTION - Collecting 1,000 samples")
    print("="*60)

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    try:
        samples = collector.collect_all_advisories(target_samples=1000)
        print(f"\nCollection successful! Collected {len(samples)} samples.")
        print(f"Output: data/raw/github/github_advisories.jsonl")

    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
        print(f"Collected {collector.samples_collected} samples before interruption.")

    except Exception as e:
        print(f"\nCollection failed: {str(e)}")
        import traceback
        traceback.print_exc()


def full_collection():
    """Run a full collection (10,000 samples)."""
    print("="*60)
    print("FULL COLLECTION - Collecting 10,000 samples")
    print("="*60)
    print("This will take several hours due to rate limiting.")
    print("Progress will be saved periodically.")
    print("You can safely interrupt (Ctrl+C) and resume later.\n")

    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    try:
        samples = collector.collect_all_advisories(target_samples=10000)
        print(f"\nCollection successful! Collected {len(samples)} samples.")
        print(f"Output: data/raw/github/github_advisories.jsonl")

    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
        print(f"Collected {collector.samples_collected} samples before interruption.")
        print("Progress has been saved. Re-run to continue from cache.")

    except Exception as e:
        print(f"\nCollection failed: {str(e)}")
        import traceback
        traceback.print_exc()


def custom_collection():
    """Run a custom collection with user input."""
    print("="*60)
    print("CUSTOM COLLECTION")
    print("="*60)

    # Get target samples
    while True:
        try:
            target = input("Number of samples to collect (default: 1000): ").strip()
            target_samples = int(target) if target else 1000
            if target_samples <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Get cache preference
    cache_input = input("Enable caching? (Y/n): ").strip().lower()
    cache_enabled = cache_input != 'n'

    print(f"\nConfiguration:")
    print(f"  Target samples: {target_samples}")
    print(f"  Caching: {'Enabled' if cache_enabled else 'Disabled'}")
    print()

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=cache_enabled
    )

    try:
        samples = collector.collect_all_advisories(target_samples=target_samples)
        print(f"\nCollection successful! Collected {len(samples)} samples.")
        print(f"Output: data/raw/github/github_advisories.jsonl")

    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
        print(f"Collected {collector.samples_collected} samples before interruption.")

    except Exception as e:
        print(f"\nCollection failed: {str(e)}")
        import traceback
        traceback.print_exc()


def ecosystem_specific():
    """Collect for a specific ecosystem."""
    print("="*60)
    print("ECOSYSTEM-SPECIFIC COLLECTION")
    print("="*60)

    ecosystems = ["PIP", "NPM", "MAVEN", "RUBYGEMS", "GO", "COMPOSER", "NUGET", "CARGO"]
    severities = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

    print("\nAvailable ecosystems:")
    for i, eco in enumerate(ecosystems, 1):
        print(f"  {i}. {eco}")

    while True:
        try:
            choice = input("\nSelect ecosystem (1-8): ").strip()
            eco_idx = int(choice) - 1
            if 0 <= eco_idx < len(ecosystems):
                ecosystem = ecosystems[eco_idx]
                break
            print("Please select a number between 1 and 8.")
        except ValueError:
            print("Please enter a valid number.")

    print("\nAvailable severities:")
    for i, sev in enumerate(severities, 1):
        print(f"  {i}. {sev}")

    while True:
        try:
            choice = input("\nSelect severity (1-4, or 0 for all): ").strip()
            sev_idx = int(choice) - 1
            if choice == '0':
                severity = None  # All severities
                break
            elif 0 <= sev_idx < len(severities):
                severity = severities[sev_idx]
                break
            print("Please select a number between 0 and 4.")
        except ValueError:
            print("Please enter a valid number.")

    target_samples = int(input("\nNumber of samples (default: 500): ").strip() or "500")

    print(f"\nCollecting {target_samples} samples for {ecosystem}" +
          (f" / {severity}" if severity else " / ALL severities"))

    collector = GitHubAdvisoryCollectorEnhanced(
        output_dir="data/raw/github",
        cache_enabled=True
    )

    try:
        if severity:
            samples = collector.collect_by_ecosystem_severity(
                ecosystem=ecosystem,
                severity=severity,
                max_samples=target_samples
            )
        else:
            # Collect across all severities
            samples = []
            for sev in severities:
                sev_samples = collector.collect_by_ecosystem_severity(
                    ecosystem=ecosystem,
                    severity=sev,
                    max_samples=target_samples // len(severities)
                )
                samples.extend(sev_samples)

        # Save samples
        output_file = collector.save_samples(
            samples,
            f"github_advisories_{ecosystem.lower()}.jsonl"
        )

        print(f"\nCollection successful! Collected {len(samples)} samples.")
        print(f"Output: {output_file}")

    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")

    except Exception as e:
        print(f"\nCollection failed: {str(e)}")
        import traceback
        traceback.print_exc()


def main_menu():
    """Display main menu and handle user selection."""
    while True:
        print("\n" + "="*60)
        print("GitHub Security Advisory Collector")
        print("="*60)
        print("\nSelect an option:")
        print("  1. Quick Test (100 samples, ~5 minutes)")
        print("  2. Small Collection (1,000 samples, ~30 minutes)")
        print("  3. Full Collection (10,000 samples, ~4-6 hours)")
        print("  4. Custom Collection")
        print("  5. Ecosystem-Specific Collection")
        print("  0. Exit")

        choice = input("\nYour choice: ").strip()

        if choice == "1":
            quick_test()
        elif choice == "2":
            small_collection()
        elif choice == "3":
            full_collection()
        elif choice == "4":
            custom_collection()
        elif choice == "5":
            ecosystem_specific()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("GitHub Security Advisory Collector - Quick Start")
    print("="*60)
    print()

    # Check prerequisites
    check_prerequisites()

    # Show main menu
    main_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
