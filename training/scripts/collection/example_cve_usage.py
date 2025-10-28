"""Example usage of the Enhanced CVE Collector.

This script demonstrates various ways to use the CVE collector
with different configurations and options.
"""

import os
from pathlib import Path
from cve_collector_enhanced import CVECollectorEnhanced


def example_basic_collection():
    """Example 1: Basic collection with default settings."""
    print("\n" + "="*80)
    print("Example 1: Basic Collection")
    print("="*80)

    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "cves"

    collector = CVECollectorEnhanced(
        output_dir=str(output_dir),
        cache_enabled=True
    )

    # Collect small sample for demonstration
    # Override the target to collect just 100 samples
    collector.TARGET_SAMPLES = 100

    samples = collector.collect()

    print(f"\nCollected {len(samples)} samples")
    print(f"Samples with code: {sum(1 for s in samples if s.get('vulnerable_code'))}")

    # Show first sample
    if samples:
        print("\nFirst sample:")
        print(f"  CVE ID: {samples[0]['cve_id']}")
        print(f"  Type: {samples[0]['vulnerability_type']}")
        print(f"  Severity: {samples[0]['severity']}")
        print(f"  Has code: {'Yes' if samples[0].get('vulnerable_code') else 'No'}")


def example_with_github_token():
    """Example 2: Collection with GitHub token for better rate limits."""
    print("\n" + "="*80)
    print("Example 2: Collection with GitHub Token")
    print("="*80)

    # Get GitHub token from environment variable
    github_token = os.environ.get('GITHUB_TOKEN')

    if not github_token:
        print("Warning: GITHUB_TOKEN environment variable not set")
        print("Set it with: export GITHUB_TOKEN='your_token_here'")
        return

    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "cves"

    collector = CVECollectorEnhanced(
        output_dir=str(output_dir),
        cache_enabled=True,
        github_token=github_token
    )

    collector.TARGET_SAMPLES = 100

    samples = collector.collect()

    print(f"\nCollected {len(samples)} samples with GitHub token")


def example_specific_keywords():
    """Example 3: Collect only specific vulnerability types."""
    print("\n" + "="*80)
    print("Example 3: Collection for Specific Keywords")
    print("="*80)

    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "cves"

    collector = CVECollectorEnhanced(
        output_dir=str(output_dir),
        cache_enabled=True
    )

    # Override keywords to focus on specific vulnerabilities
    collector.KEYWORDS = ["SQL injection", "XSS", "command injection"]

    collector.TARGET_SAMPLES = 300  # 100 per keyword

    samples = collector.collect()

    # Count by vulnerability type
    type_counts = {}
    for sample in samples:
        vuln_type = sample.get('vulnerability_type', 'unknown')
        type_counts[vuln_type] = type_counts.get(vuln_type, 0) + 1

    print("\nSamples by vulnerability type:")
    for vuln_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {vuln_type}: {count}")


def example_analyze_collection():
    """Example 4: Analyze collected data."""
    print("\n" + "="*80)
    print("Example 4: Analyze Collected Data")
    print("="*80)

    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "cves"
    data_file = output_dir / "cve_data.jsonl"

    if not data_file.exists():
        print(f"No data file found at {data_file}")
        print("Run collection first!")
        return

    import json

    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"\nTotal samples in file: {len(samples)}")

    # Statistics
    with_code = sum(1 for s in samples if s.get('vulnerable_code'))
    print(f"Samples with code: {with_code} ({with_code/len(samples)*100:.1f}%)")

    # By severity
    severity_counts = {}
    for sample in samples:
        severity = sample.get('severity', 'UNKNOWN')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    print("\nBy severity:")
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN']:
        count = severity_counts.get(severity, 0)
        if count > 0:
            print(f"  {severity}: {count}")

    # By vulnerability type
    type_counts = {}
    for sample in samples:
        vuln_type = sample.get('vulnerability_type', 'unknown')
        type_counts[vuln_type] = type_counts.get(vuln_type, 0) + 1

    print("\nTop 10 vulnerability types:")
    for vuln_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {vuln_type}: {count}")

    # By year
    year_counts = {}
    for sample in samples:
        date = sample.get('published_date', '')
        if date:
            year = date[:4]
            year_counts[year] = year_counts.get(year, 0) + 1

    print("\nBy year:")
    for year in sorted(year_counts.keys()):
        count = year_counts[year]
        print(f"  {year}: {count}")

    # Code sources
    source_counts = {}
    for sample in samples:
        if sample.get('vulnerable_code'):
            source = sample.get('source', 'unknown')
            # Extract just the platform (github, nvd, etc.)
            platform = source.split(':')[0] if ':' in source else source
            source_counts[platform] = source_counts.get(platform, 0) + 1

    print("\nCode sources:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")

    # CWE statistics
    cwe_counts = {}
    for sample in samples:
        cwes = sample.get('cwes', [])
        for cwe in cwes:
            cwe_counts[cwe] = cwe_counts.get(cwe, 0) + 1

    print("\nTop 10 CWEs:")
    for cwe, count in sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cwe}: {count}")


def example_incremental_collection():
    """Example 5: Incremental collection (resume interrupted collection)."""
    print("\n" + "="*80)
    print("Example 5: Incremental Collection")
    print("="*80)

    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "cves"
    data_file = output_dir / "cve_data.jsonl"

    # Check existing samples
    existing_samples = 0
    existing_cves = set()

    if data_file.exists():
        import json
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                existing_cves.add(sample['cve_id'])
                existing_samples += 1

        print(f"Found {existing_samples} existing samples")

    # Collect more
    collector = CVECollectorEnhanced(
        output_dir=str(output_dir),
        cache_enabled=True
    )

    collector.TARGET_SAMPLES = 100

    new_samples = collector.collect()

    # Filter out duplicates
    unique_new = [s for s in new_samples if s['cve_id'] not in existing_cves]

    print(f"\nCollected {len(new_samples)} samples")
    print(f"New unique samples: {len(unique_new)}")

    # Append new samples to existing file
    if unique_new:
        import json
        with open(data_file, 'a', encoding='utf-8') as f:
            for sample in unique_new:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        print(f"Appended {len(unique_new)} new samples to {data_file}")


def main():
    """Run examples based on user choice."""
    print("\n" + "="*80)
    print("Enhanced CVE Collector - Usage Examples")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. Basic collection (default settings)")
    print("  2. Collection with GitHub token")
    print("  3. Collection for specific keywords")
    print("  4. Analyze existing collection")
    print("  5. Incremental collection (resume)")
    print("  6. Run all examples")
    print("  0. Exit")

    while True:
        try:
            choice = input("\nSelect example (0-6): ").strip()

            if choice == '0':
                print("Exiting...")
                break
            elif choice == '1':
                example_basic_collection()
            elif choice == '2':
                example_with_github_token()
            elif choice == '3':
                example_specific_keywords()
            elif choice == '4':
                example_analyze_collection()
            elif choice == '5':
                example_incremental_collection()
            elif choice == '6':
                example_basic_collection()
                example_with_github_token()
                example_specific_keywords()
                example_analyze_collection()
                example_incremental_collection()
            else:
                print("Invalid choice. Please select 0-6.")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError running example: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
