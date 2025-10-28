"""Example usage of the Synthetic Data Generator.

This script demonstrates how to use the SyntheticGenerator class
to create synthetic vulnerability samples.
"""

import json
from pathlib import Path
from synthetic_generator import SyntheticGenerator


def example_basic_generation():
    """Example 1: Basic synthetic data generation."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Generation (500 samples)")
    print("=" * 60)

    # Create generator
    generator = SyntheticGenerator(
        output_dir="data/raw/synthetic",
        seed=42
    )

    # Generate all samples
    samples = generator.generate_all(total_samples=500)

    # Save to file
    output_path = generator.save_samples(samples, "synthetic_data.jsonl")
    print(f"\nSaved {len(samples)} samples to: {output_path}")


def example_specific_type():
    """Example 2: Generate samples for specific vulnerability type."""
    print("\n" + "=" * 60)
    print("Example 2: Generate SQL Injection Samples")
    print("=" * 60)

    generator = SyntheticGenerator(output_dir="data/raw/synthetic")

    # Generate 10 pairs (20 samples) for SQL injection
    samples = generator.generate_for_type("sql_injection_concat", num_pairs=10)

    print(f"\nGenerated {len(samples)} SQL injection samples")
    print("\nExample vulnerable code:")
    vuln_sample = [s for s in samples if s["vulnerable"]][0]
    print(f"  {vuln_sample['code']}")
    print(f"\nCounterfactual (safe) code:")
    print(f"  {vuln_sample['counterfactual']}")


def example_validation():
    """Example 3: Generate and validate dataset."""
    print("\n" + "=" * 60)
    print("Example 3: Generate and Validate")
    print("=" * 60)

    generator = SyntheticGenerator(output_dir="data/raw/synthetic", seed=123)

    # Generate samples
    samples = generator.generate_all(total_samples=500)

    # Validate
    stats = generator.validate_dataset(samples)

    print("\nValidation Results:")
    print(f"  Total: {stats['total_samples']}")
    print(f"  Vulnerable: {stats['vulnerable_count']}")
    print(f"  Safe: {stats['safe_count']}")
    print(f"  Avg code length: {stats['avg_code_length']:.1f} chars")

    print("\n  By vulnerability type:")
    for vuln_type, counts in stats["by_type"].items():
        total = counts["vulnerable"] + counts["safe"]
        print(f"    {vuln_type}: {total} samples")

    print("\n  By language:")
    for lang, count in sorted(stats["by_language"].items()):
        print(f"    {lang}: {count} samples")


def example_custom_pairs():
    """Example 4: Create custom counterfactual pairs."""
    print("\n" + "=" * 60)
    print("Example 4: Create Custom Counterfactual Pairs")
    print("=" * 60)

    generator = SyntheticGenerator(output_dir="data/raw/synthetic")

    # Create specific pairs for different vulnerability types
    vuln_types = [
        "sql_injection_concat",
        "xss_output",
        "command_injection",
        "path_traversal",
        "ssrf"
    ]

    print("\nCreating sample pairs for each vulnerability type:\n")

    for vuln_type in vuln_types:
        vuln_sample, safe_sample = generator.create_vulnerable_safe_pair(
            vuln_type=vuln_type,
            vuln_idx=0,
            safe_idx=0
        )

        print(f"{vuln_type}:")
        print(f"  Vulnerable: {vuln_sample['code']}")
        print(f"  Safe: {safe_sample['code']}")
        print(f"  Language: {vuln_sample['language']}")
        print()


def example_load_and_analyze():
    """Example 5: Load generated data and analyze."""
    print("\n" + "=" * 60)
    print("Example 5: Load and Analyze Generated Data")
    print("=" * 60)

    # Path to generated data
    data_file = Path("data/raw/synthetic/synthetic_data.jsonl")

    if not data_file.exists():
        print(f"\nData file not found: {data_file}")
        print("Run example_basic_generation() first.")
        return

    # Load samples
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"\nLoaded {len(samples)} samples from {data_file}")

    # Analyze
    vuln_count = sum(1 for s in samples if s["vulnerable"])
    safe_count = len(samples) - vuln_count

    print(f"\nBreakdown:")
    print(f"  Vulnerable: {vuln_count} ({vuln_count/len(samples)*100:.1f}%)")
    print(f"  Safe: {safe_count} ({safe_count/len(samples)*100:.1f}%)")

    # Show some examples
    print("\nExample samples:")
    for i, sample in enumerate(samples[:3]):
        status = "VULNERABLE" if sample["vulnerable"] else "SAFE"
        print(f"\n  Sample {i+1} [{status}] ({sample['vulnerability_type']}):")
        print(f"    Code: {sample['code'][:80]}...")
        print(f"    Language: {sample['language']}")


def example_all_vulnerability_types():
    """Example 6: Generate samples for all vulnerability types."""
    print("\n" + "=" * 60)
    print("Example 6: All Vulnerability Types")
    print("=" * 60)

    generator = SyntheticGenerator(output_dir="data/raw/synthetic")

    # Get all vulnerability types
    vuln_types = list(generator.vulnerability_templates.keys())

    print(f"\nSupported vulnerability types ({len(vuln_types)}):")
    for vuln_type in vuln_types:
        templates = generator.vulnerability_templates[vuln_type]
        num_vuln = len(templates["vulnerable"])
        num_safe = len(templates["safe"])
        print(f"  - {vuln_type}: {num_vuln} vulnerable templates, {num_safe} safe templates")

    # Generate small dataset with all types
    print("\nGenerating 50 samples (10 samples per type)...")
    all_samples = []

    for vuln_type in vuln_types:
        samples = generator.generate_for_type(vuln_type, num_pairs=5)  # 5 pairs = 10 samples
        all_samples.extend(samples)

    print(f"Generated {len(all_samples)} samples total")

    # Show distribution
    type_dist = {}
    for sample in all_samples:
        vtype = sample["vulnerability_type"]
        type_dist[vtype] = type_dist.get(vtype, 0) + 1

    print("\nDistribution by type:")
    for vtype, count in sorted(type_dist.items()):
        print(f"  {vtype}: {count}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" " * 15 + "SYNTHETIC DATA GENERATOR EXAMPLES")
    print("=" * 70)

    examples = [
        ("Basic Generation", example_basic_generation),
        ("Specific Type", example_specific_type),
        ("Validation", example_validation),
        ("Custom Pairs", example_custom_pairs),
        ("All Vulnerability Types", example_all_vulnerability_types),
        ("Load and Analyze", example_load_and_analyze),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
