"""Display example outputs from synthetic data generator."""

import json
from pathlib import Path


def show_examples():
    """Display sample outputs from generated data."""

    data_file = Path("data/raw/synthetic/synthetic_data.jsonl")

    if not data_file.exists():
        print("Data file not found. Run synthetic_generator.py first.")
        return

    # Load samples
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    # Group by vulnerability type
    by_type = {}
    for sample in samples:
        vtype = sample['vulnerability_type']
        if vtype not in by_type:
            by_type[vtype] = {'vulnerable': [], 'safe': []}

        if sample['vulnerable']:
            by_type[vtype]['vulnerable'].append(sample)
        else:
            by_type[vtype]['safe'].append(sample)

    # Print examples for each type
    print('=' * 80)
    print(' ' * 20 + 'SYNTHETIC DATA GENERATOR - SAMPLE OUTPUT')
    print('=' * 80)
    print()

    for vtype in sorted(by_type.keys()):
        print(f"### {vtype.upper().replace('_', ' ')}")
        print()

        # Get one example of each
        vuln = by_type[vtype]['vulnerable'][0]
        safe = by_type[vtype]['safe'][0]

        print(f"Vulnerable Example ({vuln['language']}):")
        print(f"  {vuln['code']}")
        print()

        print(f"Safe Counterfactual ({vuln['language']}):")
        print(f"  {vuln['counterfactual']}")
        print()

        print(f"Another Safe Example ({safe['language']}):")
        print(f"  {safe['code']}")
        print()

        print(f"Its Vulnerable Counterfactual ({safe['language']}):")
        print(f"  {safe['counterfactual']}")
        print()

        print('-' * 80)
        print()

    # Print statistics
    print('=' * 80)
    print(' ' * 30 + 'DATASET STATISTICS')
    print('=' * 80)
    print()

    print(f"Total Samples: {len(samples)}")
    print(f"Vulnerable: {sum(1 for s in samples if s['vulnerable'])}")
    print(f"Safe: {sum(1 for s in samples if not s['vulnerable'])}")
    print()

    print("By Vulnerability Type:")
    for vtype, data in sorted(by_type.items()):
        total = len(data['vulnerable']) + len(data['safe'])
        print(f"  {vtype}: {total} samples ({len(data['vulnerable'])} vuln, {len(data['safe'])} safe)")
    print()

    print("By Language:")
    lang_counts = {}
    for sample in samples:
        lang = sample['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count} samples")
    print()


if __name__ == "__main__":
    show_examples()
