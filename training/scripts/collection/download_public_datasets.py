"""
Download Public Vulnerability Datasets

This script downloads and processes public datasets for vulnerability detection:
1. Devign - Graph-based vulnerability detection dataset
2. CodeXGLUE - Microsoft's code understanding benchmark
3. Juliet Test Suite - NIST/SARD synthetic vulnerabilities
4. SARD - Software Assurance Reference Dataset

Usage:
    python download_public_datasets.py --all
    python download_public_datasets.py --datasets devign codexglue
    python download_public_datasets.py --datasets juliet --output-dir data/public
"""

import os
import sys
import json
import requests
import zipfile
import tarfile
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import csv
from tqdm import tqdm


class PublicDatasetDownloader:
    """Download and process public vulnerability datasets."""

    def __init__(self, output_dir: str = "data/public"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dataset configurations
        self.datasets = {
            "devign": {
                "name": "Devign",
                "description": "Graph-based vulnerability detection dataset (C projects)",
                "url": "https://raw.githubusercontent.com/saikat107/Devign/master/Devign.json",
                "format": "json",
                "samples": 27000,
                "license": "MIT"
            },
            "codexglue": {
                "name": "CodeXGLUE Defect Detection",
                "description": "Microsoft's defect detection benchmark (C projects)",
                "urls": {
                    "train": "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/Defect-detection/dataset/train.jsonl",
                    "valid": "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/Defect-detection/dataset/valid.jsonl",
                    "test": "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/Defect-detection/dataset/test.jsonl"
                },
                "format": "jsonl",
                "samples": 21854,
                "license": "MIT"
            },
            "juliet": {
                "name": "Juliet Test Suite v1.3",
                "description": "NIST synthetic vulnerability test cases (C/C++/Java)",
                "info_url": "https://samate.nist.gov/SARD/test-suites/112",
                "download_url": "https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip",
                "format": "source_files",
                "samples": 64000,
                "license": "Public Domain",
                "note": "Large download (~300MB), requires manual processing"
            },
            "sard": {
                "name": "SARD (Software Assurance Reference Dataset)",
                "description": "NIST curated vulnerability dataset",
                "info_url": "https://samate.nist.gov/SARD/",
                "api_url": "https://samate.nist.gov/SARD/api/test-cases",
                "format": "xml/json",
                "samples": 176000,
                "license": "Public Domain",
                "note": "Requires API access or web scraping"
            }
        }

    def download_file(self, url: str, output_path: Path, desc: str = "Downloading") -> bool:
        """Download a file with progress bar."""
        try:
            print(f"\nDownloading from: {url}")
            print(f"Saving to: {output_path}")

            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

            print(f"[+] Downloaded: {output_path}")
            return True

        except Exception as e:
            print(f"[!] Error downloading {url}: {e}")
            return False

    def download_devign(self) -> Dict:
        """Download Devign dataset."""
        print("\n" + "="*70)
        print("Downloading Devign Dataset")
        print("="*70)

        dataset_dir = self.output_dir / "devign"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download main dataset
        raw_file = dataset_dir / "Devign.json"
        processed_file = dataset_dir / "devign_processed.jsonl"

        if not raw_file.exists():
            success = self.download_file(
                self.datasets["devign"]["url"],
                raw_file,
                "Devign dataset"
            )
            if not success:
                return {"status": "failed", "samples": 0}
        else:
            print(f"[+] Using cached file: {raw_file}")

        # Process to JSONL format
        print("\nProcessing Devign dataset...")
        samples_processed = self._process_devign(raw_file, processed_file)

        return {
            "status": "success",
            "samples": samples_processed,
            "output_file": str(processed_file),
            "format": "jsonl"
        }

    def _process_devign(self, input_file: Path, output_file: Path) -> int:
        """Process Devign JSON to standardized JSONL format."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            samples = []
            vulnerable_count = 0
            safe_count = 0

            for item in data:
                # Extract relevant fields
                sample = {
                    "vulnerability_id": f"DEVIGN-{item.get('id', 'unknown')}",
                    "description": f"Function from project: {item.get('project', 'unknown')}",
                    "vulnerable_code": item.get('func', ''),
                    "fixed_code": "",  # Devign doesn't provide fixes
                    "ecosystem": "C/C++",
                    "severity": "HIGH" if item.get('target') == 1 else "SAFE",
                    "source": "devign",
                    "metadata": {
                        "project": item.get('project', ''),
                        "commit_id": item.get('commit_id', ''),
                        "target": item.get('target', 0),  # 0 = safe, 1 = vulnerable
                        "file_name": item.get('file_name', '')
                    }
                }

                if item.get('target') == 1:
                    vulnerable_count += 1
                else:
                    safe_count += 1

                samples.append(sample)

            # Write to JSONL
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            print(f"\n[+] Processed {len(samples)} samples:")
            print(f"    Vulnerable: {vulnerable_count}")
            print(f"    Safe: {safe_count}")
            print(f"    Output: {output_file}")

            return len(samples)

        except Exception as e:
            print(f"[!] Error processing Devign: {e}")
            return 0

    def download_codexglue(self) -> Dict:
        """Download CodeXGLUE Defect Detection dataset."""
        print("\n" + "="*70)
        print("Downloading CodeXGLUE Defect Detection Dataset")
        print("="*70)

        dataset_dir = self.output_dir / "codexglue"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        urls = self.datasets["codexglue"]["urls"]
        files_downloaded = []
        total_samples = 0

        # Download train, valid, test splits
        for split, url in urls.items():
            output_file = dataset_dir / f"{split}.jsonl"

            if not output_file.exists():
                success = self.download_file(url, output_file, f"CodeXGLUE {split}")
                if not success:
                    continue
            else:
                print(f"[+] Using cached file: {output_file}")

            files_downloaded.append(output_file)

        # Process and merge all splits
        processed_file = dataset_dir / "codexglue_processed.jsonl"
        total_samples = self._process_codexglue(files_downloaded, processed_file)

        return {
            "status": "success",
            "samples": total_samples,
            "output_file": str(processed_file),
            "format": "jsonl",
            "splits": {
                "train": str(dataset_dir / "train.jsonl"),
                "valid": str(dataset_dir / "valid.jsonl"),
                "test": str(dataset_dir / "test.jsonl")
            }
        }

    def _process_codexglue(self, input_files: List[Path], output_file: Path) -> int:
        """Process CodeXGLUE JSONL to standardized format."""
        try:
            all_samples = []
            vulnerable_count = 0
            safe_count = 0

            for input_file in input_files:
                print(f"\nProcessing: {input_file.name}")

                with open(input_file, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        try:
                            item = json.loads(line.strip())

                            # CodeXGLUE format: {"func": "...", "target": 0/1, "idx": ...}
                            sample = {
                                "vulnerability_id": f"CODEXGLUE-{item.get('idx', idx)}",
                                "description": "Defect detection sample from CodeXGLUE benchmark",
                                "vulnerable_code": item.get('func', ''),
                                "fixed_code": "",  # CodeXGLUE doesn't provide fixes
                                "ecosystem": "C",
                                "severity": "HIGH" if item.get('target') == 1 else "SAFE",
                                "source": "codexglue",
                                "metadata": {
                                    "target": item.get('target', 0),
                                    "split": input_file.stem,
                                    "idx": item.get('idx', idx)
                                }
                            }

                            if item.get('target') == 1:
                                vulnerable_count += 1
                            else:
                                safe_count += 1

                            all_samples.append(sample)

                        except json.JSONDecodeError:
                            continue

            # Write merged dataset
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in all_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            print(f"\n[+] Processed {len(all_samples)} total samples:")
            print(f"    Vulnerable: {vulnerable_count}")
            print(f"    Safe: {safe_count}")
            print(f"    Output: {output_file}")

            return len(all_samples)

        except Exception as e:
            print(f"[!] Error processing CodeXGLUE: {e}")
            return 0

    def download_juliet_info(self) -> Dict:
        """Download Juliet Test Suite information and instructions."""
        print("\n" + "="*70)
        print("Juliet Test Suite Information")
        print("="*70)

        dataset_dir = self.output_dir / "juliet"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        info_file = dataset_dir / "README.md"

        info_content = """# Juliet Test Suite v1.3

## Overview
The Juliet Test Suite is a comprehensive set of test cases for static analysis tools.
It contains synthetic C/C++ and Java code with known vulnerabilities.

## Download Instructions

**Size:** ~300MB (compressed), ~4GB (extracted)
**Samples:** 64,000+ test cases
**Languages:** C, C++, Java

### Manual Download Required

Due to the large size, this dataset requires manual download:

1. **Visit NIST SARD:**
   https://samate.nist.gov/SARD/test-suites/112

2. **Download the ZIP file:**
   https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip

3. **Extract to this directory:**
   ```bash
   # Place the downloaded ZIP in this directory, then:
   unzip 2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip
   ```

4. **Process the test cases:**
   ```bash
   python process_juliet.py --input ./C/testcases --output juliet_processed.jsonl
   ```

## CWE Coverage

The Juliet suite covers 118 CWE categories including:
- CWE-119: Buffer Overflow
- CWE-120: Buffer Copy without Checking Size
- CWE-78: OS Command Injection
- CWE-89: SQL Injection
- CWE-79: Cross-site Scripting
- And many more...

## File Structure

```
testcases/
├── CWE119_Buffer_Overflow/
├── CWE120_Buffer_Copy_without_Checking_Size_of_Input/
├── CWE78_OS_Command_Injection/
└── ...
```

Each CWE directory contains:
- Good examples (no vulnerability)
- Bad examples (with vulnerability)

## Processing Script

A helper script `process_juliet.py` will be created to:
1. Scan all CWE directories
2. Extract good/bad code pairs
3. Convert to standardized JSONL format
4. Label with CWE IDs and vulnerability types

## License

Public Domain - NIST Software

## More Information

- SARD Home: https://samate.nist.gov/SARD/
- Juliet Documentation: https://samate.nist.gov/SARD/test-suites
"""

        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(info_content)

        print(f"\n[+] Created information file: {info_file}")
        print(f"\n[!] Juliet Test Suite requires manual download due to size (~300MB)")
        print(f"[!] See {info_file} for download instructions")

        return {
            "status": "info_only",
            "samples": 64000,
            "info_file": str(info_file),
            "download_url": self.datasets["juliet"]["download_url"],
            "note": "Manual download required - see README.md"
        }

    def download_sard_info(self) -> Dict:
        """Download SARD information and create access script."""
        print("\n" + "="*70)
        print("SARD (Software Assurance Reference Dataset) Information")
        print("="*70)

        dataset_dir = self.output_dir / "sard"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        info_file = dataset_dir / "README.md"

        info_content = """# SARD - Software Assurance Reference Dataset

## Overview
SARD is NIST's comprehensive database of software security flaws.
It contains over 176,000 test cases across multiple languages and vulnerability types.

## Access Options

### Option 1: Web Interface (Recommended for browsing)
Visit: https://samate.nist.gov/SARD/

### Option 2: API Access (For automated download)
```python
import requests

# Search for test cases
url = "https://samate.nist.gov/SARD/api/test-cases"
params = {
    "language": "C",
    "cwe": "120",
    "limit": 100
}

response = requests.get(url, params=params)
test_cases = response.json()
```

### Option 3: Bulk Download (Selected subsets)
SARD provides curated subsets for download:

1. **C Test Suite 2008**:
   https://samate.nist.gov/SARD/testsuite.php

2. **Java Test Suite**:
   https://samate.nist.gov/SARD/downloads/java-test-suite.zip

3. **Specific CWE Collections**:
   Navigate to SARD → Browse → Filter by CWE

## Dataset Statistics

- **Total Test Cases:** 176,000+
- **Languages:** C, C++, Java, PHP, C#, Python, and more
- **CWE Coverage:** 150+ weakness types
- **Contributors:** Multiple organizations worldwide

## Recommended Approach for StreamGuard

Given the large size, we recommend:

1. **Start with Juliet** (included in SARD, well-structured)
2. **Add specific CWE categories** relevant to your use case
3. **Use API** to fetch additional samples as needed

## Example: Fetch 1000 samples via script

```bash
python fetch_sard_samples.py --cwe 120,119,78,89 --limit 1000 --output sard_samples.jsonl
```

## Processing

SARD test cases require parsing:
- Extract source files
- Identify vulnerable vs. safe code
- Extract CWE labels and metadata
- Convert to standardized format

## License

Public Domain - NIST Software

## More Information

- SARD Home: https://samate.nist.gov/SARD/
- API Documentation: https://samate.nist.gov/SARD/api
- CWE Database: https://cwe.mitre.org/
"""

        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(info_content)

        print(f"\n[+] Created information file: {info_file}")
        print(f"\n[!] SARD is a large dataset (176,000+ samples) requiring selective download")
        print(f"[!] See {info_file} for access methods")
        print(f"[!] Recommendation: Use Juliet (subset of SARD) instead for initial training")

        return {
            "status": "info_only",
            "samples": 176000,
            "info_file": str(info_file),
            "api_url": self.datasets["sard"]["api_url"],
            "note": "Large dataset - use API or selective download"
        }

    def generate_summary_report(self, results: Dict) -> None:
        """Generate summary report of downloaded datasets."""
        report_file = self.output_dir / "DOWNLOAD_SUMMARY.md"

        total_samples = sum(r.get("samples", 0) for r in results.values() if r.get("status") == "success")

        report = f"""# Public Datasets Download Summary

**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Samples Downloaded:** {total_samples:,}

---

## Downloaded Datasets

"""

        for dataset_name, result in results.items():
            status_icon = "✅" if result.get("status") == "success" else "ℹ️" if result.get("status") == "info_only" else "❌"

            report += f"""### {status_icon} {self.datasets[dataset_name]["name"]}

**Status:** {result.get("status", "unknown").upper()}
**Samples:** {result.get("samples", 0):,}
**Description:** {self.datasets[dataset_name]["description"]}
"""

            if result.get("output_file"):
                report += f"**Output File:** `{result['output_file']}`\n"

            if result.get("info_file"):
                report += f"**Info File:** `{result['info_file']}`\n"

            if result.get("note"):
                report += f"**Note:** {result['note']}\n"

            report += "\n---\n\n"

        report += """## Next Steps

### 1. Verify Downloaded Data

```bash
# Count samples in each dataset
wc -l data/public/devign/devign_processed.jsonl
wc -l data/public/codexglue/codexglue_processed.jsonl
```

### 2. Combine with Collector Data

```bash
# Run collectors to gather 8000-9000 additional samples
python run_full_collection.py --collectors synthetic osv exploitdb \\
  --synthetic-samples 3000 --osv-samples 4000 --exploitdb-samples 2000 \\
  --sequential --no-dashboard
```

### 3. Merge All Datasets

```bash
python merge_datasets.py \\
  --public data/public/devign/devign_processed.jsonl \\
  --public data/public/codexglue/codexglue_processed.jsonl \\
  --collectors data/raw/synthetic/synthetic_data.jsonl \\
  --collectors data/raw/osv/osv_vulnerabilities.jsonl \\
  --collectors data/raw/exploitdb/exploitdb_exploits.jsonl \\
  --output data/training/merged_dataset.jsonl
```

### 4. Start Training

```bash
cd training
python train_model.py --dataset ../data/training/merged_dataset.jsonl
```

---

## Dataset Licenses

- **Devign:** MIT License
- **CodeXGLUE:** MIT License
- **Juliet:** Public Domain (NIST)
- **SARD:** Public Domain (NIST)

All datasets are approved for research and commercial use.
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n{'='*70}")
        print(f"[+] Summary report saved: {report_file}")
        print(f"[+] Total samples downloaded: {total_samples:,}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Download public vulnerability detection datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['devign', 'codexglue', 'juliet', 'sard', 'all'],
        default=['all'],
        help='Datasets to download (default: all)'
    )

    parser.add_argument(
        '--output-dir',
        default='data/public',
        help='Output directory for datasets (default: data/public)'
    )

    args = parser.parse_args()

    # Expand 'all' to all datasets
    if 'all' in args.datasets:
        datasets_to_download = ['devign', 'codexglue', 'juliet', 'sard']
    else:
        datasets_to_download = args.datasets

    # Initialize downloader
    downloader = PublicDatasetDownloader(output_dir=args.output_dir)

    print("\n" + "="*70)
    print("StreamGuard - Public Dataset Downloader")
    print("="*70)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Datasets to download: {', '.join(datasets_to_download)}\n")

    results = {}

    # Download each dataset
    for dataset in datasets_to_download:
        if dataset == 'devign':
            results['devign'] = downloader.download_devign()
        elif dataset == 'codexglue':
            results['codexglue'] = downloader.download_codexglue()
        elif dataset == 'juliet':
            results['juliet'] = downloader.download_juliet_info()
        elif dataset == 'sard':
            results['sard'] = downloader.download_sard_info()

    # Generate summary report
    downloader.generate_summary_report(results)

    print("\n" + "="*70)
    print("[+] Download process complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review downloaded datasets in:", args.output_dir)
    print("2. For Juliet/SARD: Follow instructions in README.md files")
    print("3. Run collectors to gather additional 8000-9000 samples")
    print("4. Merge all datasets for training")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
