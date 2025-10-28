"""
Download CodeXGLUE Defect Detection Dataset

Downloads the official CodeXGLUE defect detection dataset (based on Devign)
from Hugging Face Hub - the most reliable source.

Dataset Details:
- Total Samples: 27,318
- Split: 80/10/10 (train/valid/test)
- Label Distribution: 50/50 (vulnerable/safe)
- Language: C/C++
- Projects: FFmpeg, Qemu, Linux Kernel, Wireshark, etc.

Usage:
    python download_codexglue.py --output data/codexglue
    python download_codexglue.py --output data/codexglue --source github  # fallback
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import requests
from tqdm import tqdm


class CodeXGLUEDownloader:
    """Download CodeXGLUE Defect Detection dataset."""

    # Hugging Face Hub URLs (most reliable)
    HF_BASE = "https://huggingface.co/datasets/code_x_glue_cc_defect_detection"
    HF_URLS = {
        "train": f"{HF_BASE}/resolve/main/train.jsonl",
        "valid": f"{HF_BASE}/resolve/main/valid.jsonl",
        "test": f"{HF_BASE}/resolve/main/test.jsonl"
    }

    # GitHub raw URLs (fallback)
    GH_BASE = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/Defect-detection/dataset"
    GH_URLS = {
        "train": f"{GH_BASE}/train.jsonl",
        "valid": f"{GH_BASE}/valid.jsonl",
        "test": f"{GH_BASE}/test.jsonl"
    }

    def __init__(self, output_dir: str = "data/codexglue", source: str = "huggingface"):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Select URLs based on source
        if source == "huggingface":
            self.urls = self.HF_URLS
            self.source_name = "Hugging Face Hub"
        elif source == "github":
            self.urls = self.GH_URLS
            self.source_name = "GitHub"
        else:
            raise ValueError(f"Invalid source: {source}. Choose 'huggingface' or 'github'")

        self.stats = {
            "train": {"samples": 0, "vulnerable": 0, "safe": 0},
            "valid": {"samples": 0, "vulnerable": 0, "safe": 0},
            "test": {"samples": 0, "vulnerable": 0, "safe": 0}
        }

    def download_file(self, url: str, output_path: Path) -> bool:
        """Download a file with progress bar."""
        try:
            print(f"\nDownloading: {url}")
            print(f"Saving to: {output_path}")

            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                    print("[+] Downloaded (size unknown)")
                else:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Progress") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

            print(f"[+] Download complete: {output_path}")
            return True

        except Exception as e:
            print(f"[!] Error downloading {url}: {e}")
            return False

    def verify_and_count(self, file_path: Path, split_name: str) -> bool:
        """Verify downloaded file and count samples."""
        try:
            print(f"\nVerifying {split_name} split...")

            vulnerable_count = 0
            safe_count = 0
            total_count = 0

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())

                        # Validate required fields
                        if 'func' not in sample or 'target' not in sample:
                            print(f"[!] Warning: Missing required fields in {split_name}")
                            continue

                        total_count += 1

                        # Count by label
                        if sample['target'] == 1:
                            vulnerable_count += 1
                        else:
                            safe_count += 1

                    except json.JSONDecodeError:
                        print(f"[!] Warning: Invalid JSON line in {split_name}")
                        continue

            # Update stats
            self.stats[split_name]["samples"] = total_count
            self.stats[split_name]["vulnerable"] = vulnerable_count
            self.stats[split_name]["safe"] = safe_count

            print(f"[+] {split_name.capitalize()} split verified:")
            print(f"    Total: {total_count:,}")
            print(f"    Vulnerable: {vulnerable_count:,} ({100*vulnerable_count/total_count:.1f}%)")
            print(f"    Safe: {safe_count:,} ({100*safe_count/total_count:.1f}%)")

            return True

        except Exception as e:
            print(f"[!] Error verifying {split_name}: {e}")
            return False

    def download_all(self) -> bool:
        """Download all splits."""
        print("="*70)
        print("CodeXGLUE Defect Detection Dataset Downloader")
        print("="*70)
        print(f"\nSource: {self.source_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Raw files: {self.raw_dir}\n")

        success = True

        for split_name, url in self.urls.items():
            output_file = self.raw_dir / f"{split_name}.jsonl"

            # Download
            if not output_file.exists():
                if not self.download_file(url, output_file):
                    success = False
                    continue
            else:
                print(f"\n[+] Using cached file: {output_file}")

            # Verify
            if not self.verify_and_count(output_file, split_name):
                success = False

        return success

    def generate_summary(self) -> None:
        """Generate summary report."""
        total_samples = sum(s["samples"] for s in self.stats.values())
        total_vulnerable = sum(s["vulnerable"] for s in self.stats.values())
        total_safe = sum(s["safe"] for s in self.stats.values())

        summary_file = self.output_dir / "DOWNLOAD_SUMMARY.md"

        summary = f"""# CodeXGLUE Defect Detection Dataset

**Download Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Source:** {self.source_name}
**Total Samples:** {total_samples:,}

---

## Dataset Statistics

| Split | Samples | Vulnerable | Safe | Balance |
|-------|---------|------------|------|---------|
| **Train** | {self.stats['train']['samples']:,} | {self.stats['train']['vulnerable']:,} | {self.stats['train']['safe']:,} | {100*self.stats['train']['vulnerable']/max(1,self.stats['train']['samples']):.1f}% |
| **Valid** | {self.stats['valid']['samples']:,} | {self.stats['valid']['vulnerable']:,} | {self.stats['valid']['safe']:,} | {100*self.stats['valid']['vulnerable']/max(1,self.stats['valid']['samples']):.1f}% |
| **Test** | {self.stats['test']['samples']:,} | {self.stats['test']['vulnerable']:,} | {self.stats['test']['safe']:,} | {100*self.stats['test']['vulnerable']/max(1,self.stats['test']['samples']):.1f}% |
| **TOTAL** | **{total_samples:,}** | **{total_vulnerable:,}** | **{total_safe:,}** | **{100*total_vulnerable/max(1,total_samples):.1f}%** |

---

## File Locations

```
{self.output_dir}/
├── raw/
│   ├── train.jsonl ({self.stats['train']['samples']:,} samples)
│   ├── valid.jsonl ({self.stats['valid']['samples']:,} samples)
│   └── test.jsonl ({self.stats['test']['samples']:,} samples)
└── DOWNLOAD_SUMMARY.md (this file)
```

---

## Sample Format

Each line contains a JSON object:

```json
{{
  "func": "void foo() {{ ... }}",  // C/C++ function code
  "target": 1,                      // 0 = safe, 1 = vulnerable
  "project": "FFmpeg",              // Source project
  "commit_id": "abc123..."          // Git commit hash
}}
```

---

## Next Steps

### 1. Preprocess for StreamGuard

```bash
python training/scripts/data/preprocess_codexglue.py \\
  --input {self.raw_dir} \\
  --output {self.output_dir}/processed \\
  --format streamguard_explainable
```

### 2. Verify Quality

```bash
python training/scripts/data/verify_dataset.py \\
  --dataset {self.output_dir}/processed
```

### 3. Upload to S3 for SageMaker

```bash
aws s3 cp {self.output_dir}/processed/ \\
  s3://streamguard-training/phase1/data/ --recursive
```

### 4. Train on AWS SageMaker

```bash
python training/sagemaker/launch_training.py \\
  --phase phase1 \\
  --data-path s3://streamguard-training/phase1/data/ \\
  --model-type multi_agent_explainable
```

---

## References

- **Paper:** CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
- **Authors:** Lu, Shuai, et al.
- **ArXiv:** https://arxiv.org/abs/2102.04664
- **Dataset:** https://huggingface.co/datasets/code_x_glue_cc_defect_detection
- **Base Dataset:** Devign (Zhou et al., NeurIPS 2019)

---

## License

**MIT License** - Free for research and commercial use

---

**Last Updated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status:** ✅ Download Complete
"""

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        print("\n" + "="*70)
        print(f"[+] Summary saved: {summary_file}")
        print(f"[+] Total samples downloaded: {total_samples:,}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download CodeXGLUE Defect Detection dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--output',
        default='data/codexglue',
        help='Output directory (default: data/codexglue)'
    )

    parser.add_argument(
        '--source',
        choices=['huggingface', 'github'],
        default='huggingface',
        help='Download source (default: huggingface)'
    )

    args = parser.parse_args()

    # Download
    downloader = CodeXGLUEDownloader(output_dir=args.output, source=args.source)

    success = downloader.download_all()

    if not success:
        print("\n[!] Some downloads failed. Check errors above.")
        if args.source == 'huggingface':
            print("[!] Try fallback source: --source github")
        sys.exit(1)

    # Generate summary
    downloader.generate_summary()

    print("\n[+] Download complete!")
    print("\nNext steps:")
    print(f"1. Review: {downloader.output_dir}/DOWNLOAD_SUMMARY.md")
    print(f"2. Preprocess: python training/scripts/data/preprocess_codexglue.py --input {downloader.raw_dir}")
    print("3. Train: python training/sagemaker/launch_training.py --phase phase1\n")


if __name__ == "__main__":
    main()
