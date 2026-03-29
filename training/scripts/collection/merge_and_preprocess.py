#!/usr/bin/env python3
"""
Comprehensive Merge and Preprocessing Script for StreamGuard

This script:
1. Merges data from all 6 collection sources
2. Deduplicates samples
3. Validates data quality
4. Prepares for preprocessing pipeline
5. Generates comprehensive statistics

Usage:
    python merge_and_preprocess.py --input-dir ../../data/raw --output ../../data/raw/merged
"""

import json
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter, defaultdict
from datetime import datetime


class ComprehensiveMerger:
    """Merge and validate data from all collection sources."""
    
    SOURCES = {
        'cves': {'file_pattern': 'cve*.jsonl', 'expected_min': 10000},
        'github': {'file_pattern': 'github*.jsonl', 'expected_min': 7000},
        'opensource': {'file_pattern': 'mined*.jsonl', 'expected_min': 14000},
        'osv': {'file_pattern': 'osv*.jsonl', 'expected_min': 14000},
        'exploitdb': {'file_pattern': 'exploitdb*.jsonl', 'expected_min': 7000},
        'synthetic': {'file_pattern': 'synthetic*.jsonl', 'expected_min': 3500}
    }
    
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize merger."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples = []
        self.code_hashes = set()
        self.duplicates_removed = 0
        self.stats = defaultdict(lambda: defaultdict(int))
    
    def load_source(self, source_name: str) -> int:
        """Load samples from a specific source."""
        source_dir = self.input_dir / source_name
        if not source_dir.exists():
            print(f"  ⚠ Source directory not found: {source_dir}")
            return 0
        
        file_pattern = self.SOURCES[source_name]['file_pattern']
        jsonl_files = list(source_dir.glob(file_pattern))
        
        if not jsonl_files:
            print(f"  ⚠ No files matching {file_pattern} in {source_dir}")
            return 0
        
        count = 0
        for jsonl_file in jsonl_files:
            count += self._load_jsonl_file(jsonl_file, source_name)
        
        return count
    
    def _load_jsonl_file(self, file_path: Path, source_name: str) -> int:
        """Load samples from a JSONL file."""
        count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line)
                        
                        # Add source metadata
                        if 'metadata' not in sample:
                            sample['metadata'] = {}
                        sample['metadata']['collection_source'] = source_name
                        sample['metadata']['original_file'] = file_path.name
                        
                        # Normalize sample format
                        normalized = self._normalize_sample(sample, source_name)
                        
                        if normalized:
                            # Check for duplicates
                            code_hash = self._get_code_hash(normalized)
                            if code_hash not in self.code_hashes:
                                self.code_hashes.add(code_hash)
                                self.samples.append(normalized)
                                count += 1
                                
                                # Update stats
                                self.stats[source_name]['loaded'] += 1
                                lang = normalized.get('language', 'unknown')
                                self.stats[source_name][f'lang_{lang}'] += 1
                            else:
                                self.duplicates_removed += 1
                                self.stats[source_name]['duplicates'] += 1
                    
                    except json.JSONDecodeError as e:
                        self.stats[source_name]['json_errors'] += 1
                        if line_num <= 5:  # Only print first few errors
                            print(f"    JSON error at line {line_num}: {e}")
                    except Exception as e:
                        self.stats[source_name]['other_errors'] += 1
                        if line_num <= 5:
                            print(f"    Error at line {line_num}: {e}")
        
        except Exception as e:
            print(f"    Error reading file: {e}")
            return 0
        
        return count
    
    def _normalize_sample(self, sample: Dict, source: str) -> Dict:
        """Normalize sample to unified format."""
        normalized = {}
        
        # Extract code (handle different field names)
        code = None
        code_fields = ['code', 'vulnerable_code', 'exploit_code', 'func']
        for field in code_fields:
            if field in sample and sample[field]:
                code = sample[field]
                break
        
        if not code or len(code) < 50:
            return None
        
        normalized['code'] = code
        
        # Extract language
        language = sample.get('language', 'unknown')
        if language == 'unknown':
            # Try to infer from code or metadata
            if 'def ' in code or 'import ' in code:
                language = 'python'
            elif 'function ' in code or 'const ' in code or 'let ' in code:
                language = 'javascript'
        
        normalized['language'] = language
        
        # Extract vulnerability info
        if 'vulnerable' in sample:
            normalized['vulnerable'] = sample['vulnerable']
        elif 'label' in sample:
            normalized['vulnerable'] = sample['label'] == 1
        else:
            # Assume vulnerable if from exploit/vuln sources
            normalized['vulnerable'] = source in ['cves', 'osv', 'exploitdb']
        
        # Extract vulnerability type
        vuln_type = sample.get('vulnerability_type') or sample.get('vuln_type') or 'unknown'
        normalized['vulnerability_type'] = vuln_type
        
        # Extract fixed code if available
        if 'fixed_code' in sample and sample['fixed_code']:
            normalized['fixed_code'] = sample['fixed_code']
        
        # Preserve metadata
        normalized['metadata'] = sample.get('metadata', {})
        
        # Add additional fields if present
        for field in ['severity', 'cve_id', 'commit_sha', 'repository']:
            if field in sample:
                normalized[field] = sample[field]
        
        return normalized
    
    def _get_code_hash(self, sample: Dict) -> str:
        """Generate hash for code deduplication."""
        code = sample.get('code', '')
        # Normalize whitespace for better deduplication
        normalized_code = ' '.join(code.split())
        return hashlib.md5(normalized_code.encode()).hexdigest()
    
    def merge_all_sources(self) -> bool:
        """Merge data from all sources."""
        print("\n" + "="*70)
        print("StreamGuard Data Merge and Preprocessing")
        print("="*70)
        
        print("\nLoading data from all sources...")
        
        total_loaded = 0
        for source_name in self.SOURCES.keys():
            print(f"\n[{source_name.upper()}]")
            count = self.load_source(source_name)
            total_loaded += count
            
            expected_min = self.SOURCES[source_name]['expected_min']
            if count < expected_min:
                print(f"  ⚠ Below expected minimum ({count:,} < {expected_min:,})")
            else:
                print(f"  ✓ Loaded {count:,} samples")
        
        print(f"\nTotal samples loaded: {total_loaded:,}")
        print(f"Duplicates removed: {self.duplicates_removed:,}")
        print(f"Unique samples: {len(self.samples):,}")
        
        return len(self.samples) > 0
    
    def analyze_dataset(self) -> Dict:
        """Analyze merged dataset."""
        print("\n" + "="*70)
        print("Dataset Analysis")
        print("="*70)
        
        analysis = {
            'total_samples': len(self.samples),
            'duplicates_removed': self.duplicates_removed,
            'languages': Counter(),
            'vulnerability_types': Counter(),
            'vulnerable_vs_safe': Counter(),
            'avg_code_length': 0,
            'sources': dict(self.stats)
        }
        
        total_length = 0
        for sample in self.samples:
            # Language distribution
            lang = sample.get('language', 'unknown')
            analysis['languages'][lang] += 1
            
            # Vulnerability type distribution
            vuln_type = sample.get('vulnerability_type', 'unknown')
            analysis['vulnerability_types'][vuln_type] += 1
            
            # Vulnerable vs safe
            is_vuln = sample.get('vulnerable', True)
            analysis['vulnerable_vs_safe']['vulnerable' if is_vuln else 'safe'] += 1
            
            # Code length
            code_length = len(sample.get('code', ''))
            total_length += code_length
        
        analysis['avg_code_length'] = total_length / len(self.samples) if self.samples else 0
        
        # Print analysis
        print(f"\nTotal samples: {analysis['total_samples']:,}")
        print(f"Duplicates removed: {analysis['duplicates_removed']:,}")
        print(f"Average code length: {analysis['avg_code_length']:.0f} chars")
        
        print("\nLanguage distribution:")
        for lang, count in analysis['languages'].most_common():
            percentage = (count / analysis['total_samples']) * 100
            print(f"  {lang}: {count:,} ({percentage:.1f}%)")
        
        print("\nVulnerable vs Safe:")
        for label, count in analysis['vulnerable_vs_safe'].items():
            percentage = (count / analysis['total_samples']) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        print("\nTop vulnerability types:")
        for vuln_type, count in analysis['vulnerability_types'].most_common(10):
            percentage = (count / analysis['total_samples']) * 100
            print(f"  {vuln_type}: {count:,} ({percentage:.1f}%)")
        
        return analysis
    
    def save_merged_dataset(self, filename: str = 'merged_samples.jsonl') -> Path:
        """Save merged dataset to JSONL."""
        output_file = self.output_dir / filename
        
        print(f"\nSaving merged dataset to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ Saved {len(self.samples):,} samples")
        
        return output_file
    
    def save_statistics(self, filename: str = 'merge_statistics.json') -> Path:
        """Save merge statistics."""
        stats_file = self.output_dir / filename
        
        analysis = self.analyze_dataset()
        
        # Convert Counters to dicts for JSON serialization
        analysis['languages'] = dict(analysis['languages'])
        analysis['vulnerability_types'] = dict(analysis['vulnerability_types'])
        analysis['vulnerable_vs_safe'] = dict(analysis['vulnerable_vs_safe'])
        
        # Add timestamp
        analysis['merge_timestamp'] = datetime.now().isoformat()
        
        with open(stats_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✓ Statistics saved to: {stats_file}")
        
        return stats_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Merge and preprocess StreamGuard collected data'
    )
    parser.add_argument(
        '--input-dir',
        default='../../data/raw',
        help='Input directory containing collected data'
    )
    parser.add_argument(
        '--output-dir',
        default='../../data/raw/merged',
        help='Output directory for merged data'
    )
    parser.add_argument(
        '--output-file',
        default='merged_samples.jsonl',
        help='Output filename'
    )
    
    args = parser.parse_args()
    
    # Create merger
    merger = ComprehensiveMerger(args.input_dir, args.output_dir)
    
    # Merge all sources
    if not merger.merge_all_sources():
        print("\n✗ No data loaded. Check input directory and file patterns.")
        return 1
    
    # Analyze dataset
    merger.analyze_dataset()
    
    # Save merged dataset
    output_file = merger.save_merged_dataset(args.output_file)
    
    # Save statistics
    merger.save_statistics()
    
    print("\n" + "="*70)
    print("✓ Merge completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print(f"  1. Review statistics: cat {merger.output_dir}/merge_statistics.json")
    print(f"  2. Run preprocessing: python ../../preprocessing/enhanced_preprocessing.py")
    print(f"     --input {output_file}")
    print(f"     --output ../../data/processed/streamguard/")
    print(f"  3. Start training with StreamGuard_Production_Training.ipynb")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Merge interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
