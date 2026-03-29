"""
Post-Collection Validation Script for StreamGuard Data Collection

Validates collected data meets quality standards:
- File existence and sample counts
- Schema validation
- Language distribution
- Code length ranges
- Duplicate detection
- Label balance
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import hashlib
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Try to import tree-sitter for parse validation
TREE_SITTER_AVAILABLE = False

# Method 1: Try tree-sitter-languages (pre-built, recommended)
try:
    from tree_sitter_languages import get_parser
    TREE_SITTER_AVAILABLE = True
    USING_TREE_SITTER_LANGUAGES = True
except ImportError:
    USING_TREE_SITTER_LANGUAGES = False
    
    # Method 2: Try individual packages (requires Rust)
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python
        import tree_sitter_javascript
        TREE_SITTER_AVAILABLE = True
    except ImportError:
        print("Warning: tree-sitter not available. Parse validation will be skipped.")


class CollectionValidator:
    """Validates collected data quality."""
    
    # Expected targets from collection plan
    EXPECTED_SAMPLES = {
        'cves': 15000,
        'github': 10000,
        'opensource': 20000,
        'osv': 20000,
        'exploitdb': 10000,
        'synthetic': 5000
    }
    
    TOTAL_TARGET = 80000
    MIN_ACCEPTABLE_RATIO = 0.70  # Accept if 70% of target reached
    
    def __init__(self, base_dir: str = '../../data/raw'):
        """Initialize validator."""
        self.base_dir = Path(base_dir)
        self.results = {}
        self.errors = []
        self.warnings = []
        
        # Initialize tree-sitter parsers for parse validation
        self.parsers = {}
        if TREE_SITTER_AVAILABLE:
            self._init_parsers()
    
    def _init_parsers(self):
        """Initialize tree-sitter parsers for multiple language versions."""
        try:
            # Method 1: Use tree-sitter-languages (pre-built, recommended)
            if USING_TREE_SITTER_LANGUAGES:
                py_parser = get_parser('python')
                js_parser = get_parser('javascript')
                ts_parser = get_parser('typescript')
                
                self.parsers['python'] = [('python3', py_parser)]
                self.parsers['javascript'] = [('es2022', js_parser)]
                self.parsers['typescript'] = [('typescript', ts_parser)]
                return
            
            # Method 2: Use individual language packages
            PY_LANGUAGE = Language(tree_sitter_python.language())
            py_parser = Parser(PY_LANGUAGE)
            self.parsers['python'] = [('python3', py_parser)]
            
            # JavaScript/TypeScript parser
            JS_LANGUAGE = Language(tree_sitter_javascript.language())
            js_parser = Parser(JS_LANGUAGE)
            self.parsers['javascript'] = [('es2022', js_parser)]
            self.parsers['typescript'] = [('typescript', js_parser)]
            
        except Exception as e:
            print(f"Warning: Could not initialize tree-sitter parsers: {e}")
            self.parsers = {}
    
    def validate_source(self, source: str) -> Dict:
        """Validate a single data source."""
        source_dir = self.base_dir / source
        result = {
            'source': source,
            'exists': False,
            'sample_count': 0,
            'target': self.EXPECTED_SAMPLES[source],
            'completion_rate': 0.0,
            'languages': {},
            'avg_code_length': 0,
            'duplicate_rate': 0.0,
            'label_balance': {},
            'parse_validation': {},
            'errors': []
        }
        
        # Check if directory exists
        if not source_dir.exists():
            result['errors'].append(f"Directory not found: {source_dir}")
            return result
        
        result['exists'] = True
        
        # Find JSONL files
        jsonl_files = list(source_dir.glob('*.jsonl'))
        if not jsonl_files:
            result['errors'].append(f"No JSONL files found in {source_dir}")
            return result
        
        # Load and validate samples
        samples = []
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            sample = json.loads(line)
                            samples.append(sample)
                        except json.JSONDecodeError as e:
                            result['errors'].append(
                                f"Invalid JSON at {jsonl_file.name}:{line_num}: {e}"
                            )
            except Exception as e:
                result['errors'].append(f"Error reading {jsonl_file}: {e}")
        
        result['sample_count'] = len(samples)
        result['completion_rate'] = (len(samples) / result['target']) * 100
        
        if samples:
            # Analyze samples
            result['languages'] = self._analyze_languages(samples)
            result['avg_code_length'] = self._analyze_code_length(samples)
            result['duplicate_rate'] = self._analyze_duplicates(samples)
            result['label_balance'] = self._analyze_labels(samples)
            
            # Parse validation (sample 5% of data)
            if TREE_SITTER_AVAILABLE and self.parsers:
                result['parse_validation'] = self._validate_parseability(samples)
        
        return result
    
    def _analyze_languages(self, samples: List[Dict]) -> Dict[str, int]:
        """Analyze language distribution."""
        languages = Counter()
        for sample in samples:
            lang = sample.get('language', 'unknown')
            languages[lang] += 1
        return dict(languages)
    
    def _analyze_code_length(self, samples: List[Dict]) -> float:
        """Calculate average code length."""
        total_length = 0
        count = 0
        
        for sample in samples:
            # Check various possible code fields
            code_fields = ['code', 'vulnerable_code', 'exploit_code']
            for field in code_fields:
                if field in sample and sample[field]:
                    total_length += len(sample[field])
                    count += 1
                    break
        
        return total_length / count if count > 0 else 0
    
    def _analyze_duplicates(self, samples: List[Dict]) -> float:
        """Calculate duplicate rate based on code hash."""
        code_hashes = set()
        duplicates = 0
        
        for sample in samples:
            # Get code content
            code = None
            code_fields = ['code', 'vulnerable_code', 'exploit_code']
            for field in code_fields:
                if field in sample and sample[field]:
                    code = sample[field]
                    break
            
            if code:
                code_hash = hashlib.md5(code.encode()).hexdigest()
                if code_hash in code_hashes:
                    duplicates += 1
                else:
                    code_hashes.add(code_hash)
        
        return (duplicates / len(samples)) * 100 if samples else 0.0
    
    def _analyze_labels(self, samples: List[Dict]) -> Dict:
        """Analyze label distribution."""
        labels = Counter()
        for sample in samples:
            # Check various possible label fields
            if 'vulnerable' in sample:
                label = 'vulnerable' if sample['vulnerable'] else 'safe'
            elif 'label' in sample:
                label = 'vulnerable' if sample['label'] == 1 else 'safe'
            else:
                label = 'unknown'
            labels[label] += 1
        
        total = sum(labels.values())
        return {
            'counts': dict(labels),
            'percentages': {k: (v/total)*100 for k, v in labels.items()} if total > 0 else {}
        }
    
    def _validate_parseability(self, samples: List[Dict]) -> Dict:
        """
        Validate that code samples are parseable with tree-sitter.
        Samples 5% of data for efficiency.
        """
        # Sample 5% of data
        sample_size = max(10, int(len(samples) * 0.05))
        sampled = random.sample(samples, min(sample_size, len(samples)))
        
        results_by_lang = defaultdict(lambda: {
            'total': 0,
            'parseable': 0,
            'parse_failures': [],
            'by_parser': defaultdict(int)
        })
        
        for sample in sampled:
            # Get code and language
            code = self._extract_code_from_sample(sample)
            if not code:
                continue
            
            language = sample.get('language', 'unknown').lower()
            if language in ['js', 'jsx', 'tsx', 'ts']:
                language = 'javascript'
            
            if language not in self.parsers:
                continue
            
            results_by_lang[language]['total'] += 1
            
            # Try parsing with multiple parser configs
            success, parser_name, failure_reason = self._validate_parseable(code, language)
            
            if success:
                results_by_lang[language]['parseable'] += 1
                results_by_lang[language]['by_parser'][parser_name] += 1
            else:
                results_by_lang[language]['parse_failures'].append({
                    'reason': failure_reason,
                    'code_snippet': code[:200]  # First 200 chars
                })
        
        # Calculate percentages
        summary = {}
        for lang, data in results_by_lang.items():
            total = data['total']
            parseable = data['parseable']
            pct = (parseable / total * 100) if total > 0 else 0
            
            summary[lang] = {
                'total_tested': total,
                'parseable': parseable,
                'parse_success_rate': f"{pct:.1f}%",
                'parser_breakdown': dict(data['by_parser']),
                'failure_count': len(data['parse_failures']),
                'sample_failures': data['parse_failures'][:3]  # First 3 failures
            }
        
        return summary
    
    def _extract_code_from_sample(self, sample: Dict) -> Optional[str]:
        """Extract code from a sample (handles different field names)."""
        code_fields = ['code', 'vulnerable_code', 'exploit_code', 'fixed_code']
        for field in code_fields:
            if field in sample and sample[field]:
                code = sample[field]
                # Skip placeholder code
                if isinstance(code, str) and not code.startswith('#'):
                    return code
        return None
    
    def _validate_parseable(
        self,
        code: str,
        language: str
    ) -> Tuple[bool, str, str]:
        """
        Try parsing code with multiple parser configurations.
        
        Returns:
            Tuple of (success, parser_name, failure_reason)
        """
        if language not in self.parsers:
            return False, "", "No parser available"
        
        parsers = self.parsers[language]
        
        for parser_name, parser in parsers:
            try:
                tree = parser.parse(bytes(code, 'utf-8'))
                if not tree.root_node.has_error:
                    return True, parser_name, ""
            except Exception as e:
                continue
        
        # All parsers failed - diagnose
        failure_reason = self._diagnose_parse_failure(code, language)
        return False, "", failure_reason
    
    def _diagnose_parse_failure(self, code: str, language: str) -> str:
        """Diagnose why parsing failed for logging."""
        indicators = {
            'python2': ['print "', 'except Exception, e:', 'xrange(', '<>'],
            'es5': ['var ', 'function()', '.prototype.'],
            'syntax_error': ['SyntaxError', 'IndentationError'],
        }
        
        code_lower = code.lower()
        
        # Check for Python 2 syntax
        if language == 'python':
            for keyword in indicators['python2']:
                if keyword in code:
                    return "Python 2 syntax detected"
        
        # Check for old JS syntax
        if language == 'javascript':
            for keyword in indicators['es5']:
                if keyword in code:
                    return "ES5/legacy syntax detected"
        
        # Check for incomplete code
        if len(code.strip()) < 10:
            return "Code too short/incomplete"
        
        return "Unknown parse error"
    
    def validate_all_sources(self) -> bool:
        """Validate all data sources."""
        print("="*70)
        print("StreamGuard Post-Collection Validation")
        print("="*70)
        
        all_passed = True
        total_samples = 0
        
        for source in self.EXPECTED_SAMPLES.keys():
            print(f"\n[{source.upper()}]")
            result = self.validate_source(source)
            self.results[source] = result
            
            # Print results
            if not result['exists']:
                print(f"  ✗ Directory not found")
                all_passed = False
                continue
            
            print(f"  Samples: {result['sample_count']:,} / {result['target']:,} "
                  f"({result['completion_rate']:.1f}%)")
            
            if result['completion_rate'] < self.MIN_ACCEPTABLE_RATIO * 100:
                print(f"  ✗ Below minimum threshold ({self.MIN_ACCEPTABLE_RATIO*100}%)")
                all_passed = False
            else:
                print(f"  ✓ Meets minimum threshold")
            
            if result['languages']:
                print(f"  Languages: {', '.join(f'{k}={v}' for k, v in result['languages'].items())}")
            
            if result['avg_code_length'] > 0:
                print(f"  Avg code length: {result['avg_code_length']:.0f} chars")
            
            if result['duplicate_rate'] > 5.0:
                print(f"  ⚠ Duplicate rate: {result['duplicate_rate']:.1f}% (>5%)")
                self.warnings.append(f"{source}: High duplicate rate")
            elif result['duplicate_rate'] > 0:
                print(f"  ✓ Duplicate rate: {result['duplicate_rate']:.1f}%")
            
            if result['label_balance'].get('percentages'):
                percentages = result['label_balance']['percentages']
                print(f"  Label balance: {', '.join(f'{k}={v:.1f}%' for k, v in percentages.items())}")
            
            # Parse validation results
            if result.get('parse_validation'):
                print(f"  Parse Validation:")
                for lang, parse_data in result['parse_validation'].items():
                    success_rate = parse_data['parse_success_rate']
                    tested = parse_data['total_tested']
                    parseable = parse_data['parseable']
                    print(f"    {lang}: {success_rate} ({parseable}/{tested} parseable)")
                    
                    # Warn if parse rate is low
                    rate_float = float(success_rate.rstrip('%'))
                    if rate_float < 80:
                        print(f"    ⚠ Low parse success rate for {lang}")
                        self.warnings.append(f"{source}/{lang}: Parse rate {success_rate} < 80%")
                        if parse_data.get('sample_failures'):
                            print(f"    Sample failures: {parse_data['failure_count']}")
            
            if result['errors']:
                print(f"  ✗ Errors: {len(result['errors'])}")
                for error in result['errors'][:3]:  # Show first 3 errors
                    print(f"    - {error}")
                all_passed = False
            
            total_samples += result['sample_count']
        
        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Total samples collected: {total_samples:,} / {self.TOTAL_TARGET:,} "
              f"({(total_samples/self.TOTAL_TARGET)*100:.1f}%)")
        
        # Check overall targets
        if total_samples < self.TOTAL_TARGET * self.MIN_ACCEPTABLE_RATIO:
            print(f"✗ Below minimum threshold ({self.MIN_ACCEPTABLE_RATIO*100}% of {self.TOTAL_TARGET:,})")
            all_passed = False
        else:
            print(f"✓ Meets minimum threshold")
        
        # Aggregate language distribution
        all_languages = defaultdict(int)
        for result in self.results.values():
            for lang, count in result.get('languages', {}).items():
                all_languages[lang] += count
        
        if all_languages:
            print(f"\nOverall language distribution:")
            total_lang = sum(all_languages.values())
            for lang, count in sorted(all_languages.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_lang) * 100
                print(f"  {lang}: {count:,} ({percentage:.1f}%)")
        
        # Check Python/JS ratio
        python_count = all_languages.get('python', 0)
        js_count = sum(all_languages.get(lang, 0) for lang in ['javascript', 'typescript'])
        total_py_js = python_count + js_count
        
        if total_py_js > 0:
            py_ratio = (python_count / total_py_js) * 100
            js_ratio = (js_count / total_py_js) * 100
            print(f"\nPython/JS ratio: {py_ratio:.1f}% / {js_ratio:.1f}%")
            
            if py_ratio < 30 or py_ratio > 70:
                self.warnings.append(f"Python/JS ratio imbalanced: {py_ratio:.1f}% / {js_ratio:.1f}%")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        print("\n" + "="*70)
        
        if all_passed:
            print("✓ All validations passed! Data quality is acceptable.")
            print("\nNext steps:")
            print("  1. Merge datasets: python merge_datasets.py")
            print("  2. Run preprocessing: python ../../preprocessing/enhanced_preprocessing.py")
            return True
        else:
            print("✗ Some validations failed. Review errors above.")
            print("\nRecommended actions:")
            print("  1. Re-run underperforming collectors")
            print("  2. Check error logs in collection_results.json")
            print("  3. Verify API keys and network connectivity")
            return False
    
    def save_report(self, output_file: str = 'validation_report.json'):
        """Save validation report to JSON."""
        report = {
            'results': self.results,
            'warnings': self.warnings,
            'errors': self.errors,
            'summary': {
                'total_samples': sum(r['sample_count'] for r in self.results.values()),
                'target_samples': self.TOTAL_TARGET,
                'completion_rate': (sum(r['sample_count'] for r in self.results.values()) / self.TOTAL_TARGET) * 100
            }
        }
        
        output_path = self.base_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Validation report saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate collected data')
    parser.add_argument(
        '--base-dir',
        default='../../data/raw',
        help='Base directory containing collected data'
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save validation report to JSON'
    )
    
    args = parser.parse_args()
    
    validator = CollectionValidator(args.base_dir)
    success = validator.validate_all_sources()
    
    if args.save_report:
        validator.save_report()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
