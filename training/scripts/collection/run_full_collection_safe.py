#!/usr/bin/env python3
"""
Safe wrapper script for running StreamGuard data collection.

This script:
1. Runs pre-collection validation
2. Executes master orchestrator with proper error handling
3. Runs post-collection validation
4. Provides clear status updates

Usage:
    python run_full_collection_safe.py [--parallel] [--collectors cve github repo synthetic osv exploitdb]
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {description} interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed with error: {e}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Safe wrapper for StreamGuard data collection'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run collectors in parallel (default: sequential)'
    )
    parser.add_argument(
        '--collectors',
        nargs='+',
        choices=['cve', 'github', 'repo', 'synthetic', 'osv', 'exploitdb'],
        default=['synthetic', 'osv', 'exploitdb', 'github', 'cve', 'repo'],
        help='Collectors to run (default: all in optimal order)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip pre-collection validation'
    )
    parser.add_argument(
        '--skip-post-validation',
        action='store_true',
        help='Skip post-collection validation'
    )
    parser.add_argument(
        '--output-dir',
        default='../../data/raw',
        help='Output directory (default: ../../data/raw)'
    )
    
    args = parser.parse_args()
    
    # Start timestamp
    start_time = datetime.now()
    
    print_header("StreamGuard Data Collection - Safe Execution Wrapper")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
    print(f"Collectors: {', '.join(args.collectors)}")
    print(f"Output directory: {args.output_dir}")
    
    # Step 1: Pre-collection validation
    if not args.skip_validation:
        print_header("Step 1: Pre-Collection Validation")
        
        if not run_command(
            [sys.executable, 'pre_collection_check.py'],
            "Pre-collection validation"
        ):
            print("\n❌ Pre-collection validation failed!")
            print("Please fix the errors above before proceeding.")
            print("\nTo skip validation (not recommended), use: --skip-validation")
            return 1
    else:
        print_header("Step 1: Pre-Collection Validation (SKIPPED)")
    
    # Step 2: Run collection
    print_header("Step 2: Data Collection")
    print("⚠️  This may take 12-24 hours (parallel) or 24-48 hours (sequential)")
    print("You can safely interrupt with Ctrl+C - progress will be saved\n")
    
    # Build command
    cmd = [
        sys.executable,
        'master_orchestrator.py',
        '--collectors'
    ] + args.collectors + [
        '--output-dir', args.output_dir
    ]
    
    if not args.parallel:
        cmd.append('--sequential')
    
    collection_success = run_command(cmd, "Data collection")
    
    # Step 3: Post-collection validation
    if not args.skip_post_validation:
        print_header("Step 3: Post-Collection Validation")
        
        validation_success = run_command(
            [sys.executable, 'validate_collection.py', '--base-dir', args.output_dir, '--save-report'],
            "Post-collection validation"
        )
    else:
        print_header("Step 3: Post-Collection Validation (SKIPPED)")
        validation_success = True
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("EXECUTION SUMMARY")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"\nCollection: {'✓ Success' if collection_success else '✗ Failed'}")
    print(f"Validation: {'✓ Success' if validation_success else '✗ Failed'}")
    
    # Check results file
    results_file = Path(args.output_dir) / 'collection_results.json'
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
                summary = results.get('summary', {})
                total_samples = summary.get('total_samples_collected', 0)
                target_samples = summary.get('total_target_samples', 80000)
                
                print(f"\nSamples collected: {total_samples:,} / {target_samples:,} "
                      f"({(total_samples/target_samples)*100:.1f}%)")
        except Exception as e:
            print(f"\n⚠ Could not read results file: {e}")
    
    print("\n" + "="*70)
    
    if collection_success and validation_success:
        print("\n✓ Data collection completed successfully!")
        print("\nNext steps:")
        print("  1. Merge datasets: python merge_datasets.py")
        print("  2. Run preprocessing: python ../../preprocessing/enhanced_preprocessing.py")
        print("  3. Start training with StreamGuard_Production_Training.ipynb")
        return 0
    else:
        print("\n✗ Data collection encountered issues.")
        print("\nRecommended actions:")
        print("  1. Review error logs above")
        print("  2. Check collection_results.json for details")
        print("  3. Re-run failed collectors individually")
        print("  4. Use --resume flag to continue from checkpoint")
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        print("Partial results may be available in data/raw/")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
