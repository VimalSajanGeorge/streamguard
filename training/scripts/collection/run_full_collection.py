#!/usr/bin/env python
"""
StreamGuard Full Collection Runner

CLI entry point for running the complete data collection pipeline.
Runs all collectors in parallel with real-time progress monitoring
and generates comprehensive reports.

Usage:
    # Run all collectors with default settings
    python run_full_collection.py

    # Run specific collectors
    python run_full_collection.py --collectors cve github

    # Sequential mode
    python run_full_collection.py --sequential

    # Custom output directory
    python run_full_collection.py --output-dir /path/to/output

    # Generate specific report formats
    python run_full_collection.py --report-formats json pdf sarif

    # Quick test run (reduced samples)
    python run_full_collection.py --quick-test
"""

import argparse
import sys
import os
import signal
from pathlib import Path
import traceback
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from master_orchestrator import MasterOrchestrator
from progress_dashboard import create_dashboard
from report_generator import ReportGenerator

# Global reference for graceful shutdown
_orchestrator = None


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='StreamGuard Full Data Collection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all collectors in parallel
  python run_full_collection.py

  # Run specific collectors
  python run_full_collection.py --collectors cve github synthetic

  # Quick test run (100 samples each)
  python run_full_collection.py --quick-test

  # Sequential mode with custom output
  python run_full_collection.py --sequential --output-dir /custom/path

  # Generate only JSON and CSV reports
  python run_full_collection.py --report-formats json csv
        """
    )

    # Collector selection
    parser.add_argument(
        '--collectors',
        nargs='+',
        choices=['cve', 'github', 'repo', 'synthetic', 'osv', 'exploitdb', 'all'],
        default=['all'],
        help='Collectors to run (default: all)'
    )

    # Execution mode
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Run collectors in parallel (default)'
    )
    execution_group.add_argument(
        '--sequential',
        action='store_true',
        help='Run collectors sequentially'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        default='data/raw',
        help='Output directory for collected data (default: data/raw)'
    )

    # Dashboard
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Disable progress dashboard'
    )

    # Report formats
    parser.add_argument(
        '--report-formats',
        nargs='+',
        choices=['json', 'csv', 'pdf', 'sarif', 'all'],
        default=['all'],
        help='Report formats to generate (default: all available)'
    )

    # Sample targets
    parser.add_argument(
        '--cve-samples',
        type=int,
        default=15000,
        help='Target CVE samples (default: 15000)'
    )
    parser.add_argument(
        '--github-samples',
        type=int,
        default=10000,
        help='Target GitHub advisory samples (default: 10000)'
    )
    parser.add_argument(
        '--repo-samples',
        type=int,
        default=20000,
        help='Target repository mining samples (default: 20000)'
    )
    parser.add_argument(
        '--synthetic-samples',
        type=int,
        default=5000,
        help='Target synthetic samples (default: 5000)'
    )
    parser.add_argument(
        '--osv-samples',
        type=int,
        default=20000,
        help='Target OSV vulnerability samples (default: 20000)'
    )
    parser.add_argument(
        '--exploitdb-samples',
        type=int,
        default=10000,
        help='Target ExploitDB exploit samples (default: 10000)'
    )

    # Quick test mode
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (100 samples per collector)'
    )

    # API tokens
    parser.add_argument(
        '--github-token',
        help='GitHub API token (or set GITHUB_TOKEN env var)'
    )

    # Caching
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )

    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Checkpoint/resume
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )

    return parser.parse_args()


def graceful_shutdown(signum, frame):
    """
    Graceful shutdown handler for SIGINT (Ctrl+C) and SIGTERM.

    Saves partial progress and terminates cleanly.
    """
    global _orchestrator

    print("\n\n" + "="*70)
    print("SHUTDOWN SIGNAL RECEIVED")
    print("="*70)
    print("\nAttempting graceful shutdown...")
    print("Saving partial progress and cleaning up...\n")

    if _orchestrator:
        try:
            # Save whatever progress we have
            print("Saving partial results...")
            _orchestrator.save_partial_results()

            # Print summary of what was collected
            print("\nPartial Collection Summary:")
            _orchestrator.print_summary()

            print("\n" + "="*70)
            print("Graceful shutdown complete.")
            print("="*70)
            print("\nPartial results saved. You can:")
            print("  1. Resume collection with the same parameters")
            print("  2. Run with --no-dashboard if VS Code crashed")
            print("  3. Check data/raw/ for collected samples\n")

        except Exception as e:
            print(f"Error during shutdown: {e}")
            traceback.print_exc()

    sys.exit(130)  # Standard exit code for Ctrl+C


def main():
    """Main entry point."""
    global _orchestrator

    # Load environment variables from .env file FIRST
    load_dotenv()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, graceful_shutdown)   # Ctrl+C
    if hasattr(signal, 'SIGTERM'):  # Not available on Windows
        signal.signal(signal.SIGTERM, graceful_shutdown)  # Termination signal

    args = parse_arguments()

    # Banner
    print("\n" + "="*70)
    print("StreamGuard Full Data Collection Pipeline")
    print("="*70)
    print("\nGathering vulnerability data from multiple sources...")
    print("This may take several hours depending on configuration.\n")

    # Handle 'all' in collectors and report formats
    if 'all' in args.collectors:
        collectors = ['cve', 'github', 'repo', 'synthetic', 'osv', 'exploitdb']
    else:
        collectors = args.collectors

    if 'all' in args.report_formats:
        report_formats = ['json', 'csv', 'pdf', 'sarif']
    else:
        report_formats = args.report_formats

    # Quick test mode
    if args.quick_test:
        print("⚡ Quick Test Mode: Running with 100 samples per collector\n")
        sample_config = {
            'cve_samples': 100,
            'github_samples': 100,
            'repo_samples': 100,
            'synthetic_samples': 100,
            'osv_samples': 100,
            'exploitdb_samples': 100
        }
    else:
        sample_config = {
            'cve_samples': args.cve_samples,
            'github_samples': args.github_samples,
            'repo_samples': args.repo_samples,
            'synthetic_samples': args.synthetic_samples,
            'osv_samples': args.osv_samples,
            'exploitdb_samples': args.exploitdb_samples
        }

    # Get GitHub token
    github_token = args.github_token or os.environ.get('GITHUB_TOKEN')
    if not github_token and ('cve' in collectors or 'github' in collectors):
        print("Warning: No GitHub token provided. API rate limits will be lower.")
        print("  Set GITHUB_TOKEN environment variable or use --github-token\n")

    # Configuration
    config = {
        **sample_config,
        'cache_enabled': not args.no_cache,
        'github_token': github_token,
        'seed': args.seed,
        'resume': args.resume
    }

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    print(f"  Collectors: {', '.join(collectors)}")
    print(f"  Mode: {'Parallel' if not args.sequential else 'Sequential'}")
    print(f"  Output: {output_dir}")
    print(f"  Dashboard: {'Enabled' if not args.no_dashboard else 'Disabled'}")
    print(f"  Caching: {'Enabled' if not args.no_cache else 'Disabled'}")
    print(f"  Resume: {'Enabled' if args.resume else 'Disabled'}")
    print(f"  Report Formats: {', '.join(report_formats)}")
    print(f"  Total Target Samples: {sum(sample_config.values()):,}")
    print()

    try:
        # Create orchestrator
        orchestrator = MasterOrchestrator(
            collectors=collectors,
            output_dir=str(output_dir),
            parallel=not args.sequential,
            show_dashboard=not args.no_dashboard,
            config=config
        )

        # Store global reference for graceful shutdown
        _orchestrator = orchestrator

        # Run collection
        print("-"*70)
        print("Starting collection...")
        print("-"*70 + "\n")

        results = orchestrator.run_collection()

        # Print summary
        orchestrator.print_summary()

        # Save raw results
        orchestrator.save_results()

        # Generate reports
        if report_formats:
            print("\nGenerating reports...")
            generator = ReportGenerator(results, output_dir=str(output_dir))
            generated_reports = generator.generate_all_reports(formats=report_formats)

            print("\n" + "="*70)
            print(" Collection Complete!")
            print("="*70)
            print(f"\nTotal Samples: {results['summary']['total_samples_collected']:,}")
            print(f"Duration: {results['total_duration']:.1f}s")
            print(f"\nReports saved in: {output_dir}/")
            print("="*70 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n[WARNING] Collection interrupted by user.")
        return 1

    except Exception as e:
        print(f"\n[ERROR] Error during collection: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
