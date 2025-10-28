"""
Master Orchestrator for StreamGuard Data Collection

Runs all 6 collectors in parallel with real-time progress monitoring,
error handling, and comprehensive reporting.

Collectors:
- CVE (NVD): 15,000 samples
- GitHub Advisories: 10,000 samples
- Open Source Repos: 20,000 samples
- Synthetic: 5,000 samples
- OSV Database: 20,000 samples
- ExploitDB: 10,000 samples
Total: 80,000 samples

Features:
- Parallel execution with multiprocessing
- Real-time Rich progress dashboard
- Graceful error handling and recovery
- Comprehensive statistics and reporting
- Multiple export formats (JSON, CSV, PDF, SARIF)
"""

import json
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import traceback
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cve_collector_enhanced import CVECollectorEnhanced
from github_advisory_collector_enhanced import GitHubAdvisoryCollectorEnhanced
from repo_miner_enhanced import EnhancedRepoMiner
from synthetic_generator import SyntheticGenerator
from osv_collector import OSVCollector
from exploitdb_collector import ExploitDBCollector

# Import dashboard functionality
try:
    from progress_dashboard import create_dashboard, RICH_AVAILABLE
except ImportError:
    RICH_AVAILABLE = False
    def create_dashboard(*args, **kwargs):
        return None


class CollectorProcess:
    """Wrapper for running a collector in a separate process."""

    def __init__(self, name: str, collector_class, target_samples: int,
                 output_dir: str, config: Dict, queue: Queue):
        """
        Initialize collector process wrapper.

        Args:
            name: Collector name (cve, github, repo, synthetic)
            collector_class: Collector class to instantiate
            target_samples: Target number of samples
            output_dir: Output directory
            config: Collector-specific configuration
            queue: Queue for progress updates
        """
        self.name = name
        self.collector_class = collector_class
        self.target_samples = target_samples
        self.output_dir = output_dir
        self.config = config
        self.queue = queue
        self.process = None
        self.start_time = None
        self.end_time = None

    def run(self):
        """Run the collector in a separate process."""
        self.start_time = time.time()
        self.process = Process(
            target=self._collect_worker,
            args=(self.name, self.collector_class, self.target_samples,
                  self.output_dir, self.config, self.queue)
        )
        self.process.start()
        return self.process

    @staticmethod
    def _collect_worker(name: str, collector_class, target_samples: int,
                       output_dir: str, config: Dict, queue: Queue):
        """
        Worker function that runs in separate process.

        Args:
            name: Collector name
            collector_class: Collector class
            target_samples: Target samples
            output_dir: Output directory
            config: Configuration
            queue: Progress queue
        """
        try:
            # CRITICAL: Load .env in subprocess (each subprocess needs its own load)
            from dotenv import load_dotenv
            load_dotenv()

            queue.put({
                'collector': name,
                'status': 'starting',
                'message': f'Initializing {name} collector...',
                'timestamp': datetime.now().isoformat()
            })

            # Initialize collector
            if name == 'cve':
                collector = collector_class(
                    output_dir=output_dir,
                    cache_enabled=config.get('cache_enabled', True),
                    github_token=None  # Read from environment after load_dotenv()
                )
                collector.TARGET_SAMPLES = target_samples

            elif name == 'github':
                collector = collector_class(
                    output_dir=output_dir,
                    cache_enabled=config.get('cache_enabled', True),
                    github_token=None  # Read from environment after load_dotenv()
                )

            elif name == 'repo':
                collector = collector_class(
                    output_dir=output_dir,
                    cache_enabled=config.get('cache_enabled', True)
                )

            elif name == 'synthetic':
                collector = collector_class(
                    output_dir=output_dir,
                    seed=config.get('seed', 42)
                )

            elif name == 'osv':
                collector = collector_class(
                    output_dir=output_dir,
                    cache_enabled=config.get('cache_enabled', True),
                    resume=config.get('resume', False)
                )

            elif name == 'exploitdb':
                collector = collector_class(
                    output_dir=output_dir,
                    cache_enabled=config.get('cache_enabled', True),
                    resume=config.get('resume', False)
                )

            else:
                raise ValueError(f"Unknown collector: {name}")

            queue.put({
                'collector': name,
                'status': 'running',
                'message': f'{name} collector started',
                'samples_collected': 0,
                'target_samples': target_samples,
                'timestamp': datetime.now().isoformat()
            })

            # Run collection with progress updates
            start_time = time.time()

            if name == 'cve':
                samples = collector.collect()
            elif name == 'github':
                samples = collector.collect_all_advisories(target_samples)
            elif name == 'repo':
                samples = collector.collect()
            elif name == 'synthetic':
                samples = collector.generate_samples(target_samples)
            elif name == 'osv':
                samples = collector.collect_all_vulnerabilities(target_samples)
            elif name == 'exploitdb':
                samples = collector.collect_all_exploits(target_samples)

            end_time = time.time()
            duration = end_time - start_time

            # Send completion update
            queue.put({
                'collector': name,
                'status': 'completed',
                'message': f'{name} collector completed successfully',
                'samples_collected': len(samples),
                'target_samples': target_samples,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'success': True
            })

        except Exception as e:
            # Send error update
            queue.put({
                'collector': name,
                'status': 'error',
                'message': f'{name} collector failed: {str(e)}',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat(),
                'success': False
            })


class MasterOrchestrator:
    """
    Master orchestrator for parallel data collection.

    Coordinates all collectors, monitors progress, handles errors,
    and generates comprehensive reports.
    """

    def __init__(self,
                 collectors: List[str] = None,
                 output_dir: str = 'data/raw',
                 parallel: bool = True,
                 show_dashboard: bool = True,
                 config: Optional[Dict] = None):
        """
        Initialize master orchestrator.

        Args:
            collectors: List of collectors to run (cve, github, repo, synthetic)
            output_dir: Base output directory
            parallel: Run collectors in parallel
            show_dashboard: Show Rich progress dashboard
            config: Configuration dictionary
        """
        self.collectors = collectors or ['cve', 'github', 'repo', 'synthetic', 'osv', 'exploitdb']
        self.output_dir = Path(output_dir)
        self.parallel = parallel
        self.show_dashboard = show_dashboard
        self.config = config or {}

        # Collector configurations
        self.collector_configs = {
            'cve': {
                'class': CVECollectorEnhanced,
                'target': self.config.get('cve_samples', 15000),
                'output_dir': str(self.output_dir / 'cves'),
                'config': {
                    'cache_enabled': self.config.get('cache_enabled', True),
                    'github_token': None  # Read from environment in subprocess
                }
            },
            'github': {
                'class': GitHubAdvisoryCollectorEnhanced,
                'target': self.config.get('github_samples', 10000),
                'output_dir': str(self.output_dir / 'github'),
                'config': {
                    'cache_enabled': self.config.get('cache_enabled', True),
                    'github_token': None  # Read from environment in subprocess
                }
            },
            'repo': {
                'class': EnhancedRepoMiner,
                'target': self.config.get('repo_samples', 20000),
                'output_dir': str(self.output_dir / 'opensource'),
                'config': {
                    'cache_enabled': self.config.get('cache_enabled', True)
                }
            },
            'synthetic': {
                'class': SyntheticGenerator,
                'target': self.config.get('synthetic_samples', 5000),
                'output_dir': str(self.output_dir / 'synthetic'),
                'config': {
                    'seed': self.config.get('seed', 42)
                }
            },
            'osv': {
                'class': OSVCollector,
                'target': self.config.get('osv_samples', 20000),
                'output_dir': str(self.output_dir / 'osv'),
                'config': {
                    'cache_enabled': self.config.get('cache_enabled', True),
                    'resume': self.config.get('resume', False)
                }
            },
            'exploitdb': {
                'class': ExploitDBCollector,
                'target': self.config.get('exploitdb_samples', 10000),
                'output_dir': str(self.output_dir / 'exploitdb'),
                'config': {
                    'cache_enabled': self.config.get('cache_enabled', True),
                    'resume': self.config.get('resume', False)
                }
            }
        }

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for collector_name in self.collectors:
            Path(self.collector_configs[collector_name]['output_dir']).mkdir(
                parents=True, exist_ok=True
            )

        # Shared progress queue
        self.progress_queue = Queue()
        self.results = {}

    def run_collection(self) -> Dict:
        """
        Run all collectors and return results.

        Returns:
            Dictionary with collection results and statistics
        """
        print("\n" + "="*70)
        print("StreamGuard Data Collection - Master Orchestrator")
        print("="*70)
        print(f"\nCollectors to run: {', '.join(self.collectors)}")
        print(f"Mode: {'Parallel' if self.parallel else 'Sequential'}")
        print(f"Output directory: {self.output_dir}")
        print(f"Dashboard: {'Enabled' if self.show_dashboard else 'Disabled'}")
        print("\n" + "-"*70 + "\n")

        start_time = time.time()

        if self.parallel:
            results = self._run_parallel()
        else:
            results = self._run_sequential()

        end_time = time.time()
        total_duration = end_time - start_time

        # Compile final results
        final_results = {
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'total_duration': total_duration,
            'mode': 'parallel' if self.parallel else 'sequential',
            'collectors': results,
            'summary': self._generate_summary(results)
        }

        self.results = final_results
        return final_results

    def _run_parallel(self) -> Dict:
        """Run all collectors in parallel."""
        print("Starting parallel collection...\n")

        # Create collector processes
        collector_processes = {}
        for collector_name in self.collectors:
            config = self.collector_configs[collector_name]
            process_wrapper = CollectorProcess(
                name=collector_name,
                collector_class=config['class'],
                target_samples=config['target'],
                output_dir=config['output_dir'],
                config=config['config'],
                queue=self.progress_queue
            )
            collector_processes[collector_name] = process_wrapper

        # Start all processes
        for name, wrapper in collector_processes.items():
            print(f"+ Starting {name} collector...")
            wrapper.run()

        # Monitor progress
        results = self._monitor_progress(collector_processes)

        # Wait for all processes to complete
        for name, wrapper in collector_processes.items():
            wrapper.process.join()

        return results

    def _run_sequential(self) -> Dict:
        """Run all collectors sequentially."""
        print("Starting sequential collection...\n")

        results = {}
        for collector_name in self.collectors:
            print(f"\n{'='*70}")
            print(f"Running {collector_name} collector...")
            print('='*70 + "\n")

            config = self.collector_configs[collector_name]

            # Run collector
            wrapper = CollectorProcess(
                name=collector_name,
                collector_class=config['class'],
                target_samples=config['target'],
                output_dir=config['output_dir'],
                config=config['config'],
                queue=self.progress_queue
            )

            wrapper.run()
            wrapper.process.join()

            # Get result
            while not self.progress_queue.empty():
                update = self.progress_queue.get()
                if update['collector'] == collector_name and \
                   update['status'] == 'completed':
                    results[collector_name] = update

        return results

    def _monitor_progress(self, collector_processes: Dict) -> Dict:
        """
        Monitor progress of all collectors.

        Args:
            collector_processes: Dictionary of collector process wrappers

        Returns:
            Dictionary with results for each collector
        """
        results = {name: {} for name in collector_processes.keys()}
        active_collectors = set(collector_processes.keys())

        # Create dashboard if enabled and Rich is available
        dashboard = None
        if self.show_dashboard and RICH_AVAILABLE:
            try:
                dashboard = create_dashboard(list(collector_processes.keys()), use_rich=True)

                # Initialize dashboard with target samples
                for name, wrapper in collector_processes.items():
                    dashboard.update(name, {
                        'status': 'pending',
                        'message': 'Waiting to start...',
                        'samples_collected': 0,
                        'target_samples': wrapper.target_samples
                    })

                dashboard.start()
                print("Rich dashboard started\n")
            except Exception as e:
                print(f"WARNING: Could not start Rich dashboard: {e}")
                print("Falling back to simple text output...\n")
                dashboard = None

        if not dashboard:
            print("Monitoring progress...\n")

        while active_collectors:
            try:
                # Check for updates (non-blocking with timeout)
                try:
                    update = self.progress_queue.get(timeout=1)
                except:
                    # Check if processes are still alive
                    for name in list(active_collectors):
                        if not collector_processes[name].process.is_alive():
                            active_collectors.remove(name)

                    # Refresh dashboard if active
                    if dashboard:
                        dashboard.refresh()

                    continue

                collector_name = update['collector']
                status = update['status']

                # Update dashboard OR print to console
                if dashboard:
                    dashboard.update(collector_name, update)
                    dashboard.refresh()
                else:
                    # Simple text output (fallback)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] {collector_name}: {update['message']}")

                # Store result
                results[collector_name] = update

                # Remove from active if completed or errored
                if status in ['completed', 'error']:
                    if collector_name in active_collectors:
                        active_collectors.remove(collector_name)

                    # Print completion details for simple mode
                    if not dashboard:
                        if status == 'completed':
                            duration = update.get('duration', 0)
                            samples = update.get('samples_collected', 0)
                            print(f"  + Completed in {duration:.1f}s - {samples} samples collected")
                        elif status == 'error':
                            print(f"  X Error: {update['message']}")

            except KeyboardInterrupt:
                print("\n\n! Keyboard interrupt detected. Stopping collectors...")
                for name, wrapper in collector_processes.items():
                    if wrapper.process.is_alive():
                        wrapper.process.terminate()
                break

        # Stop dashboard and show final summary
        if dashboard:
            dashboard.stop()
            dashboard.print_final_summary()

        return results

    def _generate_summary(self, results: Dict) -> Dict:
        """
        Generate summary statistics from results.

        Args:
            results: Dictionary of collector results

        Returns:
            Summary statistics dictionary
        """
        summary = {
            'total_collectors': len(self.collectors),
            'successful_collectors': 0,
            'failed_collectors': 0,
            'total_samples_collected': 0,
            'total_target_samples': 0,
            'total_duration': 0,
            'by_collector': {}
        }

        for collector_name, result in results.items():
            if result.get('success', False):
                summary['successful_collectors'] += 1
                summary['total_samples_collected'] += result.get('samples_collected', 0)
            else:
                summary['failed_collectors'] += 1

            summary['total_target_samples'] += result.get('target_samples', 0)
            summary['total_duration'] += result.get('duration', 0)

            summary['by_collector'][collector_name] = {
                'status': result.get('status', 'unknown'),
                'samples_collected': result.get('samples_collected', 0),
                'target_samples': result.get('target_samples', 0),
                'duration': result.get('duration', 0),
                'success': result.get('success', False)
            }

        # Calculate completion rate
        if summary['total_target_samples'] > 0:
            summary['completion_rate'] = (
                summary['total_samples_collected'] / summary['total_target_samples']
            ) * 100

        return summary

    def print_summary(self):
        """Print collection summary to console."""
        if not self.results:
            print("No results available. Run collection first.")
            return

        summary = self.results['summary']

        print("\n" + "="*70)
        print("COLLECTION SUMMARY")
        print("="*70)

        print(f"\nTotal Duration: {self.results['total_duration']:.1f}s")
        print(f"Mode: {self.results['mode'].title()}")
        print(f"\nCollectors: {summary['successful_collectors']}/{summary['total_collectors']} successful")
        print(f"Total Samples: {summary['total_samples_collected']:,}/{summary['total_target_samples']:,} "
              f"({summary.get('completion_rate', 0):.1f}%)")

        print("\n" + "-"*70)
        print("By Collector:")
        print("-"*70)

        for name, stats in summary['by_collector'].items():
            status_icon = "+" if stats['success'] else "X"
            print(f"\n{status_icon} {name.upper()}")
            print(f"  Status: {stats['status']}")
            print(f"  Samples: {stats['samples_collected']:,}/{stats['target_samples']:,}")
            print(f"  Duration: {stats['duration']:.1f}s")

        print("\n" + "="*70 + "\n")

    def save_results(self, output_file: str = None):
        """
        Save results to JSON file.

        Args:
            output_file: Output file path (default: data/raw/collection_results.json)
        """
        if not self.results:
            print("No results to save. Run collection first.")
            return

        output_file = output_file or str(self.output_dir / 'collection_results.json')

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n+ Results saved to: {output_file}")

    def save_partial_results(self):
        """
        Save partial results during graceful shutdown.

        This method can be called when collection is interrupted (Ctrl+C)
        to save whatever progress has been made so far.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        partial_file = self.output_dir / f'collection_partial_{timestamp}.json'

        # Gather whatever results we have from the queue
        partial_results = {}
        while not self.progress_queue.empty():
            try:
                update = self.progress_queue.get_nowait()
                collector_name = update['collector']
                # Keep the latest update for each collector
                if collector_name not in partial_results or \
                   update.get('timestamp', '') > partial_results[collector_name].get('timestamp', ''):
                    partial_results[collector_name] = update
            except:
                break

        # If we have results from run_collection, use those
        if self.results:
            results_to_save = self.results
            results_to_save['interrupted'] = True
            results_to_save['interrupted_at'] = datetime.now().isoformat()
        else:
            # Create minimal results structure
            results_to_save = {
                'start_time': datetime.now().isoformat(),
                'interrupted': True,
                'interrupted_at': datetime.now().isoformat(),
                'mode': 'parallel' if self.parallel else 'sequential',
                'collectors': partial_results,
                'summary': self._generate_summary(partial_results)
            }

        # Save to file
        with open(partial_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"+ Partial results saved to: {partial_file}")

        # Also check what data files were actually written
        print("\nData files created:")
        for collector_name in self.collectors:
            output_dir = Path(self.collector_configs[collector_name]['output_dir'])
            if output_dir.exists():
                files = list(output_dir.glob('*.jsonl')) + list(output_dir.glob('*.json'))
                if files:
                    total_size = sum(f.stat().st_size for f in files)
                    print(f"  {collector_name}: {len(files)} file(s), {total_size/1024/1024:.2f} MB")
                else:
                    print(f"  {collector_name}: No data files yet")

        return str(partial_file)


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description='StreamGuard Master Data Collection Orchestrator'
    )
    parser.add_argument(
        '--collectors',
        nargs='+',
        choices=['cve', 'github', 'repo', 'synthetic', 'osv', 'exploitdb'],
        default=['cve', 'github', 'repo', 'synthetic', 'osv', 'exploitdb'],
        help='Collectors to run'
    )
    parser.add_argument(
        '--output-dir',
        default='data/raw',
        help='Output directory'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run collectors sequentially instead of parallel'
    )
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Disable progress dashboard'
    )
    parser.add_argument(
        '--github-token',
        help='GitHub API token'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )

    args = parser.parse_args()

    # Create configuration
    config = {
        'cache_enabled': not args.no_cache,
        'github_token': args.github_token
    }

    # Create and run orchestrator
    orchestrator = MasterOrchestrator(
        collectors=args.collectors,
        output_dir=args.output_dir,
        parallel=not args.sequential,
        show_dashboard=not args.no_dashboard,
        config=config
    )

    try:
        results = orchestrator.run_collection()
        orchestrator.print_summary()
        orchestrator.save_results()
    except KeyboardInterrupt:
        print("\n\n! Collection interrupted by user.")
    except Exception as e:
        print(f"\nX Error: {e}")
        traceback.print_exc()
