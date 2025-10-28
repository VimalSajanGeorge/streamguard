"""
Rich Progress Dashboard for StreamGuard Data Collection

Provides real-time visual progress monitoring with Rich library,
including progress bars, statistics, and status indicators.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeRemainingColumn, TimeElapsedColumn
    )
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("WARNING: Rich library not available. Install with: pip install rich")


class ProgressDashboard:
    """
    Real-time progress dashboard using Rich library.

    Features:
    - Live progress bars for each collector
    - Real-time statistics
    - Color-coded status indicators
    - ETA calculations
    - Error tracking
    """

    def __init__(self, collectors: List[str]):
        """
        Initialize progress dashboard.

        Args:
            collectors: List of collector names
        """
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required for dashboard. Install with: pip install rich")

        self.collectors = collectors
        self.console = Console()

        # Collector states
        self.states = {
            name: {
                'status': 'pending',
                'samples_collected': 0,
                'target_samples': 0,
                'start_time': None,
                'end_time': None,
                'message': 'Waiting to start...',
                'error': None
            }
            for name in collectors
        }

        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )

        # Create progress tasks
        self.tasks = {}
        for name in collectors:
            task_id = self.progress.add_task(
                f"{name.upper()}", total=100, completed=0
            )
            self.tasks[name] = task_id

        self.start_time = time.time()
        self.live = None

    def update(self, collector: str, update_data: Dict):
        """
        Update collector status.

        Args:
            collector: Collector name
            update_data: Update data from collector
        """
        if collector not in self.states:
            return

        state = self.states[collector]

        # Update state
        state['status'] = update_data.get('status', state['status'])
        state['message'] = update_data.get('message', state['message'])
        state['samples_collected'] = update_data.get('samples_collected', state['samples_collected'])
        state['target_samples'] = update_data.get('target_samples', state['target_samples'])
        state['error'] = update_data.get('error', state['error'])

        if update_data.get('status') == 'starting':
            state['start_time'] = time.time()
        elif update_data.get('status') in ['completed', 'error']:
            state['end_time'] = time.time()

        # Update progress bar
        task_id = self.tasks[collector]
        if state['target_samples'] > 0:
            percentage = (state['samples_collected'] / state['target_samples']) * 100
            self.progress.update(task_id, completed=percentage, total=100)

            # Update task description with status emoji
            status_emoji = self._get_status_emoji(state['status'])
            self.progress.update(
                task_id,
                description=f"{status_emoji} {collector.upper()}"
            )

    def _get_status_emoji(self, status: str) -> str:
        """Get ASCII-safe status indicator."""
        status_indicators = {
            'pending': '[-]',
            'starting': '[>]',
            'running': '[*]',
            'completed': '[+]',
            'error': '[X]'
        }
        return status_indicators.get(status, '[?]')

    def generate_layout(self) -> Layout:
        """Generate Rich layout with all dashboard components."""
        layout = Layout()

        # Split layout
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=len(self.collectors) + 4),
            Layout(name="statistics", size=10),
            Layout(name="footer", size=3)
        )

        # Header
        elapsed = time.time() - self.start_time
        layout["header"].update(
            Panel(
                Text("StreamGuard Data Collection Dashboard", justify="center", style="bold cyan"),
                subtitle=f"Elapsed: {str(timedelta(seconds=int(elapsed)))}"
            )
        )

        # Progress bars
        layout["progress"].update(Panel(self.progress, title="Collection Progress", border_style="blue"))

        # Statistics table
        layout["statistics"].update(self._generate_statistics_table())

        # Footer
        active_count = sum(1 for s in self.states.values() if s['status'] == 'running')
        completed_count = sum(1 for s in self.states.values() if s['status'] == 'completed')
        error_count = sum(1 for s in self.states.values() if s['status'] == 'error')

        footer_text = f"Active: {active_count} | Completed: {completed_count} | Errors: {error_count}"
        layout["footer"].update(Panel(footer_text, style="dim"))

        return layout

    def _generate_statistics_table(self) -> Table:
        """Generate statistics table."""
        table = Table(title="Collector Statistics", show_header=True, header_style="bold magenta")

        table.add_column("Collector", style="cyan", width=12)
        table.add_column("Status", width=12)
        table.add_column("Samples", justify="right", width=15)
        table.add_column("Duration", justify="right", width=12)
        table.add_column("Rate", justify="right", width=12)
        table.add_column("Message", width=30)

        for name, state in self.states.items():
            # Status with color
            status = state['status']
            if status == 'completed':
                status_display = Text("[+] Completed", style="green")
            elif status == 'error':
                status_display = Text("[X] Error", style="red")
            elif status == 'running':
                status_display = Text("[*] Running", style="yellow")
            elif status == 'starting':
                status_display = Text("[>] Starting", style="blue")
            else:
                status_display = Text("[-] Pending", style="dim")

            # Samples
            samples_display = f"{state['samples_collected']:,}/{state['target_samples']:,}"

            # Duration
            if state['start_time']:
                if state['end_time']:
                    duration = state['end_time'] - state['start_time']
                else:
                    duration = time.time() - state['start_time']
                duration_display = f"{int(duration)}s"
            else:
                duration_display = "-"

            # Collection rate
            if state['start_time'] and state['samples_collected'] > 0:
                elapsed = (state['end_time'] or time.time()) - state['start_time']
                if elapsed > 0:
                    rate = state['samples_collected'] / elapsed
                    rate_display = f"{rate:.1f}/s"
                else:
                    rate_display = "-"
            else:
                rate_display = "-"

            # Message (truncated)
            message = state['message'][:28] + "..." if len(state['message']) > 30 else state['message']

            table.add_row(
                name.upper(),
                status_display,
                samples_display,
                duration_display,
                rate_display,
                message
            )

        # Add totals row
        total_collected = sum(s['samples_collected'] for s in self.states.values())
        total_target = sum(s['target_samples'] for s in self.states.values())
        table.add_row(
            "TOTAL",
            "",
            f"{total_collected:,}/{total_target:,}",
            "",
            "",
            "",
            style="bold"
        )

        return table

    def start(self):
        """Start live dashboard."""
        self.live = Live(
            self.generate_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=False
        )
        self.live.start()

    def stop(self):
        """Stop live dashboard."""
        if self.live:
            self.live.stop()

    def refresh(self):
        """Refresh dashboard display."""
        if self.live:
            self.live.update(self.generate_layout())

    def print_final_summary(self):
        """Print final summary after collection."""
        self.console.print("\n" + "="*70, style="bold")
        self.console.print("FINAL COLLECTION SUMMARY", style="bold cyan", justify="center")
        self.console.print("="*70, style="bold")

        # Overall statistics
        total_collected = sum(s['samples_collected'] for s in self.states.values())
        total_target = sum(s['target_samples'] for s in self.states.values())
        completed_count = sum(1 for s in self.states.values() if s['status'] == 'completed')
        error_count = sum(1 for s in self.states.values() if s['status'] == 'error')
        total_duration = time.time() - self.start_time

        self.console.print(f"\nTotal Duration: {str(timedelta(seconds=int(total_duration)))}", style="bold")
        self.console.print(f"Collectors: {completed_count}/{len(self.collectors)} successful", style="green")
        self.console.print(f"Total Samples: {total_collected:,}/{total_target:,} ({(total_collected/total_target*100):.1f}%)", style="cyan")

        if error_count > 0:
            self.console.print(f"Errors: {error_count}", style="red")

        # By collector
        self.console.print("\n" + "-"*70, style="dim")
        self.console.print("By Collector:", style="bold")
        self.console.print("-"*70, style="dim")

        for name, state in self.states.items():
            if state['status'] == 'completed':
                icon = "[+]"
                style = "green"
            elif state['status'] == 'error':
                icon = "[X]"
                style = "red"
            else:
                icon = "[!]"
                style = "yellow"

            self.console.print(f"\n{icon} {name.upper()}", style=f"bold {style}")
            self.console.print(f"  Samples: {state['samples_collected']:,}/{state['target_samples']:,}")

            if state['start_time']:
                duration = (state['end_time'] or time.time()) - state['start_time']
                self.console.print(f"  Duration: {int(duration)}s")

            if state['error']:
                self.console.print(f"  Error: {state['error']}", style="red")

        self.console.print("\n" + "="*70 + "\n", style="bold")


class SimpleDashboard:
    """
    Simple text-based dashboard for systems without Rich.

    Fallback dashboard that works with basic console output.
    """

    def __init__(self, collectors: List[str]):
        """
        Initialize simple dashboard.

        Args:
            collectors: List of collector names
        """
        self.collectors = collectors
        self.states = {
            name: {
                'status': 'pending',
                'samples_collected': 0,
                'target_samples': 0,
                'message': 'Waiting...'
            }
            for name in collectors
        }
        self.start_time = time.time()

    def update(self, collector: str, update_data: Dict):
        """Update collector status."""
        if collector not in self.states:
            return

        self.states[collector].update(update_data)

        # Print update
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = update_data.get('status', '')
        message = update_data.get('message', '')

        status_icons = {
            'starting': '[>]',
            'running': '[*]',
            'completed': '[+]',
            'error': '[X]'
        }
        icon = status_icons.get(status, '[-]')

        print(f"[{timestamp}] {icon} {collector.upper()}: {message}")

    def start(self):
        """Start dashboard."""
        print("\n" + "="*70)
        print("StreamGuard Data Collection - Progress Monitor")
        print("="*70 + "\n")

    def stop(self):
        """Stop dashboard."""
        pass

    def refresh(self):
        """Refresh dashboard (no-op for simple)."""
        pass

    def print_final_summary(self):
        """Print final summary."""
        print("\n" + "="*70)
        print("COLLECTION COMPLETE")
        print("="*70)

        total_collected = sum(s['samples_collected'] for s in self.states.values())
        total_target = sum(s['target_samples'] for s in self.states.values())
        duration = time.time() - self.start_time

        print(f"\nDuration: {int(duration)}s")
        print(f"Total Samples: {total_collected:,}/{total_target:,}")

        print("\nBy Collector:")
        for name, state in self.states.items():
            status = state['status']
            icon = "[+]" if status == 'completed' else "[X]" if status == 'error' else "[-]"
            print(f"  {icon} {name}: {state['samples_collected']:,}/{state['target_samples']:,}")

        print("\n" + "="*70 + "\n")


def create_dashboard(collectors: List[str], use_rich: bool = True) -> 'ProgressDashboard | SimpleDashboard':
    """
    Create appropriate dashboard based on availability.

    Args:
        collectors: List of collector names
        use_rich: Try to use Rich library if available

    Returns:
        Dashboard instance (Rich or Simple)
    """
    if use_rich and RICH_AVAILABLE:
        try:
            return ProgressDashboard(collectors)
        except Exception as e:
            print(f"[!] Could not create Rich dashboard: {e}")
            print("Falling back to simple dashboard...")

    return SimpleDashboard(collectors)


if __name__ == '__main__':
    # Test dashboard
    import random

    collectors = ['cve', 'github', 'repo', 'synthetic']
    dashboard = create_dashboard(collectors, use_rich=True)

    dashboard.start()

    # Simulate collection
    for i in range(20):
        for collector in collectors:
            progress = random.randint(i*5, min((i+1)*5, 100))
            dashboard.update(collector, {
                'status': 'running' if progress < 100 else 'completed',
                'samples_collected': progress * 100,
                'target_samples': 10000,
                'message': f'Collecting samples... {progress}%'
            })

        dashboard.refresh()
        time.sleep(0.5)

    dashboard.stop()
    dashboard.print_final_summary()
