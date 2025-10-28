"""
Checkpoint Manager for StreamGuard Data Collection.

Provides checkpoint save/load functionality to enable pause/resume of long-running
data collection operations. Supports graceful recovery from interruptions including
laptop shutdown, sleep, or process termination.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

# Platform-specific imports for file locking
if os.name == 'nt':
    # Windows
    import msvcrt
else:
    # Unix/Linux/Mac
    import fcntl


def _lock_file(file_obj, exclusive=True):
    """Lock a file object (platform-independent)."""
    if os.name == 'nt':
        # Windows locking
        try:
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBLCK, 1)
        except:
            pass  # Non-blocking, ignore if can't lock
    else:
        # Unix locking
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


def _unlock_file(file_obj):
    """Unlock a file object (platform-independent)."""
    if os.name == 'nt':
        # Windows unlocking
        try:
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        except:
            pass
    else:
        # Unix unlocking
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


class CheckpointManager:
    """Manages checkpoint save/load operations for data collectors."""

    def __init__(self, checkpoint_dir: str = "data/raw/checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        collector_name: str,
        state: Dict[str, Any],
        samples: List[Dict]
    ) -> str:
        """
        Save checkpoint for a collector.

        Args:
            collector_name: Name of the collector
            state: Current state information (progress, config, etc.)
            samples: Collected samples so far

        Returns:
            Path to checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"{collector_name}_checkpoint.json"

        checkpoint_data = {
            "collector": collector_name,
            "timestamp": timestamp,
            "state": state,
            "samples_count": len(samples),
            "samples": samples
        }

        # Write checkpoint atomically with file locking
        temp_file = checkpoint_file.with_suffix('.tmp')

        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                # Acquire exclusive lock
                _lock_file(f, exclusive=True)
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                _unlock_file(f)

            # Atomic rename
            temp_file.replace(checkpoint_file)

            return str(checkpoint_file)

        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise Exception(f"Failed to save checkpoint: {str(e)}")

    def load_checkpoint(self, collector_name: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for a collector.

        Args:
            collector_name: Name of the collector

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        checkpoint_file = self.checkpoint_dir / f"{collector_name}_checkpoint.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                # Acquire shared lock for reading
                _lock_file(f, exclusive=False)
                checkpoint_data = json.load(f)
                _unlock_file(f)

            return checkpoint_data

        except Exception as e:
            print(f"WARNING: Failed to load checkpoint for {collector_name}: {str(e)}")
            return None

    def checkpoint_exists(self, collector_name: str) -> bool:
        """
        Check if checkpoint exists for a collector.

        Args:
            collector_name: Name of the collector

        Returns:
            True if checkpoint exists
        """
        checkpoint_file = self.checkpoint_dir / f"{collector_name}_checkpoint.json"
        return checkpoint_file.exists()

    def delete_checkpoint(self, collector_name: str) -> bool:
        """
        Delete checkpoint for a collector.

        Args:
            collector_name: Name of the collector

        Returns:
            True if checkpoint was deleted
        """
        checkpoint_file = self.checkpoint_dir / f"{collector_name}_checkpoint.json"

        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                return True
            except Exception as e:
                print(f"WARNING: Failed to delete checkpoint for {collector_name}: {str(e)}")
                return False

        return False

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint information
        """
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    checkpoints.append({
                        "collector": data.get("collector"),
                        "timestamp": data.get("timestamp"),
                        "samples_count": data.get("samples_count"),
                        "file": str(checkpoint_file)
                    })
            except Exception as e:
                print(f"WARNING: Failed to read checkpoint {checkpoint_file}: {str(e)}")
                continue

        return checkpoints

    def save_orchestrator_checkpoint(
        self,
        collectors_state: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Save orchestrator-level checkpoint (tracks all collectors).

        Args:
            collectors_state: State information for all collectors

        Returns:
            Path to checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / "orchestrator_checkpoint.json"

        checkpoint_data = {
            "timestamp": timestamp,
            "collectors": collectors_state
        }

        # Write checkpoint atomically
        temp_file = checkpoint_file.with_suffix('.tmp')

        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                _lock_file(f, exclusive=True)
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                _unlock_file(f)

            temp_file.replace(checkpoint_file)

            return str(checkpoint_file)

        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise Exception(f"Failed to save orchestrator checkpoint: {str(e)}")

    def load_orchestrator_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load orchestrator-level checkpoint.

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        checkpoint_file = self.checkpoint_dir / "orchestrator_checkpoint.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                _lock_file(f, exclusive=False)
                checkpoint_data = json.load(f)
                _unlock_file(f)

            return checkpoint_data

        except Exception as e:
            print(f"WARNING: Failed to load orchestrator checkpoint: {str(e)}")
            return None

    def delete_orchestrator_checkpoint(self) -> bool:
        """
        Delete orchestrator-level checkpoint.

        Returns:
            True if checkpoint was deleted
        """
        checkpoint_file = self.checkpoint_dir / "orchestrator_checkpoint.json"

        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                return True
            except Exception as e:
                print(f"WARNING: Failed to delete orchestrator checkpoint: {str(e)}")
                return False

        return False

    def get_checkpoint_info(self, collector_name: str) -> Optional[Dict[str, Any]]:
        """
        Get checkpoint information without loading full data.

        Args:
            collector_name: Name of the collector

        Returns:
            Checkpoint metadata or None
        """
        checkpoint_file = self.checkpoint_dir / f"{collector_name}_checkpoint.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    "collector": data.get("collector"),
                    "timestamp": data.get("timestamp"),
                    "samples_count": data.get("samples_count"),
                    "state": data.get("state", {})
                }
        except Exception as e:
            print(f"WARNING: Failed to get checkpoint info for {collector_name}: {str(e)}")
            return None
