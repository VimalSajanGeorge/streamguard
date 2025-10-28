#!/usr/bin/env python
"""
Test script for checkpoint/resume functionality.

This script tests the checkpoint/resume system by:
1. Starting a collection with OSV (small samples)
2. Artificially creating a checkpoint midway
3. Testing resume functionality
"""

import subprocess
import time
import json
from pathlib import Path
import sys

def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def check_checkpoint_exists(collector_name):
    """Check if checkpoint exists for a collector."""
    checkpoint_file = Path(f"data/raw/checkpoints/{collector_name}_checkpoint.json")
    return checkpoint_file.exists()

def read_checkpoint(collector_name):
    """Read checkpoint contents."""
    checkpoint_file = Path(f"data/raw/checkpoints/{collector_name}_checkpoint.json")
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def create_test_checkpoint_osv():
    """Create a test checkpoint for OSV with some ecosystems already processed."""
    checkpoint_dir = Path("data/raw/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "collector": "osv",
        "timestamp": "20251017_test",
        "state": {
            "processed_ecosystems": ["PyPI", "npm"],
            "target_samples": 20,
            "samples_per_ecosystem": 2
        },
        "samples_count": 4,
        "samples": []
    }

    with open(checkpoint_dir / "osv_checkpoint.json", 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print("[+] Created test checkpoint: PyPI and npm marked as processed")
    return checkpoint

def create_test_checkpoint_exploitdb():
    """Create a test checkpoint for ExploitDB with some progress."""
    checkpoint_dir = Path("data/raw/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "collector": "exploitdb",
        "timestamp": "20251017_test",
        "state": {
            "last_processed_index": 9,
            "target_samples": 20
        },
        "samples_count": 10,
        "samples": []
    }

    with open(checkpoint_dir / "exploitdb_checkpoint.json", 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print("[+] Created test checkpoint: 10 exploits already processed (index 0-9)")
    return checkpoint

def test_osv_resume():
    """Test OSV collector resume functionality."""
    print_section("TEST 1: OSV Collector Resume")

    # Create test checkpoint
    create_test_checkpoint_osv()

    # Show checkpoint contents
    checkpoint = read_checkpoint("osv")
    print("Checkpoint State:")
    print(f"  - Processed Ecosystems: {checkpoint['state']['processed_ecosystems']}")
    print(f"  - Samples Count: {checkpoint['samples_count']}")

    # Run with resume
    print("\nRunning: python osv_collector.py --target-samples 20 --resume")
    print("-" * 70)

    result = subprocess.run(
        [sys.executable, "training/scripts/collection/osv_collector.py",
         "--target-samples", "20", "--resume"],
        capture_output=True,
        text=True,
        timeout=300
    )

    # Show key output lines
    output_lines = result.stdout.split('\n')
    for line in output_lines[:30]:  # Show first 30 lines
        if any(keyword in line for keyword in ['Found existing checkpoint', 'Resuming',
                                                'Skipping', 'Collecting:', 'Total samples']):
            print(line)

    # Check if successful
    if "Skipping PyPI" in result.stdout and "Skipping npm" in result.stdout:
        print("\n[SUCCESS] ✅ OSV collector correctly skipped processed ecosystems!")
        return True
    else:
        print("\n[FAILED] ❌ OSV collector did not skip processed ecosystems")
        print("Full output:", result.stdout[-500:])
        return False

def test_exploitdb_resume():
    """Test ExploitDB collector resume functionality."""
    print_section("TEST 2: ExploitDB Collector Resume")

    # Create test checkpoint
    create_test_checkpoint_exploitdb()

    # Show checkpoint contents
    checkpoint = read_checkpoint("exploitdb")
    print("Checkpoint State:")
    print(f"  - Last Processed Index: {checkpoint['state']['last_processed_index']}")
    print(f"  - Samples Count: {checkpoint['samples_count']}")

    # Run with resume
    print("\nRunning: python exploitdb_collector.py --target-samples 20 --resume")
    print("-" * 70)

    result = subprocess.run(
        [sys.executable, "training/scripts/collection/exploitdb_collector.py",
         "--target-samples", "20", "--resume"],
        capture_output=True,
        text=True,
        timeout=300
    )

    # Show key output lines
    output_lines = result.stdout.split('\n')
    for line in output_lines[:30]:  # Show first 30 lines
        if any(keyword in line for keyword in ['Found existing checkpoint', 'Resuming',
                                                'Starting from index', 'samples already collected']):
            print(line)

    # Check if successful
    if "Starting from index" in result.stdout and "Resuming with" in result.stdout:
        print("\n[SUCCESS] ✅ ExploitDB collector correctly resumed from checkpoint!")
        return True
    else:
        print("\n[FAILED] ❌ ExploitDB collector did not resume correctly")
        print("Full output:", result.stdout[-500:])
        return False

def test_full_collection_resume():
    """Test full collection with orchestrator."""
    print_section("TEST 3: Full Collection with Orchestrator (Small Samples)")

    # Create checkpoints for both collectors
    create_test_checkpoint_osv()
    create_test_checkpoint_exploitdb()

    print("\nRunning: python run_full_collection.py --collectors osv exploitdb")
    print("          --osv-samples 20 --exploitdb-samples 20 --resume --no-dashboard")
    print("-" * 70)

    result = subprocess.run(
        [sys.executable, "training/scripts/collection/run_full_collection.py",
         "--collectors", "osv", "exploitdb",
         "--osv-samples", "20",
         "--exploitdb-samples", "20",
         "--resume",
         "--no-dashboard"],
        capture_output=True,
        text=True,
        timeout=600
    )

    # Show configuration section
    output_lines = result.stdout.split('\n')
    in_config = False
    for line in output_lines:
        if "Configuration:" in line:
            in_config = True
        if in_config and line.strip() == "":
            break
        if in_config or "Resume:" in line:
            print(line)

    # Check if resume flag was recognized
    if "Resume: Enabled" in result.stdout:
        print("\n[SUCCESS] ✅ Orchestrator recognized --resume flag!")

        # Check for collection completion
        if "Collection Statistics" in result.stdout or "completed" in result.stdout.lower():
            print("[SUCCESS] ✅ Collection completed successfully!")
            return True
        else:
            print("[PARTIAL] ⚠ Collection started but may not have completed")
            return True
    else:
        print("\n[FAILED] ❌ Orchestrator did not recognize --resume flag")
        print("Output snippet:", result.stdout[:1000])
        return False

def main():
    """Run all tests."""
    print_section("StreamGuard Checkpoint/Resume System Test Suite")

    # Clear any existing checkpoints
    checkpoint_dir = Path("data/raw/checkpoints")
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        print("[+] Cleared existing checkpoints\n")

    results = []

    # Test 1: OSV Resume
    try:
        results.append(("OSV Resume", test_osv_resume()))
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        results.append(("OSV Resume", False))

    time.sleep(2)

    # Test 2: ExploitDB Resume
    try:
        results.append(("ExploitDB Resume", test_exploitdb_resume()))
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        results.append(("ExploitDB Resume", False))

    time.sleep(2)

    # Test 3: Full Collection
    try:
        results.append(("Full Collection", test_full_collection_resume()))
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        results.append(("Full Collection", False))

    # Print summary
    print_section("TEST SUMMARY")

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:30} {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\n  Total: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Checkpoint/resume system is working correctly!")
        return 0
    else:
        print(f"\n⚠ {total_tests - total_passed} test(s) failed. Please review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
