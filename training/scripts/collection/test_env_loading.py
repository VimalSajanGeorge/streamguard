#!/usr/bin/env python
"""
Test script to verify .env loading works in both parent and child processes.
"""

import os
import sys
from pathlib import Path
from multiprocessing import Process, Queue
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def subprocess_check_with_load(queue: Queue):
    """Function that runs in subprocess to check env vars WITH load_dotenv()."""
    # Load .env in subprocess
    load_dotenv()

    github_token = os.getenv("GITHUB_TOKEN")

    result = {
        'subprocess': True,
        'github_token_present': github_token is not None,
        'token_preview': github_token[:10] + '...' if github_token else None
    }

    queue.put(result)


def subprocess_check_no_load(queue: Queue):
    """Function that runs in subprocess to check env vars WITHOUT load_dotenv()."""
    github_token = os.getenv("GITHUB_TOKEN")

    result = {
        'subprocess': True,
        'github_token_present': github_token is not None,
        'token_preview': github_token[:10] + '...' if github_token else None
    }

    queue.put(result)


def main():
    """Test environment variable loading."""
    print("="*70)
    print("Environment Variable Loading Test")
    print("="*70)

    # Test 1: Parent process
    print("\n[Test 1] Parent process BEFORE load_dotenv():")
    token_before = os.getenv("GITHUB_TOKEN")
    print(f"  GITHUB_TOKEN present: {token_before is not None}")
    if token_before:
        print(f"  Token preview: {token_before[:10]}...")

    # Load .env in parent
    load_dotenv()

    print("\n[Test 2] Parent process AFTER load_dotenv():")
    token_after = os.getenv("GITHUB_TOKEN")
    print(f"  GITHUB_TOKEN present: {token_after is not None}")
    if token_after:
        print(f"  Token preview: {token_after[:10]}...")

    # Test 3: Subprocess without load_dotenv()
    print("\n[Test 3] Subprocess WITHOUT load_dotenv():")
    queue1 = Queue()

    p1 = Process(target=subprocess_check_no_load, args=(queue1,))
    p1.start()
    p1.join()

    result1 = queue1.get()
    print(f"  GITHUB_TOKEN present: {result1['github_token_present']}")
    if result1['token_preview']:
        print(f"  Token preview: {result1['token_preview']}")

    # Test 4: Subprocess WITH load_dotenv()
    print("\n[Test 4] Subprocess WITH load_dotenv():")
    queue2 = Queue()

    p2 = Process(target=subprocess_check_with_load, args=(queue2,))
    p2.start()
    p2.join()

    result2 = queue2.get()
    print(f"  GITHUB_TOKEN present: {result2['github_token_present']}")
    if result2['token_preview']:
        print(f"  Token preview: {result2['token_preview']}")

    # Summary
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)

    if not token_after:
        print("[FAILED] .env not loaded in parent process")
        print("   Action: Check if .env file exists in project root")
        return 1

    if not result2['github_token_present']:
        print("[FAILED] .env not loaded in subprocess")
        print("   Action: Ensure load_dotenv() is called in subprocess")
        return 1

    print("[SUCCESS] Environment variables loaded correctly!")
    print("   - Parent process: OK")
    print("   - Subprocess: OK")

    # Note about Test 3
    print("\nNote: Test 3 shows token present even without load_dotenv() in subprocess")
    print("This is because on Windows, child processes inherit parent env vars.")
    print("However, load_dotenv() in subprocess ensures .env is always loaded.")

    # Check .env file location
    env_file = Path.cwd() / '.env'
    print(f"\n.env file location: {env_file}")
    print(f".env file exists: {env_file.exists()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
