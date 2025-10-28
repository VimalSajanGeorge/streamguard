"""Check if CVE collector setup is ready.

This script verifies all dependencies, permissions, and configurations
needed to run the Enhanced CVE Collector.
"""

import sys
import os
from pathlib import Path

# Fix Unicode output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_dependencies():
    """Check required Python packages."""
    print("\nChecking dependencies...")
    required = [
        'requests',
        'pytest',
    ]

    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            all_ok = False

    return all_ok


def check_modules():
    """Check if collector modules can be imported."""
    print("\nChecking collector modules...")

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from base_collector import BaseCollector
        print("  ✓ base_collector.py")
    except Exception as e:
        print(f"  ✗ base_collector.py ({e})")
        return False

    try:
        from cve_collector_enhanced import CVECollectorEnhanced
        print("  ✓ cve_collector_enhanced.py")
    except Exception as e:
        print(f"  ✗ cve_collector_enhanced.py ({e})")
        return False

    try:
        from cve_config import get_config
        print("  ✓ cve_config.py")
    except Exception as e:
        print(f"  ✗ cve_config.py ({e})")
        return False

    return True


def check_output_directory():
    """Check output directory permissions."""
    print("\nChecking output directory...")

    # Get the project root (4 levels up from this file)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    output_dir = project_root / "data" / "raw" / "cves"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Output directory: {output_dir}")

        # Test write permission
        test_file = output_dir / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
        print(f"  ✓ Write permission OK")

        return True
    except Exception as e:
        print(f"  ✗ Cannot create/write to output directory: {e}")
        return False


def check_network():
    """Check network connectivity to APIs."""
    print("\nChecking network connectivity...")

    import requests

    # Check NVD API
    try:
        response = requests.get(
            "https://services.nvd.nist.gov/rest/json/cves/2.0",
            params={'resultsPerPage': 1},
            timeout=10
        )
        if response.status_code == 200:
            print("  ✓ NVD API accessible")
            nvd_ok = True
        else:
            print(f"  ✗ NVD API returned status {response.status_code}")
            nvd_ok = False
    except Exception as e:
        print(f"  ✗ Cannot reach NVD API: {e}")
        nvd_ok = False

    # Check GitHub API
    try:
        response = requests.get(
            "https://api.github.com",
            timeout=10
        )
        if response.status_code == 200:
            print("  ✓ GitHub API accessible")
            github_ok = True
        else:
            print(f"  ✗ GitHub API returned status {response.status_code}")
            github_ok = False
    except Exception as e:
        print(f"  ✗ Cannot reach GitHub API: {e}")
        github_ok = False

    return nvd_ok and github_ok


def check_github_token():
    """Check for GitHub token."""
    print("\nChecking GitHub token...")

    token = os.environ.get('GITHUB_TOKEN')

    if token:
        print(f"  ✓ GitHub token found (length: {len(token)})")

        # Verify token works
        try:
            import requests
            response = requests.get(
                "https://api.github.com/user",
                headers={'Authorization': f'token {token}'},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Token valid (user: {data.get('login', 'unknown')})")
                print(f"  ✓ Rate limit: {response.headers.get('X-RateLimit-Limit', 'unknown')}/hour")
                return True
            else:
                print(f"  ⚠ Token may be invalid (status: {response.status_code})")
                return False
        except Exception as e:
            print(f"  ⚠ Could not verify token: {e}")
            return False
    else:
        print("  ⚠ No GitHub token found (will use lower rate limits)")
        print("    Set token with: export GITHUB_TOKEN='your_token_here'")
        print("    Or pass to collector: --github-token YOUR_TOKEN")
        return False


def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent

    try:
        import shutil
        total, used, free = shutil.disk_usage(project_root)

        free_gb = free // (2**30)
        print(f"  ✓ Free space: {free_gb} GB")

        if free_gb < 1:
            print("  ⚠ Low disk space (< 1 GB)")
            return False
        elif free_gb < 5:
            print("  ⚠ Disk space is limited (< 5 GB)")

        return True
    except Exception as e:
        print(f"  ⚠ Could not check disk space: {e}")
        return True  # Don't fail on this


def check_existing_data():
    """Check for existing collected data."""
    print("\nChecking existing data...")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    data_file = project_root / "data" / "raw" / "cves" / "cve_data.jsonl"

    if data_file.exists():
        import json

        count = 0
        with_code = 0

        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    count += 1
                    sample = json.loads(line)
                    if sample.get('vulnerable_code'):
                        with_code += 1

            print(f"  ℹ Found existing data: {count} samples")
            print(f"    With code: {with_code} ({with_code/count*100:.1f}%)")

        except Exception as e:
            print(f"  ⚠ Could not read existing data: {e}")

    else:
        print("  ℹ No existing data found")

    return True


def print_summary(results):
    """Print summary of checks."""
    print("\n" + "="*80)
    print("SETUP CHECK SUMMARY")
    print("="*80)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")

    print("="*80)

    if all_passed:
        print("\n✓ All checks passed! Ready to collect CVE data.")
        print("\nTo start collection:")
        print("  python cve_collector_enhanced.py")
        print("\nOr with GitHub token:")
        print("  python cve_collector_enhanced.py --github-token YOUR_TOKEN")
        print("\nFor examples:")
        print("  python example_cve_usage.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above before running the collector.")

    return all_passed


def main():
    """Run all checks."""
    print("="*80)
    print("CVE COLLECTOR SETUP CHECK")
    print("="*80)

    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Modules': check_modules(),
        'Output Directory': check_output_directory(),
        'Network': check_network(),
        'Disk Space': check_disk_space(),
    }

    # Optional checks (don't affect pass/fail)
    check_github_token()
    check_existing_data()

    # Print summary
    success = print_summary(results)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
