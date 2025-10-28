"""Verify StreamGuard v3.0 setup."""

import sys
import subprocess
from importlib import import_module
from pathlib import Path
import requests

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def check_python_packages():
    """Check Python packages."""
    required = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('captum', 'Captum (Explainability)'),
        ('neo4j', 'Neo4j Driver'),
        ('fastapi', 'FastAPI'),
        ('z3', 'Z3 Solver')
        # Note: angr removed due to Python 3.12 compatibility issues
    ]

    print("📦 Python Packages:")
    all_ok = True
    for package, name in required:
        try:
            mod = import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✅ {name}: {version}")
        except ImportError:
            print(f"  ❌ {name}: not installed")
            all_ok = False

    return all_ok

def check_neo4j():
    """Check Neo4j connection."""
    print("\n🔗 Neo4j:")
    try:
        response = requests.get("http://localhost:7474", timeout=5)
        if response.status_code == 200:
            print("  ✅ Neo4j running on http://localhost:7474")
            return True
        else:
            print("  ❌ Neo4j not responding correctly")
            return False
    except:
        print("  ❌ Neo4j not running")
        return False

def check_docker():
    """Check Docker."""
    print("\n🐳 Docker:")
    try:
        result = subprocess.run(
            ['docker', 'ps'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✅ Docker is running")
            return True
        else:
            print("  ❌ Docker not running")
            return False
    except:
        print("  ❌ Docker not found")
        return False

def check_directory_structure():
    """Check project structure."""
    print("\n📁 Project Structure:")
    required_dirs = [
        'core/agents',
        'core/explainability',
        'core/graph',
        'core/feedback',
        'core/verification',
        'dashboard',
        'training',
        'data/raw',
        'tests/unit',
        '.claude/agents'
    ]

    all_ok = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ (missing)")
            all_ok = False

    return all_ok

def check_aws_credentials():
    """Check AWS configuration."""
    print("\n☁️  AWS Configuration:")
    try:
        result = subprocess.run(
            ['aws', 'sts', 'get-caller-identity'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✅ AWS credentials configured")
            return True
        else:
            print("  ❌ AWS credentials not configured")
            return False
    except:
        print("  ❌ AWS CLI not found")
        return False

def check_data_collection():
    """Check if data has been collected."""
    print("\n📊 Data Collection:")
    data_files = [
        'data/raw/cves/cve_data.jsonl',
        'data/raw/synthetic/synthetic_data.jsonl'
    ]

    collected = 0
    for file_path in data_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"  ✅ {file_path} ({size // 1024} KB)")
            collected += 1
        else:
            print(f"  ⚠️  {file_path} (not collected yet)")

    return collected > 0

def main():
    """Run all checks."""
    print("🔍 Verifying StreamGuard v3.0 Setup\n")
    print("="*60)

    checks = [
        check_python_packages(),
        check_neo4j(),
        check_docker(),
        check_directory_structure(),
        check_aws_credentials(),
        check_data_collection()
    ]

    print("\n" + "="*60)

    if all(checks):
        print("\n✅ All checks passed! Environment is ready.")
        print("\nNext steps:")
        print("  1. Review docs/CLAUDE.md")
        print("  2. Start with: claude --plan 'Begin Phase 1: ML Training'")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        print("\nRefer to docs/01_setup.md for troubleshooting.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
