"""
Pre-Collection Validation Script for StreamGuard Data Collection

Validates environment setup before running data collection:
- API keys are configured
- Dependencies are installed
- Sufficient disk space available
- Git is installed
- Network connectivity to APIs
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def _supports_unicode_output() -> bool:
    """Best-effort check whether stdout can print unicode symbols on this terminal."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "✓✗⚠❌".encode(enc)
        return True
    except Exception:
        return False


_UNICODE = _supports_unicode_output()
OK = "✓" if _UNICODE else "OK"
FAIL = "✗" if _UNICODE else "X"
WARN = "⚠" if _UNICODE else "!"
ERROR = "❌" if _UNICODE else "ERROR"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def _load_dotenv_repo_root() -> None:
    """
    Load `.env` reliably even when running from subfolders.

    `load_dotenv()` without a path only checks the current working directory,
    which fails if the user runs scripts from `training/scripts/collection`.
    """
    try:
        here = Path(__file__).resolve()
        candidates = [
            Path.cwd() / ".env",
            here.parents[3] / ".env",  # repo root (collection -> scripts -> training -> root)
        ]
        for p in candidates:
            if p.exists():
                load_dotenv(dotenv_path=p, override=False)
                return
        load_dotenv(override=False)
    except Exception:
        # Never fail pre-check due to dotenv loading
        load_dotenv(override=False)


class PreCollectionValidator:
    """Validates environment before data collection."""
    
    REQUIRED_DISK_SPACE_GB = 20  # Minimum disk space required
    REQUIRED_DEPENDENCIES = [
        'git',
        'requests',
        'python-dotenv',
        'rich',
        'tqdm',
        'gitpython'
    ]
    
    API_ENDPOINTS = {
        'nvd': 'https://services.nvd.nist.gov/rest/json/cves/2.0',
        'github': 'https://api.github.com',
        'osv': 'https://api.osv.dev/v1',
        'exploitdb': 'https://gitlab.com/exploit-database/exploitdb'
    }
    
    def __init__(self):
        """Initialize validator."""
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_failed = 0
        
        # Load environment variables
        _load_dotenv_repo_root()
    
    def check_api_keys(self) -> bool:
        """Check if required API keys are configured."""
        print("\n[1/7] Checking API Keys...")
        
        nvd_key = os.getenv('NVD_API_KEY')
        github_token = os.getenv('GITHUB_TOKEN') or os.getenv('GITHUB_TOKENS')
        
        if not nvd_key or nvd_key == 'your_nvd_api_key_here':
            self.errors.append("NVD_API_KEY not configured in .env file")
            print(f"  {FAIL} NVD_API_KEY: NOT FOUND")
            return False
        else:
            print(f"  {OK} NVD_API_KEY: Found ({nvd_key[:8]}...)")
        
        if not github_token or github_token == 'your_github_token_here':
            self.errors.append("GitHub token not configured in .env file (GITHUB_TOKEN or GITHUB_TOKENS)")
            print(f"  {FAIL} GitHub token: NOT FOUND")
            return False
        else:
            token_preview = github_token.split(",")[0].strip()
            print(f"  {OK} GitHub token: Found ({token_preview[:8]}...)")
        
        print(f"  {OK} All required API keys configured")
        return True
    
    def check_dependencies(self) -> bool:
        """Check if required Python packages are installed."""
        print("\n[2/7] Checking Python Dependencies...")
        
        missing = []
        for dep in self.REQUIRED_DEPENDENCIES:
            try:
                if dep == 'gitpython':
                    import git
                elif dep == 'python-dotenv':
                    import dotenv
                else:
                    __import__(dep)
                print(f"  {OK} {dep}: Installed")
            except ImportError:
                missing.append(dep)
                print(f"  {FAIL} {dep}: NOT INSTALLED")
        
        if missing:
            self.errors.append(f"Missing dependencies: {', '.join(missing)}")
            print(f"\n  Install with: pip install {' '.join(missing)}")
            return False
        
        print(f"  {OK} All dependencies installed")
        return True
    
    def check_git_installed(self) -> bool:
        """Check if Git is installed."""
        print("\n[3/7] Checking Git Installation...")
        
        try:
            result = subprocess.run(
                ['git', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"  {OK} Git installed: {version}")
                return True
            else:
                self.errors.append("Git is not installed or not in PATH")
                print(f"  {FAIL} Git: NOT FOUND")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.errors.append("Git is not installed or not in PATH")
            print(f"  {FAIL} Git: NOT FOUND")
            return False
    
    def check_disk_space(self) -> bool:
        """Check if sufficient disk space is available."""
        print("\n[4/7] Checking Disk Space...")
        
        try:
            # Get disk usage for current directory
            usage = shutil.disk_usage('.')
            free_gb = usage.free / (1024**3)
            
            print(f"  Available: {free_gb:.2f} GB")
            print(f"  Required: {self.REQUIRED_DISK_SPACE_GB} GB")
            
            if free_gb < self.REQUIRED_DISK_SPACE_GB:
                self.errors.append(
                    f"Insufficient disk space: {free_gb:.2f} GB available, "
                    f"{self.REQUIRED_DISK_SPACE_GB} GB required"
                )
                print(f"  {FAIL} Insufficient disk space")
                return False
            else:
                print(f"  {OK} Sufficient disk space available")
                return True
        except Exception as e:
            self.warnings.append(f"Could not check disk space: {e}")
            print(f"  {WARN} Could not check disk space: {e}")
            return True  # Don't fail on this
    
    def check_network_connectivity(self) -> bool:
        """Check network connectivity to required APIs."""
        print("\n[5/7] Checking Network Connectivity...")
        
        all_reachable = True
        for name, endpoint in self.API_ENDPOINTS.items():
            try:
                # OSV's base `/v1` returns 404 for GET; use the correct endpoint.
                if name == "osv":
                    response = requests.post(
                        f"{endpoint}/query",
                        json={"package": {"ecosystem": "PyPI", "name": "django"}},
                        timeout=10,
                    )
                else:
                    response = requests.get(endpoint, timeout=5)
                # 200=OK, 401/403=Auth required (API exists), 404=Resource not found (API responding)
                if response.status_code in [200, 401, 403, 404]:
                    print(f"  {OK} {name}: Reachable")
                else:
                    self.warnings.append(f"{name} API returned status {response.status_code}")
                    print(f"  {WARN} {name}: Status {response.status_code}")
            except requests.RequestException as e:
                self.warnings.append(f"Could not reach {name} API: {e}")
                print(f"  {WARN} {name}: Not reachable ({str(e)[:50]}...)")
                all_reachable = False
        
        if all_reachable:
            print(f"  {OK} All APIs reachable")
        else:
            print(f"  {WARN} Some APIs not reachable (may work during collection)")
        
        return True  # Don't fail on network issues
    
    def check_output_directories(self) -> bool:
        """Check if output directories can be created."""
        print("\n[6/7] Checking Output Directories...")
        
        base_dir = PROJECT_ROOT / "data" / "raw"
        collectors = ['cves', 'github', 'opensource', 'osv', 'exploitdb', 'synthetic']
        
        try:
            for collector in collectors:
                output_dir = base_dir / collector
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"  {OK} {collector}: {output_dir}")
            
            print(f"  {OK} All output directories ready")
            return True
        except Exception as e:
            self.errors.append(f"Could not create output directories: {e}")
            print(f"  {FAIL} Error creating directories: {e}")
            return False
    
    def check_existing_data(self) -> bool:
        """Check for existing data files."""
        print("\n[7/7] Checking for Existing Data...")
        
        base_dir = PROJECT_ROOT / "data" / "raw"
        collectors = ['cves', 'github', 'opensource', 'osv', 'exploitdb', 'synthetic']
        
        existing_files = []
        for collector in collectors:
            output_dir = base_dir / collector
            if output_dir.exists():
                jsonl_files = list(output_dir.glob('*.jsonl'))
                if jsonl_files:
                    total_size = sum(f.stat().st_size for f in jsonl_files)
                    existing_files.append((collector, len(jsonl_files), total_size))
        
        if existing_files:
            print(f"  {WARN} Found existing data files:")
            for collector, count, size in existing_files:
                size_mb = size / (1024**2)
                print(f"    - {collector}: {count} file(s), {size_mb:.2f} MB")
            print("\n  Note: Existing files will be overwritten during collection")
        else:
            print(f"  {OK} No existing data files found")
        
        return True
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("="*70)
        print("StreamGuard Pre-Collection Validation")
        print("="*70)
        
        checks = [
            self.check_api_keys,
            self.check_dependencies,
            self.check_git_installed,
            self.check_disk_space,
            self.check_network_connectivity,
            self.check_output_directories,
            self.check_existing_data
        ]
        
        for check in checks:
            try:
                if check():
                    self.checks_passed += 1
                else:
                    self.checks_failed += 1
            except Exception as e:
                self.checks_failed += 1
                self.errors.append(f"Check failed with exception: {e}")
                print(f"  {FAIL} Exception: {e}")
        
        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Checks passed: {self.checks_passed}/{len(checks)}")
        print(f"Checks failed: {self.checks_failed}/{len(checks)}")
        
        if self.errors:
            print(f"\n{ERROR} ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n{WARN} WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        print("\n" + "="*70)
        
        if self.checks_failed == 0:
            print(f"{OK} All checks passed! Ready to start data collection.")
            print("\nNext step:")
            print("  python master_orchestrator.py --collectors cve github repo synthetic osv exploitdb --parallel")
            return True
        else:
            print(f"{FAIL} Some checks failed. Please fix the errors above before proceeding.")
            return False


def main():
    """Main entry point."""
    validator = PreCollectionValidator()
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
