"""
Pre-Collection Comprehensive Test Suite for StreamGuard

Tests all components before running full data collection:
1. Environment configuration
2. API connectivity
3. Module imports
4. Unit tests for each collector
5. Edge case handling
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PreCollectionTester:
    """Comprehensive pre-collection test suite."""
    
    def __init__(self):
        self.results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'tests': []
        }
        self.github_tokens = []
        self.nvd_api_key = None
    
    def log_test(self, name, status, message, details=None):
        """Log test result."""
        result = {
            'name': name,
            'status': status,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.results['tests'].append(result)
        
        if status == 'PASS':
            self.results['passed'] += 1
            icon = '[OK]'
        elif status == 'FAIL':
            self.results['failed'] += 1
            icon = '[FAIL]'
        else:
            self.results['warnings'] += 1
            icon = '[WARN]'
        
        print(f"  {icon} {name}: {message}")
        if details and status == 'FAIL':
            print(f"    Details: {details}")
    
    def run_all_tests(self):
        """Run all pre-collection tests."""
        print("=" * 70)
        print("STREAMGUARD PRE-COLLECTION TEST SUITE")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test 1: Environment Configuration
        self.test_environment_config()
        
        # Test 2: Dependencies
        self.test_dependencies()
        
        # Test 3: Module Imports
        self.test_module_imports()
        
        # Test 4: Tree-Sitter Functionality
        self.test_tree_sitter()
        
        # Test 5: API Connectivity
        self.test_api_connectivity()
        
        # Test 6: GitHub API with Tokens
        self.test_github_api()
        
        # Test 7: NVD API
        self.test_nvd_api()
        
        # Test 8: Disk Space
        self.test_disk_space()
        
        # Test 9: Output Directories
        self.test_output_directories()
        
        # Test 10: Unit Tests for Collectors
        self.test_collectors_unit()
        
        # Test 11: Edge Cases
        self.test_edge_cases()
        
        # Print Summary
        self.print_summary()
        
        return self.results['failed'] == 0
    
    def test_environment_config(self):
        """Test environment configuration."""
        print("\n[1/11] ENVIRONMENT CONFIGURATION")
        print("-" * 40)
        
        # Check .env file exists
        env_file = Path(__file__).parent.parent.parent.parent / '.env'
        if env_file.exists():
            self.log_test("ENV File", "PASS", f"Found at {env_file}")
        else:
            self.log_test("ENV File", "FAIL", "Not found", str(env_file))
        
        # Check GitHub tokens (try both GITHUB_TOKENS and GITHUB_TOKEN)
        github_tokens_str = os.getenv('GITHUB_TOKENS', '') or os.getenv('GITHUB_TOKEN', '')
        if github_tokens_str:
            tokens = [t.strip() for t in github_tokens_str.split(',') if t.strip()]
            self.github_tokens = tokens
            
            if len(tokens) >= 1:
                self.log_test("GitHub Tokens", "PASS", f"{len(tokens)} tokens configured")
                
                # Verify token format
                valid_count = 0
                for t in tokens:
                    if t.startswith('ghp_') or t.startswith('github_pat_'):
                        valid_count += 1
                
                if valid_count == len(tokens):
                    self.log_test("Token Format", "PASS", "All tokens have valid format")
                else:
                    self.log_test("Token Format", "WARN", f"{len(tokens) - valid_count} tokens may have invalid format")
                
                rate_limit = len(tokens) * 5000
                self.log_test("Rate Limit", "PASS", f"{rate_limit:,} requests/hour available")
            else:
                self.log_test("GitHub Tokens", "FAIL", "No valid tokens found")
        else:
            self.log_test("GitHub Tokens", "FAIL", "GITHUB_TOKENS not set in .env")
        
        # Check NVD API key
        nvd_key = os.getenv('NVD_API_KEY', '')
        if nvd_key:
            self.nvd_api_key = nvd_key
            self.log_test("NVD API Key", "PASS", f"Configured (length: {len(nvd_key)} chars)")
        else:
            self.log_test("NVD API Key", "WARN", "Not set (will use slower public rate limit)")
    
    def test_dependencies(self):
        """Test required dependencies."""
        print("\n[2/11] DEPENDENCIES")
        print("-" * 40)
        
        dependencies = [
            ('requests', 'HTTP requests'),
            ('git', 'GitPython for repo cloning'),
            ('dotenv', 'Environment variables'),
            ('rich', 'Progress display'),
            ('tqdm', 'Progress bars'),
            ('ratelimit', 'Rate limiting'),
            ('tree_sitter', 'Code parsing'),
            ('tree_sitter_languages', 'Language parsers'),
            ('numpy', 'Numerical operations'),
            ('pandas', 'Data processing'),
        ]
        
        for module, description in dependencies:
            try:
                __import__(module)
                self.log_test(module, "PASS", description)
            except ImportError as e:
                self.log_test(module, "FAIL", f"Not installed: {e}")
    
    def test_module_imports(self):
        """Test all collector module imports."""
        print("\n[3/11] MODULE IMPORTS")
        print("-" * 40)
        
        modules = [
            'base_collector',
            'github_api_manager',
            'tree_sitter_extractor',
            'synthetic_generator',
            'cve_collector_enhanced',
            'repo_miner_enhanced',
            'github_advisory_collector_enhanced',
            'osv_collector',
            'exploitdb_collector',
            'dynamic_repo_discoverer',
            'validate_collection',
        ]
        
        for module in modules:
            try:
                __import__(module)
                self.log_test(module, "PASS", "Imported successfully")
            except Exception as e:
                self.log_test(module, "FAIL", str(e)[:50])
    
    def test_tree_sitter(self):
        """Test tree-sitter functionality."""
        print("\n[4/11] TREE-SITTER PARSING")
        print("-" * 40)
        
        try:
            from tree_sitter_extractor import TreeSitterFunctionExtractor
            extractor = TreeSitterFunctionExtractor()
            
            # Check parsers initialized
            if extractor.parsers:
                self.log_test("Parser Init", "PASS", f"Languages: {list(extractor.parsers.keys())}")
            else:
                self.log_test("Parser Init", "WARN", "No parsers initialized (will use fallback)")
            
            # Test Python parsing
            python_code = '''
def vulnerable_function(user_input):
    query = "SELECT * FROM users WHERE id=" + user_input
    return execute(query)
'''
            result = extractor.extract_function_body(python_code, [3], 'python')
            if result and 'SELECT' in result:
                self.log_test("Python Parse", "PASS", "Function extraction working")
            else:
                self.log_test("Python Parse", "WARN", "Extraction returned unexpected result")
            
            # Test JavaScript parsing
            js_code = '''
function vulnerableQuery(userInput) {
    const query = "SELECT * FROM users WHERE id=" + userInput;
    return db.execute(query);
}
'''
            result = extractor.extract_function_body(js_code, [3], 'javascript')
            if result:
                self.log_test("JavaScript Parse", "PASS", "Function extraction working")
            else:
                self.log_test("JavaScript Parse", "WARN", "Extraction returned unexpected result")
                
        except Exception as e:
            self.log_test("Tree-Sitter", "FAIL", str(e)[:100])
    
    def test_api_connectivity(self):
        """Test basic API connectivity."""
        print("\n[5/11] API CONNECTIVITY")
        print("-" * 40)
        
        import requests
        
        # Test GitHub API
        try:
            response = requests.get("https://api.github.com", timeout=10)
            if response.status_code == 200:
                self.log_test("GitHub API", "PASS", "Reachable")
            else:
                self.log_test("GitHub API", "WARN", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("GitHub API", "FAIL", str(e)[:50])
        
        # Test NVD API
        try:
            response = requests.get("https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=1", timeout=15)
            if response.status_code == 200:
                self.log_test("NVD API", "PASS", "Reachable")
            else:
                self.log_test("NVD API", "WARN", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("NVD API", "FAIL", str(e)[:50])
        
        # Test OSV API
        try:
            response = requests.get("https://api.osv.dev/v1/vulns/GHSA-test", timeout=10)
            # 404 is expected for non-existent vuln, but means API is reachable
            if response.status_code in [200, 404]:
                self.log_test("OSV API", "PASS", "Reachable")
            else:
                self.log_test("OSV API", "WARN", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("OSV API", "FAIL", str(e)[:50])
    
    def test_github_api(self):
        """Test GitHub API with tokens."""
        print("\n[6/11] GITHUB API WITH TOKENS")
        print("-" * 40)
        
        if not self.github_tokens:
            self.log_test("Token Test", "FAIL", "No tokens available")
            return
        
        import requests
        
        for i, token in enumerate(self.github_tokens[:3], 1):  # Test first 3 tokens
            try:
                headers = {'Authorization': f'token {token}'}
                response = requests.get("https://api.github.com/user", headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    username = data.get('login', 'Unknown')
                    self.log_test(f"Token #{i}", "PASS", f"Valid (user: {username})")
                elif response.status_code == 401:
                    self.log_test(f"Token #{i}", "FAIL", "Invalid or expired")
                else:
                    self.log_test(f"Token #{i}", "WARN", f"Status: {response.status_code}")
                    
                # Check rate limit
                remaining = response.headers.get('X-RateLimit-Remaining', 'Unknown')
                limit = response.headers.get('X-RateLimit-Limit', 'Unknown')
                self.log_test(f"Token #{i} Rate", "PASS", f"{remaining}/{limit} remaining")
                
            except Exception as e:
                self.log_test(f"Token #{i}", "FAIL", str(e)[:50])
    
    def test_nvd_api(self):
        """Test NVD API with key."""
        print("\n[7/11] NVD API WITH KEY")
        print("-" * 40)
        
        import requests
        
        # Test without key
        try:
            response = requests.get(
                "https://services.nvd.nist.gov/rest/json/cves/2.0",
                params={"keywordSearch": "SQL injection", "resultsPerPage": 1},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                total = data.get('totalResults', 0)
                self.log_test("NVD Query", "PASS", f"Found {total:,} SQL injection CVEs")
            else:
                self.log_test("NVD Query", "WARN", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("NVD Query", "FAIL", str(e)[:50])
        
        # Test with key if available
        if self.nvd_api_key:
            try:
                headers = {'apiKey': self.nvd_api_key}
                response = requests.get(
                    "https://services.nvd.nist.gov/rest/json/cves/2.0",
                    params={"keywordSearch": "SQL injection", "resultsPerPage": 1},
                    headers=headers,
                    timeout=30
                )
                if response.status_code == 200:
                    self.log_test("NVD API Key", "PASS", "Key working (faster rate limit)")
                elif response.status_code == 403:
                    self.log_test("NVD API Key", "FAIL", "Key invalid or expired")
                else:
                    self.log_test("NVD API Key", "WARN", f"Status: {response.status_code}")
            except Exception as e:
                self.log_test("NVD API Key", "FAIL", str(e)[:50])
    
    def test_disk_space(self):
        """Test available disk space."""
        print("\n[8/11] DISK SPACE")
        print("-" * 40)
        
        import shutil
        
        try:
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024 ** 3)
            
            if free_gb >= 20:
                self.log_test("Disk Space", "PASS", f"{free_gb:.1f} GB available (need 20 GB)")
            elif free_gb >= 10:
                self.log_test("Disk Space", "WARN", f"{free_gb:.1f} GB available (recommend 20 GB)")
            else:
                self.log_test("Disk Space", "FAIL", f"Only {free_gb:.1f} GB available (need 20 GB)")
        except Exception as e:
            self.log_test("Disk Space", "WARN", f"Could not check: {e}")
    
    def test_output_directories(self):
        """Test output directory creation."""
        print("\n[9/11] OUTPUT DIRECTORIES")
        print("-" * 40)
        
        base_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        
        directories = ['cves', 'github', 'opensource', 'osv', 'exploitdb', 'synthetic']
        
        for dir_name in directories:
            dir_path = base_dir / dir_name
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                if dir_path.exists():
                    self.log_test(f"Dir: {dir_name}", "PASS", f"Created/exists at {dir_path}")
                else:
                    self.log_test(f"Dir: {dir_name}", "FAIL", "Could not create")
            except Exception as e:
                self.log_test(f"Dir: {dir_name}", "FAIL", str(e)[:50])
    
    def test_collectors_unit(self):
        """Unit tests for each collector."""
        print("\n[10/11] COLLECTOR UNIT TESTS")
        print("-" * 40)
        
        # Test Synthetic Generator
        try:
            from synthetic_generator import SyntheticGenerator
            gen = SyntheticGenerator(output_dir="test_output", seed=42)
            samples = gen.generate_all(total_samples=10)
            
            if len(samples) == 10:
                self.log_test("Synthetic Gen", "PASS", f"Generated {len(samples)} samples")
            else:
                self.log_test("Synthetic Gen", "WARN", f"Expected 10, got {len(samples)}")
            
            # Check sample structure
            sample = samples[0]
            required_fields = ['code', 'vulnerable', 'vulnerability_type', 'source']
            missing = [f for f in required_fields if f not in sample]
            if not missing:
                self.log_test("Sample Schema", "PASS", "All required fields present")
            else:
                self.log_test("Sample Schema", "FAIL", f"Missing: {missing}")
                
        except Exception as e:
            self.log_test("Synthetic Gen", "FAIL", str(e)[:100])
        
        # Test GitHub API Manager
        try:
            from github_api_manager import GitHubAPIManager
            cache_dir = Path("test_cache")
            cache_dir.mkdir(exist_ok=True)
            
            manager = GitHubAPIManager(
                tokens=self.github_tokens[:1] if self.github_tokens else [],
                cache_dir=cache_dir
            )
            
            # Test URL parsing
            parsed = manager.parse_github_url("https://github.com/django/django/commit/abc123")
            if parsed and parsed[0] == 'django' and parsed[1] == 'django':
                self.log_test("URL Parser", "PASS", "GitHub URL parsing working")
            else:
                self.log_test("URL Parser", "FAIL", f"Got: {parsed}")
                
        except Exception as e:
            self.log_test("API Manager", "FAIL", str(e)[:100])
        
        # Test Base Collector
        try:
            from base_collector import BaseCollector
            
            # Test vulnerability type extraction
            class TestCollector(BaseCollector):
                def collect(self):
                    return []
            
            collector = TestCollector(output_dir="test_output")
            
            vuln_type = collector.extract_vulnerability_type("SQL injection vulnerability in login")
            if vuln_type == "sql_injection":
                self.log_test("Vuln Extract", "PASS", "Correctly identified SQL injection")
            else:
                self.log_test("Vuln Extract", "WARN", f"Got: {vuln_type}")
                
        except Exception as e:
            self.log_test("Base Collector", "FAIL", str(e)[:100])
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n[11/11] EDGE CASES & ERROR HANDLING")
        print("-" * 40)
        
        # Test code validation
        try:
            from base_collector import BaseCollector
            
            class TestCollector(BaseCollector):
                def collect(self):
                    return []
            
            collector = TestCollector(output_dir="test_output")
            
            # Test empty code
            if not collector.validate_code(""):
                self.log_test("Empty Code", "PASS", "Correctly rejected empty code")
            else:
                self.log_test("Empty Code", "FAIL", "Should reject empty code")
            
            # Test too short code
            if not collector.validate_code("x=1"):
                self.log_test("Short Code", "PASS", "Correctly rejected short code")
            else:
                self.log_test("Short Code", "WARN", "Accepted very short code")
            
            # Test valid code
            valid_code = "def test():\n    return 'hello world' * 10\n" * 5
            if collector.validate_code(valid_code):
                self.log_test("Valid Code", "PASS", "Accepted valid code")
            else:
                self.log_test("Valid Code", "FAIL", "Rejected valid code")
                
        except Exception as e:
            self.log_test("Code Validation", "FAIL", str(e)[:100])
        
        # Test deduplication
        try:
            samples = [
                {"code": "same code here", "id": 1},
                {"code": "same code here", "id": 2},
                {"code": "different code", "id": 3},
            ]
            deduped = collector.deduplicate_samples(samples, key="code")
            
            if len(deduped) == 2:
                self.log_test("Deduplication", "PASS", f"Removed {len(samples) - len(deduped)} duplicates")
            else:
                self.log_test("Deduplication", "WARN", f"Expected 2, got {len(deduped)}")
                
        except Exception as e:
            self.log_test("Deduplication", "FAIL", str(e)[:100])
        
        # Test rate limiting
        try:
            from github_api_manager import TokenBucket
            bucket = TokenBucket(capacity=10, refill_rate=1.0)
            
            # Should be able to consume immediately
            start = time.time()
            bucket.consume(1)
            elapsed = time.time() - start
            
            if elapsed < 0.1:
                self.log_test("Rate Limiter", "PASS", "Token bucket working")
            else:
                self.log_test("Rate Limiter", "WARN", f"Took {elapsed:.2f}s (expected instant)")
                
        except Exception as e:
            self.log_test("Rate Limiter", "FAIL", str(e)[:100])
    
    def print_summary(self):
        """Print test summary."""
        print()
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        total = self.results['passed'] + self.results['failed'] + self.results['warnings']
        
        print(f"Total Tests: {total}")
        print(f"  [OK]   Passed:   {self.results['passed']}")
        print(f"  [FAIL] Failed:   {self.results['failed']}")
        print(f"  [WARN] Warnings: {self.results['warnings']}")
        print()
        
        if self.results['failed'] == 0:
            print("=" * 70)
            print("STATUS: ALL CRITICAL TESTS PASSED [SUCCESS]")
            print("=" * 70)
            print()
            print("You are ready to start data collection!")
            print()
            print("Next steps:")
            print("  1. Run SQL injection collection:")
            print("     python master_orchestrator.py --collectors cve github repo osv --keywords 'SQL injection'")
            print()
        else:
            print("=" * 70)
            print("STATUS: SOME TESTS FAILED [ERROR]")
            print("=" * 70)
            print()
            print("Please fix the failed tests before proceeding.")
            print()
            print("Failed tests:")
            for test in self.results['tests']:
                if test['status'] == 'FAIL':
                    print(f"  - {test['name']}: {test['message']}")


def main():
    """Run pre-collection tests."""
    tester = PreCollectionTester()
    success = tester.run_all_tests()
    
    # Cleanup test directories
    import shutil
    for dir_name in ['test_output', 'test_cache']:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name, ignore_errors=True)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
