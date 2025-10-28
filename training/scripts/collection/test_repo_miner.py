"""Test script for Enhanced Repository Miner."""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Change to project root to ensure relative imports work
os.chdir(str(project_root))

from training.scripts.collection.repo_miner_enhanced import EnhancedRepoMiner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_miner_initialization():
    """Test miner initialization."""
    logger.info("Testing miner initialization...")

    miner = EnhancedRepoMiner(output_dir="data/raw/opensource")

    assert miner.repos_dir.exists(), "Repos directory should be created"
    assert len(miner.REPOSITORIES) == 12, "Should have 12 repositories configured"
    assert len(miner.SECURITY_KEYWORDS) > 0, "Should have security keywords"

    logger.info("Initialization test passed!")


def test_security_commit_detection():
    """Test security commit detection logic."""
    logger.info("Testing security commit detection...")

    miner = EnhancedRepoMiner(output_dir="data/raw/opensource")

    # Mock commit object
    class MockCommit:
        def __init__(self, message):
            self.message = message

    # Test positive cases
    security_messages = [
        "Fix SQL injection vulnerability in user authentication",
        "Security patch: XSS protection for user input",
        "CVE-2023-1234: Fix command injection",
        "Sanitize user input to prevent CSRF attacks"
    ]

    for msg in security_messages:
        commit = MockCommit(msg)
        assert miner.is_security_commit(commit), f"Should detect security commit: {msg}"

    # Test negative cases
    normal_messages = [
        "Add new feature for user profile",
        "Update documentation",
        "Refactor code for better performance",
        "Fix typo in README"
    ]

    for msg in normal_messages:
        commit = MockCommit(msg)
        assert not miner.is_security_commit(commit), f"Should not detect as security: {msg}"

    logger.info("Security commit detection test passed!")


def test_file_relevance():
    """Test file relevance checking."""
    logger.info("Testing file relevance checking...")

    miner = EnhancedRepoMiner(output_dir="data/raw/opensource")

    # Test Python files
    assert miner._is_relevant_file("auth.py", "python")
    assert miner._is_relevant_file("views/user.py", "python")
    assert not miner._is_relevant_file("config.yml", "python")
    assert not miner._is_relevant_file("README.md", "python")

    # Test JavaScript files
    assert miner._is_relevant_file("app.js", "javascript")
    assert miner._is_relevant_file("index.ts", "javascript")
    assert miner._is_relevant_file("component.jsx", "javascript")
    assert miner._is_relevant_file("component.tsx", "javascript")
    assert not miner._is_relevant_file("package.json", "javascript")

    logger.info("File relevance test passed!")


def test_vulnerability_extraction():
    """Test vulnerability type extraction."""
    logger.info("Testing vulnerability type extraction...")

    miner = EnhancedRepoMiner(output_dir="data/raw/opensource")

    test_cases = [
        ("Fix SQL injection in login form", "sql_injection"),
        ("Prevent XSS attack in comment section", "xss"),
        ("Security: Fix command injection vulnerability", "command_injection"),
        ("Patch CSRF vulnerability", "csrf"),
        ("CVE-2023-1234: Path traversal fix", "path_traversal"),
        ("Fix SSRF in URL fetcher", "ssrf"),
        ("XXE vulnerability patched", "xxe"),
        ("Authentication bypass fix", "auth_bypass"),
        ("Fix insecure deserialization", "deserialization"),
    ]

    for message, expected_type in test_cases:
        extracted_type = miner.extract_vulnerability_type(message)
        assert extracted_type == expected_type, \
            f"Expected {expected_type}, got {extracted_type} for: {message}"

    logger.info("Vulnerability extraction test passed!")


def test_code_validation():
    """Test code validation."""
    logger.info("Testing code validation...")

    miner = EnhancedRepoMiner(output_dir="data/raw/opensource")

    # Valid code
    valid_code = """
def authenticate_user(username, password):
    query = "SELECT * FROM users WHERE username='%s' AND password='%s'" % (username, password)
    result = db.execute(query)
    return result
"""
    assert miner.validate_code(valid_code), "Should validate proper code"

    # Too short
    short_code = "x = 1"
    assert not miner.validate_code(short_code), "Should reject too short code"

    # Empty
    empty_code = ""
    assert not miner.validate_code(empty_code), "Should reject empty code"

    # Just whitespace
    whitespace_code = "   \n   \n   "
    assert not miner.validate_code(whitespace_code), "Should reject whitespace-only code"

    logger.info("Code validation test passed!")


def test_repository_config():
    """Test repository configurations."""
    logger.info("Testing repository configurations...")

    miner = EnhancedRepoMiner(output_dir="data/raw/opensource")

    # Check all repositories are configured
    python_repos = [name for name, config in miner.REPOSITORIES.items()
                   if config["language"] == "python"]
    js_repos = [name for name, config in miner.REPOSITORIES.items()
               if config["language"] == "javascript"]

    assert len(python_repos) == 6, "Should have 6 Python repositories"
    assert len(js_repos) == 6, "Should have 6 JavaScript repositories"

    # Check target counts sum to approximately 33,000 (to account for filtering/deduplication to reach 20k)
    total_target = sum(config["target"] for config in miner.REPOSITORIES.values())
    assert 32000 <= total_target <= 34000, \
        f"Total target should be around 33,000, got {total_target}"

    # Verify specific repositories
    assert "django/django" in miner.REPOSITORIES
    assert "expressjs/express" in miner.REPOSITORIES
    assert "pallets/flask" in miner.REPOSITORIES
    assert "nodejs/node" in miner.REPOSITORIES

    logger.info("Repository configuration test passed!")


def run_all_tests():
    """Run all tests."""
    logger.info("="*60)
    logger.info("Running Enhanced Repository Miner Tests")
    logger.info("="*60)

    tests = [
        test_miner_initialization,
        test_security_commit_detection,
        test_file_relevance,
        test_vulnerability_extraction,
        test_code_validation,
        test_repository_config,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            logger.error(f"Test {test.__name__} failed: {str(e)}")
            failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} raised exception: {str(e)}")
            failed += 1

    logger.info("="*60)
    logger.info(f"Tests completed: {passed} passed, {failed} failed")
    logger.info("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
