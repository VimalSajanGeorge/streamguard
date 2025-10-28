"""Configuration for CVE Collector.

This module contains configuration constants and settings for the
Enhanced CVE Collector. Modify these values to customize collection behavior.
"""

from typing import List, Dict

# Collection targets
TARGET_SAMPLES = 15000  # Total CVE samples to collect
SAMPLES_PER_KEYWORD = None  # Auto-calculated if None

# Time range for CVE search
START_YEAR = 2020
END_YEAR = 2025

# API Configuration
NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
GITHUB_API_BASE = "https://api.github.com"

# Rate limiting (NVD public API)
RATE_LIMIT_REQUESTS = 5
RATE_LIMIT_WINDOW = 30  # seconds

# Pagination
RESULTS_PER_PAGE = 2000  # Max allowed by NVD API

# Multiprocessing
MAX_WORKERS = 4  # Maximum parallel processes

# Extended vulnerability keywords
VULNERABILITY_KEYWORDS = [
    "SQL injection",
    "XSS",
    "cross-site scripting",
    "command injection",
    "path traversal",
    "SSRF",
    "XXE",
    "CSRF",
    "deserialization",
    "remote code execution",
    "directory traversal",
    "insecure deserialization",
]

# Additional focused keywords (optional)
ADDITIONAL_KEYWORDS = [
    "buffer overflow",
    "format string",
    "integer overflow",
    "use after free",
    "null pointer dereference",
    "race condition",
    "authentication bypass",
    "privilege escalation",
    "information disclosure",
    "denial of service"
]

# Code validation settings
MIN_CODE_LENGTH = 50
MAX_CODE_LENGTH = 10000
MIN_NON_WHITESPACE_RATIO = 0.5

# File type filters for GitHub code extraction
CODE_FILE_EXTENSIONS = {
    '.py',      # Python
    '.js',      # JavaScript
    '.java',    # Java
    '.c',       # C
    '.cpp',     # C++
    '.cc',      # C++
    '.cxx',     # C++
    '.cs',      # C#
    '.rb',      # Ruby
    '.php',     # PHP
    '.go',      # Go
    '.rs',      # Rust
    '.swift',   # Swift
    '.kt',      # Kotlin
    '.scala',   # Scala
    '.ts',      # TypeScript
    '.tsx',     # TypeScript React
    '.jsx',     # JavaScript React
    '.m',       # Objective-C
    '.h',       # C/C++ header
    '.hpp',     # C++ header
    '.pl',      # Perl
    '.sh',      # Shell script
    '.bash',    # Bash script
    '.ps1',     # PowerShell
}

# Severity mapping
SEVERITY_LEVELS = {
    'CRITICAL': 9.0,
    'HIGH': 7.0,
    'MEDIUM': 4.0,
    'LOW': 0.0,
    'UNKNOWN': -1.0
}

# Output configuration
OUTPUT_DIR = 'data/raw/cves'
OUTPUT_FILENAME = 'cve_data.jsonl'
CACHE_ENABLED = True
CACHE_SUBDIR = '.cache'

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.0
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# Timeout settings
API_TIMEOUT = 30  # seconds
GITHUB_TIMEOUT = 30  # seconds

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N samples

# GitHub API settings
GITHUB_COMMIT_PATTERN = r'github\.com/([^/]+)/([^/]+)/commit/([a-f0-9]+)'
GITHUB_PR_PATTERN = r'github\.com/([^/]+)/([^/]+)/pull/(\d+)'

# CVE patterns for filtering
CVE_ID_PATTERN = r'CVE-\d{4}-\d{4,}'

# Data quality thresholds
MIN_DESCRIPTION_LENGTH = 20
MIN_CWES_COUNT = 0  # 0 means no minimum
REQUIRE_SEVERITY = False  # Whether to require severity score
REQUIRE_CODE = False  # Whether to only collect CVEs with code samples

# Vulnerability type mapping
VULNERABILITY_TYPE_PATTERNS = {
    "sql_injection": [
        "sql injection",
        "sqli",
        "sql inject"
    ],
    "xss": [
        "xss",
        "cross-site scripting",
        "cross site scripting"
    ],
    "command_injection": [
        "command injection",
        "rce",
        "remote code execution",
        "os command injection"
    ],
    "path_traversal": [
        "path traversal",
        "directory traversal",
        "file traversal"
    ],
    "ssrf": [
        "ssrf",
        "server-side request forgery",
        "server side request forgery"
    ],
    "xxe": [
        "xxe",
        "xml external entity"
    ],
    "csrf": [
        "csrf",
        "cross-site request forgery",
        "cross site request forgery"
    ],
    "deserialization": [
        "deserialization",
        "insecure deserialization",
        "unsafe deserialization"
    ],
    "auth_bypass": [
        "authentication bypass",
        "auth bypass",
        "authorization bypass",
        "access control"
    ],
    "code_injection": [
        "code injection",
        "eval injection",
        "script injection"
    ],
    "buffer_overflow": [
        "buffer overflow",
        "stack overflow",
        "heap overflow"
    ],
    "format_string": [
        "format string",
        "string format"
    ],
    "integer_overflow": [
        "integer overflow",
        "integer underflow"
    ],
    "use_after_free": [
        "use after free",
        "use-after-free"
    ]
}

# CWE to vulnerability type mapping
CWE_TO_VULN_TYPE: Dict[str, str] = {
    "CWE-89": "sql_injection",
    "CWE-79": "xss",
    "CWE-78": "command_injection",
    "CWE-77": "command_injection",
    "CWE-22": "path_traversal",
    "CWE-918": "ssrf",
    "CWE-611": "xxe",
    "CWE-352": "csrf",
    "CWE-502": "deserialization",
    "CWE-287": "auth_bypass",
    "CWE-94": "code_injection",
    "CWE-95": "code_injection",
    "CWE-120": "buffer_overflow",
    "CWE-121": "buffer_overflow",
    "CWE-122": "buffer_overflow",
    "CWE-134": "format_string",
    "CWE-190": "integer_overflow",
    "CWE-416": "use_after_free",
}


def get_config() -> Dict:
    """
    Get configuration as dictionary.

    Returns:
        Configuration dictionary
    """
    return {
        'target_samples': TARGET_SAMPLES,
        'start_year': START_YEAR,
        'end_year': END_YEAR,
        'rate_limit_requests': RATE_LIMIT_REQUESTS,
        'rate_limit_window': RATE_LIMIT_WINDOW,
        'keywords': VULNERABILITY_KEYWORDS,
        'output_dir': OUTPUT_DIR,
        'cache_enabled': CACHE_ENABLED,
        'min_code_length': MIN_CODE_LENGTH,
        'max_code_length': MAX_CODE_LENGTH,
    }


def get_keywords(include_additional: bool = False) -> List[str]:
    """
    Get list of keywords for CVE search.

    Args:
        include_additional: Whether to include additional keywords

    Returns:
        List of keywords
    """
    keywords = VULNERABILITY_KEYWORDS.copy()
    if include_additional:
        keywords.extend(ADDITIONAL_KEYWORDS)
    return keywords


def validate_config():
    """Validate configuration settings."""
    assert TARGET_SAMPLES > 0, "TARGET_SAMPLES must be positive"
    assert START_YEAR < END_YEAR, "START_YEAR must be before END_YEAR"
    assert RATE_LIMIT_REQUESTS > 0, "RATE_LIMIT_REQUESTS must be positive"
    assert RATE_LIMIT_WINDOW > 0, "RATE_LIMIT_WINDOW must be positive"
    assert MIN_CODE_LENGTH < MAX_CODE_LENGTH, "MIN_CODE_LENGTH must be less than MAX_CODE_LENGTH"
    assert 0 <= MIN_NON_WHITESPACE_RATIO <= 1, "MIN_NON_WHITESPACE_RATIO must be between 0 and 1"
    assert len(VULNERABILITY_KEYWORDS) > 0, "Must have at least one keyword"


# Validate on import
validate_config()
