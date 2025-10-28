

"""Base collector class with common utilities."""

import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import requests
from datetime import datetime


class BaseCollector(ABC):
    """Base class for all data collectors."""

    def __init__(self, output_dir: str, cache_enabled: bool = True):
        """
        Initialize base collector.

        Args:
            output_dir: Directory to save collected data
            cache_enabled: Whether to cache API responses
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.output_dir / ".cache"
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_enabled = cache_enabled
        self.session = requests.Session()
        self.samples_collected = 0
        self.errors = []

    @abstractmethod
    def collect(self) -> List[Dict]:
        """
        Collect data from source.

        Returns:
            List of collected samples
        """
        pass

    def save_samples(self, samples: List[Dict], filename: str):
        """
        Save samples to JSONL file.

        Args:
            samples: List of sample dictionaries
            filename: Output filename
        """
        output_file = self.output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        return output_file

    def load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache."""
        if not self.cache_enabled:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def save_cache(self, cache_key: str, data: Dict):
        """Save data to cache."""
        if not self.cache_enabled:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def make_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def deduplicate_samples(self, samples: List[Dict], key: str = "code") -> List[Dict]:
        """
        Remove duplicate samples based on key.

        Args:
            samples: List of samples
            key: Key to check for duplicates

        Returns:
            Deduplicated list
        """
        seen = set()
        unique_samples = []

        for sample in samples:
            if key in sample:
                code_hash = hashlib.md5(sample[key].encode()).hexdigest()
                if code_hash not in seen:
                    seen.add(code_hash)
                    unique_samples.append(sample)
            else:
                unique_samples.append(sample)

        return unique_samples

    def rate_limit(self, requests_per_second: float = 1.0):
        """Simple rate limiting."""
        time.sleep(1.0 / requests_per_second)

    def log_error(self, error: str, context: Dict = None):
        """Log an error with context."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context or {}
        }
        self.errors.append(error_entry)

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "samples_collected": self.samples_collected,
            "errors_count": len(self.errors),
            "errors": self.errors[-10:]  # Last 10 errors
        }

    def validate_code(self, code: str, min_length: int = 50, max_length: int = 10000) -> bool:
        """
        Validate code sample meets quality criteria.

        Args:
            code: Code string to validate
            min_length: Minimum code length
            max_length: Maximum code length

        Returns:
            True if valid, False otherwise
        """
        if not code or not isinstance(code, str):
            return False

        code = code.strip()

        # Check length
        if len(code) < min_length or len(code) > max_length:
            return False

        # Check if it's not just whitespace or comments
        non_whitespace = ''.join(code.split())
        if len(non_whitespace) < min_length // 2:
            return False

        return True

    def extract_vulnerability_type(self, text: str) -> Optional[str]:
        """
        Extract vulnerability type from text.

        Args:
            text: Text to analyze (description, commit message, etc.)

        Returns:
            Vulnerability type or None
        """
        text_lower = text.lower()

        patterns = {
            "sql_injection": ["sql injection", "sqli", "sql inject"],
            "xss": ["xss", "cross-site scripting", "cross site scripting"],
            "command_injection": ["command injection", "rce", "remote code execution"],
            "path_traversal": ["path traversal", "directory traversal"],
            "ssrf": ["ssrf", "server-side request forgery"],
            "xxe": ["xxe", "xml external entity"],
            "csrf": ["csrf", "cross-site request forgery"],
            "deserialization": ["deserialization", "insecure deserialization"],
            "auth_bypass": ["authentication bypass", "auth bypass", "authorization bypass"],
            "code_injection": ["code injection", "eval injection"]
        }

        for vuln_type, keywords in patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return vuln_type

        return "unknown"
