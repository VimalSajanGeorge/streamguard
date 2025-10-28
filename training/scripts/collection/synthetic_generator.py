"""Synthetic Data Generator for StreamGuard.

Generates synthetic vulnerability samples with counterfactual pairs.
Target: 500 samples (250 vulnerable + 250 safe).
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from base_collector import BaseCollector


class SyntheticGenerator(BaseCollector):
    """Generate synthetic vulnerability samples with counterfactual pairs."""

    def __init__(self, output_dir: str = "data/raw/synthetic", seed: int = 42):
        """
        Initialize synthetic data generator.

        Args:
            output_dir: Directory to save synthetic data
            seed: Random seed for reproducibility
        """
        super().__init__(output_dir, cache_enabled=False)
        random.seed(seed)

        # Vulnerability templates for each type
        self.vulnerability_templates = {
            "sql_injection_concat": {
                "vulnerable": [
                    'query = "SELECT * FROM {table} WHERE {column}=" + {user_input}',
                    'sql = "DELETE FROM {table} WHERE id=" + str({user_input})',
                    'cursor.execute("UPDATE {table} SET {column}=\'{value}\' WHERE id=" + {user_input})',
                    '{table}_query = "INSERT INTO {table} ({column}) VALUES (" + {user_input} + ")"',
                    'db.execute("SELECT * FROM {table} WHERE {column} LIKE \'" + {user_input} + "%\'")',
                ],
                "safe": [
                    'cursor.execute("SELECT * FROM {table} WHERE {column}=?", ({user_input},))',
                    'stmt = conn.prepareStatement("SELECT * FROM {table} WHERE {column}=?")',
                    'query = session.query({table}).filter({table}.{column} == {user_input})',
                    'cursor.execute("DELETE FROM {table} WHERE id=%s", ({user_input},))',
                    'db.execute("UPDATE {table} SET {column}=? WHERE id=?", [{value}, {user_input}])',
                ]
            },
            "xss_output": {
                "vulnerable": [
                    'return "<div>" + {user_input} + "</div>"',
                    'html = "<h1>Welcome " + {user_input} + "</h1>"',
                    'response.write("<p>Search results for: " + {user_input} + "</p>")',
                    'output = "<span class=\'username\'>" + {user_input} + "</span>"',
                    'document.getElementById("{element}").innerHTML = {user_input}',
                ],
                "safe": [
                    'return "<div>" + escape({user_input}) + "</div>"',
                    'html = f"<h1>Welcome {{htmlspecialchars({user_input})}}</h1>"',
                    'response.write("<p>Search results for: " + sanitize({user_input}) + "</p>")',
                    'output = "<span class=\'username\'>" + html.escape({user_input}) + "</span>"',
                    'document.getElementById("{element}").textContent = {user_input}',
                ]
            },
            "command_injection": {
                "vulnerable": [
                    'os.system("ping " + {user_input})',
                    'exec("ls " + {user_input})',
                    'subprocess.call("wget " + {user_input}, shell=True)',
                    'Runtime.getRuntime().exec("curl " + {user_input})',
                    'shell_exec("tar -xzf " . {user_input})',
                ],
                "safe": [
                    'subprocess.run(["ping", {user_input}], shell=False)',
                    'subprocess.call(["ls", {user_input}])',
                    'subprocess.run(["wget", {user_input}], check=True)',
                    'ProcessBuilder pb = new ProcessBuilder("curl", {user_input})',
                    'escapeshellarg({user_input}); exec("tar -xzf " . $safe_input)',
                ]
            },
            "path_traversal": {
                "vulnerable": [
                    'with open("/var/www/files/" + {user_input}, "r") as f:',
                    'file_path = os.path.join(base_dir, {user_input})',
                    'return send_file("./uploads/" + {user_input})',
                    'fs.readFileSync("/app/data/" + {user_input})',
                    'File file = new File(uploadDir + {user_input})',
                ],
                "safe": [
                    'safe_path = os.path.basename({user_input}); open("/var/www/files/" + safe_path)',
                    'file_path = os.path.realpath(os.path.join(base_dir, {user_input}))',
                    'if not ".." in {user_input}: return send_file("./uploads/" + {user_input})',
                    r'const safePath = path.normalize({user_input}).replace(/^(\.\.(\\/|\\\\))+/, "")',
                    'Path filePath = Paths.get(uploadDir).resolve().normalize()',
                ]
            },
            "ssrf": {
                "vulnerable": [
                    'response = requests.get({user_input})',
                    'urllib.request.urlopen({user_input})',
                    'fetch({user_input}).then(r => r.json())',
                    'HttpURLConnection conn = (HttpURLConnection) new URL({user_input}).openConnection()',
                    'file_get_contents({user_input})',
                ],
                "safe": [
                    'if {user_input}.startswith("https://"): response = requests.get({user_input})',
                    'parsed = urlparse({user_input}); if parsed.netloc in ALLOWED_HOSTS: urllib.request.urlopen({user_input})',
                    'if (isValidUrl({user_input})) {{ fetch({user_input}) }}',
                    'if (ALLOWED_DOMAINS.contains(url.getHost())) {{ conn = url.openConnection() }}',
                    'if (filter_var({user_input}, FILTER_VALIDATE_URL)) {{ file_get_contents({user_input}) }}',
                ]
            }
        }

        # Vocabulary for template expansion
        self.vocabulary = {
            "table": ["users", "products", "orders", "customers", "accounts", "sessions"],
            "column": ["id", "name", "email", "username", "password", "status"],
            "user_input": [
                "user_id",
                'request.args.get("id")',
                'params["name"]',
                'request.form["email"]',
                'req.query.search',
                '$_GET["id"]',
                'request.getParameter("username")'
            ],
            "value": ["user_data", "new_value", "updated_field", "input_value"],
            "element": ["result", "output", "content", "userInfo", "searchBox"],
        }

        # Language mappings for different templates
        self.language_hints = {
            'cursor.execute': 'python',
            'os.system': 'python',
            'subprocess': 'python',
            'requests.get': 'python',
            'conn.prepareStatement': 'java',
            'Runtime.getRuntime': 'java',
            'HttpURLConnection': 'java',
            'Paths.get': 'java',
            'fs.readFileSync': 'javascript',
            'fetch(': 'javascript',
            'document.getElementById': 'javascript',
            'shell_exec': 'php',
            'file_get_contents': 'php',
            'escapeshellarg': 'php',
            '$_GET': 'php',
        }

    def expand_template(self, template: str) -> Tuple[str, str]:
        """
        Expand template with random vocabulary.

        Args:
            template: Template string with {placeholders}

        Returns:
            Tuple of (expanded_code, detected_language)
        """
        expanded = template

        # Replace placeholders with random vocabulary
        for placeholder, options in self.vocabulary.items():
            if "{" + placeholder + "}" in expanded:
                expanded = expanded.replace(
                    "{" + placeholder + "}",
                    random.choice(options)
                )

        # Detect language based on code patterns
        language = self._detect_language(expanded)

        return expanded, language

    def _detect_language(self, code: str) -> str:
        """
        Detect programming language from code.

        Args:
            code: Code string

        Returns:
            Detected language
        """
        for hint, lang in self.language_hints.items():
            if hint in code:
                return lang

        # Default detection based on syntax
        if 'def ' in code or 'import ' in code:
            return 'python'
        elif 'function' in code or 'const ' in code or 'let ' in code:
            return 'javascript'
        elif 'public class' in code or 'new ' in code:
            return 'java'
        elif '$' in code or 'php' in code.lower():
            return 'php'

        return 'python'  # Default

    def create_vulnerable_safe_pair(
        self,
        vuln_type: str,
        vuln_idx: int,
        safe_idx: int
    ) -> Tuple[Dict, Dict]:
        """
        Create a counterfactual pair of vulnerable and safe code.

        Args:
            vuln_type: Vulnerability type
            vuln_idx: Index of vulnerable template to use
            safe_idx: Index of safe template to use

        Returns:
            Tuple of (vulnerable_sample, safe_sample)
        """
        templates = self.vulnerability_templates[vuln_type]

        # Expand templates
        vuln_code, vuln_lang = self.expand_template(
            templates["vulnerable"][vuln_idx]
        )
        safe_code, safe_lang = self.expand_template(
            templates["safe"][safe_idx]
        )

        # Create vulnerable sample
        vulnerable_sample = {
            "code": vuln_code,
            "vulnerable": True,
            "vulnerability_type": vuln_type,
            "counterfactual": safe_code,
            "source": "synthetic",
            "language": vuln_lang,
            "generated_at": datetime.now().isoformat()
        }

        # Create safe sample
        safe_sample = {
            "code": safe_code,
            "vulnerable": False,
            "vulnerability_type": vuln_type,
            "counterfactual": vuln_code,
            "source": "synthetic",
            "language": safe_lang,
            "generated_at": datetime.now().isoformat()
        }

        return vulnerable_sample, safe_sample

    def generate_for_type(
        self,
        vuln_type: str,
        num_pairs: int
    ) -> List[Dict]:
        """
        Generate samples for a specific vulnerability type.

        Args:
            vuln_type: Vulnerability type to generate
            num_pairs: Number of vulnerable/safe pairs to generate

        Returns:
            List of generated samples
        """
        if vuln_type not in self.vulnerability_templates:
            raise ValueError(f"Unknown vulnerability type: {vuln_type}")

        templates = self.vulnerability_templates[vuln_type]
        num_vuln_templates = len(templates["vulnerable"])
        num_safe_templates = len(templates["safe"])

        samples = []

        for _ in range(num_pairs):
            # Randomly select template indices
            vuln_idx = random.randint(0, num_vuln_templates - 1)
            safe_idx = random.randint(0, num_safe_templates - 1)

            # Create pair
            vuln_sample, safe_sample = self.create_vulnerable_safe_pair(
                vuln_type, vuln_idx, safe_idx
            )

            samples.extend([vuln_sample, safe_sample])

        return samples

    def generate_all(self, total_samples: int = 500) -> List[Dict]:
        """
        Generate all synthetic samples.

        Args:
            total_samples: Total number of samples to generate (must be even)

        Returns:
            List of all generated samples
        """
        if total_samples % 2 != 0:
            raise ValueError("total_samples must be even (pairs of vulnerable/safe)")

        num_pairs = total_samples // 2
        vuln_types = list(self.vulnerability_templates.keys())

        # Distribute pairs evenly across vulnerability types
        pairs_per_type = num_pairs // len(vuln_types)
        remaining_pairs = num_pairs % len(vuln_types)

        all_samples = []

        print(f"\nGenerating {total_samples} synthetic samples...")
        print(f"Target: {num_pairs} pairs ({pairs_per_type} pairs per type)")
        print(f"Vulnerability types: {', '.join(vuln_types)}\n")

        for i, vuln_type in enumerate(vuln_types):
            # Add extra pair to first types if there's a remainder
            num_pairs_for_type = pairs_per_type + (1 if i < remaining_pairs else 0)

            print(f"Generating {num_pairs_for_type} pairs for {vuln_type}...")
            samples = self.generate_for_type(vuln_type, num_pairs_for_type)
            all_samples.extend(samples)

            self.samples_collected += len(samples)

        # Shuffle to mix vulnerable and safe samples
        random.shuffle(all_samples)

        print(f"\nGenerated {len(all_samples)} total samples")
        print(f"Vulnerable: {sum(1 for s in all_samples if s['vulnerable'])}")
        print(f"Safe: {sum(1 for s in all_samples if not s['vulnerable'])}")

        return all_samples

    def generate_samples(self, num_samples: int) -> List[Dict]:
        """
        Generate synthetic samples (API compatible with orchestrator).

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of generated samples
        """
        samples = self.generate_all(total_samples=num_samples)

        # Save the generated samples
        output_file = self.save_samples(samples, "synthetic_data.jsonl")
        print(f"\nSaved {len(samples)} samples to: {output_file}\n")

        return samples

    def collect(self) -> List[Dict]:
        """
        Collect synthetic data (implements BaseCollector.collect).

        Returns:
            List of generated samples
        """
        return self.generate_all(total_samples=500)

    def validate_dataset(self, samples: List[Dict]) -> Dict:
        """
        Validate the generated dataset.

        Args:
            samples: List of samples to validate

        Returns:
            Validation statistics
        """
        stats = {
            "total_samples": len(samples),
            "vulnerable_count": 0,
            "safe_count": 0,
            "by_type": {},
            "by_language": {},
            "has_counterfactuals": 0,
            "avg_code_length": 0
        }

        total_length = 0

        for sample in samples:
            # Count by vulnerability status
            if sample["vulnerable"]:
                stats["vulnerable_count"] += 1
            else:
                stats["safe_count"] += 1

            # Count by type
            vuln_type = sample["vulnerability_type"]
            if vuln_type not in stats["by_type"]:
                stats["by_type"][vuln_type] = {"vulnerable": 0, "safe": 0}

            if sample["vulnerable"]:
                stats["by_type"][vuln_type]["vulnerable"] += 1
            else:
                stats["by_type"][vuln_type]["safe"] += 1

            # Count by language
            lang = sample["language"]
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1

            # Count counterfactuals
            if sample.get("counterfactual"):
                stats["has_counterfactuals"] += 1

            # Track length
            total_length += len(sample["code"])

        stats["avg_code_length"] = total_length / len(samples) if samples else 0

        return stats


def main():
    """Main entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic vulnerability samples for StreamGuard"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/synthetic",
        help="Output directory for synthetic data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Total number of samples to generate (must be even)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="synthetic_data.jsonl",
        help="Output filename"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after generation"
    )

    args = parser.parse_args()

    # Create generator
    print("=" * 60)
    print("StreamGuard Synthetic Data Generator")
    print("=" * 60)

    generator = SyntheticGenerator(
        output_dir=args.output_dir,
        seed=args.seed
    )

    # Generate samples
    samples = generator.generate_all(total_samples=args.num_samples)

    # Save to file
    print(f"\nSaving to {args.output_file}...")
    output_path = generator.save_samples(samples, args.output_file)
    print(f"Saved to: {output_path}")

    # Validate if requested
    if args.validate:
        print("\n" + "=" * 60)
        print("Dataset Validation")
        print("=" * 60)

        stats = generator.validate_dataset(samples)

        print(f"\nTotal samples: {stats['total_samples']}")
        print(f"Vulnerable: {stats['vulnerable_count']}")
        print(f"Safe: {stats['safe_count']}")
        print(f"With counterfactuals: {stats['has_counterfactuals']}")
        print(f"Average code length: {stats['avg_code_length']:.1f} characters")

        print("\nBreakdown by vulnerability type:")
        for vuln_type, counts in stats["by_type"].items():
            print(f"  {vuln_type}:")
            print(f"    Vulnerable: {counts['vulnerable']}")
            print(f"    Safe: {counts['safe']}")

        print("\nBreakdown by language:")
        for lang, count in stats["by_language"].items():
            print(f"  {lang}: {count}")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
