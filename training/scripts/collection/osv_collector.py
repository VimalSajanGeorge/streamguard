"""
OSV (Open Source Vulnerabilities) Database Collector for StreamGuard.

Collects vulnerability data from OSV.dev, which aggregates vulnerabilities
from multiple sources including GitHub Security Advisories, NVD, and more.

OSV covers 20+ ecosystems with 100,000+ vulnerabilities.
API: https://osv.dev/docs/
"""

import json
import time
import requests
import zipfile
import io
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))

from base_collector import BaseCollector
from checkpoint_manager import CheckpointManager


class OSVCollector(BaseCollector):
    """Collector for OSV (Open Source Vulnerabilities) database."""

    # OSV API endpoint
    OSV_API = "https://api.osv.dev/v1"

    # OSV GCS bucket for bulk data
    OSV_GCS_BASE = "https://osv-vulnerabilities.storage.googleapis.com"

    # Supported ecosystems
    ECOSYSTEMS = [
        "PyPI",           # Python
        "npm",            # JavaScript/Node.js
        "Maven",          # Java
        "Go",             # Go
        "crates.io",      # Rust
        "RubyGems",       # Ruby
        "Packagist",      # PHP
        "NuGet",          # .NET
        "Hex",            # Erlang/Elixir
        "Pub",            # Dart
    ]

    def __init__(self, output_dir: str = "data/raw/osv", cache_enabled: bool = True, resume: bool = False):
        """
        Initialize OSV Collector.

        Args:
            output_dir: Directory to save collected data
            cache_enabled: Whether to enable caching
            resume: Whether to resume from checkpoint
        """
        super().__init__(output_dir, cache_enabled)

        # Collection statistics
        self.stats = {
            "total_vulnerabilities": 0,
            "by_ecosystem": defaultdict(int),
            "by_severity": defaultdict(int),
            "with_code_refs": 0,
            "with_patches": 0
        }

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager()
        self.resume = resume
        self.checkpoint_interval = 300  # Save checkpoint every 5 minutes
        self.last_checkpoint_time = time.time()

    def collect(self) -> List[Dict]:
        """
        Main collection method - collects OSV vulnerabilities.

        Returns:
            List of collected vulnerability samples
        """
        print("Starting OSV Database Collection")
        print(f"Target: 20,000 samples across {len(self.ECOSYSTEMS)} ecosystems")

        return self.collect_all_vulnerabilities(target_samples=20000)

    def collect_all_vulnerabilities(self, target_samples: int = 20000) -> List[Dict]:
        """
        Collect vulnerabilities from OSV across all ecosystems.

        Args:
            target_samples: Target number of samples to collect

        Returns:
            List of collected vulnerability samples
        """
        all_samples = []
        samples_per_ecosystem = target_samples // len(self.ECOSYSTEMS)
        processed_ecosystems = []

        # Check for existing checkpoint
        if self.resume:
            checkpoint = self.checkpoint_manager.load_checkpoint("osv")
            if checkpoint:
                print("\n[+] Found existing checkpoint, resuming collection...")
                all_samples = checkpoint.get("samples", [])
                processed_ecosystems = checkpoint.get("state", {}).get("processed_ecosystems", [])
                self.samples_collected = len(all_samples)
                print(f"[+] Resuming with {len(all_samples)} samples already collected")
                print(f"[+] Already processed ecosystems: {', '.join(processed_ecosystems)}")
            else:
                print("\n[!] No checkpoint found, starting fresh collection...")

        print(f"\nTarget per ecosystem: {samples_per_ecosystem}")

        for ecosystem in self.ECOSYSTEMS:
            # Skip already processed ecosystems
            if ecosystem in processed_ecosystems:
                print(f"\n[*] Skipping {ecosystem} (already processed)")
                continue
            print(f"\n{'='*60}")
            print(f"Collecting: {ecosystem}")
            print(f"{'='*60}")

            try:
                samples = self.collect_by_ecosystem(
                    ecosystem=ecosystem,
                    max_samples=samples_per_ecosystem
                )

                all_samples.extend(samples)
                processed_ecosystems.append(ecosystem)

                print(f"Collected {len(samples)} samples for {ecosystem}")
                print(f"Total samples so far: {len(all_samples)}")

                # Save checkpoint after each ecosystem
                self._save_checkpoint(all_samples, processed_ecosystems, target_samples)

                # Save intermediate results
                if len(all_samples) % 5000 == 0:
                    self._save_intermediate_results(all_samples)

            except Exception as e:
                error_msg = f"Error collecting {ecosystem}: {str(e)}"
                print(f"ERROR: {error_msg}")
                self.log_error(error_msg, {"ecosystem": ecosystem})

            # Rate limiting
            time.sleep(1)  # Be nice to OSV API

        # Deduplicate samples
        unique_samples = self.deduplicate_samples(all_samples, key="vulnerability_id")
        print(f"\nTotal unique samples collected: {len(unique_samples)}")

        # Save final results
        if len(unique_samples) > 0:
            output_file = self.save_samples(unique_samples, "osv_vulnerabilities.jsonl")
            print(f"\nSaved to: {output_file}")
        else:
            print("\nWARNING: No samples to save!")

        # Print statistics
        self._print_statistics()

        # Delete checkpoint after successful completion
        if self.checkpoint_manager.checkpoint_exists("osv"):
            self.checkpoint_manager.delete_checkpoint("osv")
            print("\n[+] Checkpoint deleted (collection completed)")

        return unique_samples

    def collect_by_ecosystem(self, ecosystem: str, max_samples: int = 2000) -> List[Dict]:
        """
        Collect vulnerabilities for a specific ecosystem.

        Args:
            ecosystem: Package ecosystem (PyPI, npm, etc.)
            max_samples: Maximum samples to collect

        Returns:
            List of vulnerability samples
        """
        samples = []

        # Download vulnerability IDs from GCS
        vuln_ids = self._download_ecosystem_vulns_from_gcs(ecosystem)

        if not vuln_ids:
            print(f"No vulnerabilities found for {ecosystem}")
            return samples

        print(f"Found {len(vuln_ids)} vulnerabilities for {ecosystem}")
        print(f"Will collect up to {max_samples} samples")

        # Fetch and process vulnerabilities up to max_samples
        successful = 0
        failed = 0

        for idx, vuln_id in enumerate(vuln_ids[:max_samples]):
            if idx % 10 == 0 and idx > 0:
                success_rate = (successful / idx) * 100 if idx > 0 else 0
                print(f"  Processed {idx}/{min(len(vuln_ids), max_samples)} - Success: {successful} ({success_rate:.1f}%), Failed: {failed}")

            try:
                # Fetch vulnerability details via API
                vuln_data = self._fetch_vulnerability_details(vuln_id)
                if not vuln_data:
                    failed += 1
                    continue

                # Process vulnerability data
                sample = self._process_vulnerability(vuln_data, ecosystem)
                if sample:
                    samples.append(sample)
                    self.samples_collected += 1
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                self.log_error(f"Failed to process {vuln_id}: {str(e)}")
                failed += 1
                continue

        # Final success rate
        total_processed = successful + failed
        success_rate = (successful / total_processed) * 100 if total_processed > 0 else 0
        print(f"\nFinal: {successful}/{total_processed} successful ({success_rate:.1f}%)")

        return samples

    def _download_ecosystem_vulns_from_gcs(self, ecosystem: str) -> List[str]:
        """
        Download vulnerability IDs from OSV GCS bucket.

        OSV provides complete vulnerability lists at:
        https://osv-vulnerabilities.storage.googleapis.com/{ecosystem}/all.zip

        Args:
            ecosystem: Package ecosystem

        Returns:
            List of vulnerability IDs
        """
        # Check cache first
        cache_key = self.make_cache_key("gcs_vuln_list", ecosystem)
        cached_result = self.load_cache(cache_key)
        if cached_result:
            print(f"Using cached vulnerability list for {ecosystem}")
            return cached_result.get("vuln_ids", [])

        vuln_ids = []

        try:
            # Construct GCS URL
            url = f"{self.OSV_GCS_BASE}/{ecosystem}/all.zip"
            print(f"Downloading vulnerability list from {url}...")

            # Download ZIP file
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()

            # Parse ZIP file
            print(f"Parsing ZIP file for {ecosystem}...")
            zip_data = io.BytesIO(response.content)

            with zipfile.ZipFile(zip_data) as zip_file:
                # Iterate through JSON files in ZIP
                json_files = [f for f in zip_file.namelist() if f.endswith('.json')]
                print(f"Found {len(json_files)} vulnerability files")

                for json_file in json_files:
                    try:
                        # Read JSON file and extract ID only
                        with zip_file.open(json_file) as f:
                            vuln_data = json.load(f)
                            vuln_id = vuln_data.get("id")
                            if vuln_id:
                                vuln_ids.append(vuln_id)
                    except Exception as e:
                        self.log_error(f"Failed to parse {json_file}: {str(e)}")
                        continue

            print(f"Extracted {len(vuln_ids)} vulnerability IDs from {ecosystem}")

            # Cache the IDs
            self.save_cache(cache_key, {"vuln_ids": vuln_ids})

            return vuln_ids

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"No GCS data found for {ecosystem} (404)")
            else:
                print(f"HTTP error downloading {ecosystem}: {e}")
            self.log_error(f"GCS download failed for {ecosystem}: {str(e)}")
            return []

        except Exception as e:
            print(f"Error downloading from GCS for {ecosystem}: {str(e)}")
            self.log_error(f"GCS download failed for {ecosystem}: {str(e)}")
            return []

    def _query_ecosystem_vulnerabilities(
        self,
        ecosystem: str,
        page: int = 0,
        page_size: int = 100
    ) -> Optional[Dict]:
        """
        Query OSV for vulnerabilities in an ecosystem.

        Args:
            ecosystem: Package ecosystem
            page: Page number
            page_size: Results per page

        Returns:
            Query result or None
        """
        # Check cache
        cache_key = self.make_cache_key("osv_query", ecosystem, page)
        cached_result = self.load_cache(cache_key)
        if cached_result:
            return cached_result

        try:
            # OSV query endpoint
            url = f"{self.OSV_API}/query"

            # Query body
            body = {
                "package": {
                    "ecosystem": ecosystem
                },
                "page_token": str(page * page_size) if page > 0 else None
            }

            response = requests.post(url, json=body, timeout=30)
            response.raise_for_status()

            result = response.json()

            # Save to cache
            self.save_cache(cache_key, result)

            return result

        except Exception as e:
            self.log_error(f"Query failed: {str(e)}", {"ecosystem": ecosystem, "page": page})
            return None

    def _fetch_vulnerability_details(self, vuln_id: str) -> Optional[Dict]:
        """
        Fetch detailed information for a vulnerability.

        Args:
            vuln_id: OSV vulnerability ID

        Returns:
            Vulnerability details or None
        """
        # Check cache
        cache_key = self.make_cache_key("osv_vuln", vuln_id)
        cached_result = self.load_cache(cache_key)
        if cached_result:
            return cached_result

        try:
            url = f"{self.OSV_API}/vulns/{vuln_id}"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            result = response.json()

            # Save to cache
            self.save_cache(cache_key, result)

            return result

        except Exception as e:
            self.log_error(f"Failed to fetch {vuln_id}: {str(e)}")
            return None

    def _process_vulnerability(self, vuln: Dict, ecosystem: str) -> Optional[Dict]:
        """
        Process a vulnerability and extract relevant information.

        Args:
            vuln: Vulnerability data from OSV
            ecosystem: Package ecosystem

        Returns:
            Processed sample or None
        """
        vuln_id = vuln.get("id")
        summary = vuln.get("summary", "")
        details = vuln.get("details", "")
        published = vuln.get("published")
        modified = vuln.get("modified")

        # Get affected packages
        affected = vuln.get("affected", [])
        if not affected:
            return None

        # Get first affected package
        first_affected = affected[0]
        package_info = first_affected.get("package", {})
        package_name = package_info.get("name")

        if not package_name:
            return None

        # Get version information
        ranges = first_affected.get("ranges", [])
        versions = first_affected.get("versions", [])

        # Get severity
        severity = self._extract_severity(vuln)

        # Get references
        references = vuln.get("references", [])
        reference_urls = [ref.get("url") for ref in references if ref.get("url")]

        # Try to extract code from references
        vulnerable_code, fixed_code = self._extract_code_from_references(reference_urls)

        # Update statistics
        self.stats["total_vulnerabilities"] += 1
        self.stats["by_ecosystem"][ecosystem] += 1
        self.stats["by_severity"][severity] += 1

        if vulnerable_code or fixed_code:
            self.stats["with_code_refs"] += 1

        # Create sample
        sample = {
            "vulnerability_id": vuln_id,
            "description": f"{summary}\n\n{details}".strip(),
            "vulnerable_code": vulnerable_code or f"# Vulnerable package: {package_name} (see references)",
            "fixed_code": fixed_code or f"# Fixed package: {package_name} (see references)",
            "ecosystem": ecosystem,
            "severity": severity,
            "published_at": published,
            "modified_at": modified,
            "source": "osv",
            "metadata": {
                "package_name": package_name,
                "affected_versions": versions,
                "ranges": ranges,
                "references": reference_urls,
                "vulnerability_type": self.extract_vulnerability_type(details or summary),
                "aliases": vuln.get("aliases", [])
            }
        }

        return sample

    def _extract_severity(self, vuln: Dict) -> str:
        """
        Extract severity from vulnerability data.

        Args:
            vuln: Vulnerability data

        Returns:
            Severity string
        """
        # OSV uses severity field
        severity_list = vuln.get("severity", [])

        if not severity_list:
            return "UNKNOWN"

        # Get first severity entry
        first_severity = severity_list[0]
        severity_type = first_severity.get("type", "")

        if severity_type == "CVSS_V3":
            score = first_severity.get("score", "")
            # Extract numeric score from CVSS vector
            # e.g., "CVSS:3.1/AV:N/AC:L/..."
            if isinstance(score, str) and "/" in score:
                return "HIGH"  # Default for CVSS
            return "MEDIUM"
        else:
            return first_severity.get("severity", "UNKNOWN").upper()

    def _extract_code_from_references(
        self,
        references: List[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract code from reference URLs (if they point to commits/diffs).

        Args:
            references: List of reference URLs

        Returns:
            Tuple of (vulnerable_code, fixed_code)
        """
        # For now, return None - could be enhanced to fetch from GitHub commits
        # This would require GitHub API integration similar to GitHub collector

        return None, None

    def _save_checkpoint(self, samples: List[Dict], processed_ecosystems: List[str], target_samples: int):
        """Save checkpoint for resuming collection."""
        try:
            state = {
                "processed_ecosystems": processed_ecosystems,
                "target_samples": target_samples,
                "samples_per_ecosystem": target_samples // len(self.ECOSYSTEMS)
            }
            checkpoint_file = self.checkpoint_manager.save_checkpoint("osv", state, samples)
            print(f"[+] Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            print(f"[!] Warning: Failed to save checkpoint: {str(e)}")

    def _save_intermediate_results(self, samples: List[Dict]):
        """Save intermediate results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"osv_vulnerabilities_intermediate_{timestamp}.jsonl"
        self.save_samples(samples, filename)
        print(f"\nSaved intermediate results: {filename}")

    def _print_statistics(self):
        """Print collection statistics."""
        print("\n" + "="*60)
        print("COLLECTION STATISTICS")
        print("="*60)
        print(f"Total Vulnerabilities: {self.stats['total_vulnerabilities']}")
        print(f"With Code References: {self.stats['with_code_refs']}")

        print("\nBy Ecosystem:")
        for ecosystem, count in sorted(self.stats['by_ecosystem'].items()):
            print(f"  {ecosystem}: {count}")

        print("\nBy Severity:")
        for severity, count in sorted(self.stats['by_severity'].items()):
            print(f"  {severity}: {count}")

        print("\nErrors:", len(self.errors))
        if self.errors:
            print("\nLast 5 errors:")
            for error in self.errors[-5:]:
                print(f"  - {error['error']}")


def main():
    """Main entry point for the collector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="OSV (Open Source Vulnerabilities) Collector"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/osv",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=20000,
        help="Target number of samples to collect"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    parser.add_argument(
        "--ecosystem",
        choices=OSVCollector.ECOSYSTEMS,
        help="Collect from specific ecosystem only"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    args = parser.parse_args()

    # Initialize collector
    collector = OSVCollector(
        output_dir=args.output_dir,
        cache_enabled=not args.no_cache,
        resume=args.resume
    )

    # Collect data
    try:
        if args.ecosystem:
            # Collect from specific ecosystem
            samples = collector.collect_by_ecosystem(
                ecosystem=args.ecosystem,
                max_samples=args.target_samples
            )
            output_file = collector.save_samples(samples, f"osv_{args.ecosystem.lower()}.jsonl")
            print(f"\nSaved to: {output_file}")
        else:
            # Collect from all ecosystems
            samples = collector.collect_all_vulnerabilities(target_samples=args.target_samples)

        print(f"\nCollection complete! Collected {len(samples)} samples.")

    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
        print(f"Collected {collector.samples_collected} samples before interruption.")

    except Exception as e:
        print(f"\n\nCollection failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Print final statistics
        stats = collector.get_stats()
        print(f"\nFinal statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
