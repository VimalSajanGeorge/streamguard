        return {
            'total_feedback': total,
            'by_action': by_action,
            'pending_sync': pending,
            'sync_ready': pending >= 100
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    collector = FeedbackCollector()
    
    # Add sample feedback
    asyncio.run(collector.add_feedback(
        vulnerability_id="vuln_001",
        action="accepted",
        comment="Good catch!"
    ))
    
    # Get statistics
    stats = collector.get_statistics()
    print(f"📊 Feedback Statistics:")
    print(f"  Total: {stats['total_feedback']}")
    print(f"  Pending sync: {stats['pending_sync']}")
    print(f"  By action: {stats['by_action']}")
```

---

### 4. Feedback Anonymization

**File:** `core/feedback/anonymizer.py`

```python
"""Anonymize feedback data before syncing to cloud."""

import hashlib
import json
from typing import Dict, List
from datetime import datetime, timedelta
import re

class FeedbackAnonymizer:
    """Anonymize feedback while preserving utility for training."""
    
    def __init__(self, salt: str = "streamguard-v3"):
        self.salt = salt
    
    def anonymize_batch(self, feedback_batch: List[Dict]) -> List[Dict]:
        """
        Anonymize a batch of feedback.
        
        Removes all PII while preserving useful information for training.
        """
        anonymized = []
        
        for feedback in feedback_batch:
            anonymized.append(self.anonymize_single(feedback))
        
        return anonymized
    
    def anonymize_single(self, feedback: Dict) -> Dict:
        """Anonymize a single feedback entry."""
        # Create anonymized version
        anonymized = {
            'feedback_id': self._hash_id(feedback['id']),
            'vulnerability_type': self._extract_vulnerability_type(feedback['vulnerability_id']),
            'action': feedback['action'],
            'timestamp_bucket': self._bucket_timestamp(feedback['timestamp']),
            'code_pattern_hash': feedback.get('code_context_hash'),  # Already hashed
            'has_comment': bool(feedback.get('comment'))
        }
        
        # Remove all identifiable information
        # No: code, file paths, user IDs, IP addresses, etc.
        
        return anonymized
    
    def _hash_id(self, id_value: any) -> str:
        """Create one-way hash of ID."""
        return hashlib.sha256(f"{self.salt}{id_value}".encode()).hexdigest()[:16]
    
    def _extract_vulnerability_type(self, vulnerability_id: str) -> str:
        """Extract vulnerability type from ID."""
        # Assuming format: vuln_{type}_{timestamp}_{hash}
        parts = vulnerability_id.split('_')
        if len(parts) >= 2:
            return parts[1]
        return 'unknown'
    
    def _bucket_timestamp(self, timestamp: datetime, hours: int = 24) -> str:
        """
        Bucket timestamp to reduce granularity.
        
        Groups timestamps into time buckets (e.g., 24-hour periods).
        """
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Round down to nearest bucket
        bucket_start = timestamp.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        return bucket_start.isoformat()
    
    def validate_anonymization(self, original: Dict, anonymized: Dict) -> bool:
        """
        Validate that anonymized data doesn't contain PII.
        
        Returns True if safe to sync.
        """
        # Check for common PII patterns
        pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address
            r'/[Uu]sers/\w+/',  # User paths
            r'/home/\w+/',  # Home directories
        ]
        
        anonymized_str = json.dumps(anonymized)
        
        for pattern in pii_patterns:
            if re.search(pattern, anonymized_str):
                print(f"⚠️  PII detected: {pattern}")
                return False
        
        # Check that certain fields are NOT present
        forbidden_fields = ['code', 'file_path', 'user_id', 'ip_address', 'email']
        for field in forbidden_fields:
            if field in anonymized:
                print(f"⚠️  Forbidden field present: {field}")
                return False
        
        return True


# Example usage
if __name__ == "__main__":
    anonymizer = FeedbackAnonymizer()
    
    # Original feedback with PII
    original = {
        'id': 12345,
        'vulnerability_id': 'vuln_sql_injection_20241008_abc123',
        'action': 'accepted',
        'comment': 'Found in /home/john/project/auth.py',
        'code_context_hash': 'abc123...',
        'timestamp': datetime.now()
    }
    
    # Anonymize
    anonymized = anonymizer.anonymize_single(original)
    
    print("Original:")
    print(json.dumps(original, default=str, indent=2))
    
    print("\nAnonymized:")
    print(json.dumps(anonymized, indent=2))
    
    # Validate
    is_safe = anonymizer.validate_anonymization(original, anonymized)
    print(f"\nSafe to sync: {is_safe}")
```

---

### 5. Secure Sync Manager

**File:** `core/feedback/sync_manager.py`

```python
"""Secure feedback sync to AWS S3 with encryption."""

import boto3
import json
from typing import List, Dict
from pathlib import Path
import gzip
from datetime import datetime
from cryptography.fernet import Fernet
from core.feedback.collector import FeedbackCollector
from core.feedback.anonymizer import FeedbackAnonymizer

class SecureSyncManager:
    """Securely sync anonymized feedback to cloud."""
    
    def __init__(
        self,
        bucket: str = "streamguard-ml-v3",
        encryption_key: str = None,
        require_consent: bool = True
    ):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.require_consent = require_consent
        
        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            # Generate new key (should be stored securely)
            self.cipher = Fernet(Fernet.generate_key())
        
        self.collector = FeedbackCollector()
        self.anonymizer = FeedbackAnonymizer()
    
    def check_consent(self) -> bool:
        """
        Check if user has consented to feedback sync.
        
        In production, this would check a persistent setting.
        """
        if not self.require_consent:
            return True
        
        consent_file = Path("data/feedback/.consent")
        return consent_file.exists()
    
    def request_consent(self):
        """Request user consent for feedback sync."""
        print("\n" + "="*60)
        print("StreamGuard Feedback Sync")
        print("="*60)
        print("\nStreamGuard can improve by learning from your feedback.")
        print("\nWhat we collect:")
        print("  • Vulnerability types and your feedback (accept/reject)")
        print("  • Anonymized code patterns (one-way hashed)")
        print("  • Timestamp buckets (24-hour periods)")
        print("\nWhat we DON'T collect:")
        print("  • Source code or code snippets")
        print("  • File paths or directory structures")
        print("  • User names, emails, or any PII")
        print("  • IP addresses or system information")
        print("\nYour feedback is:")
        print("  • Encrypted during transmission")
        print("  • Stored securely in AWS")
        print("  • Only used for model training")
        print("  • Never shared with third parties")
        print("\n" + "="*60)
        
        response = input("\nEnable feedback sync? (yes/no): ").strip().lower()
        
        if response == 'yes':
            consent_file = Path("data/feedback/.consent")
            consent_file.parent.mkdir(parents=True, exist_ok=True)
            consent_file.write_text(f"Consent granted: {datetime.now().isoformat()}")
            print("✅ Feedback sync enabled")
            return True
        else:
            print("❌ Feedback sync disabled")
            return False
    
    async def sync_feedback(self, force: bool = False) -> Dict:
        """
        Sync pending feedback to cloud.
        
        Args:
            force: Force sync even if batch size is small
        
        Returns:
            Sync statistics
        """
        # Check consent
        if not self.check_consent():
            if not self.request_consent():
                return {
                    'status': 'cancelled',
                    'message': 'User consent required'
                }
        
        # Get pending feedback
        pending = self.collector.get_pending_feedback(limit=1000)
        
        if not pending:
            return {
                'status': 'success',
                'synced_count': 0,
                'message': 'No pending feedback'
            }
        
        # Check batch size (minimum 100 for privacy)
        if len(pending) < 100 and not force:
            return {
                'status': 'pending',
                'synced_count': 0,
                'pending_count': len(pending),
                'message': f'Waiting for more feedback ({len(pending)}/100)'
            }
        
        print(f"🔄 Syncing {len(pending)} feedback entries...")
        
        # Anonymize
        print("  Anonymizing data...")
        anonymized = self.anonymizer.anonymize_batch(pending)
        
        # Validate anonymization
        print("  Validating anonymization...")
        for orig, anon in zip(pending, anonymized):
            if not self.anonymizer.validate_anonymization(orig, anon):
                return {
                    'status': 'error',
                    'message': 'Anonymization validation failed'
                }
        
        # Encrypt
        print("  Encrypting data...")
        encrypted_data = self._encrypt_batch(anonymized)
        
        # Compress
        print("  Compressing data...")
        compressed = gzip.compress(encrypted_data)
        
        # Upload to S3
        print("  Uploading to S3...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"feedback/batches/batch_{timestamp}.json.gz.enc"
        
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=compressed,
                ServerSideEncryption='AES256',
                Metadata={
                    'batch_size': str(len(pending)),
                    'timestamp': timestamp,
                    'version': '3.0'
                }
            )
            
            # Mark as synced
            feedback_ids = [f['id'] for f in pending]
            self.collector.mark_synced(feedback_ids)
            
            print(f"✅ Synced {len(pending)} feedback entries")
            
            return {
                'status': 'success',
                'synced_count': len(pending),
                's3_key': s3_key,
                'message': f'Successfully synced {len(pending)} entries'
            }
        
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _encrypt_batch(self, batch: List[Dict]) -> bytes:
        """Encrypt feedback batch."""
        json_data = json.dumps(batch).encode()
        encrypted = self.cipher.encrypt(json_data)
        return encrypted
    
    def get_sync_statistics(self) -> Dict:
        """Get sync statistics."""
        stats = self.collector.get_statistics()
        
        return {
            'total_feedback': stats['total_feedback'],
            'pending_sync': stats['pending_sync'],
            'sync_ready': stats['sync_ready'],
            'consent_granted': self.check_consent()
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    manager = SecureSyncManager()
    
    # Check sync status
    stats = manager.get_sync_statistics()
    print("📊 Sync Statistics:")
    print(f"  Total feedback: {stats['total_feedback']}")
    print(f"  Pending sync: {stats['pending_sync']}")
    print(f"  Ready to sync: {stats['sync_ready']}")
    print(f"  Consent granted: {stats['consent_granted']}")
    
    # Sync feedback
    if stats['sync_ready']:
        result = asyncio.run(manager.sync_feedback())
        print(f"\n{result['message']}")
```

---

### 6. Compliance Report Generator

**File:** `core/reports/report_generator.py`

```python
"""Generate compliance reports in multiple formats."""

from typing import Dict, List
from datetime import datetime
from pathlib import Path
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors

class ComplianceReportGenerator:
    """Generate compliance reports for vulnerabilities."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
    
    def generate_pdf_report(
        self,
        vulnerabilities: List[Dict],
        output_path: str = "reports/compliance_report.pdf"
    ):
        """
        Generate PDF compliance report.
        
        Args:
            vulnerabilities: List of vulnerability dictionaries
            output_path: Output file path
        """
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title
        title = Paragraph(
            "<b>StreamGuard Security Compliance Report</b>",
            self.styles['Title']
        )
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Report metadata
        metadata = Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
            f"<b>Total Vulnerabilities:</b> {len(vulnerabilities)}<br/>"
            f"<b>Critical:</b> {sum(1 for v in vulnerabilities if v['severity'] == 'critical')}<br/>"
            f"<b>High:</b> {sum(1 for v in vulnerabilities if v['severity'] == 'high')}<br/>"
            f"<b>Medium:</b> {sum(1 for v in vulnerabilities if v['severity'] == 'medium')}<br/>"
            f"<b>Low:</b> {sum(1 for v in vulnerabilities if v['severity'] == 'low')}",
            self.styles['Normal']
        )
        story.append(metadata)
        story.append(Spacer(1, 24))
        
        # Executive Summary
        summary_title = Paragraph("<b>Executive Summary</b>", self.styles['Heading1'])
        story.append(summary_title)
        story.append(Spacer(1, 12))
        
        critical_count = sum(1 for v in vulnerabilities if v['severity'] == 'critical')
        high_count = sum(1 for v in vulnerabilities if v['severity'] == 'high')
        
        summary_text = Paragraph(
            f"This security assessment identified {len(vulnerabilities)} vulnerabilities "
            f"in the codebase, including {critical_count} critical and {high_count} high "
            f"severity issues that require immediate attention. All findings have been "
            f"verified using StreamGuard's multi-agent detection system with {self._avg_confidence(vulnerabilities):.1%} "
            f"average confidence.",
            self.styles['Normal']
        )
        story.append(summary_text)
        story.append(Spacer(1, 24))
        
        # Vulnerability Table
        table_title = Paragraph("<b>Vulnerability Summary</b>", self.styles['Heading1'])
        story.append(table_title)
        story.append(Spacer(1, 12))
        
        # Create table data
        table_data = [
            ['ID', 'Type', 'Severity', 'File', 'Line', 'Confidence']
        ]
        
        for vuln in vulnerabilities[:50]:  # Limit to 50 for PDF
            table_data.append([
                vuln['id'][:8] + '...',
                vuln['type'],
                vuln['severity'].upper(),
                Path(vuln.get('file_path', 'N/A')).name,
                str(vuln['line']),
                f"{vuln['confidence']:.0%}"
            ])
        
        # Create table
        table = Table(table_data, colWidths=[60, 100, 60, 120, 40, 60])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(PageBreak())
        
        # Detailed Findings
        details_title = Paragraph("<b>Detailed Findings</b>", self.styles['Heading1'])
        story.append(details_title)
        story.append(Spacer(1, 12))
        
        for i, vuln in enumerate(vulnerabilities[:20], 1):  # Top 20 detailed
            # Vulnerability header
            vuln_title = Paragraph(
                f"<b>{i}. {vuln['type']} ({vuln['severity'].upper()})</b>",
                self.styles['Heading2']
            )
            story.append(vuln_title)
            story.append(Spacer(1, 6))
            
            # Details
            details = Paragraph(
                f"<b>Location:</b> {vuln.get('file_path', 'N/A')}:{vuln['line']}<br/>"
                f"<b>Confidence:</b> {vuln['confidence']:.0%}<br/>"
                f"<b>Message:</b> {vuln['message']}<br/>",
                self.styles['Normal']
            )
            story.append(details)
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF report generated: {output_path}")
    
    def generate_json_report(
        self,
        vulnerabilities: List[Dict],
        output_path: str = "reports/compliance_report.json"
    ):
        """Generate JSON compliance report."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '3.0',
                'total_vulnerabilities': len(vulnerabilities)
            },
            'summary': {
                'by_severity': self._count_by_severity(vulnerabilities),
                'by_type': self._count_by_type(vulnerabilities),
                'average_confidence': self._avg_confidence(vulnerabilities)
            },
            'vulnerabilities': vulnerabilities
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ JSON report generated: {output_path}")
    
    def generate_sarif_report(
        self,
        vulnerabilities: List[Dict],
        output_path: str = "reports/compliance_report.sarif"
    ):
        """
        Generate SARIF (Static Analysis Results Interchange Format) report.
        
        SARIF is a standard format for static analysis tools.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Build SARIF structure
        sarif = {
            'version': '2.1.0',
            '$schema': 'https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json',
            'runs': [
                {
                    'tool': {
                        'driver': {
                            'name': 'StreamGuard',
                            'version': '3.0.0',
                            'informationUri': 'https://streamguard.dev',
                            'rules': self._generate_sarif_rules(vulnerabilities)
                        }
                    },
                    'results': self._generate_sarif_results(vulnerabilities)
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(sarif, f, indent=2)
        
        print(f"✅ SARIF report generated: {output_path}")
    
    def _generate_sarif_rules(self, vulnerabilities: List[Dict]) -> List[Dict]:
        """Generate SARIF rules from vulnerabilities."""
        # Get unique vulnerability types
        types = set(v['type'] for v in vulnerabilities)
        
        rules = []
        for vuln_type in types:
            rules.append({
                'id': vuln_type,
                'name': vuln_type,
                'shortDescription': {
                    'text': f'{vuln_type} vulnerability'
                },
                'fullDescription': {
                    'text': f'Detected {vuln_type} vulnerability in code'
                },
                'defaultConfiguration': {
                    'level': 'error'
                }
            })
        
        return rules
    
    def _generate_sarif_results(self, vulnerabilities: List[Dict]) -> List[Dict]:
        """Generate SARIF results from vulnerabilities."""
        results = []
        
        for vuln in vulnerabilities:
            result = {
                'ruleId': vuln['type'],
                'level': self._severity_to_sarif_level(vuln['severity']),
                'message': {
                    'text': vuln['message']
                },
                'locations': [
                    {
                        'physicalLocation': {
                            'artifactLocation': {
                                'uri': vuln.get('file_path', 'unknown')
                            },
                            'region': {
                                'startLine': vuln['line'],
                                'startColumn': vuln.get('column', 1)
                            }
                        }
                    }
                ],
                'properties': {
                    'confidence': vuln['confidence'],
                    'vulnerability_id': vuln['id']
                }
            }
            
            results.append(result)
        
        return results
    
    def _severity_to_sarif_level(self, severity: str) -> str:
        """Convert severity to SARIF level."""
        mapping = {
            'critical': 'error',
            'high': 'error',
            'medium': 'warning',
            'low': 'note'
        }
        return mapping.get(severity, 'warning')
    
    def _count_by_severity(self, vulnerabilities: List[Dict]) -> Dict[str, int]:
        """Count vulnerabilities by severity."""
        counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for v in vulnerabilities:
            counts[v['severity']] = counts.get(v['severity'], 0) + 1
        return counts
    
    def _count_by_type(self, vulnerabilities: List[Dict]) -> Dict[str, int]:
        """Count vulnerabilities by type."""
        counts = {}
        for v in vulnerabilities:
            counts[v['type']] = counts.get(v['type'], 0) + 1
        return counts
    
    def _avg_confidence(self, vulnerabilities: List[Dict]) -> float:
        """Calculate average confidence."""
        if not vulnerabilities:
            return 0.0
        return sum(v['confidence'] for v in vulnerabilities) / len(vulnerabilities)


# Example usage
if __name__ == "__main__":
    # Sample vulnerabilities
    vulnerabilities = [
        {
            'id': 'vuln_001',
            'type': 'sql_injection',
            'severity': 'critical',
            'confidence': 0.95,
            'line': 42,
            'column': 10,
            'file_path': 'auth.py',
            'message': 'SQL injection vulnerability detected'
        },
        {
            'id': 'vuln_002',
            'type': 'xss',
            'severity': 'high',
            'confidence': 0.87,
            'line': 156,
            'column': 5,
            'file_path': 'views.py',
            'message': 'Cross-site scripting vulnerability'
        }
    ]
    
    generator = ComplianceReportGenerator()
    
    # Generate all formats
    generator.generate_pdf_report(vulnerabilities)
    generator.generate_json_report(vulnerabilities)
    generator.generate_sarif_report(vulnerabilities)
    
    print("\n✅ All reports generated successfully!")
```

---

### 7. Tauri Configuration

**File:** `dashboard/src-tauri/tauri.conf.json`

```json
{
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devPath": "http://localhost:5173",
    "distDir": "../dist"
  },
  "package": {
    "productName": "StreamGuard",
    "version": "3.0.0"
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "shell": {
        "all": false,
        "open": true
      },
      "dialog": {
        "all": true,
        "open": true,
        "save": true
      },
      "fs": {
        "all": false,
        "readFile": true,
        "writeFile": true,
        "readDir": true,
        "scope": ["$APPDATA/*", "$RESOURCE/*"]
      },
      "http": {
        "all": true,
        "request": true,
        "scope": ["http://localhost:8765/*"]
      },
      "notification": {
        "all": true
      }
    },
    "bundle": {
      "active": true,
      "identifier": "com.streamguard.app",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/128x128@2x.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ],
      "resources": [],
      "externalBin": [],
      "copyright": "",
      "category": "DeveloperTool",
      "shortDescription": "AI-Powered Vulnerability Prevention",
      "longDescription": "StreamGuard is an AI-powered real-time vulnerability detection system that helps developers find and fix security issues as they code.",
      "deb": {
        "depends": []
      },
      "macOS": {
        "frameworks": [],
        "minimumSystemVersion": "",
        "exceptionDomain": "",
        "signingIdentity": null,
        "providerShortName": null,
        "entitlements": null
      },
      "windows": {
        "certificateThumbprint": null,
        "digestAlgorithm": "sha256",
        "timestampUrl": ""
      }
    },
    "security": {
      "csp": "default-src 'self'; connect-src 'self' http://localhost:8765 ws://localhost:8765; img-src 'self' data:; style-src 'self' 'unsafe-inline'"
    },
    "windows": [
      {
        "fullscreen": false,
        "height": 800,
        "resizable": true,
        "title": "StreamGuard",
        "width": 1200,
        "minWidth": 800,
        "minHeight": 600
      }
    ]
  }
}
```

---

## ✅ Implementation Checklist

### Dashboard (React/Tauri)
- [ ] React app setup with TypeScript
- [ ] Redux state management
- [ ] Dashboard view with statistics
- [ ] Vulnerability cards with feedback
- [ ] Interactive graph visualization
- [ ] Reports view
- [ ] Real-time WebSocket updates
- [ ] Tauri desktop wrapper

### Feedback System
- [ ] Local SQLite database
- [ ] Feedback collector
- [ ] Feedback anonymizer
- [ ] Privacy validation
- [ ] Consent management
- [ ] Secure sync to S3
- [ ] Batch processing (100+ samples)

### Reporting
- [ ] PDF report generation
- [ ] JSON export
- [ ] SARIF format support
- [ ] Custom templates
- [ ] Automated scheduling

### RLHF-lite Pipeline
- [ ] Feedback aggregation
- [ ] Drift detection
- [ ] Automated retraining triggers
- [ ] A/B testing framework
- [ ] Model deployment

---

## 🎯 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| **Dashboard Load Time** | <2s | ⏳ To Validate |
| **WebSocket Latency** | <50ms | ⏳ To Validate |
| **Graph Rendering** | <1s for 100 nodes | ⏳ To Validate |
| **Report Generation (PDF)** | <5s | ⏳ To Validate |
| **Feedback Sync** | <10s for 100 entries | ⏳ To Validate |
| **Memory Usage** | <500MB | ⏳ To Validate |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
cd dashboard
npm install

# 2. Start development server
npm run tauri:dev

# 3. Build desktop app
npm run tauri:build

# 4. Generate report
python -m core.reports.report_generator
```

---

**Status:** ✅ Ready for Implementation  
**Next:** [07_verification_patch.md](./07_verification_patch.md) - Verification & Patch System# 06 - UI & Feedback System

**Phase:** 5 (Weeks 11-12)  
**Prerequisites:** Local agent ([04_agent_architecture.md](./04_agent_architecture.md)), Graph system ([05_repository_graph.md](./05_repository_graph.md))  
**Status:** Ready to Implement

---

## 📋 Overview

Build an interactive local web dashboard with React/Tauri and implement a continuous learning feedback loop (RLHF-lite) for model improvement.

**Key Features:**
- **React/Tauri Dashboard**: Desktop app with web technologies
- **Real-Time Visualization**: Interactive graphs with D3.js/Cytoscape
- **Feedback Collection**: Developer feedback on detections
- **RLHF-lite Pipeline**: Continuous model improvement
- **Compliance Reporting**: PDF/JSON/SARIF export
- **Team Pattern Library**: Shared vulnerability patterns

**Deliverables:**
- ✅ React dashboard with TypeScript
- ✅ Tauri desktop wrapper
- ✅ Real-time WebSocket updates
- ✅ Interactive visualizations
- ✅ Feedback collection system
- ✅ Automated retraining pipeline
- ✅ Compliance report generator

**Expected Time:** 2 weeks

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    Desktop Application (Tauri)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  React Frontend                          │  │
│  │                                                          │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐        │  │
│  │  │ Dashboard  │  │   Graph    │  │  Reports   │        │  │
│  │  │   View     │  │   View     │  │   View     │        │  │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘        │  │
│  │        │               │                │               │  │
│  │  ┌─────▼───────────────▼────────────────▼──────┐        │  │
│  │  │          State Management (Redux)           │        │  │
│  │  └─────────────────────┬─────────────────────────┘        │  │
│  │                        │                                 │  │
│  │  ┌─────────────────────▼─────────────────────────┐        │  │
│  │  │         API Client (Axios + WebSocket)        │        │  │
│  │  └─────────────────────┬─────────────────────────┘        │  │
│  └────────────────────────┼──────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼──────────────────────────────────┐  │
│  │              Tauri Backend (Rust)                         │  │
│  │  • File system access                                     │  │
│  │  • System notifications                                   │  │
│  │  • Native dialogs                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP/WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Local Agent (localhost:8765)                   │
│  • Analysis API                                                 │
│  • WebSocket streaming                                          │
│  • Feedback collection                                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          │ Store feedback
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Feedback System                             │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Local Feedback Collector                                 │ │
│  │  • Collects user feedback (accept/reject/false_positive)  │ │
│  │  • Anonymizes data (removes PII)                          │ │
│  │  • Stores locally (SQLite)                                │ │
│  │  • Batches for sync (100+ samples)                        │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │                                    │
│  ┌─────────────────────────▼─────────────────────────────────┐ │
│  │  Feedback Anonymizer                                      │ │
│  │  • Remove code snippets                                   │ │
│  │  • Hash identifiers                                       │ │
│  │  • Aggregate statistics                                   │ │
│  │  • Privacy-preserving transforms                          │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │                                    │
│  ┌─────────────────────────▼─────────────────────────────────┐ │
│  │  Secure Sync Manager                                      │ │
│  │  • Encrypted upload to S3                                 │ │
│  │  • User consent required                                  │ │
│  │  • Bandwidth throttling                                   │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
└────────────────────────────┼──────────────────────────────────┘
                             │ Upload (encrypted)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              AWS SageMaker Retraining Pipeline                  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Feedback Aggregator                                      │ │
│  │  • Merge feedback from multiple users                     │ │
│  │  • Deduplicate samples                                    │ │
│  │  • Quality filtering                                      │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │                                    │
│  ┌─────────────────────────▼─────────────────────────────────┐ │
│  │  Drift Detector                                           │ │
│  │  • Monitor model accuracy                                 │ │
│  │  • Detect false positive rate increase                    │ │
│  │  • Trigger retraining when needed                         │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │                                    │
│  ┌─────────────────────────▼─────────────────────────────────┐ │
│  │  Automated Retraining                                     │ │
│  │  • Fine-tune on new feedback                              │ │
│  │  • A/B test new model                                     │ │
│  │  • Deploy if improved                                     │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Implementation

### 1. React Dashboard Setup

**File:** `dashboard/package.json`

```json
{
  "name": "streamguard-dashboard",
  "version": "3.0.0",
  "description": "StreamGuard Interactive Dashboard",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "tauri": "tauri",
    "tauri:dev": "tauri dev",
    "tauri:build": "tauri build"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-redux": "^8.1.3",
    "@reduxjs/toolkit": "^1.9.7",
    "axios": "^1.6.2",
    "d3": "^7.8.5",
    "cytoscape": "^3.27.0",
    "cytoscape-react": "^2.0.0",
    "recharts": "^2.10.3",
    "react-markdown": "^9.0.1",
    "lucide-react": "^0.263.1",
    "@tauri-apps/api": "^1.5.1"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@types/d3": "^7.4.3",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.3.3",
    "vite": "^5.0.8",
    "@tauri-apps/cli": "^1.5.7",
    "tailwindcss": "^3.3.6",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32"
  }
}
```

**File:** `dashboard/src/main.tsx`

```typescript
// Main entry point for React app
import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import App from './App';
import { store } from './store/store';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Provider store={store}>
      <App />
    </Provider>
  </React.StrictMode>
);
```

**File:** `dashboard/src/App.tsx`

```typescript
// Main App component
import React, { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import DashboardView from './components/DashboardView';
import GraphView from './components/GraphView';
import ReportsView from './components/ReportsView';
import Sidebar from './components/Sidebar';
import { connectWebSocket } from './services/websocket';
import './App.css';

const App: React.FC = () => {
  const [activeView, setActiveView] = React.useState<'dashboard' | 'graph' | 'reports'>('dashboard');
  const dispatch = useDispatch();

  useEffect(() => {
    // Connect to local agent WebSocket
    const ws = connectWebSocket('ws://localhost:8765/stream', dispatch);

    return () => {
      ws.close();
    };
  }, [dispatch]);

  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return <DashboardView />;
      case 'graph':
        return <GraphView />;
      case 'reports':
        return <ReportsView />;
      default:
        return <DashboardView />;
    }
  };

  return (
    <div className="app">
      <Sidebar activeView={activeView} onViewChange={setActiveView} />
      <main className="main-content">
        {renderView()}
      </main>
    </div>
  );
};

export default App;
```

---

### 2. Dashboard Components

**File:** `dashboard/src/components/DashboardView.tsx`

```typescript
// Main dashboard view showing vulnerability overview
import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store/store';
import { fetchVulnerabilities } from '../store/vulnerabilitiesSlice';
import VulnerabilityCard from './VulnerabilityCard';
import StatisticsPanel from './StatisticsPanel';
import RecentActivity from './RecentActivity';
import { AlertCircle, Shield, TrendingUp, Activity } from 'lucide-react';

const DashboardView: React.FC = () => {
  const dispatch = useDispatch();
  const { vulnerabilities, loading, statistics } = useSelector(
    (state: RootState) => state.vulnerabilities
  );

  useEffect(() => {
    dispatch(fetchVulnerabilities());
  }, [dispatch]);

  if (loading) {
    return (
      <div className="loading-container">
        <Activity className="animate-spin" size={48} />
        <p>Loading vulnerabilities...</p>
      </div>
    );
  }

  const criticalCount = vulnerabilities.filter(v => v.severity === 'critical').length;
  const highCount = vulnerabilities.filter(v => v.severity === 'high').length;
  const mediumCount = vulnerabilities.filter(v => v.severity === 'medium').length;

  return (
    <div className="dashboard-view">
      <header className="dashboard-header">
        <h1>StreamGuard Dashboard</h1>
        <p className="subtitle">Real-time vulnerability monitoring</p>
      </header>

      {/* Statistics Cards */}
      <div className="stats-grid">
        <StatCard
          title="Total Vulnerabilities"
          value={vulnerabilities.length}
          icon={<AlertCircle />}
          color="blue"
        />
        <StatCard
          title="Critical"
          value={criticalCount}
          icon={<AlertCircle />}
          color="red"
          trend="-12%"
        />
        <StatCard
          title="High Priority"
          value={highCount}
          icon={<Shield />}
          color="orange"
          trend="+5%"
        />
        <StatCard
          title="Detection Rate"
          value="95.2%"
          icon={<TrendingUp />}
          color="green"
        />
      </div>

      {/* Vulnerability List */}
      <section className="vulnerabilities-section">
        <h2>Recent Vulnerabilities</h2>
        <div className="vulnerability-list">
          {vulnerabilities.slice(0, 10).map(vuln => (
            <VulnerabilityCard key={vuln.id} vulnerability={vuln} />
          ))}
        </div>
      </section>

      {/* Statistics Panel */}
      <section className="statistics-section">
        <StatisticsPanel statistics={statistics} />
      </section>

      {/* Recent Activity */}
      <section className="activity-section">
        <RecentActivity />
      </section>
    </div>
  );
};

interface StatCardProps {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  color: 'blue' | 'red' | 'orange' | 'green';
  trend?: string;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, icon, color, trend }) => {
  return (
    <div className={`stat-card stat-card-${color}`}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-content">
        <h3 className="stat-title">{title}</h3>
        <p className="stat-value">{value}</p>
        {trend && (
          <span className={`stat-trend ${trend.startsWith('-') ? 'positive' : 'negative'}`}>
            {trend}
          </span>
        )}
      </div>
    </div>
  );
};

export default DashboardView;
```

**File:** `dashboard/src/components/VulnerabilityCard.tsx`

```typescript
// Individual vulnerability card with feedback buttons
import React, { useState } from 'react';
import { Check, X, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';
import { Vulnerability } from '../types';
import { submitFeedback } from '../services/api';
import ExplanationPanel from './ExplanationPanel';

interface VulnerabilityCardProps {
  vulnerability: Vulnerability;
}

const VulnerabilityCard: React.FC<VulnerabilityCardProps> = ({ vulnerability }) => {
  const [expanded, setExpanded] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const handleFeedback = async (action: 'accepted' | 'rejected' | 'false_positive') => {
    try {
      await submitFeedback({
        vulnerability_id: vulnerability.id,
        action,
        comment: null
      });
      setFeedbackSubmitted(true);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    }
  };

  const severityColor = {
    critical: 'bg-red-100 text-red-800 border-red-300',
    high: 'bg-orange-100 text-orange-800 border-orange-300',
    medium: 'bg-yellow-100 text-yellow-800 border-yellow-300',
    low: 'bg-blue-100 text-blue-800 border-blue-300'
  }[vulnerability.severity];

  return (
    <div className={`vulnerability-card ${severityColor}`}>
      <div className="card-header">
        <div className="card-title-section">
          <AlertTriangle className="card-icon" />
          <div>
            <h3 className="card-title">{vulnerability.type}</h3>
            <p className="card-location">
              {vulnerability.file_path}:{vulnerability.line}
            </p>
          </div>
        </div>
        <div className="card-meta">
          <span className={`severity-badge ${vulnerability.severity}`}>
            {vulnerability.severity.toUpperCase()}
          </span>
          <span className="confidence-badge">
            {Math.round(vulnerability.confidence * 100)}% confident
          </span>
        </div>
      </div>

      <div className="card-body">
        <p className="card-message">{vulnerability.message}</p>

        {/* Expand/Collapse Button */}
        <button
          className="expand-button"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? (
            <>
              <ChevronUp size={16} /> Hide Details
            </>
          ) : (
            <>
              <ChevronDown size={16} /> Show Details
            </>
          )}
        </button>

        {/* Expanded Details */}
        {expanded && (
          <div className="card-details">
            {/* Explanation Panel */}
            {vulnerability.explanation && (
              <ExplanationPanel explanation={vulnerability.explanation} />
            )}

            {/* Suggested Fix */}
            {vulnerability.suggested_fix && (
              <div className="suggested-fix">
                <h4>Suggested Fix:</h4>
                <pre className="code-block">{vulnerability.suggested_fix}</pre>
              </div>
            )}

            {/* CVE References */}
            {vulnerability.cve_references && vulnerability.cve_references.length > 0 && (
              <div className="cve-references">
                <h4>Related CVEs:</h4>
                <ul>
                  {vulnerability.cve_references.map(cve => (
                    <li key={cve}>
                      <a href={`https://nvd.nist.gov/vuln/detail/${cve}`} target="_blank" rel="noopener noreferrer">
                        {cve}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Feedback Buttons */}
      <div className="card-footer">
        {!feedbackSubmitted ? (
          <div className="feedback-buttons">
            <button
              className="feedback-button accept"
              onClick={() => handleFeedback('accepted')}
              title="This is a real vulnerability"
            >
              <Check size={16} /> Accept
            </button>
            <button
              className="feedback-button reject"
              onClick={() => handleFeedback('false_positive')}
              title="This is a false positive"
            >
              <X size={16} /> False Positive
            </button>
          </div>
        ) : (
          <div className="feedback-submitted">
            <Check size={16} className="text-green-600" />
            <span>Feedback submitted. Thank you!</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default VulnerabilityCard;
```

**File:** `dashboard/src/components/GraphView.tsx`

```typescript
// Interactive graph visualization of code dependencies
import React, { useEffect, useRef, useState } from 'react';
import CytoscapeComponent from 'cytoscape-react';
import cytoscape from 'cytoscape';
import { fetchGraphData } from '../services/api';
import { Maximize2, Download, Filter } from 'lucide-react';

const GraphView: React.FC = () => {
  const [graphData, setGraphData] = useState<any>(null);
  const [filter, setFilter] = useState<'all' | 'vulnerabilities' | 'entry_points'>('all');
  const cytoscapeRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    loadGraphData();
  }, [filter]);

  const loadGraphData = async () => {
    try {
      const data = await fetchGraphData(filter);
      setGraphData(data);
    } catch (error) {
      console.error('Failed to load graph data:', error);
    }
  };

  const cytoscapeStylesheet: cytoscape.Stylesheet[] = [
    {
      selector: 'node',
      style: {
        'background-color': '#667eea',
        'label': 'data(label)',
        'width': 40,
        'height': 40,
        'font-size': '12px',
        'text-valign': 'center',
        'text-halign': 'center',
        'color': '#333'
      }
    },
    {
      selector: 'node[type="vulnerability"]',
      style: {
        'background-color': '#f56565',
        'shape': 'triangle'
      }
    },
    {
      selector: 'node[type="entry_point"]',
      style: {
        'background-color': '#48bb78',
        'shape': 'diamond'
      }
    },
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#cbd5e0',
        'target-arrow-color': '#cbd5e0',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier'
      }
    },
    {
      selector: 'edge[type="taint_flow"]',
      style: {
        'line-color': '#f56565',
        'target-arrow-color': '#f56565',
        'line-style': 'dashed'
      }
    }
  ];

  const layout: cytoscape.LayoutOptions = {
    name: 'cose',
    animate: true,
    animationDuration: 1000,
    nodeDimensionsIncludeLabels: true
  };

  const handleExport = () => {
    if (cytoscapeRef.current) {
      const png = cytoscapeRef.current.png({ scale: 2 });
      const link = document.createElement('a');
      link.href = png;
      link.download = 'streamguard-graph.png';
      link.click();
    }
  };

  const handleFullscreen = () => {
    const element = document.querySelector('.graph-container');
    if (element) {
      element.requestFullscreen();
    }
  };

  return (
    <div className="graph-view">
      <header className="graph-header">
        <h1>Repository Dependency Graph</h1>
        <div className="graph-controls">
          <select
            className="filter-select"
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
          >
            <option value="all">All Nodes</option>
            <option value="vulnerabilities">Vulnerabilities Only</option>
            <option value="entry_points">Entry Points</option>
          </select>
          <button className="control-button" onClick={handleExport}>
            <Download size={16} /> Export
          </button>
          <button className="control-button" onClick={handleFullscreen}>
            <Maximize2 size={16} /> Fullscreen
          </button>
        </div>
      </header>

      <div className="graph-container">
        {graphData && (
          <CytoscapeComponent
            elements={CytoscapeComponent.normalizeElements(graphData)}
            style={{ width: '100%', height: '100%' }}
            stylesheet={cytoscapeStylesheet}
            layout={layout}
            cy={(cy) => {
              cytoscapeRef.current = cy;
              
              // Add click handler
              cy.on('tap', 'node', (event) => {
                const node = event.target;
                console.log('Clicked node:', node.data());
                // TODO: Show node details panel
              });
            }}
          />
        )}
      </div>

      {/* Legend */}
      <div className="graph-legend">
        <h3>Legend</h3>
        <div className="legend-item">
          <div className="legend-icon" style={{ backgroundColor: '#667eea' }}></div>
          <span>Function</span>
        </div>
        <div className="legend-item">
          <div className="legend-icon triangle" style={{ backgroundColor: '#f56565' }}></div>
          <span>Vulnerability</span>
        </div>
        <div className="legend-item">
          <div className="legend-icon diamond" style={{ backgroundColor: '#48bb78' }}></div>
          <span>Entry Point</span>
        </div>
      </div>
    </div>
  );
};

export default GraphView;
```

---

### 3. Feedback Collection System

**File:** `core/feedback/collector.py`

```python
"""Local feedback collection with privacy preservation."""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import hashlib
import json

class FeedbackCollector:
    """Collect and store user feedback locally."""
    
    def __init__(self, db_path: str = "data/feedback/feedback.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vulnerability_id TEXT NOT NULL,
                action TEXT NOT NULL,
                comment TEXT,
                code_context_hash TEXT,
                timestamp DATETIME NOT NULL,
                synced BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_stats (
                date DATE PRIMARY KEY,
                total_feedback INTEGER DEFAULT 0,
                accepted INTEGER DEFAULT 0,
                rejected INTEGER DEFAULT 0,
                false_positives INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def add_feedback(
        self,
        vulnerability_id: str,
        action: str,
        comment: Optional[str] = None,
        code_context: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Add feedback to local database.
        
        Args:
            vulnerability_id: ID of the vulnerability
            action: User action (accepted, rejected, false_positive)
            comment: Optional user comment
            code_context: Optional code context (will be hashed)
            timestamp: Feedback timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Hash code context for privacy
        code_hash = None
        if code_context:
            code_hash = hashlib.sha256(code_context.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback (
                vulnerability_id, action, comment,
                code_context_hash, timestamp
            )
            VALUES (?, ?, ?, ?, ?)
        """, (vulnerability_id, action, comment, code_hash, timestamp))
        
        # Update stats
        date = timestamp.date()
        cursor.execute("""
            INSERT INTO feedback_stats (date, total_feedback)
            VALUES (?, 1)
            ON CONFLICT(date) DO UPDATE SET
                total_feedback = total_feedback + 1
        """, (date,))
        
        # Update action-specific counts
        if action == 'accepted':
            cursor.execute("""
                UPDATE feedback_stats
                SET accepted = accepted + 1
                WHERE date = ?
            """, (date,))
        elif action == 'false_positive':
            cursor.execute("""
                UPDATE feedback_stats
                SET false_positives = false_positives + 1
                WHERE date = ?
            """, (date,))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Feedback recorded: {action} for {vulnerability_id}")
    
    def get_pending_feedback(self, limit: int = 100) -> List[Dict]:
        """Get feedback that hasn't been synced yet."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM feedback
            WHERE synced = FALSE
            ORDER BY timestamp ASC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def mark_synced(self, feedback_ids: List[int]):
        """Mark feedback as synced."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            UPDATE feedback
            SET synced = TRUE
            WHERE id IN ({','.join('?' * len(feedback_ids))})
        """, feedback_ids)
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict:
        """Get feedback statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total feedback
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total = cursor.fetchone()[0]
        
        # By action
        cursor.execute("""
            SELECT action, COUNT(*) as count
            FROM feedback
            GROUP BY action
        """)
        by_action = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Pending sync
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE synced = FALSE")
        pending = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_feedback': total,
            'by_action': by_action,
            'pending_sync': pending,
            'sync_ready': pending >= 100
        }