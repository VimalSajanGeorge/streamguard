# Dataset Collection Strategy - Deep Dive

**Supplement to:** `docs/02_ml_training.md`  
**Focus:** Complete dataset collection workflow for StreamGuard  
**Target:** 50,000+ high-quality vulnerability samples

---

## 📊 Dataset Composition (50K+ Samples)

```
Total Dataset: 50,000+ samples
├── CVE Database (NVD): 10,000 samples (20%)
├── GitHub Security Advisories: 5,000 samples (10%)
├── Open Source Mining: 15,000 samples (30%)
├── Synthetic Generation: 15,000 samples (30%)
└── Counterfactual Pairs: 5,000 samples (10%)
```

---

## 🎯 Collection Workflow

### Reference: `02_ml_training.md` Section "Enhanced Data Collection"

**The base collector is already in 02_ml_training.md:**
- Location: `training/scripts/collection/enhanced_collector.py`
- Methods: `collect_cve_data()`, `collect_github_advisories()`, etc.

**Here's the DETAILED execution plan:**

---

## 📋 Step-by-Step Collection Process

### **Step 1: Setup Collection Environment**

```bash
# Ensure you're in the project root
cd streamguard
source venv/bin/activate

# Create collection directories (if not exists)
mkdir -p data/raw/{cves,github,opensource/repos,synthetic,counterfactuals}

# Set up API keys (create .env file)
cat > .env << EOF
# NVD API Key (get from https://nvd.nist.gov/developers/request-an-api-key)
NVD_API_KEY=your_nvd_api_key_here

# GitHub Token (get from https://github.com/settings/tokens)
GITHUB_TOKEN=your_github_token_here

# Optional: OpenAI for enhanced synthetic (if using)
OPENAI_API_KEY=your_openai_key_here
EOF
```

### **Step 2: Run Full Collection**

**Method 1: All-in-One (Recommended)**
```bash
# Run the enhanced collector from 02_ml_training.md
python training/scripts/collection/enhanced_collector.py

# This will:
# 1. Collect CVEs from NVD (10K samples) - ~2 hours
# 2. Collect GitHub advisories (5K samples) - ~1 hour  
# 3. Mine open source repos (15K samples) - ~4 hours
# 4. Generate synthetic data (15K samples) - ~30 minutes
# 5. Create counterfactuals (5K samples) - ~30 minutes
# Total time: ~8 hours
```

**Method 2: Incremental Collection (for monitoring)**
```bash
# Collect each source separately

# 1. CVE Database
python -c "
from training.scripts.collection.enhanced_collector import EnhancedDataCollector
collector = EnhancedDataCollector()
cve_samples = collector.collect_cve_data()
print(f'Collected {len(cve_samples)} CVE samples')
"

# 2. GitHub Advisories  
python -c "
from training.scripts.collection.enhanced_collector import EnhancedDataCollector
collector = EnhancedDataCollector()
github_samples = collector.collect_github_advisories()
print(f'Collected {len(github_samples)} GitHub samples')
"

# 3. Open Source Mining
python -c "
from training.scripts.collection.enhanced_collector import EnhancedDataCollector
collector = EnhancedDataCollector()
opensource_samples = collector.mine_opensource_repos()
print(f'Collected {len(opensource_samples)} open source samples')
"

# 4. Synthetic Generation
python -c "
from training.scripts.collection.enhanced_collector import EnhancedDataCollector
collector = EnhancedDataCollector()
synthetic_samples = collector.generate_synthetic_data()
print(f'Generated {len(synthetic_samples)} synthetic samples')
"

# 5. Counterfactuals
python -c "
from training.scripts.collection.enhanced_collector import EnhancedDataCollector
collector = EnhancedDataCollector()
# Load existing samples first
import json
from pathlib import Path
all_samples = []
for file in Path('data/raw').rglob('*.jsonl'):
    with open(file) as f:
        all_samples.extend([json.loads(line) for line in f])
counterfactuals = collector.generate_counterfactuals(all_samples)
print(f'Generated {len(counterfactuals)} counterfactual samples')
"
```

---

## 🔍 Detailed Collection Strategies

### **1. CVE Database Collection** (10,000 samples)

**What we collect:**
- SQL injection CVEs (CWE-89)
- XSS vulnerabilities (CWE-79)
- Path traversal (CWE-22)
- Command injection (CWE-77)
- All with code examples when available

**Enhancement needed:**
```python
# Add to enhanced_collector.py (line ~95 in collect_cve_data)

# ENHANCED: Search multiple vulnerability types
vulnerability_types = [
    ("SQL injection", "CWE-89"),
    ("XSS", "CWE-79"),
    ("path traversal", "CWE-22"),
    ("command injection", "CWE-77"),
    ("SSRF", "CWE-918"),
    ("XXE", "CWE-611")
]

all_cve_samples = []
for vuln_type, cwe_id in vulnerability_types:
    print(f"  Collecting {vuln_type} CVEs...")
    params = {
        "keywordSearch": vuln_type,
        "resultsPerPage": 100,
        "startIndex": 0
    }
    # ... rest of collection logic
    all_cve_samples.extend(cve_samples)
```

**Expected output:**
```
data/raw/cves/
├── cve_data.jsonl (all CVEs)
├── cve_sql_injection.jsonl (filtered)
├── cve_xss.jsonl (filtered)
└── cve_metadata.json (stats)
```

---

### **2. GitHub Security Advisories** (5,000 samples)

**Target repositories:**
```python
# Add this list to enhanced_collector.py
GITHUB_REPOS = [
    # Python frameworks
    "django/django",
    "pallets/flask", 
    "encode/django-rest-framework",
    "sqlalchemy/sqlalchemy",
    
    # JavaScript/Node.js
    "expressjs/express",
    "nestjs/nest",
    "nodejs/node",
    
    # Security-focused repos
    "OWASP/owasp-mstg",
    "OWASP/CheatSheetSeries"
]

# Collection strategy
def collect_github_advisories_enhanced(self):
    """Enhanced GitHub collection with pagination."""
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    all_advisories = []
    for repo in GITHUB_REPOS:
        print(f"  Collecting from {repo}...")
        
        # Use GitHub GraphQL API for advisories
        query = """
        query($owner: String!, $name: String!, $after: String) {
          repository(owner: $owner, name: $name) {
            vulnerabilityAlerts(first: 100, after: $after) {
              nodes {
                securityVulnerability {
                  advisory {
                    description
                    identifiers { type value }
                    severity
                  }
                  vulnerableVersionRange
                }
              }
              pageInfo {
                hasNextPage
                endCursor
              }
            }
          }
        }
        """
        
        # Pagination logic here
        # ... (implement full GraphQL query)
    
    return all_advisories
```

**Expected output:**
```
data/raw/github/
├── github_advisories.jsonl
├── github_by_repo/
│   ├── django.jsonl
│   ├── flask.jsonl
│   └── express.jsonl
└── github_metadata.json
```

---

### **3. Open Source Repository Mining** (15,000 samples)

**This is the MOST VALUABLE source** - real vulnerabilities with fixes!

**Repositories to mine:**
```python
REPOS_TO_MINE = [
    # High-quality Python projects
    "https://github.com/django/django",
    "https://github.com/pallets/flask",
    "https://github.com/psf/requests",
    "https://github.com/sqlalchemy/sqlalchemy",
    
    # JavaScript/Node.js
    "https://github.com/expressjs/express",
    "https://github.com/nodejs/node",
    "https://github.com/sequelize/sequelize",
    
    # Security tools (have vulnerability examples)
    "https://github.com/sqlmapproject/sqlmap",
    "https://github.com/zaproxy/zaproxy",
    
    # Add 20+ more popular repos
]
```

**What we extract:**
```python
def mine_repository_detailed(self, repo_url: str):
    """Enhanced mining with better filtering."""
    repo_name = repo_url.split('/')[-1]
    local_path = self.output_dir / "opensource" / "repos" / repo_name
    
    # Clone if needed
    if not local_path.exists():
        print(f"  Cloning {repo_name}...")
        git.Repo.clone_from(repo_url, local_path)
    
    repo = git.Repo(local_path)
    
    # Find security commits
    security_keywords = [
        "security", "vulnerability", "CVE", "XSS", "SQL injection",
        "sqli", "fix injection", "security fix", "patch",
        "sanitize", "validate input", "escape"
    ]
    
    security_commits = []
    for commit in repo.iter_commits(max_count=5000):  # Last 5000 commits
        message = commit.message.lower()
        if any(keyword in message for keyword in security_keywords):
            security_commits.append(commit)
    
    print(f"  Found {len(security_commits)} security-related commits")
    
    # Extract vulnerable → fixed pairs
    samples = []
    for commit in security_commits:
        commit_samples = self._extract_vulnerability_fix_pairs(commit, repo_name)
        samples.extend(commit_samples)
    
    return samples

def _extract_vulnerability_fix_pairs(self, commit, repo_name):
    """Extract before/after code from commit."""
    samples = []
    
    if not commit.parents:
        return samples
    
    parent = commit.parents[0]
    diffs = parent.diff(commit, create_patch=True)
    
    for diff in diffs:
        # Only Python/JavaScript files
        if not diff.a_path or not diff.a_path.endswith(('.py', '.js', '.ts')):
            continue
        
        try:
            # Before (vulnerable)
            before_code = diff.a_blob.data_stream.read().decode('utf-8', errors='ignore')
            
            # After (fixed)
            after_code = diff.b_blob.data_stream.read().decode('utf-8', errors='ignore')
            
            # Only include if there's actual change
            if before_code != after_code and len(before_code) < 5000:
                samples.append({
                    "vulnerable_code": before_code,
                    "safe_code": after_code,
                    "repository": repo_name,
                    "commit_sha": commit.hexsha,
                    "commit_message": commit.message,
                    "file_path": diff.a_path,
                    "source": "opensource_mining",
                    "timestamp": commit.committed_datetime.isoformat()
                })
        except Exception as e:
            print(f"    Warning: Could not extract from {diff.a_path}: {e}")
    
    return samples
```

**Expected output:**
```
data/raw/opensource/
├── repos/
│   ├── django/
│   ├── flask/
│   └── express/
├── mined_samples.jsonl (15K samples)
└── mining_stats.json
```

---

### **4. Synthetic Data Generation** (15,000 samples)

**Reference:** Already in `02_ml_training.md` (lines 280-350)

**Enhancement - Add more patterns:**
```python
# Add to generate_synthetic_data() in enhanced_collector.py

EXTENDED_PATTERNS = {
    # SQL Injection
    "sql_injection_concat": [
        'query = "SELECT * FROM {table} WHERE {col}=" + {input}',
        'sql = "DELETE FROM {table} WHERE id=" + str({input})',
        'cmd = f"UPDATE {table} SET {col}={{{input}}}"',
    ],
    
    "sql_injection_format": [
        'query = "SELECT * FROM {} WHERE {}={}".format({table}, {col}, {input})',
        'sql = "INSERT INTO {} VALUES ({})".format({table}, {input})',
    ],
    
    # XSS
    "xss_unsafe": [
        'html = "<div>" + {input} + "</div>"',
        'output = f"<span>{{{input}}}</span>"',
        'render_template("page.html", content={input})',
    ],
    
    # Path Traversal
    "path_traversal": [
        'file_path = base_path + "/" + {input}',
        'open(directory + {input})',
        'Path(folder) / {input}',
    ],
    
    # Command Injection
    "command_injection": [
        'os.system("ping " + {input})',
        'subprocess.call("curl " + {input}, shell=True)',
    ],
    
    # SAFE VERSIONS
    "safe_parameterized": [
        'cursor.execute("SELECT * FROM {table} WHERE id=?", ({input},))',
        'query = db.query({table}).filter({table}.id == {input})',
    ],
    
    "safe_escaped": [
        'html = "<div>" + escape({input}) + "</div>"',
        'output = Markup.escape({input})',
    ],
    
    "safe_path": [
        'file_path = Path(base_path).joinpath({input}).resolve()',
        'secure_filename({input})',
    ],
}

# Generate 15,000 samples with variety
def generate_synthetic_data_enhanced(self):
    samples = []
    
    for pattern_type, templates in EXTENDED_PATTERNS.items():
        count_per_pattern = 15000 // len(EXTENDED_PATTERNS)
        
        for i in range(count_per_pattern):
            template = templates[i % len(templates)]
            
            # Fill template with realistic values
            code = template.format(
                table=random.choice(["users", "products", "orders", "customers"]),
                col=random.choice(["id", "name", "email", "username", "status"]),
                input=random.choice([
                    "user_id", 
                    "request.args.get('id')", 
                    "params['name']",
                    "form_data['email']"
                ])
            )
            
            samples.append({
                "code": code,
                "vulnerable": "safe" not in pattern_type,
                "vulnerability_type": pattern_type.split('_')[0] if "safe" not in pattern_type else "none",
                "pattern": pattern_type,
                "source": "synthetic"
            })
    
    return samples
```

---

### **5. Counterfactual Generation** (5,000 samples)

**Reference:** `02_ml_training.md` lines 355-380

**Strategy:**
```python
# For each vulnerable sample, generate "what if it was safe?"
def generate_counterfactuals_enhanced(self, samples):
    """Generate counterfactual pairs from existing samples."""
    
    counterfactuals = []
    
    # Transformation rules
    transformations = {
        "sql_injection": [
            (r'"([^"]*)" \+ (\w+)', r'"\1?", (\2,)'),  # concat → parameterized
            (r'f"([^{]*)\{(\w+)\}([^"]*)"', r'"\1?\3", (\2,)'),  # f-string → param
        ],
        "xss": [
            (r'"\<div\>" \+ (\w+)', r'"<div>" + escape(\1)'),  # Add escape
            (r'f"\<span\>\{(\w+)\}', r'"<span>" + escape(\1)'),
        ],
        "path_traversal": [
            (r'base_path \+ "/" \+ (\w+)', r'Path(base_path).joinpath(\1).resolve()'),
        ]
    }
    
    # Apply transformations
    for sample in samples[:5000]:  # Process subset
        if 'vulnerable_code' in sample or sample.get('vulnerable', False):
            vuln_type = sample.get('vulnerability_type')
            
            if vuln_type in transformations:
                original = sample.get('code') or sample.get('vulnerable_code')
                
                for pattern, replacement in transformations[vuln_type]:
                    safe_version = re.sub(pattern, replacement, original)
                    
                    if safe_version != original:
                        counterfactuals.append({
                            "original": original,
                            "counterfactual": safe_version,
                            "transformation": f"{pattern} → {replacement}",
                            "vulnerability_type": vuln_type,
                            "source": "counterfactual"
                        })
                        break  # One counterfactual per sample
    
    return counterfactuals
```

---

## 📁 Expected Final Structure

```
data/raw/
├── cves/
│   ├── cve_data.jsonl                    # 10,000 samples
│   └── cve_metadata.json
├── github/
│   ├── github_advisories.jsonl           # 5,000 samples
│   └── github_by_repo/
├── opensource/
│   ├── repos/                            # Cloned repos
│   │   ├── django/
│   │   ├── flask/
│   │   └── ...
│   └── mined_samples.jsonl               # 15,000 samples
├── synthetic/
│   └── synthetic_data.jsonl              # 15,000 samples
└── counterfactuals/
    └── counterfactual_data.jsonl         # 5,000 samples

TOTAL: 50,000+ samples
```

---

## 🔄 Data Preprocessing Pipeline

**After collection, run preprocessing:**

```bash
# Reference: 02_ml_training.md lines 45-200
python training/scripts/preprocessing/enhanced_preprocessing.py
```

**This creates:**
```
data/processed/
├── train.jsonl      # 35,000 samples (70%)
├── val.jsonl        # 7,500 samples (15%)
└── test.jsonl       # 7,500 samples (15%)
```

**Each sample format:**
```json
{
  "code": "query = \"SELECT * FROM users WHERE id=\" + user_id",
  "tokens": [101, 2003, 102, ...],
  "attention_mask": [1, 1, 1, ...],
  "nodes": [{...}],
  "edges": [(0, 1), ...],
  "label": 1,
  "counterfactual": "cursor.execute(...)",
  "vulnerability_type": "sql_injection",
  "metadata": {...}
}
```

---

## ☁️ Upload to S3 for SageMaker

```bash
# After preprocessing
aws s3 sync data/processed/ s3://streamguard-ml-v3/data/processed/

# Upload raw for backup
aws s3 sync data/raw/ s3://streamguard-ml-v3/data/raw/
```

---

## ⏱️ Time Estimates

| Step | Time | Output |
|------|------|--------|
| CVE Collection | 2 hours | 10K samples |
| GitHub Collection | 1 hour | 5K samples |
| Open Source Mining | 4 hours | 15K samples |
| Synthetic Generation | 30 min | 15K samples |
| Counterfactual Gen | 30 min | 5K samples |
| Preprocessing | 1 hour | Train/val/test splits |
| Upload to S3 | 15 min | Ready for SageMaker |
| **TOTAL** | **~9 hours** | **50K+ samples** |

---

## 🎯 Quality Checks

**Before training, validate:**
```bash
# Check dataset statistics
python -c "
import json
from pathlib import Path
from collections import Counter

def check_dataset(path):
    samples = []
    with open(path) as f:
        samples = [json.loads(line) for line in f]
    
    labels = Counter(s['label'] for s in samples)
    vuln_types = Counter(s.get('vulnerability_type', 'unknown') for s in samples)
    
    print(f'\nDataset: {path}')
    print(f'Total samples: {len(samples)}')
    print(f'Label distribution: {dict(labels)}')
    print(f'Balance: {labels[1]/len(samples):.2%} vulnerable')
    print(f'Vulnerability types: {dict(vuln_types)}')

check_dataset('data/processed/train.jsonl')
check_dataset('data/processed/val.jsonl')
check_dataset('data/processed/test.jsonl')
"

# Expected output:
# Total: 50,000 samples
# Balance: 45-55% vulnerable (good balance)
# Types: sql_injection, xss, path_traversal, etc.
```

---

## 📋 Collection Checklist

- [ ] API keys configured (.env file)
- [ ] Run CVE collection (10K samples)
- [ ] Run GitHub collection (5K samples)
- [ ] Run open source mining (15K samples)
- [ ] Generate synthetic data (15K samples)
- [ ] Generate counterfactuals (5K samples)
- [ ] Verify 50K+ total samples
- [ ] Run preprocessing pipeline
- [ ] Check train/val/test splits
- [ ] Verify label balance (45-55%)
- [ ] Upload to S3
- [ ] Ready for SageMaker training!

---

## 🚀 Next Step

**After collection is complete:**
```bash
# Launch SageMaker training (reference: 02_ml_training.md)
python training/scripts/sagemaker/launch_enhanced_training.py
```

This starts the training job described in `02_ml_training.md` Section "Launch Training Jobs".