# 01 - Enhanced Setup Guide

**Phase:** 0 (Weeks 1-2)  
**Prerequisites:** None  
**Status:** Ready to Execute

---

## 📋 Overview

Set up the complete StreamGuard v3.0 development environment including:
- Python 3.9+ with ML dependencies
- Node.js 18+ for dashboard
- Neo4j/TigerGraph for graph database
- Docker for containerization
- AWS SageMaker configuration
- Enhanced data collection pipeline

**Expected Time:** 4-6 hours

---

## 🖥️ System Requirements

### Minimum
- **OS:** Windows 10+, macOS 12+, Ubuntu 20.04+
- **RAM:** 16GB
- **Storage:** 30GB free (20GB for Neo4j + data)
- **CPU:** 6 cores
- **GPU:** Optional (for local model training)

### Recommended
- **RAM:** 32GB+
- **Storage:** 50GB+ SSD
- **CPU:** 8+ cores
- **GPU:** NVIDIA GPU with 8GB+ VRAM

---

## 📦 Step 1: Core Tools Installation

### 1.1 Python 3.9+

```bash
# Verify Python version
python3 --version  # Should be 3.9+

# macOS
brew install python@3.9

# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev build-essential

# Windows
winget install Python.Python.3.9
```

### 1.2 Node.js 18+

```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc

nvm install 18
nvm use 18
node --version  # Should be v18.x.x
```

### 1.3 Docker & Docker Compose

```bash
# macOS
brew install --cask docker

# Ubuntu
sudo apt install docker.io docker-compose

# Windows
winget install Docker.DockerDesktop

# Verify
docker --version
docker-compose --version
```

### 1.4 Rust (for Tauri dashboard)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Verify
rustc --version
cargo --version
```

---

## 🗂️ Step 2: Project Structure

```bash
# Create project
mkdir streamguard && cd streamguard

# Initialize Git
git init

# Create enhanced directory structure
mkdir -p \
  core/{agents,engine,explainability,graph,feedback,verification,rag,utils,models} \
  dashboard/{src,src-tauri} \
  training/{scripts/{sagemaker,retraining,collection},models,configs} \
  data/{raw/{cves,github,opensource,synthetic},processed,feedback,embeddings} \
  tests/{unit,integration,e2e,benchmarks} \
  docs/{guides,prompts,decisions,architecture} \
  scripts \
  models \
  .claude/agents

# Create __init__.py for Python packages
find core tests training -type d -exec touch {}/__init__.py \;

# Create placeholder files
touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep
```

### Enhanced .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
venv/
env/
*.egg-info/
.pytest_cache/
.mypy_cache/

# Node
node_modules/
dist/
*.log

# IDE
.vscode/
.idea/
*.swp

# Data
data/raw/*
data/processed/*
data/feedback/*
!data/*/.gitkeep

# Models
models/*.pth
models/*.onnx
!models/.gitkeep

# Graph DB
neo4j/data/
tigergraph/data/

# Secrets
.env
secrets/
*.pem
*.key

# Dashboard build
dashboard/dist/
dashboard/src-tauri/target/

# Docker
*.log
docker-compose.override.yml
EOF
```

---

## 🐍 Step 3: Python Environment with UV

### 3.1 Install UV Package Manager

UV is a fast Python package manager written in Rust, replacing pip/venv for better performance.

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### 3.2 Create Project with UV

```bash
# Initialize Python project with UV
uv init

# UV automatically creates a virtual environment and manages dependencies
# No need to manually activate venv - UV handles it automatically
```

### 3.3 Enhanced Requirements

**requirements.txt:**
```txt
# Core ML/DL
torch==2.1.2
torchvision==0.16.2
transformers==4.36.0
torch-geometric==2.4.0
sentence-transformers==2.2.2

# Explainability (NEW)
captum==0.6.0  # Integrated Gradients
shap==0.42.1
lime==0.2.0.1

# Code Analysis
tree-sitter==0.20.4
tree-sitter-python==0.20.4
tree-sitter-javascript==0.21.0

# Graph Database (NEW)
neo4j==5.14.0
py2neo==2021.2.3

# Local Agent (NEW)
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
python-multipart==0.0.6

# Symbolic Execution (NEW)
angr==9.2.77
z3-solver==4.12.2.0

# Data & Processing
pandas==2.1.4
numpy==1.26.2
scipy==1.11.4
faiss-cpu==1.7.4

# API & Serialization
pydantic==2.5.3
python-jsonrpc-server==0.4.0

# Development
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
black==23.12.1
pylint==3.0.3
mypy==1.7.1
isort==5.13.2

# AWS
boto3==1.34.0
sagemaker==2.199.0

# Utilities
python-dotenv==1.0.0
rich==13.7.0
click==8.1.7
watchdog==3.0.0  # File monitoring
```

### 3.4 Install Dependencies with UV

```bash
# Install all dependencies from requirements.txt
uv pip install -r requirements.txt

# UV automatically handles virtual environment and dependency resolution
# Much faster than pip - installs packages in parallel
```

**Note:** UV automatically manages the virtual environment. No need to run `activate` scripts!

---

## 🐳 Step 4: Neo4j Setup

### 4.1 Docker Compose Configuration

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  neo4j:
    image: neo4j:5.14-community
    container_name: streamguard-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/streamguard
      - NEO4J_PLUGINS=["graph-data-science", "apoc"]
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_memory_pagecache_size=1G
    volumes:
      - ./neo4j/data:/data
      - ./neo4j/logs:/logs
      - ./neo4j/import:/var/lib/neo4j/import
    networks:
      - streamguard-network

  redis:
    image: redis:7-alpine
    container_name: streamguard-redis
    ports:
      - "6379:6379"
    volumes:
      - ./redis/data:/data
    networks:
      - streamguard-network

networks:
  streamguard-network:
    driver: bridge
EOF

# Start services
docker-compose up -d

# Verify
docker ps
curl http://localhost:7474  # Neo4j Browser
```

### 4.2 Neo4j Initialization Script

```python
# scripts/init_neo4j.py
"""Initialize Neo4j with StreamGuard schema."""

from neo4j import GraphDatabase
import os

class Neo4jInitializer:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="streamguard"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def initialize_schema(self):
        """Create indexes and constraints."""
        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT file_path_unique IF NOT EXISTS
                FOR (f:File) REQUIRE f.path IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT function_id_unique IF NOT EXISTS
                FOR (fn:Function) REQUIRE fn.id IS UNIQUE
            """)
            
            # Create indexes
            session.run("""
                CREATE INDEX file_name_index IF NOT EXISTS
                FOR (f:File) ON (f.name)
            """)
            
            session.run("""
                CREATE INDEX function_name_index IF NOT EXISTS
                FOR (fn:Function) ON (fn.name)
            """)
            
            session.run("""
                CREATE INDEX vulnerability_type_index IF NOT EXISTS
                FOR (v:Vulnerability) ON (v.type)
            """)
            
            print("✅ Neo4j schema initialized")
    
    def create_sample_data(self):
        """Create sample graph for testing."""
        with self.driver.session() as session:
            session.run("""
                // Create sample files
                CREATE (f1:File {path: 'auth.py', name: 'auth.py'})
                CREATE (f2:File {path: 'database.py', name: 'database.py'})
                
                // Create functions
                CREATE (fn1:Function {
                    id: 'auth.login',
                    name: 'login',
                    file: 'auth.py',
                    start_line: 10,
                    end_line: 25
                })
                CREATE (fn2:Function {
                    id: 'database.query_user',
                    name: 'query_user',
                    file: 'database.py',
                    start_line: 5,
                    end_line: 15
                })
                
                // Create relationships
                CREATE (f1)-[:CONTAINS]->(fn1)
                CREATE (f2)-[:CONTAINS]->(fn2)
                CREATE (fn1)-[:CALLS]->(fn2)
                
                // Create vulnerability
                CREATE (v:Vulnerability {
                    id: 'vuln_001',
                    type: 'sql_injection',
                    severity: 'high',
                    confidence: 0.95,
                    line: 12
                })
                CREATE (fn2)-[:HAS_VULNERABILITY]->(v)
                
                // Create taint flow
                CREATE (source:TaintSource {type: 'user_input', location: 'auth.py:10'})
                CREATE (sink:TaintSink {type: 'sql_execute', location: 'database.py:12'})
                CREATE (source)-[:FLOWS_TO]->(fn1)
                CREATE (fn1)-[:FLOWS_TO]->(fn2)
                CREATE (fn2)-[:FLOWS_TO]->(sink)
            """)
            
            print("✅ Sample data created")
    
    def close(self):
        self.driver.close()

if __name__ == "__main__":
    initializer = Neo4jInitializer()
    initializer.initialize_schema()
    initializer.create_sample_data()
    initializer.close()
```

```bash
# Run initialization
python scripts/init_neo4j.py
```

---

## ☁️ Step 5: AWS SageMaker Setup

### 5.1 AWS CLI Configuration

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output (json)

# Verify
aws sts get-caller-identity
```

### 5.2 Create S3 Bucket

```bash
# Create bucket
aws s3 mb s3://streamguard-ml-v3 --region us-east-1

# Create folder structure
aws s3api put-object --bucket streamguard-ml-v3 --key data/processed/
aws s3api put-object --bucket streamguard-ml-v3 --key data/feedback/
aws s3api put-object --bucket streamguard-ml-v3 --key models/
aws s3api put-object --bucket streamguard-ml-v3 --key checkpoints/
```

### 5.3 IAM Role Setup

```python
# scripts/setup_sagemaker_role.py
"""Create IAM role for SageMaker with enhanced permissions."""

import boto3
import json

def create_sagemaker_role():
    iam = boto3.client('iam')
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    try:
        response = iam.create_role(
            RoleName='StreamGuardSageMakerRoleV3',
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Enhanced role for StreamGuard v3.0 ML training'
        )
        
        role_arn = response['Role']['Arn']
        print(f"✅ Created role: {role_arn}")
        
        # Attach policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
            'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
        ]
        
        for policy in policies:
            iam.attach_role_policy(
                RoleName='StreamGuardSageMakerRoleV3',
                PolicyArn=policy
            )
            print(f"✅ Attached policy: {policy}")
        
        return role_arn
        
    except iam.exceptions.EntityAlreadyExistsException:
        print("Role already exists")
        response = iam.get_role(RoleName='StreamGuardSageMakerRoleV3')
        return response['Role']['Arn']

if __name__ == "__main__":
    role_arn = create_sagemaker_role()
    print(f"\n📋 Save this ARN: {role_arn}")
    
    # Save to .env file
    with open('.env', 'a') as f:
        f.write(f"\nSAGEMAKER_ROLE_ARN={role_arn}\n")
```

```bash
# Run setup
 
```

---

## 📊 Step 6: Enhanced Data Collection

### 6.1 Data Collection Pipeline

```python
# training/scripts/collection/enhanced_collector.py
"""Enhanced data collection with repository mining."""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict
import git
from datetime import datetime

class EnhancedDataCollector:
    """Collect training data from multiple sources."""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_all(self):
        """Run all collection pipelines."""
        print("🔍 Starting enhanced data collection...")
        
        # 1. CVE Database
        print("\n1️⃣ Collecting CVE data...")
        cve_samples = self.collect_cve_data()
        print(f"   ✅ Collected {len(cve_samples)} CVE samples")
        
        # 2. GitHub Security Advisories
        print("\n2️⃣ Collecting GitHub advisories...")
        github_samples = self.collect_github_advisories()
        print(f"   ✅ Collected {len(github_samples)} GitHub samples")
        
        # 3. Open Source Repository Mining
        print("\n3️⃣ Mining open source repositories...")
        opensource_samples = self.mine_opensource_repos()
        print(f"   ✅ Collected {len(opensource_samples)} open source samples")
        
        # 4. Synthetic Data Generation
        print("\n4️⃣ Generating synthetic data...")
        synthetic_samples = self.generate_synthetic_data()
        print(f"   ✅ Generated {len(synthetic_samples)} synthetic samples")
        
        # 5. Counterfactual Augmentation (NEW)
        print("\n5️⃣ Generating counterfactual examples...")
        counterfactual_samples = self.generate_counterfactuals(
            cve_samples + opensource_samples
        )
        print(f"   ✅ Generated {len(counterfactual_samples)} counterfactual samples")
        
        total = (len(cve_samples) + len(github_samples) + 
                len(opensource_samples) + len(synthetic_samples) +
                len(counterfactual_samples))
        
        print(f"\n🎉 Total samples collected: {total}")
        return total
    
    def collect_cve_data(self) -> List[Dict]:
        """Collect from NVD database."""
        output_file = self.output_dir / "cves" / "cve_data.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        cve_api = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        samples = []
        
        for start_index in range(0, 10000, 100):
            try:
                params = {
                    "keywordSearch": "SQL injection",
                    "resultsPerPage": 100,
                    "startIndex": start_index
                }
                
                response = requests.get(cve_api, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                vulnerabilities = data.get("vulnerabilities", [])
                
                if not vulnerabilities:
                    break
                
                samples.extend(vulnerabilities)
                
                # Rate limiting
                import time
                time.sleep(6)
                
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
                break
        
        # Save
        with open(output_file, 'w') as f:
            for sample in samples:
                json.dump(sample, f)
                f.write('\n')
        
        return samples
    
    def collect_github_advisories(self) -> List[Dict]:
        """Collect from GitHub Security Advisories."""
        # Implementation similar to CVE collection
        # See previous version for details
        return []
    
    def mine_opensource_repos(self) -> List[Dict]:
        """Mine open source repositories for vulnerability patterns."""
        repos = [
            "https://github.com/django/django",
            "https://github.com/pallets/flask",
            "https://github.com/sqlalchemy/sqlalchemy",
            "https://github.com/psf/requests",
            "https://github.com/nodejs/node",
            "https://github.com/expressjs/express"
        ]
        
        samples = []
        repos_dir = self.output_dir / "opensource" / "repos"
        repos_dir.mkdir(parents=True, exist_ok=True)
        
        for repo_url in repos:
            repo_name = repo_url.split('/')[-1]
            local_path = repos_dir / repo_name
            
            print(f"   Mining {repo_name}...")
            
            try:
                # Clone if not exists
                if not local_path.exists():
                    git.Repo.clone_from(repo_url, local_path)
                
                # Find security commits
                repo = git.Repo(local_path)
                security_commits = self._find_security_commits(repo)
                
                # Extract before/after code
                for commit in security_commits[:50]:  # Limit per repo
                    commit_samples = self._extract_from_commit(commit, repo_name)
                    samples.extend(commit_samples)
                
            except Exception as e:
                print(f"   ⚠️  Error mining {repo_name}: {e}")
        
        # Save
        output_file = self.output_dir / "opensource" / "mined_samples.jsonl"
        with open(output_file, 'w') as f:
            for sample in samples:
                json.dump(sample, f)
                f.write('\n')
        
        return samples
    
    def _find_security_commits(self, repo: git.Repo) -> List[git.Commit]:
        """Find commits with security fixes."""
        security_keywords = [
            "security", "vulnerability", "CVE", "sql injection",
            "sqli", "fix sql", "security fix", "XSS", "injection"
        ]
        
        security_commits = []
        for commit in repo.iter_commits(max_count=1000):
            message = commit.message.lower()
            if any(keyword in message for keyword in security_keywords):
                security_commits.append(commit)
        
        return security_commits
    
    def _extract_from_commit(
        self, 
        commit: git.Commit,
        repo_name: str
    ) -> List[Dict]:
        """Extract vulnerable and safe code from commit."""
        samples = []
        
        if not commit.parents:
            return samples
        
        parent = commit.parents[0]
        diffs = parent.diff(commit, create_patch=True)
        
        for diff in diffs:
            if not diff.a_path or not diff.a_path.endswith(('.py', '.js', '.ts')):
                continue
            
            try:
                before_code = diff.a_blob.data_stream.read().decode('utf-8', errors='ignore')
                after_code = diff.b_blob.data_stream.read().decode('utf-8', errors='ignore')
                
                if before_code and after_code and before_code != after_code:
                    samples.append({
                        "vulnerable_code": before_code,
                        "safe_code": after_code,
                        "repository": repo_name,
                        "commit_sha": commit.hexsha,
                        "file_path": diff.a_path,
                        "source": "opensource_mining"
                    })
            except:
                pass
        
        return samples
    
    def generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic vulnerable and safe code."""
        templates = {
            "sql_injection_concat": [
                'query = "SELECT * FROM {table} WHERE {column}=" + {user_input}',
                'sql = "DELETE FROM {table} WHERE id=" + str({user_input})',
                'cmd = f"SELECT * FROM {table} WHERE name={{{user_input}}}"',
            ],
            "sql_injection_format": [
                'query = "SELECT * FROM {} WHERE id={}".format({table}, {user_input})',
                'sql = "UPDATE {} SET status={}".format({table}, {status})',
            ],
            "safe_parameterized": [
                'cursor.execute("SELECT * FROM {table} WHERE id=?", ({user_input},))',
                'stmt = conn.prepareStatement("SELECT * FROM {table} WHERE id=?")',
            ]
        }
        
        samples = []
        vocabulary = {
            'table': ['users', 'products', 'orders', 'customers'],
            'column': ['id', 'name', 'email', 'username'],
            'user_input': ['user_id', 'request.args.get("id")', 'params["name"]'],
            'status': ['status', 'active', 'verified']
        }
        
        # Generate 5000 samples
        for _ in range(5000):
            category = list(templates.keys())[_ % len(templates)]
            template = templates[category][_ % len(templates[category])]
            
            # Fill template
            code = template.format(
                table=vocabulary['table'][_ % len(vocabulary['table'])],
                column=vocabulary['column'][_ % len(vocabulary['column'])],
                user_input=vocabulary['user_input'][_ % len(vocabulary['user_input'])],
                status=vocabulary['status'][_ % len(vocabulary['status'])]
            )
            
            samples.append({
                "code": code,
                "vulnerable": "safe" not in category,
                "pattern": category,
                "source": "synthetic"
            })
        
        # Save
        output_file = self.output_dir / "synthetic" / "synthetic_data.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for sample in samples:
                json.dump(sample, f)
                f.write('\n')
        
        return samples
    
    def generate_counterfactuals(self, samples: List[Dict]) -> List[Dict]:
        """Generate counterfactual examples (NEW)."""
        counterfactuals = []
        
        for sample in samples[:1000]:  # Process subset
            if 'vulnerable_code' in sample:
                vulnerable = sample['vulnerable_code']
                safe = sample.get('safe_code', '')
                
                if vulnerable and safe:
                    counterfactuals.append({
                        "original": vulnerable,
                        "counterfactual": safe,
                        "type": "vulnerable_to_safe",
                        "source": "counterfactual_generation"
                    })
        
        # Save
        output_file = self.output_dir / "counterfactuals" / "counterfactual_data.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for sample in counterfactuals:
                json.dump(sample, f)
                f.write('\n')
        
        return counterfactuals

if __name__ == "__main__":
    collector = EnhancedDataCollector()
    total_samples = collector.collect_all()
    print(f"\n✅ Collection complete! Total: {total_samples} samples")
```

```bash
# Run data collection
python training/scripts/collection/enhanced_collector.py
```

---

## 🧪 Step 7: Verification

### 7.1 Verification Script

```python
# scripts/verify_enhanced_setup.py
"""Verify StreamGuard v3.0 setup."""

import sys
import subprocess
from importlib import import_module
from pathlib import Path
import requests

def check_python_packages():
    """Check Python packages."""
    required = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('captum', 'Captum (Explainability)'),
        ('neo4j', 'Neo4j Driver'),
        ('fastapi', 'FastAPI'),
        ('angr', 'angr (Symbolic Execution)')
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
```

```bash
# Run verification
python scripts/verify_enhanced_setup.py
```

---

## 🔧 Step 8: Claude Code Sub-Agents Setup

### Create Sub-Agent Configurations

```bash
# Explainability Agent
cat > .claude/agents/explainability.yml << 'EOF'
name: explainability
description: |
  ML explainability specialist for:
  - Integrated Gradients implementation
  - Counterfactual generation
  - Saliency visualization
  - SHAP/LIME integration
tools: [file_read, file_write, bash_execute]
model: opus
instructions: |
  Focus on:
  - PyTorch gradient computation
  - Captum library usage
  - Performance optimization (<100ms overhead)
  - Clear visualization outputs
EOF

# Graph Systems Agent
cat > .claude/agents/graph-systems.yml << 'EOF'
name: graph-systems
description: |
  Neo4j/TigerGraph specialist for:
  - Repository graph construction
  - Cypher query optimization
  - Vulnerability propagation tracking
  - Attack surface analysis
tools: [file_read, file_write, bash_execute]
model: sonnet
instructions: |
  Best practices:
  - Use parameterized queries always
  - Index frequently accessed properties
  - Batch operations for performance
  - Monitor and optimize query plans
EOF

# Dashboard Agent
cat > .claude/agents/dashboard.yml << 'EOF'
name: dashboard
description: |
  React/Tauri dashboard specialist for:
  - Interactive visualizations (D3.js, Cytoscape)
  - WebSocket real-time updates
  - Compliance report generation
  - Responsive UI/UX design
tools: [file_read, file_write, bash_execute]
model: sonnet
instructions: |
  Focus on:
  - React best practices with TypeScript
  - Tauri Rust integration
  - Performance optimization
  - Accessibility standards
EOF
```

---

## ✅ Setup Completion Checklist

### Environment
- [ ] Python 3.9+ installed
- [ ] Node.js 18+ installed
- [ ] Rust installed (for Tauri)
- [ ] Docker installed and running
- [ ] Git configured

### Services
- [ ] Neo4j running (http://localhost:7474)
- [ ] Redis running (port 6379)
- [ ] Docker containers healthy

### Project
- [ ] Directory structure created
- [ ] Virtual environment activated
- [ ] All Python packages installed
- [ ] Claude Code sub-agents configured

### AWS
- [ ] AWS CLI configured
- [ ] S3 bucket created
- [ ] IAM role created
- [ ] Credentials in .env file

### Data
- [ ] Data collection script executed
- [ ] At least 10K+ samples collected
- [ ] Neo4j schema initialized

### Verification
- [ ] All verification checks passed
- [ ] Sample Neo4j queries working
- [ ] Can connect to all services

---

## 🚨 Troubleshooting

### Neo4j Issues

**Problem:** Neo4j container won't start
```bash
# Check logs
docker logs streamguard-neo4j

# Reset data
docker-compose down -v
docker-compose up -d
python scripts/init_neo4j.py
```

**Problem:** Can't connect to Neo4j
```bash
# Verify it's running
docker ps | grep neo4j

# Test connection
curl http://localhost:7474

# Check credentials
docker exec streamguard-neo4j cypher-shell -u neo4j -p streamguard "RETURN 1"
```

### Python Package Issues

**Problem:** angr installation fails
```bash
# angr has many dependencies, install separately
pip install angr --no-cache-dir

# Or use conda (recommended for angr)
conda install -c conda-forge angr
```

**Problem:** Captum import error
```bash
# Ensure PyTorch is installed first
pip install torch==2.1.2
pip install captum==0.6.0
```

### AWS Issues

**Problem:** SageMaker role creation fails
```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify IAM permissions
aws iam list-roles | grep SageMaker
```

---

## 📚 Next Steps

✅ **Setup Complete!**

**Continue to:**
- [02_ml_training.md](./02_ml_training.md) - ML model training
- [CLAUDE.md](./CLAUDE.md) - Complete project guide

**Quick Start:**
```bash
# Activate environment
source venv/bin/activate

# Start services
docker-compose up -d

# Begin Phase 1 with Claude Code
claude --plan "Begin Phase 1: ML Training with enhanced features"
```

---

**Setup Guide Complete** ✨  
**Ready for Phase 1: ML Training Pipeline**