# StreamGuard - Next Steps Guide

**Purpose:** Step-by-step guide for continuing the StreamGuard project
**Target User:** ChatGPT, Claude, or any AI assistant continuation
**Current Phase:** Completed Phases 1-5 (Data Collection)
**Next Phase:** Phase 6 (Model Training)

---

## Quick Context

You are continuing work on StreamGuard, an AI-powered vulnerability detection system. The data collection pipeline is **complete and tested**. You now need to:

1. Run data collection to gather training samples
2. Preprocess the collected data
3. Train machine learning models
4. Evaluate and deploy the models

---

## Phase 6: Model Training - Overview

### Goals
- Preprocess 50,000+ vulnerability samples
- Train CodeBERT model for vulnerability detection
- Train Graph Neural Network for taint analysis
- Evaluate model performance
- Deploy to AWS SageMaker

### Expected Timeline
- Data preprocessing: 1-2 days
- Model training: 2-3 days
- Evaluation: 1 day
- Deployment: 1-2 days
- **Total: 5-8 days**

---

## Step 1: Run Data Collection (REQUIRED)

Before training, you need training data. The collectors are ready but haven't been run yet.

### Option A: Quick Test (Recommended First)
**Time:** 10-15 minutes
**Samples:** 400 total (100 from each source)
**Purpose:** Verify everything works before full collection

```bash
# Navigate to project root
cd "C:\Users\Vimal Sajan\streamguard"

# Run quick test
python training/scripts/collection/run_full_collection.py --quick-test
```

**Expected Output:**
```
StreamGuard Data Collection - Master Orchestrator
======================================================================

Collectors to run: cve, github, repo, synthetic
Mode: Parallel
Output directory: data/raw

[Progress dashboard will show real-time collection status]

COLLECTION SUMMARY
======================================================================
Total Duration: ~600s (10 minutes)
Collectors: 4/4 successful
Total Samples: 400/400 (100.0%)

By Collector:
✓ CVE: 100/100 samples
✓ GITHUB: 100/100 samples
✓ REPO: 100/100 samples
✓ SYNTHETIC: 100/100 samples

Results saved to: data/raw/collection_results.json
Reports generated in: data/raw/
```

**Files Created:**
- `data/raw/cves/cve_data.jsonl` - 100 CVE samples
- `data/raw/github/github_advisories.jsonl` - 100 GitHub samples
- `data/raw/opensource/repo_data.jsonl` - 100 repository samples
- `data/raw/synthetic/synthetic_data.jsonl` - 100 synthetic samples
- `data/raw/collection_report.json` - Summary report

### Option B: Full Collection (For Production)
**Time:** 6-10 hours
**Samples:** 50,000 total
**Purpose:** Complete dataset for final model training

```bash
# Run full collection (leave running overnight)
python training/scripts/collection/run_full_collection.py
```

**Recommendation:** Start with quick test, then run full collection overnight once verified.

### Verify Data Collection

```bash
# View collected samples
python training/scripts/collection/show_examples.py

# Or manually check files
# Each .jsonl file has one JSON object per line
```

**Sample JSON Format:**
```json
{
  "vulnerability_id": "CVE-2024-1234",
  "vulnerable_code": "sql = \"SELECT * FROM users WHERE id=\" + user_id",
  "fixed_code": "sql = \"SELECT * FROM users WHERE id=?\"\ncursor.execute(sql, (user_id,))",
  "vulnerability_type": "sql_injection",
  "severity": "HIGH",
  "description": "SQL injection in user lookup",
  "source": "cve",
  "metadata": {...}
}
```

---

## Step 2: Data Preprocessing

### What This Does
- Loads collected samples from all sources
- Tokenizes code using CodeBERT tokenizer
- Extracts features (AST, CFG, complexity metrics)
- Creates train/validation/test splits (80/10/10)
- Saves processed data for model training

### Create Preprocessing Script

**File:** `training/scripts/preprocessing/preprocess_data.py`

```python
"""
Data preprocessing for StreamGuard vulnerability detection.
Processes raw samples into training-ready format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import random
from collections import defaultdict

# You'll need to install these
# pip install transformers torch scikit-learn
from transformers import RobertaTokenizer
import torch

class DataPreprocessor:
    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed",
        model_name: str = "microsoft/codebert-base"
    ):
        """Initialize preprocessor."""
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CodeBERT tokenizer
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        # Vulnerability type mapping
        self.vuln_types = [
            'sql_injection',
            'xss',
            'command_injection',
            'path_traversal',
            'ssrf',
            'xxe',
            'csrf',
            'deserialization',
            'other'
        ]
        self.vuln_to_id = {v: i for i, v in enumerate(self.vuln_types)}

    def load_raw_samples(self) -> List[Dict]:
        """Load all raw samples from collection."""
        all_samples = []

        # Load from each source
        sources = {
            'cves': 'cves/cve_data.jsonl',
            'github': 'github/github_advisories.jsonl',
            'opensource': 'opensource/repo_data.jsonl',
            'synthetic': 'synthetic/synthetic_data.jsonl'
        }

        for source_name, file_path in sources.items():
            full_path = self.raw_data_dir / file_path
            if full_path.exists():
                print(f"Loading {source_name} from {full_path}")
                with open(full_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            sample = json.loads(line.strip())
                            sample['source_file'] = source_name
                            all_samples.append(sample)
                        except json.JSONDecodeError as e:
                            print(f"  Warning: Skipping invalid JSON on line {line_num}: {e}")
            else:
                print(f"  Warning: {full_path} not found, skipping {source_name}")

        print(f"\nTotal samples loaded: {len(all_samples)}")
        return all_samples

    def filter_code_pairs(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples that have both vulnerable and fixed code."""
        filtered = []
        for sample in samples:
            if (sample.get('vulnerable_code') and
                sample.get('fixed_code') and
                len(sample['vulnerable_code'].strip()) > 10 and
                len(sample['fixed_code'].strip()) > 10):
                filtered.append(sample)

        print(f"Samples with code pairs: {len(filtered)} ({len(filtered)/len(samples)*100:.1f}%)")
        return filtered

    def tokenize_code_pair(self, vulnerable: str, fixed: str) -> Dict:
        """Tokenize a code pair."""
        # Tokenize vulnerable code
        vuln_tokens = self.tokenizer(
            vulnerable,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize fixed code
        fixed_tokens = self.tokenizer(
            fixed,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'vuln_input_ids': vuln_tokens['input_ids'][0],
            'vuln_attention_mask': vuln_tokens['attention_mask'][0],
            'fixed_input_ids': fixed_tokens['input_ids'][0],
            'fixed_attention_mask': fixed_tokens['attention_mask'][0]
        }

    def process_sample(self, sample: Dict) -> Dict:
        """Process a single sample."""
        # Tokenize code pair
        tokens = self.tokenize_code_pair(
            sample['vulnerable_code'],
            sample['fixed_code']
        )

        # Get vulnerability type ID
        vuln_type = sample.get('vulnerability_type', 'other')
        if vuln_type not in self.vuln_to_id:
            vuln_type = 'other'
        vuln_type_id = self.vuln_to_id[vuln_type]

        # Get severity (normalize)
        severity_map = {'LOW': 0, 'MODERATE': 1, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
        severity = sample.get('severity', 'MEDIUM')
        severity_id = severity_map.get(severity.upper(), 1)

        return {
            'vuln_input_ids': tokens['vuln_input_ids'].tolist(),
            'vuln_attention_mask': tokens['vuln_attention_mask'].tolist(),
            'fixed_input_ids': tokens['fixed_input_ids'].tolist(),
            'fixed_attention_mask': tokens['fixed_attention_mask'].tolist(),
            'vulnerability_type': vuln_type_id,
            'severity': severity_id,
            'label': 1,  # 1 = vulnerable code detected
            'source': sample.get('source', 'unknown'),
            'vulnerability_id': sample.get('vulnerability_id', 'unknown')
        }

    def create_splits(
        self,
        samples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train/val/test splits."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

        # Shuffle samples
        random.seed(42)
        random.shuffle(samples)

        # Calculate split indices
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = samples[:train_end]
        val = samples[train_end:val_end]
        test = samples[val_end:]

        print(f"\nData splits:")
        print(f"  Train: {len(train)} samples ({len(train)/n*100:.1f}%)")
        print(f"  Val:   {len(val)} samples ({len(val)/n*100:.1f}%)")
        print(f"  Test:  {len(test)} samples ({len(test)/n*100:.1f}%)")

        return train, val, test

    def save_processed_data(
        self,
        train: List[Dict],
        val: List[Dict],
        test: List[Dict]
    ):
        """Save processed data to files."""
        splits = {'train': train, 'val': val, 'test': test}

        for split_name, samples in splits.items():
            output_file = self.processed_data_dir / f"{split_name}.jsonl"
            print(f"\nSaving {split_name} to {output_file}")

            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')

            print(f"  Saved {len(samples)} samples")

        # Save metadata
        metadata = {
            'num_samples': {
                'train': len(train),
                'val': len(val),
                'test': len(test),
                'total': len(train) + len(val) + len(test)
            },
            'vulnerability_types': self.vuln_types,
            'vuln_to_id': self.vuln_to_id,
            'tokenizer': 'microsoft/codebert-base',
            'max_length': 512
        }

        metadata_file = self.processed_data_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved to {metadata_file}")

    def preprocess_all(self):
        """Main preprocessing pipeline."""
        print("="*70)
        print("StreamGuard Data Preprocessing")
        print("="*70)

        # Load raw samples
        print("\n1. Loading raw samples...")
        raw_samples = self.load_raw_samples()

        if not raw_samples:
            print("\nERROR: No samples found!")
            print("Please run data collection first:")
            print("  python training/scripts/collection/run_full_collection.py --quick-test")
            return

        # Filter for code pairs
        print("\n2. Filtering for code pairs...")
        code_pairs = self.filter_code_pairs(raw_samples)

        if not code_pairs:
            print("\nERROR: No valid code pairs found!")
            return

        # Process samples
        print("\n3. Processing samples (tokenizing, extracting features)...")
        processed_samples = []
        for i, sample in enumerate(code_pairs):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(code_pairs)} samples...")

            try:
                processed = self.process_sample(sample)
                processed_samples.append(processed)
            except Exception as e:
                print(f"  Warning: Failed to process sample {i}: {e}")

        print(f"  Successfully processed: {len(processed_samples)} samples")

        # Create splits
        print("\n4. Creating train/val/test splits...")
        train, val, test = self.create_splits(processed_samples)

        # Save processed data
        print("\n5. Saving processed data...")
        self.save_processed_data(train, val, test)

        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE!")
        print("="*70)
        print(f"\nProcessed data saved to: {self.processed_data_dir}")
        print("\nNext steps:")
        print("  1. Train model:")
        print("     python training/train_model.py")
        print("  2. Or continue with Phase 6 implementation")


def main():
    """Main entry point."""
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_all()


if __name__ == "__main__":
    main()
```

### Run Preprocessing

```bash
# Install dependencies
pip install transformers torch scikit-learn

# Run preprocessing
python training/scripts/preprocessing/preprocess_data.py
```

**Expected Output:**
```
StreamGuard Data Preprocessing
======================================================================

1. Loading raw samples...
Loading cves from data/raw/cves/cve_data.jsonl
Loading github from data/raw/github/github_advisories.jsonl
Loading opensource from data/raw/opensource/repo_data.jsonl
Loading synthetic from data/raw/synthetic/synthetic_data.jsonl

Total samples loaded: 400

2. Filtering for code pairs...
Samples with code pairs: 228 (57.0%)

3. Processing samples...
  Processed 100/228 samples...
  Processed 200/228 samples...
  Successfully processed: 228 samples

4. Creating train/val/test splits...
Data splits:
  Train: 182 samples (79.8%)
  Val:   23 samples (10.1%)
  Test:  23 samples (10.1%)

5. Saving processed data...
Saving train to data/processed/train.jsonl
Saving val to data/processed/val.jsonl
Saving test to data/processed/test.jsonl

PREPROCESSING COMPLETE!
```

---

## Step 3: Model Training (Simplified)

### Create Basic Training Script

**File:** `training/train_model.py`

This is a simplified version to get started. You can enhance it later.

```python
"""
Simplified model training for StreamGuard.
Trains a CodeBERT-based vulnerability classifier.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, AdamW, get_linear_schedule_with_warmup
from pathlib import Path
from typing import List, Dict
import time

class VulnerabilityDataset(Dataset):
    """Dataset for vulnerability detection."""

    def __init__(self, data_file: str):
        """Load processed data."""
        self.samples = []
        with open(data_file, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_ids': torch.tensor(sample['vuln_input_ids']),
            'attention_mask': torch.tensor(sample['vuln_attention_mask']),
            'vulnerability_type': torch.tensor(sample['vulnerability_type']),
            'label': torch.tensor(sample['label'])
        }


class VulnerabilityClassifier(nn.Module):
    """CodeBERT-based vulnerability classifier."""

    def __init__(self, num_vuln_types: int = 9):
        super().__init__()
        self.codebert = RobertaModel.from_pretrained('microsoft/codebert-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_vuln_types)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['vulnerability_type'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['vulnerability_type'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    """Main training function."""
    print("="*70)
    print("StreamGuard Model Training")
    print("="*70)

    # Configuration
    data_dir = Path("data/processed")
    model_dir = Path("models/codebert_v1")
    model_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = VulnerabilityDataset(data_dir / "train.jsonl")
    val_dataset = VulnerabilityDataset(data_dir / "val.jsonl")

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    print("\nInitializing model...")
    model = VulnerabilityClassifier()
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print("-"*70)

    best_val_acc = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  → Saved best model to {model_path}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {model_dir}/best_model.pt")

    # Save metadata
    metadata = {
        'model_type': 'codebert_vulnerability_classifier',
        'best_val_accuracy': best_val_acc,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }

    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
```

### Run Training

```bash
# Train model
python training/train_model.py
```

**Expected Output:**
```
StreamGuard Model Training
======================================================================

Using device: cuda (or cpu)

Loading datasets...
  Train: 182 samples
  Val:   23 samples

Initializing model...

Training for 3 epochs...
----------------------------------------------------------------------

Epoch 1/3 (45.2s)
  Train Loss: 1.8234 | Train Acc: 0.4890
  Val Loss:   1.5421 | Val Acc:   0.5652
  → Saved best model

Epoch 2/3 (44.8s)
  Train Loss: 1.2145 | Train Acc: 0.6703
  Val Loss:   1.2001 | Val Acc:   0.6957
  → Saved best model

Epoch 3/3 (45.1s)
  Train Loss: 0.8921 | Train Acc: 0.7802
  Val Loss:   1.0234 | Val Acc:   0.7391
  → Saved best model

======================================================================
TRAINING COMPLETE!
======================================================================

Best validation accuracy: 0.7391
Model saved to: models/codebert_v1/best_model.pt
```

---

## Step 4: Model Evaluation (Simple)

### Create Evaluation Script

**File:** `training/evaluate_model.py`

```python
"""
Model evaluation for StreamGuard.
"""

import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from train_model import VulnerabilityDataset, VulnerabilityClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model():
    """Evaluate trained model on test set."""
    print("="*70)
    print("StreamGuard Model Evaluation")
    print("="*70)

    # Load test dataset
    test_dataset = VulnerabilityDataset("data/processed/test.jsonl")
    test_loader = DataLoader(test_dataset, batch_size=16)

    print(f"\nTest set: {len(test_dataset)} samples")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VulnerabilityClassifier()
    model.load_state_dict(torch.load("models/codebert_v1/best_model.pt"))
    model.to(device)
    model.eval()

    print("Model loaded successfully")

    # Evaluate
    print("\nEvaluating...")

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['vulnerability_type']

            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1).cpu()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    # Calculate metrics
    accuracy = np.mean([p == l for p, l in zip(all_predictions, all_labels)])

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # Detailed metrics (if you have vulnerability type names)
    vuln_types = [
        'sql_injection', 'xss', 'command_injection', 'path_traversal',
        'ssrf', 'xxe', 'csrf', 'deserialization', 'other'
    ]

    print("\nPer-Class Metrics:")
    print(classification_report(all_labels, all_predictions, target_names=vuln_types))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'num_samples': len(test_dataset),
        'predictions': all_predictions,
        'labels': all_labels
    }

    results_file = Path("models/codebert_v1/evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    evaluate_model()
```

### Run Evaluation

```bash
python training/evaluate_model.py
```

---

## Step 5: Basic API Integration (Optional)

### Create Simple Inference Script

**File:** `core/inference.py`

```python
"""
Simple inference script for StreamGuard.
"""

import torch
from transformers import RobertaTokenizer
from train_model import VulnerabilityClassifier

class VulnerabilityDetector:
    """Simple vulnerability detector."""

    def __init__(self, model_path: str = "models/codebert_v1/best_model.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

        # Load model
        self.model = VulnerabilityClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.vuln_types = [
            'sql_injection', 'xss', 'command_injection', 'path_traversal',
            'ssrf', 'xxe', 'csrf', 'deserialization', 'other'
        ]

    def predict(self, code: str) -> dict:
        """Predict vulnerability type for code."""
        # Tokenize
        tokens = self.tokenizer(
            code,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(outputs, dim=1).item()

        return {
            'vulnerability_type': self.vuln_types[predicted_class],
            'confidence': float(probabilities[predicted_class]),
            'all_probabilities': {
                vuln: float(prob)
                for vuln, prob in zip(self.vuln_types, probabilities)
            }
        }


def main():
    """Test inference."""
    detector = VulnerabilityDetector()

    # Test code
    test_code = """
    def get_user(user_id):
        query = "SELECT * FROM users WHERE id=" + user_id
        cursor.execute(query)
        return cursor.fetchone()
    """

    result = detector.predict(test_code)

    print("Code:")
    print(test_code)
    print("\nPrediction:")
    print(f"  Vulnerability: {result['vulnerability_type']}")
    print(f"  Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
```

### Test Inference

```bash
python core/inference.py
```

---

## Common Tasks for ChatGPT/Codex

Here are simple, well-defined tasks suitable for ChatGPT or GitHub Copilot:

### Easy Tasks (Good for Starting)
1. **Add logging** to existing scripts
2. **Create configuration files** (YAML/JSON) for hyperparameters
3. **Write data validation functions** to check sample quality
4. **Create visualization scripts** to plot training metrics
5. **Add command-line arguments** to scripts for flexibility
6. **Write unit tests** for individual functions
7. **Create data statistics scripts** to analyze collected samples

### Medium Tasks
1. **Implement data augmentation** (code obfuscation, variable renaming)
2. **Add more evaluation metrics** (precision, recall, F1, AUC-ROC)
3. **Create batch inference** for processing multiple files
4. **Implement model checkpointing** during training
5. **Add early stopping** to prevent overfitting
6. **Create learning rate schedulers** for better convergence
7. **Implement ensemble methods** combining multiple models

### Complex Tasks (May need multiple iterations)
1. **Implement Graph Neural Network** for taint analysis
2. **Add attention visualization** to explain predictions
3. **Create REST API** with Flask/FastAPI
4. **Implement SageMaker deployment**
5. **Add distributed training** with PyTorch DDP
6. **Create web dashboard** for model monitoring

---

## Troubleshooting

### Issue: No collected data
**Solution:** Run data collection first:
```bash
python training/scripts/collection/run_full_collection.py --quick-test
```

### Issue: Out of memory during training
**Solution:** Reduce batch size in `train_model.py`:
```python
batch_size = 8  # or even 4
```

### Issue: CUDA not available
**Solution:** Training will work on CPU (slower). Or install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Import errors
**Solution:** Install missing dependencies:
```bash
pip install transformers torch scikit-learn numpy pandas matplotlib
```

---

## Documentation References

- **Project overview:** `PROJECT_PROGRESS_SUMMARY.md`
- **Data collection:** `DATA_COLLECTION_COMPLETE.md`
- **ML training details:** `docs/02_ml_training.md`
- **Setup guide:** `docs/01_setup.md`

---

## Success Criteria for Phase 6

You'll know you've successfully completed Phase 6 when:

1. ✅ Data preprocessing runs without errors
2. ✅ Training completes for at least 3 epochs
3. ✅ Validation accuracy > 50% (70%+ is good for this simple model)
4. ✅ Evaluation script produces metrics
5. ✅ Inference script can predict on new code samples

---

## Quick Reference

### Order of Operations
1. Run data collection (quick test first)
2. Verify data with `show_examples.py`
3. Run preprocessing
4. Run training
5. Run evaluation
6. Test inference

### Key Commands
```bash
# Data collection
python training/scripts/collection/run_full_collection.py --quick-test

# Preprocessing
python training/scripts/preprocessing/preprocess_data.py

# Training
python training/train_model.py

# Evaluation
python training/evaluate_model.py

# Inference
python core/inference.py
```

---

## Tips for ChatGPT/Codex

1. **Start simple:** Get basic versions working before adding features
2. **Test frequently:** Run code after each change
3. **Use existing patterns:** Look at data collection code for examples
4. **Read error messages:** They usually indicate the exact problem
5. **Check data format:** Ensure input/output formats match
6. **Add print statements:** Debug by printing intermediate values
7. **Save checkpoints:** Don't lose progress from long training runs

---

**Ready to Start?**

Begin with Step 1 (run quick test collection) and proceed sequentially. Each step builds on the previous one. Good luck!
