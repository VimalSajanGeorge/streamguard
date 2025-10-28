# 02 - Enhanced ML Training Pipeline

**Phase:** 1 (Weeks 3-5, Runs in Parallel)  
**Prerequisites:** [01_setup.md](./01_setup.md) completed, data collected  
**Status:** Ready to Execute

---

## 📋 Overview

Train ML models with enhanced explainability support on AWS SageMaker, featuring:
- CodeBERT/CodeLLaMA fine-tuning for vulnerability detection
- Integrated Gradients support built into models
- Counterfactual-augmented training data
- Automatic model evaluation and deployment
- Continuous retraining pipeline

**Expected Time:** 3 weeks (overlaps with other phases)

**Deliverables:**
- ✅ Fine-tuned Taint-Flow GNN (92%+ accuracy)
- ✅ Fine-tuned SQL Intent Model (88%+ accuracy)
- ✅ Models with explainability hooks
- ✅ Automatic retraining pipeline
- ✅ Model registry and versioning

---

## 🏗️ Enhanced Training Architecture

```
Data Collection (50K+ samples)
        ↓
┌──────────────────────────────────────┐
│  Preprocessing Pipeline              │
│  • AST extraction                    │
│  • Graph construction                │
│  • Counterfactual augmentation (NEW) │
│  • Embedding generation              │
└────────────┬─────────────────────────┘
             │
             ↓ Upload to S3
┌────────────┴─────────────────────────┐
│  AWS SageMaker Training              │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Taint-Flow GNN Training       │ │
│  │  • 4-layer GAT                 │ │
│  │  • Attention for explainability│ │
│  │  • Target: 92%+ accuracy       │ │
│  └────────────┬───────────────────┘ │
│               │                      │
│  ┌────────────▼───────────────────┐ │
│  │  SQL Intent Transformer        │ │
│  │  • CodeBERT fine-tuning        │ │
│  │  • IG-compatible architecture  │ │
│  │  • Target: 88%+ accuracy       │ │
│  └────────────┬───────────────────┘ │
│               │                      │
│  ┌────────────▼───────────────────┐ │
│  │  Evaluation & Validation       │ │
│  │  • Accuracy, Precision, Recall │ │
│  │  • Explainability metrics      │ │
│  │  • A/B testing                 │ │
│  └────────────┬───────────────────┘ │
└───────────────┼─────────────────────┘
                │
                ↓ Register in Model Registry
┌───────────────▼─────────────────────┐
│  Model Registry                     │
│  • Versioning                       │
│  • Metadata tracking                │
│  • Automatic deployment             │
└─────────────────────────────────────┘
```

---

## 📊 Enhanced Data Preprocessing

### Preprocessing with Counterfactual Augmentation

**File:** `training/scripts/preprocessing/enhanced_preprocessing.py`

```python
"""Enhanced preprocessing with counterfactual augmentation."""

import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import tree_sitter
import tree_sitter_python
from transformers import AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split

class EnhancedDataPreprocessor:
    """Preprocess data with counterfactual augmentation."""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parsers and tokenizer
        self.parser = self._init_tree_sitter()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    def _init_tree_sitter(self) -> tree_sitter.Parser:
        """Initialize Tree-sitter parser."""
        parser = tree_sitter.Parser()
        parser.set_language(tree_sitter.Language(
            tree_sitter_python.language(), 'python'
        ))
        return parser
    
    def load_all_data(self) -> List[Dict]:
        """Load data from all sources including counterfactuals."""
        all_data = []
        
        sources = [
            "raw/cves/cve_data.jsonl",
            "raw/github/github_advisories.jsonl",
            "raw/opensource/mined_samples.jsonl",
            "raw/synthetic/synthetic_data.jsonl",
            "raw/counterfactuals/counterfactual_data.jsonl"  # NEW
        ]
        
        for source in sources:
            file_path = Path("data") / source
            if file_path.exists():
                print(f"Loading {source}...")
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            all_data.append(data)
                        except json.JSONDecodeError:
                            continue
        
        print(f"✅ Loaded {len(all_data)} total samples")
        return all_data
    
    def preprocess_sample(self, sample: Dict) -> Dict:
        """Preprocess individual sample with enhanced features."""
        # Extract code
        if 'code' in sample:
            code = sample['code']
        elif 'vulnerable_code' in sample:
            code = sample['vulnerable_code']
        else:
            return None
        
        if not code or len(code) > 10000:
            return None
        
        try:
            # Parse AST
            tree = self.parser.parse(bytes(code, "utf8"))
            root = tree.root_node
            
            # Extract features
            nodes, edges = self._extract_graph_features(root, code)
            
            # Tokenize for transformer
            tokens = self.tokenizer.encode(
                code,
                max_length=512,
                truncation=True,
                padding='max_length'
            )
            
            # Create attention mask for explainability
            attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in tokens]
            
            # Label
            label = 1 if sample.get('vulnerable', False) else 0
            
            # Add counterfactual if available (NEW)
            counterfactual = None
            if 'counterfactual' in sample:
                cf_tokens = self.tokenizer.encode(
                    sample['counterfactual'],
                    max_length=512,
                    truncation=True,
                    padding='max_length'
                )
                counterfactual = cf_tokens
            
            return {
                "code": code,
                "tokens": tokens,
                "attention_mask": attention_mask,
                "nodes": nodes,
                "edges": edges,
                "label": label,
                "counterfactual": counterfactual,  # NEW
                "vulnerability_type": sample.get('vulnerability_type'),
                "source": sample.get('source'),
                "metadata": {
                    "num_nodes": len(nodes),
                    "num_edges": len(edges),
                    "code_length": len(code)
                }
            }
        
        except Exception as e:
            print(f"Error preprocessing: {e}")
            return None
    
    def _extract_graph_features(
        self,
        ast_node: tree_sitter.Node,
        code: str
    ) -> Tuple[List[Dict], List[Tuple[int, int]]]:
        """Extract nodes and edges for GNN."""
        nodes = []
        edges = []
        node_id_map = {}
        current_id = 0
        
        def traverse(node, parent_id=None):
            nonlocal current_id
            
            node_id = current_id
            node_id_map[id(node)] = node_id
            current_id += 1
            
            # Extract node features
            nodes.append({
                "id": node_id,
                "type": node.type,
                "type_id": hash(node.type) % 10000,  # For embedding
                "text": code[node.start_byte:node.end_byte][:100],
                "start_line": node.start_point[0],
                "end_line": node.end_point[0]
            })
            
            # Create edge from parent
            if parent_id is not None:
                edges.append((parent_id, node_id))
            
            # Recurse
            for child in node.children:
                traverse(child, node_id)
        
        traverse(ast_node)
        return nodes, edges
    
    def augment_with_counterfactuals(
        self,
        data: List[Dict]
    ) -> List[Dict]:
        """Augment dataset with counterfactual pairs (NEW)."""
        augmented = []
        
        for sample in data:
            augmented.append(sample)
            
            # If has counterfactual, create paired training example
            if sample.get('counterfactual'):
                # Add the counterfactual as a separate safe example
                cf_sample = sample.copy()
                cf_sample['code'] = sample['counterfactual']
                cf_sample['label'] = 0  # Safe
                cf_sample['source'] = f"{sample['source']}_counterfactual"
                augmented.append(cf_sample)
        
        print(f"✅ Augmented dataset: {len(data)} → {len(augmented)} samples")
        return augmented
    
    def preprocess_all(self, samples: List[Dict]) -> List[Dict]:
        """Preprocess all samples."""
        # First, augment with counterfactuals
        augmented_samples = self.augment_with_counterfactuals(samples)
        
        processed = []
        for i, sample in enumerate(augmented_samples):
            if i % 1000 == 0:
                print(f"Processing {i}/{len(augmented_samples)}...")
            
            result = self.preprocess_sample(sample)
            if result:
                processed.append(result)
        
        print(f"✅ Processed {len(processed)}/{len(augmented_samples)} samples")
        return processed
    
    def split_dataset(
        self,
        data: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split into train/val/test."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        # Stratified split
        train_data, temp_data = train_test_split(
            data,
            test_size=(val_ratio + test_ratio),
            random_state=42,
            stratify=[d["label"] for d in data]
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=42,
            stratify=[d["label"] for d in temp_data]
        )
        
        print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        return train_data, val_data, test_data
    
    def save_processed_data(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict]
    ):
        """Save processed data."""
        for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            output_file = self.output_dir / f"{split_name}.jsonl"
            
            with open(output_file, 'w') as f:
                for sample in data:
                    json.dump(sample, f)
                    f.write('\n')
            
            print(f"✅ Saved {len(data)} samples to {output_file}")
    
    def run_full_pipeline(self):
        """Run complete preprocessing pipeline."""
        print("📊 Enhanced Preprocessing Pipeline")
        print("="*50)
        
        print("\n1️⃣ Loading raw data...")
        raw_data = self.load_all_data()
        
        print("\n2️⃣ Preprocessing samples...")
        processed_data = self.preprocess_all(raw_data)
        
        print("\n3️⃣ Splitting dataset...")
        train, val, test = self.split_dataset(processed_data)
        
        print("\n4️⃣ Saving processed data...")
        self.save_processed_data(train, val, test)
        
        print("\n✅ Preprocessing complete!")
        print(f"Total: {len(processed_data)} samples")
        print(f"Train: {len(train)} ({len(train)/len(processed_data)*100:.1f}%)")
        print(f"Val: {len(val)} ({len(val)/len(processed_data)*100:.1f}%)")
        print(f"Test: {len(test)} ({len(test)/len(processed_data)*100:.1f}%)")


if __name__ == "__main__":
    preprocessor = EnhancedDataPreprocessor()
    preprocessor.run_full_pipeline()
```

---

## 🧠 Enhanced Model Architectures

### Taint-Flow GNN with Explainability

**File:** `training/models/enhanced_taint_gnn.py`

```python
"""Enhanced Taint-Flow GNN with explainability support."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Tuple, Dict

class EnhancedTaintFlowGNN(nn.Module):
    """
    GNN with built-in support for Integrated Gradients.
    
    Key features:
    - Multi-head attention for explainability
    - Gradient-friendly architecture
    - Attention weight extraction
    """
    
    def __init__(
        self,
        node_features: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.node_embedding = nn.Embedding(10000, node_features)
        
        # GAT layers with attention
        self.gat_layers = nn.ModuleList([
            GATConv(
                node_features if i == 0 else hidden_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True if i < num_layers - 1 else False,
                add_self_loops=True
            )
            for i in range(num_layers)
        ])
        
        # Path attention for taint flow
        self.path_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Store attention weights for explainability
        self.attention_weights = []
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with optional attention return.
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            return_attention: Whether to return attention weights
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
            attention_dict: Attention weights if return_attention=True
        """
        # Embed nodes
        h = self.node_embedding(x)
        
        # Store attention weights
        self.attention_weights = []
        
        # Apply GAT layers
        for i, gat in enumerate(self.gat_layers):
            h_new, (edge_index_att, alpha) = gat(
                h, edge_index, return_attention_weights=True
            )
            
            if return_attention:
                self.attention_weights.append({
                    'layer': i,
                    'edge_index': edge_index_att,
                    'alpha': alpha
                })
            
            h = h_new
            if i < len(self.gat_layers) - 1:
                h = torch.relu(h)
        
        # Path attention
        h_expanded = h.unsqueeze(0)
        h_attended, path_attn = self.path_attention(
            h_expanded, h_expanded, h_expanded
        )
        h_attended = h_attended.squeeze(0)
        
        if return_attention:
            self.attention_weights.append({
                'layer': 'path_attention',
                'weights': path_attn
            })
        
        # Global pooling
        graph_embedding = global_mean_pool(h_attended, batch)
        
        # Classify
        logits = self.classifier(graph_embedding)
        
        attention_dict = {
            'attention_weights': self.attention_weights
        } if return_attention else {}
        
        return logits, attention_dict
    
    def get_node_importance(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute node importance scores for explainability.
        
        Returns:
            importance: Node importance scores [num_nodes]
        """
        logits, attn_dict = self.forward(x, edge_index, batch, return_attention=True)
        
        # Aggregate attention weights across layers
        importance = torch.zeros(x.size(0))
        
        for layer_attn in attn_dict['attention_weights']:
            if 'alpha' in layer_attn:
                # Get attention scores
                alpha = layer_attn['alpha'].mean(dim=1)  # Average across heads
                edge_index_att = layer_attn['edge_index']
                
                # Aggregate to node level
                for i in range(edge_index_att.size(1)):
                    src, dst = edge_index_att[:, i]
                    importance[dst] += alpha[i]
        
        # Normalize
        importance = importance / importance.max()
        
        return importance
```

### SQL Intent Transformer with IG Support

**File:** `training/models/enhanced_sql_intent.py`

```python
"""Enhanced SQL Intent Transformer with IG support."""

import torch
import torch.nn as nn
from transformers import AutoModel

class EnhancedSQLIntentTransformer(nn.Module):
    """
    Transformer for SQL intent with explainability hooks.
    
    Features:
    - Gradient-compatible architecture
    - Token-level attribution support
    - Attention weight extraction
    """
    
    def __init__(
        self,
        base_model: str = "microsoft/codebert-base",
        num_classes: int = 2,
        hidden_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load pretrained transformer
        self.encoder = AutoModel.from_pretrained(base_model)
        
        # Make sure gradients flow
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        # Intent classification head
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden: Whether to return hidden states
        
        Returns:
            intent_logits: Intent classification [batch_size, num_classes]
            anomaly_score: Anomaly score [batch_size, 1]
            hidden_dict: Hidden states if return_hidden=True
        """
        # Encode with attention output
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=return_hidden
        )
        
        # Use [CLS] token
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Classify intent
        intent_logits = self.intent_classifier(pooled)
        
        # Detect anomalies
        anomaly_score = self.anomaly_detector(pooled)
        
        hidden_dict = {}
        if return_hidden:
            hidden_dict['hidden_states'] = outputs.hidden_states
            hidden_dict['attentions'] = outputs.attentions
        
        return intent_logits, anomaly_score, hidden_dict
    
    def get_token_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute token importance from attention weights.
        
        Returns:
            importance: Token importance scores [batch_size, seq_len]
        """
        _, _, hidden_dict = self.forward(
            input_ids,
            attention_mask,
            return_hidden=True
        )
        
        # Aggregate attention across layers and heads
        attentions = hidden_dict['attentions']
        
        # Average across all layers and heads
        avg_attention = torch.stack(attentions).mean(dim=0).mean(dim=1)
        
        # Get attention to [CLS] token (which is used for classification)
        cls_attention = avg_attention[:, 0, :]  # [batch_size, seq_len]
        
        return cls_attention
```

---

## ☁️ AWS SageMaker Training

### Training Script for SageMaker

**File:** `training/scripts/sagemaker/train_enhanced_models.py`

```python
"""Enhanced training script for SageMaker."""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from transformers import AdamW, get_linear_schedule_with_warmup
import boto3
from pathlib import Path

class VulnerabilityDataset(Dataset):
    """Dataset for vulnerability detection."""
    
    def __init__(self, data_path: str):
        self.samples = []
        
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def train_epoch(model, loader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        # Move to device
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward
        if 'edge_index' in batch:
            # GNN model
            logits, _ = model(
                batch['x'],
                batch['edge_index'],
                batch['batch']
            )
        else:
            # Transformer model
            logits, _, _ = model(
                batch['input_ids'],
                batch['attention_mask']
            )
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(logits, batch['labels'])
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == batch['labels']).sum().item()
        total += batch['labels'].size(0)
    
    return total_loss / len(loader), correct / total

def validate(model, loader, device):
    """Validate model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Forward
            if 'edge_index' in batch:
                logits, _ = model(batch['x'], batch['edge_index'], batch['batch'])
            else:
                logits, _, _ = model(batch['input_ids'], batch['attention_mask'])
            
            pred = logits.argmax(dim=1)
            correct += (pred == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--model-type', type=str, choices=['gnn', 'transformer'])
    
    # SageMaker paths
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_dataset = VulnerabilityDataset(
        os.path.join(args.train, 'train.jsonl')
    )
    val_dataset = VulnerabilityDataset(
        os.path.join(args.validation, 'val.jsonl')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    
    # Create model
    if args.model_type == 'gnn':
        from training.models.enhanced_taint_gnn import EnhancedTaintFlowGNN
        model = EnhancedTaintFlowGNN().to(device)
    else:
        from training.models.enhanced_sql_intent import EnhancedSQLIntentTransformer
        model = EnhancedSQLIntentTransformer().to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        val_acc = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.model_dir, 'model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_path)
            print(f"✅ Saved best model (val_acc={val_acc:.4f})")
    
    print(f"\n✅ Training complete! Best val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
```

### Launch Training Jobs

**File:** `training/scripts/sagemaker/launch_enhanced_training.py`

```python
"""Launch enhanced training jobs on SageMaker."""

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
BUCKET = "streamguard-ml-v3"
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
INSTANCE_TYPE = "ml.p3.8xlarge"  # 4x V100 GPUs

def launch_gnn_training():
    """Launch Taint-Flow GNN training."""
    session = sagemaker.Session()
    
    estimator = PyTorch(
        entry_point='train_enhanced_models.py',
        source_dir='training/scripts/sagemaker',
        role=ROLE_ARN,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version='