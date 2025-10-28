# 03 - Explainability System Implementation

**Phase:** 2 (Week 6)  
**Prerequisites:** ML models trained ([02_ml_training.md](./02_ml_training.md))  
**Status:** Ready to Implement

---

## 📋 Overview

Implement deep explainability system that helps developers understand WHY vulnerabilities are detected.

**Components:**
- **Integrated Gradients**: Token-level saliency scores
- **Counterfactual Generation**: "What if it was safe?" examples
- **CVE Retrieval**: Similar vulnerabilities from FAISS
- **Confidence Decomposition**: Per-agent confidence breakdown
- **Explanation JSON**: Structured output for UI

**Target:** <100ms explainability overhead, 85%+ developer understanding

---

## 🏗️ Architecture

```
Detection Result
      ↓
┌─────────────────────────────────────┐
│   Explainability Engine             │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  Integrated Gradients        │  │
│  │  • Compute token saliency    │  │
│  │  • Identify key tokens       │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │  Counterfactual Generator    │  │
│  │  • Generate safe versions    │  │
│  │  • Minimal edit distance     │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │  CVE Retriever (FAISS)       │  │
│  │  • Semantic search           │  │
│  │  • Hybrid ranking            │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │  Explanation Formatter       │  │
│  │  • Build JSON output         │  │
│  │  • Compute confidence        │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
      ↓
  Explanation JSON
```

---

## 💻 Implementation

### 1. Integrated Gradients

**File:** `core/explainability/integrated_gradients.py`

```python
"""Integrated Gradients for token-level saliency."""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
from captum.attr import IntegratedGradients

class TokenSaliencyComputer:
    """Compute token-level importance using Integrated Gradients."""
    
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.ig = IntegratedGradients(self._forward_func)
    
    def _forward_func(self, input_ids):
        """Forward function for Integrated Gradients."""
        # Get model output (logits for vulnerability class)
        outputs = self.model(input_ids)
        return outputs.logits[:, 1]  # Vulnerability class score
    
    def compute_saliency(
        self,
        code: str,
        baseline_strategy: str = "zero"
    ) -> Dict[str, float]:
        """
        Compute token-level saliency scores.
        
        Args:
            code: Source code to analyze
            baseline_strategy: 'zero', 'pad', or 'random'
        
        Returns:
            Dictionary mapping tokens to saliency scores
        """
        # Tokenize input
        inputs = self.tokenizer(
            code,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs['input_ids']
        
        # Create baseline
        baseline = self._create_baseline(input_ids, baseline_strategy)
        
        # Compute attributions
        attributions = self.ig.attribute(
            input_ids,
            baselines=baseline,
            target=1,  # Vulnerability class
            n_steps=50,
            internal_batch_size=32
        )
        
        # Aggregate and normalize
        token_importance = attributions.sum(dim=-1).squeeze(0)
        token_importance = token_importance / token_importance.abs().max()
        
        # Map to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        saliency_map = {}
        
        for token, importance in zip(tokens, token_importance):
            if token not in ['[PAD]', '[CLS]', '[SEP]']:
                saliency_map[token] = float(importance)
        
        return saliency_map
    
    def _create_baseline(
        self,
        input_ids: torch.Tensor,
        strategy: str
    ) -> torch.Tensor:
        """Create baseline for IG."""
        if strategy == "zero":
            # Zero baseline (all zeros)
            return torch.zeros_like(input_ids)
        
        elif strategy == "pad":
            # Pad token baseline
            pad_token_id = self.tokenizer.pad_token_id
            return torch.full_like(input_ids, pad_token_id)
        
        elif strategy == "random":
            # Random token baseline
            vocab_size = self.tokenizer.vocab_size
            return torch.randint(0, vocab_size, input_ids.shape)
        
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")
    
    def get_top_k_tokens(
        self,
        saliency_map: Dict[str, float],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top-k most important tokens."""
        sorted_tokens = sorted(
            saliency_map.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_tokens[:k]
    
    def visualize_saliency(
        self,
        code: str,
        saliency_map: Dict[str, float]
    ) -> str:
        """Create ASCII visualization of saliency."""
        tokens = code.split()
        visualization = []
        
        for token in tokens:
            # Find closest match in saliency map
            saliency = 0.0
            for saliency_token, score in saliency_map.items():
                if saliency_token in token or token in saliency_token:
                    saliency = abs(score)
                    break
            
            # Color code based on importance
            if saliency > 0.7:
                visualization.append(f"🔴 {token}")
            elif saliency > 0.4:
                visualization.append(f"🟡 {token}")
            elif saliency > 0.2:
                visualization.append(f"🟢 {token}")
            else:
                visualization.append(token)
        
        return " ".join(visualization)


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("models/semantic_model.pth")
    
    # Create saliency computer
    saliency_computer = TokenSaliencyComputer(model, tokenizer)
    
    # Test code
    code = 'query = "SELECT * FROM users WHERE id=" + user_id'
    
    # Compute saliency
    saliency_map = saliency_computer.compute_saliency(code)
    
    # Get top tokens
    top_tokens = saliency_computer.get_top_k_tokens(saliency_map, k=5)
    print("Top 5 important tokens:")
    for token, score in top_tokens:
        print(f"  {token}: {score:.3f}")
    
    # Visualize
    visualization = saliency_computer.visualize_saliency(code, saliency_map)
    print(f"\nVisualization:\n{visualization}")
```

### 2. Counterfactual Generation

**File:** `core/explainability/counterfactuals.py`

```python
"""Generate counterfactual explanations."""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import Levenshtein

@dataclass
class Counterfactual:
    """Represents a counterfactual example."""
    original_code: str
    counterfactual_code: str
    edit_distance: int
    description: str
    plausibility_score: float

class CounterfactualGenerator:
    """Generate counterfactual examples showing safe alternatives."""
    
    def __init__(self):
        self.transformation_rules = self._load_transformation_rules()
    
    def _load_transformation_rules(self) -> Dict[str, List[Dict]]:
        """Load transformation rules for different vulnerability types."""
        return {
            "sql_injection": [
                {
                    "name": "concat_to_parameterized",
                    "pattern": r'"([^"]*)" \+ (\w+)',
                    "replacement": lambda m: f'"{m.group(1)}?", ({m.group(2)},)',
                    "description": "Use parameterized query instead of concatenation"
                },
                {
                    "name": "fstring_to_parameterized",
                    "pattern": r'f"([^{]*)\{(\w+)\}([^"]*)"',
                    "replacement": lambda m: f'"{m.group(1)}?{m.group(3)}", ({m.group(2)},)',
                    "description": "Replace f-string with parameterized query"
                },
                {
                    "name": "format_to_parameterized",
                    "pattern": r'"([^"]*)\{\}([^"]*)"\.format\((\w+)\)',
                    "replacement": lambda m: f'"{m.group(1)}?{m.group(2)}", ({m.group(3)},)',
                    "description": "Replace .format() with parameterized query"
                },
            ]
        }
    
    def generate_counterfactuals(
        self,
        vulnerable_code: str,
        vulnerability_type: str,
        num_examples: int = 3
    ) -> List[Counterfactual]:
        """
        Generate counterfactual examples.
        
        Args:
            vulnerable_code: Original vulnerable code
            vulnerability_type: Type of vulnerability
            num_examples: Number of counterfactuals to generate
        
        Returns:
            List of counterfactual examples
        """
        counterfactuals = []
        
        # Get transformation rules for this vulnerability type
        rules = self.transformation_rules.get(vulnerability_type, [])
        
        for rule in rules[:num_examples]:
            # Apply transformation
            transformed = self._apply_transformation(
                vulnerable_code,
                rule
            )
            
            if transformed and transformed != vulnerable_code:
                # Compute edit distance
                edit_distance = Levenshtein.distance(
                    vulnerable_code,
                    transformed
                )
                
                # Compute plausibility
                plausibility = self._compute_plausibility(
                    vulnerable_code,
                    transformed,
                    edit_distance
                )
                
                counterfactuals.append(Counterfactual(
                    original_code=vulnerable_code,
                    counterfactual_code=transformed,
                    edit_distance=edit_distance,
                    description=rule["description"],
                    plausibility_score=plausibility
                ))
        
        # Sort by plausibility
        counterfactuals.sort(key=lambda x: x.plausibility_score, reverse=True)
        
        return counterfactuals
    
    def _apply_transformation(
        self,
        code: str,
        rule: Dict
    ) -> Optional[str]:
        """Apply a transformation rule to code."""
        pattern = rule["pattern"]
        replacement = rule["replacement"]
        
        try:
            # Apply regex substitution
            if callable(replacement):
                transformed = re.sub(pattern, replacement, code)
            else:
                transformed = re.sub(pattern, replacement, code)
            
            return transformed if transformed != code else None
        except Exception as e:
            print(f"Error applying transformation: {e}")
            return None
    
    def _compute_plausibility(
        self,
        original: str,
        transformed: str,
        edit_distance: int
    ) -> float:
        """
        Compute plausibility score for counterfactual.
        
        Higher score = more plausible alternative
        Factors:
        - Minimal edits (lower edit distance)
        - Syntactic validity
        - Semantic similarity
        """
        # Normalize edit distance
        max_len = max(len(original), len(transformed))
        normalized_distance = 1.0 - (edit_distance / max_len)
        
        # Check syntax validity (simple heuristic)
        syntax_valid = self._check_syntax_validity(transformed)
        syntax_score = 1.0 if syntax_valid else 0.5
        
        # Semantic similarity (token overlap)
        semantic_score = self._compute_semantic_similarity(original, transformed)
        
        # Weighted combination
        plausibility = (
            0.4 * normalized_distance +
            0.3 * syntax_score +
            0.3 * semantic_score
        )
        
        return plausibility
    
    def _check_syntax_validity(self, code: str) -> bool:
        """Check if code is syntactically valid."""
        try:
            import ast
            ast.parse(code)
            return True
        except:
            return False
    
    def _compute_semantic_similarity(
        self,
        original: str,
        transformed: str
    ) -> float:
        """Compute semantic similarity based on token overlap."""
        original_tokens = set(original.split())
        transformed_tokens = set(transformed.split())
        
        intersection = original_tokens & transformed_tokens
        union = original_tokens | transformed_tokens
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def explain_difference(
        self,
        counterfactual: Counterfactual
    ) -> str:
        """Generate human-readable explanation of difference."""
        explanation = f"**What changed:** {counterfactual.description}\n\n"
        explanation += f"**Original (Vulnerable):**\n```\n{counterfactual.original_code}\n```\n\n"
        explanation += f"**Fixed Version:**\n```\n{counterfactual.counterfactual_code}\n```\n\n"
        explanation += f"**Why it's safer:** Parameterized queries prevent SQL injection by separating code from data.\n"
        explanation += f"**Edit distance:** {counterfactual.edit_distance} characters changed\n"
        
        return explanation


# Example usage
if __name__ == "__main__":
    generator = CounterfactualGenerator()
    
    vulnerable = 'query = "SELECT * FROM users WHERE id=" + user_id'
    
    counterfactuals = generator.generate_counterfactuals(
        vulnerable,
        "sql_injection",
        num_examples=3
    )
    
    print("Generated Counterfactuals:\n")
    for i, cf in enumerate(counterfactuals, 1):
        print(f"{i}. {cf.description}")
        print(f"   Plausibility: {cf.plausibility_score:.2f}")
        print(f"   {cf.counterfactual_code}\n")
```

### 3. CVE Retrieval with FAISS

**File:** `core/explainability/cve_retriever.py`

```python
"""CVE retrieval using FAISS for similar vulnerabilities."""

import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

class CVERetriever:
    """Retrieve similar CVEs using semantic search."""
    
    def __init__(
        self,
        index_path: str = "data/embeddings/cve_index.faiss",
        metadata_path: str = "data/embeddings/cve_metadata.json",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.encoder = SentenceTransformer(model_name)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        # Load or create index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = None
            self.metadata = []
    
    def build_index(self, cve_data: List[Dict]):
        """
        Build FAISS index from CVE data.
        
        Args:
            cve_data: List of CVE dictionaries with 'id', 'description', 'cwe', etc.
        """
        print(f"Building FAISS index from {len(cve_data)} CVEs...")
        
        # Extract descriptions
        descriptions = [cve.get('description', '') for cve in cve_data]
        
        # Generate embeddings
        embeddings = self.encoder.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata = cve_data
        
        # Save
        self._save_index()
        
        print(f"✅ Index built with {len(cve_data)} CVEs")
    
    def retrieve_similar(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve similar CVEs for a query.
        
        Args:
            query: Query text (vulnerability description or code)
            k: Number of results to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of similar CVEs with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(
            query_embedding.astype('float32'),
            k
        )
        
        # Format results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity >= min_similarity:
                cve = self.metadata[idx].copy()
                cve['similarity'] = float(similarity)
                results.append(cve)
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        cwe_filter: Optional[str] = None,
        k: int = 5
    ) -> List[Dict]:
        """
        Hybrid search combining semantic similarity and CWE filtering.
        
        Args:
            query: Query text
            cwe_filter: CWE ID to filter by (e.g., 'CWE-89')
            k: Number of results
        
        Returns:
            Filtered and ranked results
        """
        # First, semantic search with more results
        candidates = self.retrieve_similar(query, k=k*3, min_similarity=0.3)
        
        # Apply CWE filter if provided
        if cwe_filter:
            candidates = [
                cve for cve in candidates
                if cwe_filter in cve.get('cwe_ids', [])
            ]
        
        # Re-rank and return top-k
        return candidates[:k]
    
    def explain_similarity(
        self,
        query: str,
        cve: Dict
    ) -> str:
        """Generate explanation for why CVE is similar."""
        explanation = f"**CVE-{cve['id']}** (Similarity: {cve['similarity']:.2%})\n\n"
        explanation += f"**Description:** {cve.get('description', 'N/A')}\n\n"
        
        if 'cwe_ids' in cve:
            explanation += f"**CWE Categories:** {', '.join(cve['cwe_ids'])}\n\n"
        
        if 'cvss_score' in cve:
            explanation += f"**CVSS Score:** {cve['cvss_score']}\n\n"
        
        explanation += f"**Why it's similar:** Both involve similar vulnerability patterns and attack vectors.\n"
        
        return explanation
    
    def _save_index(self):
        """Save FAISS index and metadata."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Load CVE data
    cve_data = []
    with open('data/raw/cves/cve_data.jsonl', 'r') as f:
        for line in f:
            cve = json.loads(line)
            cve_data.append({
                'id': cve.get('cve_id', 'unknown'),
                'description': cve.get('description', ''),
                'cwe_ids': cve.get('cwe_ids', []),
                'cvss_score': cve.get('cvss_score')
            })
    
    # Build index
    retriever = CVERetriever()
    retriever.build_index(cve_data)
    
    # Test retrieval
    query = "SQL injection through string concatenation in web application"
    results = retriever.retrieve_similar(query, k=5)
    
    print(f"Top 5 similar CVEs for: {query}\n")
    for i, cve in enumerate(results, 1):
        print(f"{i}. CVE-{cve['id']} (Similarity: {cve['similarity']:.2%})")
        print(f"   {cve['description'][:100]}...\n")
```

### 4. Explanation Formatter

**File:** `core/explainability/explanation_formatter.py`

```python
"""Format explainability results into structured JSON."""

from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import json

@dataclass
class ExplanationOutput:
    """Structured explanation output."""
    vulnerability_id: str
    reason: str
    token_importance: List[Dict[str, Any]]
    similar_cves: List[Dict[str, Any]]
    counterfactuals: List[str]
    confidence_breakdown: Dict[str, float]
    visualization: str

class ExplanationFormatter:
    """Format all explainability components into unified output."""
    
    def format_explanation(
        self,
        vulnerability_id: str,
        detection_result: Dict,
        saliency_map: Dict[str, float],
        counterfactuals: List,
        similar_cves: List[Dict],
        agent_confidences: Dict[str, float]
    ) -> ExplanationOutput:
        """
        Format complete explanation.
        
        Args:
            vulnerability_id: Unique ID for this vulnerability
            detection_result: Original detection result
            saliency_map: Token saliency scores
            counterfactuals: List of counterfactual examples
            similar_cves: Similar CVEs from retrieval
            agent_confidences: Per-agent confidence scores
        
        Returns:
            Structured explanation output
        """
        # Format token importance
        token_importance = self._format_token_importance(saliency_map)
        
        # Format CVEs
        formatted_cves = self._format_cves(similar_cves)
        
        # Format counterfactuals
        formatted_counterfactuals = self._format_counterfactuals(counterfactuals)
        
        # Generate reason
        reason = self._generate_reason(detection_result, saliency_map)
        
        # Confidence breakdown
        confidence_breakdown = self._compute_confidence_breakdown(agent_confidences)
        
        # Create visualization
        visualization = self._create_visualization(
            detection_result.get('code', ''),
            saliency_map
        )
        
        return ExplanationOutput(
            vulnerability_id=vulnerability_id,
            reason=reason,
            token_importance=token_importance,
            similar_cves=formatted_cves,
            counterfactuals=formatted_counterfactuals,
            confidence_breakdown=confidence_breakdown,
            visualization=visualization
        )
    
    def _format_token_importance(
        self,
        saliency_map: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Format token importance for output."""
        # Sort by absolute importance
        sorted_tokens = sorted(
            saliency_map.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        formatted = []
        for token, score in sorted_tokens[:10]:  # Top 10
            contribution = "high" if abs(score) > 0.7 else "medium" if abs(score) > 0.4 else "low"
            
            formatted.append({
                "token": token,
                "saliency": float(score),
                "contribution": contribution
            })
        
        return formatted
    
    def _format_cves(self, cves: List[Dict]) -> List[Dict[str, Any]]:
        """Format CVE results."""
        formatted = []
        for cve in cves[:3]:  # Top 3
            formatted.append({
                "cve_id": cve.get('id', 'unknown'),
                "similarity": float(cve.get('similarity', 0.0)),
                "description": cve.get('description', '')[:200],
                "cwe_ids": cve.get('cwe_ids', []),
                "cvss_score": cve.get('cvss_score')
            })
        
        return formatted
    
    def _format_counterfactuals(self, counterfactuals: List) -> List[str]:
        """Format counterfactual examples."""
        return [
            cf.counterfactual_code
            for cf in counterfactuals[:3]
        ]
    
    def _generate_reason(
        self,
        detection_result: Dict,
        saliency_map: Dict[str, float]
    ) -> str:
        """Generate human-readable reason."""
        vuln_type = detection_result.get('vulnerability_type', 'Unknown')
        
        # Get top contributing tokens
        top_tokens = sorted(
            saliency_map.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        token_names = [token for token, _ in top_tokens]
        
        reason = f"Detected {vuln_type}. "
        reason += f"Key indicators: {', '.join(token_names)}. "
        reason += detection_result.get('message', '')
        
        return reason
    
    def _compute_confidence_breakdown(
        self,
        agent_confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute overall confidence from agent confidences."""
        breakdown = agent_confidences.copy()
        
        # Compute weighted average
        weights = {
            'syntax': 0.2,
            'semantic': 0.4,
            'context': 0.3,
            'verification': 0.1
        }
        
        overall = sum(
            agent_confidences.get(agent, 0.0) * weight
            for agent, weight in weights.items()
        )
        
        breakdown['overall'] = overall
        
        return breakdown
    
    def _create_visualization(
        self,
        code: str,
        saliency_map: Dict[str, float]
    ) -> str:
        """Create ASCII visualization."""
        lines = code.split('\n')
        visualization = []
        
        for line in lines:
            tokens = line.split()
            viz_line = []
            
            for token in tokens:
                # Find saliency
                saliency = 0.0
                for sal_token, score in saliency_map.items():
                    if sal_token in token or token in sal_token:
                        saliency = abs(score)
                        break
                
                # Add color indicator
                if saliency > 0.7:
                    viz_line.append(f"🔴{token}")
                elif saliency > 0.4:
                    viz_line.append(f"🟡{token}")
                else:
                    viz_line.append(token)
            
            visualization.append(" ".join(viz_line))
        
        return "\n".join(visualization)
    
    def to_json(self, explanation: ExplanationOutput) -> str:
        """Convert explanation to JSON string."""
        return json.dumps(asdict(explanation), indent=2)
    
    def to_dict(self, explanation: ExplanationOutput) -> Dict:
        """Convert explanation to dictionary."""
        return asdict(explanation)


# Example usage
if __name__ == "__main__":
    formatter = ExplanationFormatter()
    
    # Example data
    detection_result = {
        'vulnerability_type': 'sql_injection',
        'message': 'SQL injection via string concatenation',
        'code': 'query = "SELECT * FROM users WHERE id=" + user_id'
    }
    
    saliency_map = {
        'user_id': 0.87,
        '+': 0.65,
        'execute': 0.45,
        'query': 0.32
    }
    
    counterfactuals = []  # Would be populated
    similar_cves = []  # Would be populated
    agent_confidences = {
        'syntax': 0.85,
        'semantic': 0.95,
        'context': 0.90
    }
    
    explanation = formatter.format_explanation(
        vulnerability_id="vuln_001",
        detection_result=detection_result,
        saliency_map=saliency_map,
        counterfactuals=counterfactuals,
        similar_cves=similar_cves,
        agent_confidences=agent_confidences
    )
    
    print(formatter.to_json(explanation))
```

### 5. Integrated Explainability Engine

**File:** `core/explainability/explainability_engine.py`

```python
"""Main explainability engine integrating all components."""

from typing import Dict, Any
from .integrated_gradients import TokenSaliencyComputer
from .counterfactuals import CounterfactualGenerator
from .cve_retriever import CVERetriever
from .explanation_formatter import ExplanationFormatter

class ExplainabilityEngine:
    """Main engine for generating explanations."""
    
    def __init__(self, model, tokenizer):
        self.saliency_computer = TokenSaliencyComputer(model, tokenizer)
        self.counterfactual_generator = CounterfactualGenerator()
        self.cve_retriever = CVERetriever()
        self.formatter = ExplanationFormatter()
    
    def explain(
        self,
        vulnerability_id: str,
        code: str,
        detection_result: Dict[str, Any],
        agent_confidences: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate complete explanation for a detected vulnerability.
        
        Args:
            vulnerability_id: Unique ID
            code: Source code
            detection_result: Detection result from agents
            agent_confidences: Per-agent confidence scores
        
        Returns:
            Complete explanation dictionary
        """
        # 1. Compute token saliency
        saliency_map = self.saliency_computer.compute_saliency(code)
        
        # 2. Generate counterfactuals
        counterfactuals = self.counterfactual_generator.generate_counterfactuals(
            code,
            detection_result['vulnerability_type'],
            num_examples=3
        )
        
        # 3. Retrieve similar CVEs
        query = f"{detection_result['vulnerability_type']} {detection_result.get('message', '')}"
        similar_cves = self.cve_retriever.retrieve_similar(query, k=3)
        
        # 4. Format explanation
        explanation = self.formatter.format_explanation(
            vulnerability_id=vulnerability_id,
            detection_result=detection_result,
            saliency_map=saliency_map,
            counterfactuals=counterfactuals,
            similar_cves=similar_cves,
            agent_confidences=agent_confidences
        )
        
        return self.formatter.to_dict(explanation)
```

---

## 🧪 Testing

```python
# tests/unit/test_explainability.py
"""Tests for explainability system."""

import pytest
from core.explainability.integrated_gradients import TokenSaliencyComputer
from core.explainability.counterfactuals import CounterfactualGenerator
from core.explainability.explainability_engine import ExplainabilityEngine

class TestIntegratedGradients:
    def test_compute_saliency(self, mock_model, mock_tokenizer):
        computer = TokenSaliencyComputer(mock_model, mock_tokenizer)
        code = 'query = "SELECT * FROM users WHERE id=" + user_id'
        
        saliency_map = computer.compute_saliency(code)
        
        assert isinstance(saliency_map, dict)
        assert len(saliency_map) > 0
        assert all(-1.0 <= v <= 1.0 for v in saliency_map.values())
    
    def test_top_k_tokens(self, mock_model, mock_tokenizer):
        computer = TokenSaliencyComputer(mock_model, mock_tokenizer)
        saliency_map = {'token1': 0.9, 'token2': 0.5, 'token3': 0.3}
        
        top_tokens = computer.get_top_k_tokens(saliency_map, k=2)
        
        assert len(top_tokens) == 2
        assert top_tokens[0][0] == 'token1'

class TestCounterfactualGenerator:
    def test_generate_counterfactuals(self):
        generator = CounterfactualGenerator()
        vulnerable = 'query = "SELECT * FROM users WHERE id=" + user_id'
        
        counterfactuals = generator.generate_counterfactuals(
            vulnerable,
            "sql_injection",
            num_examples=2
        )
        
        assert len(counterfactuals) > 0
        assert counterfactuals[0].counterfactual_code != vulnerable
        assert '?' in counterfactuals[0].counterfactual_code

class TestExplainabilityEngine:
    def test_full_explanation(self, mock_model, mock_tokenizer):
        engine = ExplainabilityEngine(mock_model, mock_tokenizer)
        
        code = 'query = "SELECT * FROM users WHERE id=" + user_id'
        detection_result = {
            'vulnerability_type': 'sql_injection',
            'message': 'SQL injection detected',
            'confidence': 0.95
        }
        agent_confidences = {'syntax': 0.85, 'semantic': 0.95}
        
        explanation = engine.explain(
            'vuln_001',
            code,
            detection_result,
            agent_confidences
        )
        
        assert 'reason' in explanation
        assert 'token_importance' in explanation
        assert 'counterfactuals' in explanation
        assert 'confidence_breakdown' in explanation
```

---

## ✅ Implementation Checklist

- [ ] Implement Integrated Gradients
- [ ] Create Counterfactual Generator
- [ ] Build CVE Retriever with FAISS
- [ ] Implement Explanation Formatter
- [ ] Create unified Explainability Engine
- [ ] Write comprehensive tests (>90% coverage)
- [ ] Performance optimization (<100ms overhead)
- [ ] Integration with detection pipeline
- [ ] Documentation and examples

---

## 📊 Performance Targets

| Component | Target | Maximum |
|-----------|--------|---------|
| Integrated Gradients | 50ms | 100ms |
| Counterfactual Generation | 30ms | 50ms |
| CVE Retrieval | 20ms | 40ms |
| Formatting | 10ms | 20ms |
| **Total Overhead** | **<100ms** | **200ms** |

---

**Status:** ✅ Ready for Implementation  
**Next:** [04_agent_architecture.md](./04_agent_architecture.md)