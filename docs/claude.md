# CLAUDE.md - StreamGuard Enhanced Project Guide

> **Purpose:** This file is automatically read by Claude Code when you start a conversation. It provides complete project context, architecture, conventions, and workflow guidance for building StreamGuard.

**Version:** 3.0 (Enhanced)  
**Project:** StreamGuard - AI-Powered Real-Time Vulnerability Prevention System  
**Status:** Ready for Implementation  
**Last Updated:** October 2024

---

## 📚 Table of Contents

1. [Project Overview](#-project-overview)
2. [Enhanced Architecture](#-enhanced-architecture-v30)
3. [What's New in v3.0](#-whats-new-in-v30)
4. [Rules & Conventions](#-rules--conventions)
5. [Claude Code Workflow](#-claude-code-workflow--best-practices)
6. [Implementation Roadmap](#-implementation-roadmap-enhanced)
7. [Quick Reference](#-quick-reference)

---

## 🎯 Project Overview

### What is StreamGuard?

StreamGuard is a **real-time, AI-powered vulnerability prevention system** that detects security issues as developers write code, preventing vulnerabilities before they reach production.

**Core Innovation (Unchanged):**
- Multi-agent hierarchical detection pipeline (Syntax → Semantic → Context → Verification)
- 95%+ detection accuracy with <3% false positive rate
- <1 second real-time analysis
- Privacy-first architecture
- Explainable results with automated fixes

**New Enhancements (v3.0):**
- 🔍 **Deep Explainability**: Token-level saliency, counterfactual analysis, CVE retrieval
- 🚀 **Platform-Independent Agent**: Cross-IDE local agent with REST/WebSocket APIs
- 🧠 **Continuous Learning**: RLHF-lite feedback loop for model improvement
- 📊 **Repository Context**: Graph-based dependency tracking with Neo4j/TigerGraph
- 🎨 **Enhanced UI**: Local web dashboard with compliance reporting
- ✅ **Verified Patches**: Symbolic execution + fuzzing validation

### Success Metrics (Updated)

**Technical Targets:**
```yaml
Detection Accuracy: ≥95%
False Positive Rate: <3%
False Negative Rate: <5%
Latency (P95): <1 second
Memory Usage: <2GB active
Explainability Score: >85% (developer understanding)
Patch Acceptance Rate: >70% (with verification)
```

**Business Goals:**
```yaml
Month 3: 1,000+ DAU, 5,000+ installations
Month 6: 1,000+ GitHub stars, 100+ paid conversions
Month 9: 5+ enterprise pilots
User Metrics: 80%+ fix acceptance, NPS ≥60
Feedback Quality: 90%+ actionable feedback collected
```

---

## 🏗️ Enhanced Architecture (v3.0)

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Universal IDE Integration                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ VS Code      │  │ IntelliJ     │  │ Any IDE via        │   │
│  │ Extension    │  │ Plugin       │  │ REST/WebSocket     │   │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘   │
│         │                  │                     │               │
│         └──────────────────┴─────────────────────┘               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              │ REST/WebSocket API
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│              Local Real-Time Detection Agent                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  File Monitor & Diff Tracker                             │  │
│  │  • Watches file changes (inotify/FSEvents)               │  │
│  │  • Computes incremental diffs                            │  │
│  │  • Triggers analysis on save/commit                      │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │  Detection Pipeline (Multi-Agent)                        │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ Syntax Agent (50ms) - Pattern Matching            │  │  │
│  │  └─────────────┬──────────────────────────────────────┘  │  │
│  │                │                                          │  │
│  │  ┌─────────────▼──────────────────────────────────────┐  │  │
│  │  │ Semantic Agent (200ms) - ML Detection              │  │  │
│  │  │ • Taint-Flow GNN                                   │  │  │
│  │  │ • CodeBERT/CodeLLaMA Fine-tuned                    │  │  │
│  │  │ • Integrated Gradients for Explainability          │  │  │
│  │  └─────────────┬──────────────────────────────────────┘  │  │
│  │                │                                          │  │
│  │  ┌─────────────▼──────────────────────────────────────┐  │  │
│  │  │ Context Agent (500ms) - Repository Awareness       │  │  │
│  │  │ • Neo4j/TigerGraph for dependency graph            │  │  │
│  │  │ • Cross-file taint propagation                     │  │  │
│  │  │ • Entry point & attack surface mapping            │  │  │
│  │  └─────────────┬──────────────────────────────────────┘  │  │
│  │                │                                          │  │
│  │  ┌─────────────▼──────────────────────────────────────┐  │  │
│  │  │ Verification Agent (300ms) - Attack Simulation     │  │  │
│  │  │ • Query reconstruction                             │  │  │
│  │  │ • Symbolic execution verification                  │  │  │
│  │  │ • Fuzzing validation                               │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │  Explainability Engine (NEW)                             │  │
│  │  • Token-level saliency (Integrated Gradients)           │  │
│  │  • Counterfactual analysis ("what if safe?")             │  │
│  │  • CVE retrieval with FAISS                              │  │
│  │  • Confidence decomposition                              │  │
│  │  • Output: explanation JSON + visualization             │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │  Patch Generator & Verifier (NEW)                        │  │
│  │  • Template-based + LLM patch generation                 │  │
│  │  • Symbolic execution validation                         │  │
│  │  • Fuzzing test generation                               │  │
│  │  • Differential testing (behavior preservation)          │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │  Feedback & Learning System (NEW)                        │  │
│  │  • Developer feedback collection (correct/false positive)│  │
│  │  • Local feedback aggregation                            │  │
│  │  • Secure sync to SageMaker for retraining               │  │
│  │  • Drift detection & auto-retrain triggers               │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ Results + Explanations
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              Local Web Dashboard (React/Tauri)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Visualization & Reporting                               │  │
│  │  • Inline code highlights with explanations              │  │
│  │  • Interactive taint path graphs                         │  │
│  │  • CVE evidence cards                                    │  │
│  │  • Suggested fixes with verification status              │  │
│  │  • Team pattern libraries                                │  │
│  │  • Compliance reports (PDF/JSON/SARIF)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                               │
                               │ Secure sync (encrypted)
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                 AWS SageMaker Training Pipeline                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Continuous Learning Loop                                │  │
│  │  • Feedback aggregation & preprocessing                  │  │
│  │  • Model retraining (CodeBERT/CodeLLaMA fine-tuning)     │  │
│  │  • Evaluation & A/B testing                              │  │
│  │  • Model registry & versioning                           │  │
│  │  • Automatic deployment on approval                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

Supporting Systems:
┌──────────────────────────────────────────────────────────────────┐
│  Neo4j/TigerGraph: Repository dependency graph                  │
│  FAISS: CVE/CWE vector database (100K+ examples)                │
│  PostgreSQL: Feedback storage, pattern library                  │
│  Redis: Caching layer for fast retrieval                        │
└──────────────────────────────────────────────────────────────────┘
```

### Key Architectural Changes

**🔄 What Changed:**

1. **IDE Plugin → Universal Agent**
   - **Before**: VS Code-specific extension
   - **After**: Platform-independent local agent with REST/WebSocket APIs
   - **Benefit**: Works with any IDE, CI/CD, pre-commit hooks

2. **Static Explanations → Deep Explainability**
   - **Before**: Basic taint paths and CVE links
   - **After**: Token-level saliency, counterfactual analysis, confidence decomposition
   - **Benefit**: Developers understand "why" at granular level

3. **Single Model → Continuous Learning**
   - **Before**: Static trained models
   - **After**: RLHF-lite feedback loop with automatic retraining
   - **Benefit**: Models improve over time with real usage

4. **Simple Context → Graph-Based Repository Awareness**
   - **Before**: File-level context with basic cross-file tracking
   - **After**: Full dependency graph in Neo4j/TigerGraph with vulnerability propagation
   - **Benefit**: Understand entire attack surface and propagation paths

5. **Basic Fixes → Verified Patches**
   - **Before**: Template-based suggestions
   - **After**: Symbolic execution + fuzzing validation
   - **Benefit**: Higher confidence in patch correctness

---

## 🆕 What's New in v3.0

### 1. Deep Explainability System

**Token-Level Saliency with Integrated Gradients:**
```python
# For each detected vulnerability
saliency_map = compute_integrated_gradients(
    model=semantic_model,
    input_tokens=code_tokens,
    baseline=neutral_baseline
)

# Output: Which tokens contributed most to detection
# Example: "user_input" token has 0.87 saliency score
```

**Counterfactual Analysis:**
```python
# Generate "what if it was safe?" examples
counterfactuals = generate_counterfactuals(
    vulnerable_code=original_code,
    target_label="safe",
    num_examples=3
)

# Shows minimal changes to make code safe
# Example: "If you used parameterized query, this would be safe"
```

**CVE Retrieval with Explanation:**
```json
{
  "vulnerability_id": "vuln_001",
  "explanation": {
    "reason": "String concatenation detected in SQL query construction",
    "token_importance": [
      {"token": "user_input", "saliency": 0.87, "contribution": "high"},
      {"token": "+", "saliency": 0.65, "contribution": "medium"},
      {"token": "execute", "saliency": 0.45, "contribution": "medium"}
    ],
    "similar_cves": [
      {
        "cve_id": "CVE-2023-12345",
        "similarity": 0.92,
        "description": "SQL injection via concatenation in web application",
        "fix_pattern": "Use parameterized queries"
      }
    ],
    "counterfactuals": [
      "If you used cursor.execute('SELECT * FROM users WHERE id=?', (user_id,)), this would be safe"
    ],
    "confidence_breakdown": {
      "syntax_confidence": 0.85,
      "semantic_confidence": 0.95,
      "context_confidence": 0.90,
      "overall": 0.93
    }
  }
}
```

### 2. Platform-Independent Local Agent

**Architecture:**
```
Local Agent (Python FastAPI)
    ↓
┌─────────────────────────────────┐
│  REST API (Port 8765)           │
│  • POST /analyze                │
│  • GET /status                  │
│  • POST /feedback               │
│  • GET /patterns                │
└─────────────────────────────────┘
         │
         │
┌────────┴─────────────────────────┐
│  WebSocket (Real-time updates)  │
│  ws://localhost:8765/stream      │
└──────────────────────────────────┘
         │
         │
┌────────┴─────────────────────────┐
│  File System Monitor             │
│  • inotify (Linux)               │
│  • FSEvents (macOS)              │
│  • ReadDirectoryChangesW (Win)   │
└──────────────────────────────────┘
```

**Integration Options:**
```bash
# VS Code Extension
const response = await fetch('http://localhost:8765/analyze', {
  method: 'POST',
  body: JSON.stringify({ code, language, filePath })
});

# IntelliJ Plugin
HttpClient.post("http://localhost:8765/analyze", analysisRequest);

# Pre-commit Hook
streamguard-cli analyze --staged

# CI/CD Pipeline
docker run streamguard/agent analyze --project-root=/app
```

### 3. Repository-Aware Context with Graph Database

**Neo4j/TigerGraph Integration:**
```cypher
// Create code dependency graph
CREATE (f1:File {path: 'auth.py'})
CREATE (f2:File {path: 'database.py'})
CREATE (func1:Function {name: 'login', file: 'auth.py'})
CREATE (func2:Function {name: 'query_user', file: 'database.py'})
CREATE (func1)-[:CALLS]->(func2)
CREATE (func2)-[:ACCESSES]->(db:Database {name: 'users_db'})

// Track taint propagation
MATCH path = (source:TaintSource)-[:FLOWS_TO*]->(sink:VulnerableSink)
WHERE source.input_type = 'user_input'
RETURN path, length(path) as propagation_depth

// Find attack surface
MATCH (entry:EntryPoint)-[:LEADS_TO*]->(vuln:Vulnerability)
WHERE entry.exposure = 'public'
RETURN entry, vuln, shortestPath(entry, vuln)
```

**Vulnerability Propagation Tracking:**
```python
# When vulnerability detected in function A
def track_propagation(vuln_function: str, graph: Neo4jGraph):
    # Find all callers
    callers = graph.query("""
        MATCH (caller)-[:CALLS]->(vuln:Function {name: $func_name})
        RETURN caller
    """, func_name=vuln_function)
    
    # Mark as potentially affected
    for caller in callers:
        mark_for_review(caller, reason="calls vulnerable function")
    
    # Find data flow impact
    affected_data = graph.query("""
        MATCH (vuln:Function {name: $func_name})-[:RETURNS_DATA]->(data)
        -[:USED_BY]->(other_func)
        RETURN other_func, data
    """, func_name=vuln_function)
    
    return PropagationReport(
        direct_callers=callers,
        affected_data_flows=affected_data,
        risk_level=compute_risk(callers, affected_data)
    )
```

### 4. Continuous Learning with RLHF-lite

**Developer Feedback Loop:**
```python
# Developer marks suggestion as correct or false positive
feedback = {
    "suggestion_id": "vuln_001_fix_1",
    "developer_action": "accepted",  # or "rejected"
    "timestamp": "2024-10-08T10:30:00Z",
    "code_context": hash_code_context(original_code),
    "reasoning": "Good suggestion, prevented real vulnerability"
}

# Aggregate locally (privacy-preserved)
feedback_store.add(feedback)

# Periodically sync to SageMaker (encrypted, anonymized)
if feedback_store.count() > 100:
    anonymized_feedback = anonymize_feedback_batch(
        feedback_store.get_batch(100)
    )
    securely_sync_to_sagemaker(anonymized_feedback)
```

**Retraining Trigger:**
```python
# Monitor model drift
def check_drift(current_metrics, baseline_metrics):
    if current_metrics['accuracy'] < baseline_metrics['accuracy'] - 0.05:
        trigger_retraining(reason="accuracy_drift")
    
    if current_metrics['false_positive_rate'] > baseline_metrics['fpr'] + 0.02:
        trigger_retraining(reason="fp_rate_increase")

# Automatic retraining pipeline
def trigger_retraining(reason: str):
    # Collect new training data from feedback
    new_data = collect_feedback_samples(min_samples=1000)
    
    # Launch SageMaker training job
    training_job = sagemaker_client.create_training_job(
        TrainingJobName=f"streamguard-retrain-{timestamp}",
        HyperParameters={'learning_rate': '1e-5'},
        InputDataConfig=[
            {'DataSource': {'S3DataSource': {
                'S3Uri': f's3://streamguard-training/feedback/{timestamp}/'
            }}}
        ]
    )
    
    # After training, evaluate before deployment
    if evaluate_new_model(training_job.model_artifact):
        deploy_model(training_job.model_artifact)
```

### 5. Enhanced UI with Local Web Dashboard

**Technology Stack:**
- **Frontend**: React + TypeScript
- **Desktop App**: Tauri (Rust-based, lightweight)
- **API**: REST + WebSocket to local agent
- **Visualization**: D3.js, Cytoscape.js for graphs

**Dashboard Features:**
```typescript
interface Dashboard {
  // Real-time code analysis view
  codeView: {
    highlightedVulnerabilities: Vulnerability[];
    inlineExplanations: Explanation[];
    suggededFixes: Fix[];
  };
  
  // Interactive taint path graph
  taintPathView: {
    nodes: CodeNode[];
    edges: DataFlowEdge[];
    highlightPath: (source: Node, sink: Node) => void;
  };
  
  // CVE evidence cards
  evidenceView: {
    similarCVEs: CVE[];
    cweMappings: CWE[];
    osapCategories: string[];
  };
  
  // Compliance reporting
  reportingView: {
    generatePDF: () => Promise<Blob>;
    exportJSON: () => JSON;
    exportSARIF: () => SARIF;
  };
  
  // Team pattern library
  patternsView: {
    commonVulnerabilities: Pattern[];
    teamSpecificPatterns: Pattern[];
    evolutionTimeline: PatternEvolution[];
  };
}
```

### 6. Verified Patch Generation

**Symbolic Execution Validation:**
```python
def verify_patch_with_symbolic_execution(
    original_code: str,
    patched_code: str,
    vulnerability_type: str
) -> VerificationResult:
    # Use angr or KLEE for symbolic execution
    original_paths = symbolic_executor.explore(original_code)
    patched_paths = symbolic_executor.explore(patched_code)
    
    # Check if vulnerability path still exists in patched version
    vuln_path_exists = any(
        is_vulnerable_path(path, vulnerability_type)
        for path in patched_paths
    )
    
    # Check if behavior is preserved for safe inputs
    behavior_preserved = check_behavioral_equivalence(
        original_paths, patched_paths, safe_inputs
    )
    
    return VerificationResult(
        vulnerability_fixed=not vuln_path_exists,
        behavior_preserved=behavior_preserved,
        confidence=0.98 if behavior_preserved else 0.70
    )
```

**Fuzzing Validation:**
```python
def fuzz_test_patch(
    patched_code: str,
    vulnerability_type: str,
    num_iterations: int = 1000
) -> FuzzResult:
    fuzzer = create_fuzzer_for_vulnerability(vulnerability_type)
    
    vulnerabilities_found = []
    for i in range(num_iterations):
        test_input = fuzzer.generate_input()
        result = execute_code(patched_code, test_input)
        
        if is_exploitable(result, vulnerability_type):
            vulnerabilities_found.append(test_input)
    
    return FuzzResult(
        iterations=num_iterations,
        vulnerabilities_found=len(vulnerabilities_found),
        exploit_examples=vulnerabilities_found[:5],
        is_secure=len(vulnerabilities_found) == 0
    )
```

---

## 📋 Rules & Conventions

### Naming Conventions

**Python:**
```python
# Classes: PascalCase
class ExplainabilityEngine:
    pass

# Functions/methods: snake_case
def compute_integrated_gradients(model, input_tokens) -> SaliencyMap:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_COUNTERFACTUALS = 5
DEFAULT_SALIENCY_STEPS = 50

# Private members: _leading_underscore
def _compute_baseline(self, tokens):
    pass

# Files/modules: snake_case
# explainability_engine.py, graph_builder.py
```

**TypeScript:**
```typescript
// Classes/Interfaces: PascalCase
class DashboardController {}
interface ExplanationView {}

// Functions/variables: camelCase
function renderTaintPath(path: TaintPath): void {}
const maxVisualizationNodes = 100;

// Constants: UPPER_SNAKE_CASE
const DEFAULT_WEBSOCKET_PORT = 8765;

// Files: kebab-case
// dashboard-controller.ts, explanation-view.tsx
```

### Enhanced Security Practices

**Never Execute User Code:**
```python
# ❌ NEVER DO THIS
eval(user_code)
exec(user_code)
os.system(user_input)

# ✅ ALWAYS DO THIS
tree = parser.parse(bytes(user_code, "utf8"))  # Static analysis only
```

**Secure Feedback Collection:**
```python
def anonymize_feedback(feedback: Feedback) -> AnonymizedFeedback:
    """Remove all PII before syncing to cloud."""
    return AnonymizedFeedback(
        action=feedback.action,  # "accepted" or "rejected"
        vulnerability_type=feedback.vulnerability_type,  # "sql_injection"
        code_hash=hash_code(feedback.code),  # One-way hash
        timestamp_bucket=bucket_timestamp(feedback.timestamp, hours=24),
        # NO: code, file_path, developer_id, company_name
    )
```

**Graph Database Security:**
```python
# Sanitize all inputs to Neo4j queries
def sanitize_cypher_param(value: str) -> str:
    # Prevent Cypher injection
    return value.replace("'", "\\'").replace('"', '\\"')

# Always use parameterized queries
graph.query(
    "MATCH (n:Function {name: $func_name}) RETURN n",
    func_name=sanitize_cypher_param(user_input)
)
```

---

## 🤖 Claude Code Workflow & Best Practices

### Sub-Agents for Enhanced StreamGuard

**1. Explainability Agent** (`.claude/agents/explainability.yml`)
```yaml
name: explainability
description: |
  Specialist in ML explainability techniques:
  - Integrated Gradients implementation
  - Counterfactual generation
  - Saliency visualization
  - Confidence decomposition
tools: [file_read, file_write, bash_execute]
model: opus  # Requires high capability
instructions: |
  Focus on:
  - PyTorch gradient computation
  - SHAP/LIME integration
  - Visualization with D3.js/Plotly
  - Performance optimization (<100ms overhead)
```

**2. Graph Systems Agent** (`.claude/agents/graph-systems.yml`)
```yaml
name: graph-systems
description: |
  Neo4j/TigerGraph specialist for:
  - Repository dependency graph construction
  - Cypher/GSQL query optimization
  - Vulnerability propagation tracking
  - Attack surface mapping
tools: [file_read, file_write, bash_execute]
model: sonnet
instructions: |
  Best practices:
  - Use parameterized queries
  - Index frequently accessed properties
  - Batch operations for performance
  - Monitor query performance
```

**3. Dashboard Agent** (`.claude/agents/dashboard.yml`)
```yaml
name: dashboard
description: |
  React/Tauri dashboard specialist:
  - Interactive visualization (D3.js, Cytoscape)
  - Real-time WebSocket integration
  - Compliance report generation
  - Responsive UI/UX
tools: [file_read, file_write, bash_execute]
model: sonnet
```

### Using Plan Mode for Enhanced Features

**Example: Implementing Explainability**
```bash
claude --plan "Implement deep explainability system with:
1. Integrated Gradients for token-level saliency
2. Counterfactual generation
3. CVE retrieval with FAISS
4. Output explanation JSON
Target: <100ms overhead, 85%+ developer understanding"
```

**Claude's Response:**
```
📋 Implementation Plan for Explainability System

Phase 1: Integrated Gradients (Week 1)
  1. Implement gradient computation wrapper for semantic models
  2. Create baseline selection strategy (zero-baseline, random, masked)
  3. Add interpolation between baseline and input
  4. Compute gradients at each step
  5. Aggregate and visualize saliency scores
  
  Files to create:
  - core/explainability/integrated_gradients.py
  - core/explainability/baselines.py
  - tests/unit/test_integrated_gradients.py

Phase 2: Counterfactual Generation (Week 1)
  6. Implement perturbation-based counterfactual search
  7. Add constraints (minimal changes, syntax validity)
  8. Generate top-k counterfactuals ranked by plausibility
  9. Add caching for common patterns
  
  Files to create:
  - core/explainability/counterfactuals.py
  - tests/unit/test_counterfactuals.py

Phase 3: CVE Retrieval (Week 2)
  10. Build FAISS index from 100K CVEs
  11. Implement hybrid search (semantic + structural)
  12. Add relevance ranking
  13. Cache frequently retrieved CVEs
  
  Files to create:
  - core/rag/cve_retriever.py
  - core/rag/faiss_index.py

Phase 4: JSON Output & Integration (Week 2)
  14. Define explanation JSON schema
  15. Integrate all components into pipeline
  16. Add performance profiling
  17. Optimize to <100ms overhead
  
  Files to create:
  - core/explainability/explanation_formatter.py
  - core/explainability/explainability_engine.py

Shall I proceed with Phase 1? (yes/no)
```

**Workflow with Sub-Agents:**
```bash
# Phase 1: Use explainability agent
claude --agent explainability "Implement Integrated Gradients in 
core/explainability/integrated_gradients.py following PyTorch best practices"

# Phase 2: Generate tests
claude --agent testing "Create comprehensive test suite for Integrated Gradients
with edge cases and performance benchmarks"

# Phase 3: Optimize
claude "Profile integrated_gradients.py and optimize to <100ms overhead"

# Phase 4: Document
claude "Add docstrings and usage examples with visualization code"
```

---

## 🗺️ Implementation Roadmap (Enhanced)

### Phase 0: Foundation (Weeks 1-2)
**Goal:** Setup + Enhanced data collection

**Tasks:**
- Environment setup
- AWS SageMaker configuration
- Neo4j/TigerGraph setup
- Enhanced data collection (CVEs + repository mining + synthetic)

📂 **Guide:** [01_setup.md](./01_setup.md)

---

### Phase 1: ML Training Pipeline (Weeks 3-5)
**Goal:** Fine-tune CodeBERT/CodeLLaMA with explainability

**Tasks:**
- Dataset preprocessing with counterfactual augmentation
- Model fine-tuning on SageMaker
- Integrated Gradients integration
- Model evaluation with explainability metrics

📂 **Guide:** [02_ml_training.md](./02_ml_training.md)

---

### Phase 2: Explainability System (Week 6)
**Goal:** Implement deep explainability

**Tasks:**
- Integrated Gradients implementation
- Counterfactual generation
- CVE retrieval with FAISS
- Explanation JSON formatter

📂 **Guide:** [03_explainability.md](./03_explainability.md)

---

### Phase 3: Local Agent Architecture (Weeks 7-8)
**Goal:** Platform-independent detection agent

**Tasks:**
- FastAPI REST/WebSocket server
- File system monitor (cross-platform)
- Incremental diff computation
- Agent deployment (Docker + native)

📂 **Guide:** [04_agent_architecture.md](./04_agent_architecture.md)

---

### Phase 4: Repository Graph System (Weeks 9-10)
**Goal:** Graph-based dependency tracking

**Tasks:**
- Neo4j/TigerGraph integration
- Dependency graph builder
- Vulnerability propagation tracker
- Attack surface analyzer

📂 **Guide:** [05_repository_graph.md](./05_repository_graph.md)

---

### Phase 5: UI & Feedback System (Weeks 11-12)
**Goal:** Dashboard + continuous learning

**Tasks:**
- React/Tauri dashboard
- Real-time visualization
- Feedback collection UI
- RLHF-lite implementation

📂 **Guide:** [06_ui_feedback.md](./06_ui_feedback.md)

---

### Phase 6: Verification & Patch Generation (Weeks 13-14)
**Goal:** Verified patch generation

**Tasks:**
- Symbolic execution integration (angr/KLEE)
- Fuzzing validation
- Patch verification pipeline
- Deployment and testing

📂 **Guide:** [07_verification_patch.md](./07_verification_patch.md)

---

### Phase 7: Integration & Launch (Weeks 15-16)
**Goal:** End-to-end integration and launch

**Tasks:**
- Full system integration testing
- Performance optimization
- Security audit
- Beta program (100+ users)
- Public launch

---

## 📚 Quick Reference

### Essential Commands

```bash
# Start local agent
streamguard-agent start --port 8765

# Use with Claude Code
claude --plan "Implement feature X"
claude --agent explainability "Add saliency visualization"
claude --agent graph-systems "Optimize Neo4j queries"
claude --agent dashboard "Create taint path visualization"

# Analyze code
curl -X POST http://localhost:8765/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "...", "language": "python"}'

# Submit feedback
curl -X POST http://localhost:8765/feedback \
  -d '{"suggestion_id": "vuln_001", "action": "accepted"}'

# Check agent status
curl http://localhost:8765/status
```

### Project Structure (Enhanced)

```
streamguard/
├── .claude/
│   └── agents/
│       ├── frontend.yml
│       ├── ml-training.yml
│       ├── testing.yml
│       ├── security.yml
│       ├── explainability.yml    # NEW
│       ├── graph-systems.yml     # NEW
│       └── dashboard.yml         # NEW
├── core/                         # Python backend
│   ├── agents/                   # Detection agents
│   ├── engine/
│   │   ├── orchestrator.py
│   │   └── local_agent.py        # NEW: FastAPI agent
│   ├── explainability/           # NEW
│   │   ├── integrated_gradients.py
│   │   ├── counterfactuals.py
│   │   └── explanation_formatter.py
│   ├── graph/                    # NEW
│   │   ├── neo4j_client.py
│   │   ├── graph_builder.py
│   │   └── propagation_tracker.py
│   ├── feedback/                 # NEW
│   │   ├── collector.py
│   │   ├── anonymizer.py
│   │   └── sync_manager.py
│   ├── verification/             # NEW
│   │   ├── symbolic_executor.py
│   │   ├── fuzzer.py
│   │   └── patch_verifier.py
│   └── rag/
│       └── cve_retriever.py
├── dashboard/                    # NEW: React/Tauri app
│   ├── src/
│   │   ├── components/
│   │   ├── views/
│   │   └── api/
│   ├── src-tauri/                # Rust backend
│   └── package.json
├── training/
│   ├── scripts/
│   │   ├── sagemaker/
│   │   └── retraining/           # NEW: Auto-retrain
│   └── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── feedback/                 # NEW: Feedback storage
├── docs/
│   ├── CLAUDE.md                 # This file
│   ├── 01_setup.md
│   ├── 02_ml_training.md
│   ├── 03_explainability.md      # NEW
│   ├── 04_agent_architecture.md  # NEW
│   ├── 05_repository_graph.md    # NEW
│   ├── 06_ui_feedback.md         # NEW
│   └── 07_verification_patch.md  # NEW
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

### Performance Targets (Updated)

| Component | Target | Maximum | Status |
|-----------|--------|---------|--------|
| Syntax Agent | 50ms | 100ms | Core |
| Semantic Agent | 200ms | 500ms | Core |
| Context Agent (Neo4j) | 500ms | 1000ms | Enhanced |
| Verification | 300ms | 800ms | Core |
| **Explainability Overhead** | **<100ms** | **200ms** | **NEW** |
| **Graph Query** | **<50ms** | **100ms** | **NEW** |
| **Patch Verification** | **<5s** | **10s** | **NEW** |
| **Total Pipeline** | **<1.5s** | **3s** | **Updated** |

### Key Features Summary

#### Core Features (Unchanged)
✅ Multi-agent detection pipeline  
✅ 95%+ accuracy, <3% false positives  
✅ Real-time analysis (<1s)  
✅ Privacy-first architecture  
✅ Taint-flow tracking  

#### Enhanced Features (v3.0)
🆕 Token-level explainability (Integrated Gradients)  
🆕 Counterfactual analysis  
🆕 Platform-independent local agent  
🆕 Graph-based repository context (Neo4j/TigerGraph)  
🆕 Continuous learning with RLHF-lite  
🆕 Interactive web dashboard (React/Tauri)  
🆕 Verified patch generation (symbolic execution + fuzzing)  
🆕 Compliance reporting (PDF/JSON/SARIF)  
🆕 Team pattern libraries  

---

## 🎓 Learning Resources

### Claude Code
- Official Docs: https://docs.claude.com/en/docs/claude-code
- Sub-agents: https://docs.claude.com/en/docs/claude-code/sub-agents
- Plan Mode: Use `--plan` flag for complex tasks

### ML & Explainability
- Integrated Gradients: https://arxiv.org/abs/1703.01365
- Counterfactual Explanations: https://arxiv.org/abs/1711.00399
- CodeBERT: https://arxiv.org/abs/2002.08155
- CodeLLaMA: https://arxiv.org/abs/2308.12950

### Graph Databases
- Neo4j: https://neo4j.com/docs/
- TigerGraph: https://docs.tigergraph.com/
- Cypher Query Language: https://neo4j.com/developer/cypher/

### Symbolic Execution & Fuzzing
- angr: https://docs.angr.io/
- KLEE: https://klee.github.io/
- AFL++: https://aflplus.plus/

### Security
- OWASP Top 10: https://owasp.org/www-project-top-ten/
- CWE: https://cwe.mitre.org/
- SARIF: https://sarifweb.azurewebsites.net/

---

## 🔄 Migration from v2.0 to v3.0

### For Users

**No Breaking Changes:**
- Existing detections still work
- VS Code extension can coexist with new agent
- Gradual migration path

**New Capabilities:**
```bash
# Install local agent
pip install streamguard-agent

# Start agent (replaces VS Code extension backend)
streamguard-agent start

# Connect any IDE via REST API
curl http://localhost:8765/analyze -d '{"code": "...", "language": "python"}'

# View enhanced UI
streamguard-dashboard
```

### For Developers

**Additional Dependencies:**
```bash
# Graph database
docker run -d -p 7687:7687 neo4j:latest

# Install new Python packages
pip install fastapi uvicorn neo4j faiss-cpu angr
```

**New API Endpoints:**
```python
# Explainability
POST /analyze  # Now includes explanation JSON

# Feedback
POST /feedback
GET /patterns

# Graph queries
GET /graph/dependencies?file=auth.py
GET /graph/propagation?vulnerability_id=vuln_001
```

---

## 🚀 Getting Started with Enhanced StreamGuard

### Quick Start (5 minutes)

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/streamguard.git
cd streamguard
./scripts/setup.sh

# 2. Start Neo4j (Docker)
docker run -d \
  --name streamguard-neo4j \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/streamguard \
  neo4j:latest

# 3. Start local agent
source venv/bin/activate
streamguard-agent start --port 8765

# 4. Open dashboard
streamguard-dashboard

# 5. Test with sample code
curl -X POST http://localhost:8765/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "query = \"SELECT * FROM users WHERE id=\" + user_id",
    "language": "python",
    "filePath": "test.py"
  }'
```

### Using with Claude Code

```bash
# Start a new feature
claude --plan "Add XSS detection with explainability support"

# Use specialized agents
claude --agent explainability "Implement IG for XSS detection"
claude --agent graph-systems "Track XSS propagation in Neo4j"
claude --agent dashboard "Add XSS visualization to UI"

# Test end-to-end
claude --agent testing "Create E2E tests for XSS detection with explanation verification"
```

---

## 📊 Success Metrics (Enhanced)

### Technical Metrics
```yaml
Detection Accuracy: ≥95%
False Positive Rate: <3%
Latency (P95): <1.5s (with explainability)
Explainability Score: >85% (developer understanding)
Patch Verification Success: >90%
Graph Query Performance: <50ms
Feedback Collection Rate: >60%
Model Improvement Rate: >5% accuracy gain per retrain cycle
```

### User Metrics
```yaml
Developer Understanding: >85% (via surveys)
Fix Acceptance Rate: >80% (verified patches)
Dashboard Engagement: >70% weekly active
Feedback Quality: >90% actionable
Time to Fix: <10 minutes (vs >2 hours manual)
```

### Business Metrics
```yaml
Installations: 10,000+ (Month 6)
Enterprise Pilots: 10+ (Month 9)
NPS: ≥60
Developer Satisfaction: >4.5/5
Cost Savings: $50K+ per enterprise per year
```

---

## 🔐 Privacy & Security (Enhanced)

### Data Handling

**What We Collect (Opt-in Only):**
```python
{
  "feedback_id": "uuid",
  "action": "accepted",  # or "rejected"
  "vulnerability_type": "sql_injection",
  "code_hash": "sha256_hash",  # One-way hash
  "timestamp_bucket": "2024-10-08T00:00:00Z",  # 24-hour bucket
  "explanation_useful": true,
  "patch_applied": true
}
```

**What We NEVER Collect:**
❌ Source code or code snippets  
❌ File paths or directory structures  
❌ Developer names or identifiers  
❌ Company names or IP addresses  
❌ Variable or function names  

### Security Measures

**Local Agent:**
```python
# Only listens on localhost
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*"],
    allow_methods=["POST", "GET"],
)

# Rate limiting
@app.middleware("http")
async def rate_limit(request: Request, call_next):
    # Max 100 requests per minute per client
    pass
```

**Graph Database:**
```python
# No sensitive data in graph
# Only metadata: file names, function signatures, relationships
# Code content NOT stored in Neo4j
```

**Feedback Sync:**
```python
# Encrypted in transit
requests.post(
    sagemaker_endpoint,
    json=anonymized_feedback,
    headers={"Authorization": f"Bearer {encrypted_token}"},
    verify=True  # SSL verification
)
```

---

## 🎯 Roadmap Beyond v3.0

### Phase 8: Multi-Language Support (Q1 2025)
- Java/C++ detection agents
- Go/Rust support
- Language-agnostic patterns

### Phase 9: Cloud Deployment Option (Q2 2025)
- Enterprise cloud deployment
- Team dashboards
- Centralized pattern management
- SSO integration

### Phase 10: Advanced Verification (Q3 2025)
- Formal verification (Z3 solver)
- Runtime monitoring
- Exploit generation for penetration testing

---

## 💡 Tips for Development

### Using Claude Code Effectively

**1. Break Down Complex Features:**
```bash
# Instead of:
claude "Implement entire explainability system"  # Too broad

# Do this:
claude --plan "Implement explainability system"
# Then execute phase by phase
claude --agent explainability "Implement Integrated Gradients (Phase 1)"
```

**2. Leverage Sub-Agents:**
```bash
# Use the right specialist for the job
claude --agent explainability "..."  # ML explainability
claude --agent graph-systems "..."   # Neo4j/graph work
claude --agent dashboard "..."       # UI/visualization
```

**3. Reference Documentation:**
```bash
# Provide context
claude "Following docs/03_explainability.md, implement counterfactual generator"
```

**4. Iterate with Feedback:**
```bash
# Initial implementation
claude --agent explainability "Implement IG"

# Review and improve
claude "The IG implementation is slow (500ms). Optimize to <100ms by:
- Batching gradient computations
- Reducing interpolation steps
- Caching baselines"

# Verify
python tests/benchmarks/benchmark_explainability.py
```

### Performance Optimization Tips

**Graph Queries:**
```cypher
-- ❌ Slow: Unbounded traversal
MATCH (n)-[*]->(m) RETURN n, m

-- ✅ Fast: Limited depth with index
MATCH (n:Function {name: $name})-[*1..5]->(m:Function)
RETURN n, m
```

**Explainability:**
```python
# ❌ Slow: Computing IG for all tokens
for token in all_tokens:
    saliency = compute_ig(model, token)

# ✅ Fast: Batch computation
saliencies = compute_ig_batch(model, all_tokens)
```

---

## 📞 Support & Community

- 📧 Email: support@streamguard.dev
- 💬 Discord: [StreamGuard Community](https://discord.gg/streamguard)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/streamguard/issues)
- 📖 Docs: [Documentation](./docs/)
- 🎥 Tutorials: [YouTube Channel](https://youtube.com/streamguard)

---

## 🙏 Acknowledgments

**Core Technologies:**
- Tree-sitter, PyTorch, Hugging Face, VS Code
- Neo4j/TigerGraph for graph databases
- FastAPI for local agent
- React + Tauri for dashboard
- angr/KLEE for symbolic execution

**Explainability Research:**
- Integrated Gradients (Sundararajan et al.)
- Counterfactual Explanations (Wachter et al.)

**Built with Claude Code and ❤️**

---

**This document is the single source of truth for StreamGuard Enhanced development.**

**Version:** 3.0  
**Last Updated:** October 2024  
**Status:** ✅ Ready for Implementation

**Ready to build the next generation of vulnerability prevention? Start with:**
```bash
claude --plan "Set up StreamGuard v3.0 development environment with Neo4j and local agent"
```