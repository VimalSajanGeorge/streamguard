# docs/NOVELTY.md — Research Novelties vs Existing Work

> Read this to understand WHY each architectural decision exists.
> Every claim here is backed by a specific published paper.

---

## What Exists (The Baseline Landscape as of 2025)

### Devign (NeurIPS 2019)
- Standard GGNN on composite code graph (AST+CFG+DFG)
- Word2Vec node embeddings
- Single binary classification head
- **Limitation**: Static embeddings miss contextual semantics; no CFA

### REVEAL (TSE 2022)
- Improved graph construction for vulnerability detection
- BERT-style token embeddings
- **Limitation**: No paired contrastive training; high FPR on imbalanced data

### LineVul (MSR 2022)
- Transformer (CodeBERT) sequence model for line-level detection
- **Limitation**: No structural graph reasoning; misses control/data flow

### Vul-LMGNN (arXiv April 2024) ← CLOSEST BASELINE
- First to jointly train CodeBERT + GGNN
- CodeBERT initialises GGNN node embeddings
- Implicit-explicit fusion (CodeBERT as auxiliary classifier + GGNN)
- **Achieves**: ~10% F1 improvement over 17 baselines on DiverseVul
- **Limitation**: No CFA training; 3-component CPG only; no inter-procedural; no taint graph

### VISION (AIES 2025) ← PRIMARY BASELINE FOR CFA
- LLM-generated counterfactual pairs for GNN training
- Proves CFA eliminates spurious correlations
- Achieves 97.8% accuracy on CWE-20 (from 51.8%)
- **Critical limitation**: CWE-20 ONLY; Word2Vec encoder; no inter-procedural; no TPG; research prototype

### CFExplainer (ISSTA 2024)
- Counterfactual explanations for GNN vulnerability detection
- Minimal graph perturbation that flips prediction
- **We integrate**: CFExplainer-style explanation in serving layer

### AUG-PDG (Scientific Reports 2025)
- Augmented Program Dependency Graph + optimised CodeBERT
- Context slicing: 2-hop BFS from vulnerable statements
- Outperforms GCN/BGRU/GAT baselines
- **We take**: Context slicing strategy; augmented PDG structure

### VulnSC / Inter-procedural (ICSE 2024)
- First to address inter-procedural vulnerabilities in LLM-based detection
- LLM callee summaries injected into focal function analysis
- **We integrate**: Callee summary injection as CPG node features

---

## StreamGuard Novel Contributions

### N1: Multi-CWE CFA Generalisation (PRIMARY CLAIM)

**What VISION did**: Proved CFA works for CWE-20 (Improper Input Validation).
Released CWE-20-CFA benchmark (27,556 functions).
Future work explicitly stated: "evaluate VISION across a wider range of CWEs."

**What StreamGuard adds**:
- CFA evaluated across **12 CWE types** (first multi-CWE CFA study)
- Per-CWE specialised prompts (generic prompts fail: buffer overflow fix ≠ SQL injection fix)
- Per-CWE validation heuristics in the validation gate
- Releases **StreamGuard-12CWE-CFA**: largest multi-CWE CFA benchmark

**Expected finding**: CFA gain is CWE-dependent. Higher for injection CWEs (CWE-89/78: simpler structural fix), lower for lifetime/ownership CWEs (CWE-416/UAF: harder to minimally fix). This heterogeneity IS the research contribution.

**Ablation**: Train VISION (CFA) on each CWE independently. Report CFA-gain per CWE.

---

### N2: CodeBERT + CFA vs Word2Vec + CFA

**What VISION did**: Used Word2Vec (static, context-free) embeddings for CPG node initialisation.

**What StreamGuard adds**:
- CodeBERT (contextual, pre-trained on 6M code files) for node embeddings
- Cross-attention fusion: BERT sequence attends to graph structure
- **Research question**: Does pre-trained LM amplify CFA benefit vs static Word2Vec?

**Ablation**: 
- Config A: Word2Vec + Devign + CFA (reproduce VISION)
- Config B: CodeBERT + GGNN + no CFA (reproduce Vul-LMGNN)
- Config C: CodeBERT + GGNN + CFA (StreamGuard, novel combination)
- Hypothesis: Config C > Config A + Config B independently

---

### N3: Taint Propagation Graph as 4th CPG Component

**What exists**: AUG-PDG uses AST + CFG + PDG (3 components).
Vul-LMGNN uses AST + CFG + Program Dependence Graph (3 components).
No prior work adds an explicit Taint Propagation Graph as a CPG component.

**What StreamGuard adds**:
- **TPG**: source nodes → propagation nodes → sink nodes → sanitizer nodes
- Explicit TAINT edges in addition to AST/CFG/DFG edges
- GNN can directly message-pass along taint paths
- Critical for injection CWEs where vulnerability IS the taint path

**Ablation**: 3-CPG (AST+CFG+DFG) vs 4-CPG (+TPG)
**Hypothesis**: TPG adds 3–8% F1 on CWE-89/78/79; minimal impact on CWE-416/476

**Implementation detail**: 
- Joern generates AST+CFG+DFG natively
- TPG is added as Python post-processing using DFG paths between taint-labelled nodes
- Total edge types in GGNN: 4 {AST=0, CFG=1, DFG=2, TPG=3}
- CDG edges are filtered out in `stage4_cpg.py` before graph construction
- Each edge type has dedicated GatedGraphConv weights (Option B per-edge-type architecture)

---

### N4: CFA Structural Validation Gate

**What VISION did**: LLM output accepted if it "looks like" a valid fix.
No structural verification. CFA quality measured post-hoc by dataset statistics.

**What StreamGuard adds**:
- Gate 1: `gcc -fsyntax-only` (syntax validity)
- Gate 2: Similarity bounds [0.60, 0.99] (not identical, not completely different)
- Gate 3: Vulnerability pattern heuristic (CWE-specific regex check)
- Gate 4: Joern CPG diff (optional but recommended): changed nodes ≤ 10

**Research question**: Does structural CFA validation improve:
- Pairwise contrast accuracy (both v and v' correctly classified)?
- CFA quality metrics (changed lines distribution, taint elimination rate)?

**Metrics to report**:
```
cfa_compile_rate         # % of LLM outputs that compile
cfa_pattern_removal_rate # % where vulnerability heuristic pattern is gone
cfa_similarity_dist      # distribution of similarity scores 0.60–0.99
cfa_avg_changed_lines    # should be 3–8 lines (minimal edit)
```

---

### N5: Inter-Procedural Context + CFA (COMBINED NOVELTY)

**What exists**: VulnSC (ICSE 2024) showed inter-procedural analysis improves detection.
**What exists**: VISION showed CFA improves detection within function boundaries.
**What doesn't exist**: Combining inter-procedural context WITH CFA training.

**What StreamGuard adds**:
- LLM callee summaries injected as auxiliary node features on call-site nodes in CPG
- CFA generation preserves call-site context (callee invocations unchanged)
- GNN sees: "this function calls X which does NOT sanitize SQL" (inter-proc signal)

**Test set requirement**: Include CVEs that span 2+ functions. Measure detection rate:
- With inter-proc context vs without inter-proc context on this subset

---

### N6: Continuous CFA from Production Feedback (DEPLOYMENT NOVELTY)

**What all prior work does**: Train once, deploy statically. CFA dataset is fixed.

**What StreamGuard adds**:
- Human-corrected FP/FN → new labeled samples
- CFA generator creates counterfactuals from human corrections automatically
- Weekly fine-tuning on new samples + new CFA pairs
- A/B tested deployment: shadow mode → gradual rollout

**This is the first deployment of a continuously evolving CFA-augmented vulnerability detector.**

**Metrics to report** (longitudinal, 30-day post-deployment):
```
fpr_week_0   # baseline FPR at deployment
fpr_week_4   # FPR after 4 weeks of feedback loop
fnr_week_0   # baseline FNR
fnr_week_4   # FNR after 4 weeks
new_cfa_pairs_generated  # total auto-generated CFA pairs from feedback
```

---

## What StreamGuard Explicitly Does NOT Claim

1. **Not better than all LLMs at security analysis**: GPT-4/Claude can do arbitrary code review. StreamGuard is specialised, deterministic, auditable.
2. **Not production-hardened security tooling**: It is a research system with a production-grade API. Not a certified SAST tool.
3. **Not binary analysis**: C source code only. No compiled binary analysis.
4. **Not a complete replacement for CodeQL**: CodeQL has deterministic taint analysis with user-defined rules. StreamGuard has learned taint patterns. Different tradeoffs.

---

## How to Write the Paper's Related Work Section

```
Paragraph 1: Sequence-based approaches (VulDeePecker, SySeVR, LineVul)
  → Limitation: no structural graph reasoning

Paragraph 2: Graph-based approaches (Devign, REVEAL, Vul-LMGNN)
  → Limitation: no counterfactual augmentation; static embeddings (except Vul-LMGNN)

Paragraph 3: Data augmentation approaches for vulnerability detection
  → ICSE 2025 (Ding et al.): data quality issues in vulnerability datasets
  → VISION (AIES 2025): CFA for CWE-20. Our work generalises this.

Paragraph 4: Explainability in GNN-based detection
  → CFExplainer (ISSTA 2024): counterfactual explanation
  → We integrate CFExplainer-style explanation in inference

Paragraph 5: Inter-procedural analysis
  → VulnSC (ICSE 2024): callee summaries
  → We add CFA training to inter-procedural context (first combination)

Gap statement: "No existing work combines multi-CWE CFA with CodeBERT encoder,
4-component CPG including TPG, and inter-procedural context into a single framework."
```

---

*docs/NOVELTY.md | StreamGuard v1.0 | March 2026*
