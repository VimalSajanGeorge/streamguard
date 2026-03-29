# StreamGuard — Technical Product Requirements Document (PRD)

**Version**: 1.0  
**Date**: March 2026  
**Status**: Implementation Ready

---

## 1. Product Vision

StreamGuard is a **research-grade + production-deployable** C vulnerability detection system. It is the first system to combine LLM-generated counterfactual data augmentation (VISION, AIES 2025) across 12 CWE types with a CodeBERT + GGNN joint encoder (Vul-LMGNN, 2024), a 4-component Code Property Graph including an explicit Taint Propagation Graph, and inter-procedural callee context injection.

**Primary goal**: Produce a conference-quality research paper (target: ISSTA / ICSE) AND a deployable system.  
**Secondary goal**: Release the largest multi-CWE counterfactual C vulnerability benchmark (40K+ samples, 12 CWE types).

---

## 2. Research Novelties (What This System Contributes)

### N1: Multi-CWE CFA Generalisation
VISION (AIES 2025) proved counterfactual augmentation eliminates spurious correlations for CWE-20 only. StreamGuard is the **first system to evaluate CFA across 12 CWE types** with per-CWE specialised generation prompts and Joern CPG-diff structural validation.

**Expected finding**: CFA gain varies by CWE type (high gain for injection CWEs, moderate for memory-safety CWEs). This variance IS the research contribution.

### N2: CodeBERT Encoder with CFA (vs Word2Vec in VISION)
VISION uses Word2Vec (static, context-free) embeddings. StreamGuard uses CodeBERT, which provides contextual, pre-trained representations. **Research question**: Does a pre-trained LM amplify the CFA benefit beyond what Word2Vec achieves?

### N3: Taint Propagation Graph as 4th CPG Component
AUG-PDG and related work use AST+CFG+DFG (3 components). StreamGuard adds a **Taint Propagation Graph** (TPG) as a 4th component: explicit source→propagation→sink→sanitizer edges derived from data flow. This gives the GNN direct structural signal for injection vulnerability classes.

**Ablation**: 3-CPG vs 4-CPG (with TPG). Expected: TPG improves CWE-89/78/79 F1 by 3–8%.

### N4: CFA Structural Validation Gate
VISION accepts LLM-generated counterfactuals without structural verification. StreamGuard validates every CFA via: (a) `gcc -fsyntax-only`, (b) Joern CPG diff (changed nodes ≤ 10), (c) similarity bounds [0.60, 0.99], (d) taint path elimination check.

**Research question**: Does structural CFA validation improve pairwise contrast accuracy beyond unvalidated LLM output?

### N5: Inter-Procedural Context + CFA
No prior work combines VulnSC-style inter-procedural callee summarisation with CFA training. StreamGuard injects LLM callee summaries as auxiliary node features on call-site nodes in the CPG, enabling detection of vulnerabilities that span multiple functions.

### N6: Continuous CFA from Production Feedback
All prior CFA systems are static post-training. StreamGuard implements a feedback loop: human-corrected false positives/negatives generate new CFA pairs, triggering weekly fine-tuning. This is the first deployment of a continuously evolving CFA-augmented vulnerability detector.

---

## 3. Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-01 | Detect 12 CWE types in C functions | P0 | Per-CWE F1 ≥ 0.88 on held-out test set |
| FR-02 | Binary vuln/safe classification per function | P0 | Overall F1 ≥ 0.92 |
| FR-03 | Multi-class CWE type prediction | P0 | Top-1 accuracy ≥ 0.82 |
| FR-04 | CVSS-proxy severity score [0–10] | P1 | MAE ≤ 1.8 |
| FR-05 | Taint path in prediction output | P1 | Present in ≥ 90% of TP predictions |
| FR-06 | Counterfactual explanation ("what would fix this") | P1 | Human-readable in ≥ 90% of TP |
| FR-07 | Inter-procedural analysis (callee context) | P1 | Detects ≥ 70% of test set inter-func CVEs |
| FR-08 | REST API: POST /v1/scan/function | P0 | Returns prediction JSON in < 300ms P95 |
| FR-09 | REST API: POST /v1/scan/file (full C file) | P0 | Scans 1K-function file in < 60s |
| FR-10 | Batch scan: POST /v1/scan/batch (ZIP) | P1 | Async job, ≥ 500 functions/minute |
| FR-11 | Feedback endpoint: POST /v1/feedback | P1 | Corrected label stored, triggers CFA queue |
| FR-12 | Human review queue (Label Studio) | P2 | Low-confidence predictions routed to queue |
| FR-13 | CFA generation pipeline (5-tier) | P0 | Per-CWE compile rates: Tier 1 CWEs ≥ 95%; Tier 2 CWEs ≥ 83%; Tier 3 CWEs ≥ 75%; Tier 4 CWEs ≥ 65%. Overall compile rate ≥ 80%. Pattern removal rate ≥ 70%. Fix signature rate ≥ 70% (Gate 5). Taint break rate ≥ 70% (Gate 6, injection CWEs). Results verified by `cfa_quality_report.json`. |
| FR-14 | 7-stage preprocessing pipeline | P0 | All stages complete without data loss |
| FR-15 | Dataset release (HuggingFace) | P1 | 30K+ samples, 12 CWE types, with graphs |

---

## 4. Non-Functional Requirements

| ID | Category | Requirement | Metric |
|----|----------|-------------|--------|
| NFR-01 | Accuracy | Binary F1 ≥ 0.92 | F1 = 2×(P×R)/(P+R) on test set |
| NFR-02 | Accuracy | FPR ≤ 5% | FP/(FP+TN) |
| NFR-03 | Accuracy | FNR ≤ 8% | FN/(FN+TP) |
| NFR-04 | Accuracy | CFA pairwise accuracy ≥ 88% | Both vuln and CFA correctly classified |
| NFR-05 | Accuracy | Worst-group accuracy ≥ 80% | Lowest-performing CWE subgroup |
| NFR-06 | Latency | P95 single-function scan ≤ 300ms | Measured at API gateway |
| NFR-07 | Throughput | ≥ 500 functions/minute batch | Load tested with 10K function corpus |
| NFR-08 | Reliability | 99.5% API uptime | PagerDuty / health check monitoring |
| NFR-09 | Interpretability | Every prediction has taint path + node importance | Illuminati explainer integration |
| NFR-10 | Data quality | ≥ 30K deduplicated samples before training | pre_training_audit.py must PASS |
| NFR-11 | Reproducibility | All experiments reproducible with fixed seed | seed=42 globally; logged in MLflow |
| NFR-12 | Security | API key auth + rate limiting | OWASP API Security Top 10 |

---

## 5. System Architecture Summary

```
┌──────────────────────────────────────────────────────────────────────┐
│  DATA PLANE                                                          │
│  7 Sources → 6 Collectors → Canonical JSONL Store → CFA Generator   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│  PREPROCESSING PLANE (7 stages)                                      │
│  Clean → Dedup → CFA Gen → CPG Build → Embed → Graph → Split        │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│  MODEL PLANE                                                         │
│  CodeBERT (768-d) + 3-layer GGNN (256-d) → Cross-Attn Fusion →      │
│  Binary Head + CWE Head + Severity Head                              │
│  Loss: L_CE + 0.5*L_CFA_contrastive + 0.1*L_severity               │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│  SERVING PLANE                                                       │
│  FastAPI → Inference Worker (GPU) → Prediction JSON                  │
│  (includes taint_path + counterfactual_hint + node importance)       │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│  FEEDBACK PLANE                                                      │
│  Human corrections → New CFA pairs → Weekly fine-tuning             │
└──────────────────────────────────────────────────────────────────────┘
```

See `docs/ARCHITECTURE.md` for full system diagram and all sub-component specifications.

---

## 6. Data Requirements

### 6.1 Collection Targets

| Source | Type | Target Samples | Notes |
|--------|------|----------------|-------|
| SARD (NIST) | Pre-labeled C pairs | 8,000–12,000 | Process first; no API |
| ExploitDB | Exploit C + mutation | 800–1,200 | Local git clone |
| CVE/NVD (2-phase) | Before/after commits | 3,000–5,000 pairs | Phase 1: index; Phase 2: diffs |
| GitHub Advisory | Before/after commits | 2,000–3,500 pairs | GraphQL API |
| OSV | Before/after diffs | 1,000–1,800 pairs | C ecosystems only |
| Repo Miner | Security commits | 18,000–28,000 pairs | 15 repos, API only |
| Manual (Label Studio) | Expert labeled | ≥ 800 | Quality anchor |
| **Total** | | **33,600–52,300** | Target: ≥ 30K for training |

### 6.2 CFA Augmentation Targets

| CWE | Strategy Tier | Real Samples (min) | CFA Target | CFA Ratio | Min Compile Rate |
|-----|---------------|--------------------|------------|-----------|-----------------|
| CWE-120 | Tier 1 (AST) | 3,000 | 9,000 | 3:1 | ≥ 95% |
| CWE-89 | Tier 3 (CoT) | 2,000 | 6,000 | 3:1 | ≥ 75% |
| CWE-416 | Tier 4 (Few-shot) | 1,500 | 3,000 | 2:1 | ≥ 65% |
| CWE-476 | Tier 1 (AST) | 1,500 | 4,500 | 3:1 | ≥ 90% |
| CWE-78 | Tier 3 (CoT) | 1,200 | 3,600 | 3:1 | ≥ 75% |
| CWE-122 | Tier 2 (Zero-shot) | 1,200 | 3,600 | 3:1 | ≥ 83% |
| CWE-134 | Tier 1 (AST) | 1,000 | 3,000 | 3:1 | ≥ 95% |
| CWE-121 | Tier 2 (Zero-shot) | 1,000 | 3,000 | 3:1 | ≥ 83% |
| CWE-190 | Tier 3 (CoT) | 800 | 2,400 | 3:1 | ≥ 78% |
| CWE-125 | Tier 2 (Zero-shot) | 800 | 2,400 | 3:1 | ≥ 80% |
| CWE-119 | Tier 4 (Few-shot) | 800 | 1,600 | 2:1 | ≥ 65% |
| CWE-79 | Tier 3 (CoT) | 500 | 1,000 | 2:1 | ≥ 68% |

> **Note on CWE-416 and CWE-119:** Tier 4 (few-shot) generates fewer pairs per sample than
> Tier 1-3 due to higher failure rate. Accept 2:1 ratio (not 3:1) for these CWEs.
> Supplement with SARD native pairs where available.

### 6.3 Dataset Quality Gates (pre_training_audit.py)

All 9 checks must PASS before training begins:

```python
# pre_training_audit.py — All 21 checks must PASS before training begins.
# Checks 1-9: dataset structure (unchanged from original spec)
# Checks 10-21: per-CWE CFA quality (NEW — reads cfa_quality_report.json)

REQUIRED_CHECKS = {
    # ── DATASET STRUCTURE CHECKS (1-9) ──────────────────────────────────────
    "min_train_samples":     30_000,       # absolute minimum unique samples
    "vuln_safe_balance":     (0.25, 0.75), # NOTE: relaxed from (0.45,0.55) because
                                            # CFA augmentation adds label=0 pairs,
                                            # shifting balance toward 25-30% vuln
    "cwe_diversity":         7,             # minimum CWE types with ≥ 500 samples
    "max_cwe_dominance":     0.45,          # relaxed for M1 — CWE-121/122 dominate Juliet
    "no_null_code":          0,             # zero samples with empty code field
    "test_train_no_overlap": 0,             # zero commit SHA shared between train/test
    "code_length_range":     (10, 4096),    # token count range (CodeBERT max)
    "pair_integrity":        0,             # zero broken CFA pairs
    "manual_verified_min":   800,           # minimum manually verified samples

    # ── PER-CWE CFA QUALITY CHECKS (10-21) ──────────────────────────────────
    # Read from: data/processed/cfa_quality_report.json
    # Generated by: stage3_cfa.py on completion
    # If cfa_quality_report.json does not exist: WARN but do not FAIL
    "cfa_quality_CWE_134": {"compile_rate": 0.95, "fix_signature_rate": 0.90},
    "cfa_quality_CWE_120": {"compile_rate": 0.90, "fix_signature_rate": 0.88},
    "cfa_quality_CWE_476": {"compile_rate": 0.88, "fix_signature_rate": 0.80},
    "cfa_quality_CWE_121": {"compile_rate": 0.83, "fix_signature_rate": 0.75},
    "cfa_quality_CWE_122": {"compile_rate": 0.80, "fix_signature_rate": 0.72},
    "cfa_quality_CWE_125": {"compile_rate": 0.78, "fix_signature_rate": 0.70},
    "cfa_quality_CWE_89":  {"compile_rate": 0.78, "taint_break_rate":   0.72},
    "cfa_quality_CWE_78":  {"compile_rate": 0.75, "taint_break_rate":   0.70},
    "cfa_quality_CWE_190": {"compile_rate": 0.80, "fix_signature_rate": 0.70},
    "cfa_quality_CWE_79":  {"compile_rate": 0.68, "fix_signature_rate": 0.60},
    "cfa_quality_CWE_119": {"compile_rate": 0.65, "fix_signature_rate": 0.58},
    "cfa_quality_CWE_416": {"compile_rate": 0.60, "fix_signature_rate": 0.52},
}

# M1 relaxed thresholds (use --m1 flag when Phase 2 collection is incomplete)
REQUIRED_CHECKS_M1 = {
    **{k: v for k, v in REQUIRED_CHECKS.items() if not k.startswith("cfa_quality")},
    "min_train_samples":  5_000,
    "vuln_safe_balance":  (0.25, 0.75),
    "cwe_diversity":      5,
    "min_taint_coverage": 0.20,
    # CFA quality checks: same thresholds as M2 when cfa_quality_report.json exists
    # If report does not exist yet (Stage 3 not run): skip CFA checks silently
}
```

---

## 7. Model Specification

### 7.1 Architecture

```
Input: C function (source code + 4-component CPG)
         │
    ┌────┴────────────────────────────────────────┐
    │          DUAL ENCODER                        │
    │                                              │
    │  CodeBERT encoder          GGNN encoder      │
    │  (microsoft/codebert-base) (3 layers, 256-d) │
    │  512-token BPE sequence    4-type CPG input  │
    │  Output: [CLS] 768-d       Output: 256-d     │
    └────┬──────────────────────────────┬──────────┘
         │                              │
    ┌────▼──────────────────────────────▼──────────┐
    │        CROSS-ATTENTION FUSION                 │
    │  Q = BERT [CLS] (B, 768→256)                 │
    │  K/V = GGNN per-node embeddings (N, 256)      │
    │  Attn via scatter softmax (node-level)        │
    │  Output: attended graph context (B, 256)       │
    │  Fused = concat(BERT_768, Attn_256,            │
    │                 GGNN_mean_256) = 1280-d        │
    │  MLP: 1280 → 512 → 128                       │
    └──────────────────────┬───────────────────────┘
                           │
              ┌────────────┼──────────────┐
              ▼            ▼              ▼
        Binary Head   CWE Head    Severity Head
        (BCE loss)    (CE 12-cls)  (Huber loss)
```

### 7.2 Loss Function

```
L_total = 1.0 * L_CE                          # binary cross-entropy on all samples
        + 0.5 * L_CFA_contrastive             # cosine margin loss on (v, v') pairs
        + 0.1 * L_severity                    # auxiliary severity regression

L_CFA = mean(relu(cosine_sim(emb_v, emb_v') + margin))  where margin=0.5
        # equivalent to: relu(cosine_sim - (-0.5))
        # Goal: force cosine_sim < -0.5 (opposite sides of embedding space)
        # NOTE: the sign matters — relu(sim + 0.5) penalizes when sim > -0.5
        #   (strong: forces to opposite hemispheres)
        # NOT relu(sim - 0.5) which only penalizes when sim > +0.5
        #   (weak: just prevents identical embeddings)
        for all (vuln, counterfactual) pairs in batch
```

### 7.3 Training Configuration

```python
TRAINING_CONFIG = {
    "base_model":           "microsoft/codebert-base",
    "optimizer":            "AdamW",
    "weight_decay":         0.01,
    "lr_codebert":          2e-5,       # conservative for pre-trained model
    "lr_ggnn_fusion":       1e-4,       # faster for randomly-init components
    "warmup_ratio":         0.1,        # 10% of steps for linear warmup
    "scheduler":            "cosine_annealing",
    "batch_size_graphs":    8,          # GPU memory limited
    "gradient_accumulation": 4,         # effective batch = 32
    "max_seq_len":          512,        # CodeBERT maximum
    "max_cpg_nodes":        200,        # after context slicing
    "epochs":               20,
    "early_stopping_patience": 5,       # on val F1
    "seed":                 42,
}
```

### 7.4 Node Feature Vector (824-d)

```
[0:768]   — CodeBERT [CLS] embedding of node's code statement (768-d)
[768:800] — Node type one-hot encoding (32 Joern node types)
[800:808] — Taint role: source/sink/sanitizer/propagation/none + 3 reserved (8-d)
[808:812] — CPG component: AST=0, CFG=1, DFG=2, TPG=3 (4-d)
[812:824] — Structural: in_degree, out_degree, AST_depth, taint_dist_to_sink × 3 (12-d)
```

---

## 8. API Specification

### 8.1 Core Endpoints

```
POST /v1/scan/function     → Prediction JSON (< 300ms P95)
POST /v1/scan/file         → [Prediction JSON]
POST /v1/scan/batch        → {job_id: str}  (async)
GET  /v1/scan/{job_id}     → {status, results}
POST /v1/feedback          → {acknowledged: true}
GET  /v1/models            → [{version, metrics, deployed_at}]
GET  /v1/health            → {status, model_version, uptime}
```

### 8.2 Prediction JSON Schema

```json
{
  "function_name":      "process_input",
  "file":               "src/network/handler.c:42",
  "is_vulnerable":      true,
  "confidence":         0.94,
  "cwe":                "CWE-89",
  "cwe_name":           "SQL Injection",
  "severity_score":     8.1,
  "severity_label":     "HIGH",
  "taint_path": [
    {"node": "argv[1]",       "role": "SOURCE",      "line": 44},
    {"node": "query_buf",     "role": "PROPAGATION", "line": 48},
    {"node": "mysql_query()", "role": "SINK",        "line": 51}
  ],
  "key_nodes": [
    {"node": "sprintf(query,...)", "importance": 0.87, "line": 48}
  ],
  "counterfactual_hint": {
    "description":      "Replace sprintf with parameterized query binding",
    "fix_pattern":      "CWE-89-parameterized-query",
    "minimal_fix_lines": [48, 51]
  },
  "model_version":      "streamguard-v1.0.0",
  "inference_ms":       198,
  "scan_id":            "sg-2026-abc123"
}
```

---

## 9. Evaluation Design

### 9.1 Ablation Study (6 configurations)

| Config | Description | Expected F1 |
|--------|-------------|-------------|
| A: Baseline | CodeBERT sequence only (no graph) | ~0.79 |
| B: +GGNN | CodeBERT + type-blind GGNN, 3-CPG, no CFA | ~0.83 |
| B': +Type-Aware | CodeBERT + type-aware GGNN (per-edge-type), 3-CPG, no CFA | ~0.85 |
| C: +CFA | Config B' + CFA contrastive training | ~0.89 |
| D: +TPG | Config C + Taint Propagation Graph (4-CPG) | ~0.91 |
| E: Full | Config D + inter-procedural context | ~0.93 |

### 9.2 Baseline Comparisons

- **Devign** (Zhou et al., NeurIPS 2019) — standard GGNN baseline
- **REVEAL** (Chakraborty et al., TSE 2022) — graph-based
- **LineVul** (Fu & Tantithamthavorn, MSR 2022) — token-level BERT
- **VISION** (Egea et al., AIES 2025) — CFA baseline (CWE-20 only)

### 9.3 Novel Metrics (from VISION paper)

```python
# In addition to F1, precision, recall, report:
pairwise_contrast_accuracy  # P(correct_v) AND P(correct_v') for CFA pairs
worst_group_accuracy        # F1 of lowest-performing CWE subgroup
intra_class_attribution_variance   # consistency of node importance within class
inter_class_attribution_distance   # separation between vuln/safe node importance
```

---

## 10. File Naming Conventions

```
training/scripts/collection/
    schema.py                      # canonical sample schema + validate_sample()
    base_collector.py              # BaseCollector with checkpoint/retry/dedup
    process_sard.py                # SARD collector
    exploitdb_collector.py         # ExploitDB collector
    cve_collector_enhanced.py      # CVE/NVD two-phase collector
    github_advisory_collector.py   # GitHub Advisory GraphQL collector
    osv_collector.py               # OSV collector
    repo_miner_enhanced.py         # Repository miner (API-only, no git clone)

training/scripts/preprocessing/
    stage1_clean.py                # function extraction + normalization
    stage2_dedup.py                # 4-level deduplication
    stage3_cfa.py                  # CFA generation + validation
    stage4_cpg.py                  # Joern CPG construction
    stage5_embed.py                # CodeBERT node embedding
    stage6_graphs.py               # PyG Data + HDF5 cache
    stage7_split.py                # CFA-aware train/val/test split
    validate_graphs.py             # pre-training graph validation
    pre_training_audit.py          # 9 quality gates before training

training/scripts/model/
    model.py                       # StreamGuardModel: CodeBERT+GGNN+fusion+heads
    cfa_dataloader.py              # CFA-aware DataLoader keeping pairs in same batch
    losses.py                      # L_CE, L_CFA, L_severity, L_total
    train.py                       # training loop with MLflow logging
    eval.py                        # evaluation: F1, CFA pairwise, worst-group
    run_ablations.py               # automated 6-config ablation runner

training/scripts/serving/
    api.py                         # FastAPI application
    inference_worker.py            # GPU inference worker
    prediction_schema.py           # Pydantic prediction output models
    callee_cache.py                # Redis-backed callee summary cache
```

---

*StreamGuard Technical PRD v1.0 | March 2026*
