# docs/EXPERIMENTS.md — Ablation Study & Evaluation Design

> Read before implementing: training/scripts/model/run_ablations.py, eval.py

---

## 6-Configuration Ablation Study

This is the core of the conference paper. Every configuration must be trained from
the same dataset with the same seed. Results go into Table 2 of the paper.

```python
# training/scripts/model/run_ablations.py

ABLATION_CONFIGS = {
    "A_baseline": {
        "description": "CodeBERT sequence only — no graph (reproduces LineVul-style)",
        "use_ggnn": False,
        "ggnn_type": None,
        "cpg_components": [],
        "use_cfa": False,
        "use_interproc": False,
        "expected_f1": 0.79,
    },
    "B_plus_ggnn": {
        "description": "CodeBERT + type-blind GGNN, 3-CPG (AST+CFG+DFG), no CFA (reproduces Vul-LMGNN)",
        "use_ggnn": True,
        "ggnn_type": "type_blind",           # single GatedGraphConv, edge_attr ignored
        "cpg_components": ["AST", "CFG", "DFG"],
        "use_cfa": False,
        "use_interproc": False,
        "expected_f1": 0.83,
    },
    "B_prime_type_aware": {
        "description": "CodeBERT + type-aware GGNN (per-edge-type GRU), 3-CPG, no CFA — isolates typed-edge contribution",
        "use_ggnn": True,
        "ggnn_type": "per_edge_type",        # 4 × GatedGraphConv per layer
        "cpg_components": ["AST", "CFG", "DFG"],
        "use_cfa": False,
        "use_interproc": False,
        "expected_f1": 0.85,
    },
    "C_plus_cfa": {
        "description": "+ CFA contrastive training (3-CPG) — isolates CFA contribution",
        "use_ggnn": True,
        "ggnn_type": "per_edge_type",
        "cpg_components": ["AST", "CFG", "DFG"],
        "use_cfa": True,
        "use_interproc": False,
        "expected_f1": 0.89,
    },
    "D_plus_tpg": {
        "description": "+ Taint Propagation Graph (4-CPG + CFA) — isolates TPG contribution",
        "use_ggnn": True,
        "ggnn_type": "per_edge_type",
        "cpg_components": ["AST", "CFG", "DFG", "TPG"],
        "use_cfa": True,
        "use_interproc": False,
        "expected_f1": 0.91,
    },
    "E_full": {
        "description": "Full StreamGuard: 4-CPG + CFA + inter-procedural context",
        "use_ggnn": True,
        "ggnn_type": "per_edge_type",
        "cpg_components": ["AST", "CFG", "DFG", "TPG"],
        "use_cfa": True,
        "use_interproc": True,
        "expected_f1": 0.93,
    },
}
```

---

## Novel Metrics to Report

Beyond standard F1/precision/recall, report these (from VISION + novel):

```python
METRICS_TO_REPORT = [
    "f1",                           # standard
    "precision",                    # standard
    "recall",                       # standard
    "fpr",                          # false positive rate (target ≤ 5%)
    "fnr",                          # false negative rate (target ≤ 8%)
    "pairwise_contrast_accuracy",   # VISION metric: both v and v' correct
    "worst_group_f1",               # lowest F1 across 12 CWE subgroups
    "per_cwe_f1",                   # dict: {CWE-89: 0.95, CWE-416: 0.88, ...}
    "cfa_compile_rate",             # data quality: % CFAs that compile
    "cfa_pattern_removal_rate",     # % CFAs where vuln heuristic gone
    "intra_class_attribution_var",  # node importance consistency within class
    "inter_class_attribution_dist", # node importance separation vuln vs safe
]
```

---

## Baseline Comparisons (Table 3 in paper)

Re-implement or use public checkpoints for:

| Baseline | Source | Key metric to compare |
|----------|--------|-----------------------|
| Devign | NeurIPS 2019, public implementation | F1, FPR |
| REVEAL | TSE 2022, public implementation | F1, FPR |
| LineVul | MSR 2022, HuggingFace | F1, FPR |
| VISION | AIES 2025 (CWE-20 only) | Pairwise accuracy, worst-group |

Note: VISION comparison is only valid on CWE-20 subset of our test set.
Report both per-CWE-20 numbers AND overall numbers.

---

## Per-CWE F1 Targets (Table 4 in paper)

| CWE | F1 Target | Difficulty | Primary Signal |
|-----|-----------|------------|----------------|
| CWE-89 SQL Injection | ≥ 0.95 | Medium | TPG taint path, clear sink |
| CWE-78 OS Command Injection | ≥ 0.94 | Medium | TPG taint path |
| CWE-79 XSS (C web) | ≥ 0.90 | Medium-Hard | TPG, context-dependent |
| CWE-119 Buffer Overflow | ≥ 0.96 | Easy | Bounds check pattern |
| CWE-120 Buffer Copy No Bound | ≥ 0.97 | Easy | strcpy/gets pattern |
| CWE-121 Stack Buffer Overflow | ≥ 0.95 | Medium | Array index pattern |
| CWE-122 Heap Buffer Overflow | ≥ 0.94 | Medium | Malloc/pointer arithmetic |
| CWE-125 OOB Read | ≥ 0.93 | Medium | Index bounds pattern |
| CWE-134 Format String | ≥ 0.96 | Easy | printf(var) pattern |
| CWE-190 Integer Overflow | ≥ 0.92 | Hard | Arithmetic + type inference |
| CWE-416 Use After Free | ≥ 0.90 | Very Hard | Lifetime tracking |
| CWE-476 NULL Deref | ≥ 0.91 | Hard | Conditional path analysis |

---

*docs/EXPERIMENTS.md | StreamGuard v1.0 | March 2026*
