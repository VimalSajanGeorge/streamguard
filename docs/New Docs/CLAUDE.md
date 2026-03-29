# CLAUDE.md — StreamGuard Master Implementation Guide

> **READ THIS ENTIRE FILE BEFORE WRITING A SINGLE LINE OF CODE.**
> This is the authoritative reference for every implementation decision.
> All other docs in `docs/` expand on specific sections.

---

## What StreamGuard Is

StreamGuard is a **neuro-symbolic vulnerability detection system for C code** that combines:

1. **Vul-LMGNN** — Joint CodeBERT + Gated Graph Neural Network (GGNN) training (Tang et al., arXiv 2024)
2. **VISION-style CFA** — LLM-generated counterfactual pairs for GNN training, generalised from CWE-20 to **12 CWE types** (Egea et al., AIES 2025)
3. **4-Component CPG** — AST + CFG + DFG + Taint Propagation Graph (novel 4th component beyond AUG-PDG, Zou et al., Scientific Reports 2025)
4. **Inter-procedural context** — LLM callee summaries injected as CPG node features (VulnSC-style, ICSE 2024)
5. **CFExplainer** — Counterfactual explanations for GNN predictions (ISSTA 2024)
6. **Continuous CFA loop** — Human feedback generates new CFA pairs indefinitely (novel production contribution)

**The primary research novelty**: VISION proved CFA works for CWE-20 with Word2Vec. StreamGuard proves it generalises across 12 CWE types with CodeBERT, a structural TPG component, and inter-procedural context. No existing system combines all six above.

---

## Project Structure

```
streamguard/
├── CLAUDE.md                          ← YOU ARE HERE
├── PRD.md                             ← Technical Product Requirements
├── docs/
│   ├── ARCHITECTURE.md                ← Full system architecture
│   ├── NOVELTY.md                     ← Research novelties vs existing work
│   ├── DATA_PIPELINE.md               ← 6-collector data pipeline spec
│   ├── PREPROCESSING.md               ← 7-stage preprocessing pipeline
│   ├── MODEL.md                       ← Neural architecture specification
│   ├── SERVING.md                     ← API, inference, deployment
│   └── EXPERIMENTS.md                 ← Ablation study design
├── training/
│   ├── scripts/
│   │   ├── collection/                ← 6 data collectors
│   │   ├── preprocessing/             ← 7 preprocessing stages
│   │   ├── model/                     ← Model, training, eval
│   │   └── serving/                   ← FastAPI + inference
│   ├── data/
│   │   ├── raw/                       ← Collector outputs (JSONL)
│   │   ├── processed/                 ← Cleaned + deduped
│   │   ├── graphs/                    ← PyG Data objects (HDF5)
│   │   └── final/                     ← train.h5 / val.h5 / test.h5
│   └── checkpoints/                   ← Model checkpoints
├── .env.example                       ← Required environment variables
└── docker/
    ├── Dockerfile.training
    └── Dockerfile.serving
```

---

## Mandatory Implementation Rules

### Rule 1: Read Before Writing
Before implementing ANY component, read the corresponding `docs/` file completely.
Never guess at requirements — they are fully specified.

### Rule 2: Schema Compliance
Every training sample MUST conform to the canonical schema in `docs/DATA_PIPELINE.md`.
The `validate_sample()` function in `training/scripts/collection/schema.py` must pass before any sample is saved.

### Rule 3: Checkpoint Everything
Every script that runs for more than 5 minutes MUST implement checkpoint/resume.
Use atomic writes: write to `.tmp` file, then `os.replace()`. Never lose progress.

### Rule 4: CFA Pairs Are Sacred
CFA pairs (vulnerable + counterfactual) must ALWAYS stay together.
- Same `pair_id` field
- Never split across train/val/test
- Batch loader must keep pairs in same batch for contrastive loss
Violation breaks the entire training objective.

### Rule 5: Never git clone Large Repos
The repo miner uses GitHub Commits API exclusively. `git clone` of linux/openssl/FFmpeg causes OOM.

### Rule 6: Validate CPG Before Embedding
After every Joern run, validate: `len(nodes) >= 3`, `edge_index.max() < num_nodes`, no NaN in features.
Invalid graphs silently corrupt training.

### Rule 7: Test with --dry-run First
Every collector and preprocessing script MUST accept `--dry-run` and `--max-samples N` flags.
Always test with 20 samples before running full pipeline.

---

## Implementation Order (MUST follow this sequence)

```
Phase 0: Environment                    → .env, Docker, Joern smoke test
Phase 1: Schema + Base class            → schema.py, base_collector.py
Phase 2: SARD + ExploitDB collectors    → fastest data, no API dependency
Phase 3: Preprocessing Stages 1-3      → clean, dedup, CFA generation
Phase 4: CPG Pipeline                   → Joern wrapper, taint analyzer, slicer
Phase 5: Node embedding + graph build   → CodeBERT embed, PyG Data, HDF5
Phase 6: Model                          → GGNN + CodeBERT + fusion + heads
Phase 7: Training loop                  → CFA-aware DataLoader, composite loss
Phase 8: Remaining collectors           → CVE, GitHub Advisory, OSV, RepoMiner
Phase 9: Serving                        → FastAPI, inference worker
Phase 10: Evaluation + ablations        → eval.py, run_ablations.py
```

**DO NOT skip phases.** Phase N depends on Phase N-1 being correct.

---

## Environment Requirements

```bash
# Required tools
python >= 3.10
torch >= 2.2.0 + CUDA 11.8+
torch_geometric >= 2.4.0
transformers >= 4.38.0
joern >= 2.0 (JVM 17+)
tree-sitter >= 0.21.0
tree-sitter-c
gcc (for CFA compilation validation)
redis >= 7.0
neo4j >= 5.0 (optional, for graph storage)
fastapi + uvicorn (serving)

# Python packages (full list in requirements.txt)
anthropic, requests, tenacity, loguru, python-dotenv,
datasketch, pycparser, h5py, numpy, pandas, tqdm,
networkx, matplotlib, scikit-learn, mlflow
```

### Required .env Variables
```
ANTHROPIC_API_KEY=...        # For CFA generation (Claude Haiku)
GITHUB_TOKEN=...             # For CVE phase2, GitHub Advisory, repo miner
NVD_API_KEY=...              # For CVE phase1 (optional but removes rate limit)
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=...
JOERN_BIN=/opt/joern/joern-cli/joern
JOERN_EXPORT=/opt/joern/joern-cli/joern-export
STREAMGUARD_DATA_DIR=./training/data
STREAMGUARD_CHECKPOINT_DIR=./training/checkpoints
```

---

## The 12 Target CWE Types

| ID | CWE | Name | Primary Source |
|----|-----|------|----------------|
| 1 | CWE-89 | SQL Injection | SARD, CVE |
| 2 | CWE-78 | OS Command Injection | SARD, ExploitDB |
| 3 | CWE-79 | Cross-Site Scripting (C web) | CVE, nginx/Apache |
| 4 | CWE-119 | Buffer Overflow (general) | SARD, repos |
| 5 | CWE-120 | Buffer Copy No Bound Check | SARD, ExploitDB |
| 6 | CWE-121 | Stack Buffer Overflow | SARD, Linux kernel |
| 7 | CWE-122 | Heap Buffer Overflow | SARD, OpenSSL |
| 8 | CWE-125 | Out-of-Bounds Read | Linux kernel, SARD |
| 9 | CWE-134 | Uncontrolled Format String | SARD, ExploitDB |
| 10 | CWE-190 | Integer Overflow | SARD, Linux kernel |
| 11 | CWE-416 | Use After Free | Linux, OpenSSL repos |
| 12 | CWE-476 | NULL Pointer Dereference | Linux, GitHub Advisory |

---

## Key Design Decisions and Why

| Decision | Alternative | Reason |
|----------|-------------|--------|
| CodeBERT (not GraphCodeBERT) | GraphCodeBERT | Vul-LMGNN ablation: CodeBERT+GGNN > GraphCodeBERT alone on C vulnerability tasks |
| 3-layer GGNN (not deeper) | 4-5 layers | Over-smoothing in deeper GNNs on CPG graphs (Vul-LMGNN ablation RQ4) |
| Cross-attention fusion (not concat) | Simple concatenation | Cross-attention allows BERT to attend to structural graph features directly. **Must be node-level** (BERT queries attend to N graph nodes via scatter softmax), not graph-embed level (single 256-d vector as K/V) — graph-embed cross-attention mathematically degrades to simple concatenation |
| TPG as 4th CPG component | AST+CFG+DFG only | Explicit taint paths critical for injection CWEs (CWE-89/78/79) |
| **5-Tier CFA generation (not single strategy)** | **Single LLM prompt per CWE** | **CWE difficulty varies 10x: CWE-134 (format string) needs 1-line deterministic fix; CWE-416 (use-after-free) needs lifetime tracking. Tier 1 AST rules give 100% structural validity for simple CWEs; Tier 4 few-shot exemplar CoT required for CWE-416. Single-strategy wastes API budget on simple CWEs and fails on complex ones. See `CFA_Stories.md`.** |
| CFA margin contrastive loss | BCE loss only | VISION: BCE alone allows spurious correlations; contrastive loss forces structural separation |
| tree-sitter for function extraction | regex | regex fails on nested braces, #ifdef blocks, function pointers |
| Joern subprocess (not library) | pycparser | pycparser can't produce CFG/DFG; Joern is the only open-source tool for full CPG |
| HDF5 cache for graphs | re-run Joern | Joern takes ~700ms/function; HDF5 reload is ~2ms |

---

## Critical Paths — What Breaks Everything

These are the single points of failure in the whole system:

1. **`run.ossdataflow` in Joern script** — Without this, DFG edges don't exist. The GGNN has no data flow signal.
2. **`pair_id` linkage in CFA pairs** — If broken, contrastive loss trains on random pairs. F1 will not improve.
3. **Edge index bounds validation** — `edge_index.max() < num_nodes` must be checked before every training run.
4. **CFA-aware DataLoader** — Standard PyG DataLoader shuffles pairs apart. Custom loader required.
5. **Taint role labels on nodes** — If all nodes have `taint_role='none'`, the TPG component has zero signal.
6. **Edge type masking in `encode_graph()`** — Each of the 4 GatedGraphConv modules MUST receive only its own edge type (mask by `edge_attr == etype`). If any module receives the full unmasked `edge_index`, the per-type architecture silently degrades to type-blind. No error is thrown.
7. **`CWE_TIER_MAP` dispatch in `stage3_cfa.py`** — Assigning a CWE to the wrong tier degrades CFA quality silently. Tier 1 applied to a CWE without a clear AST fix rule returns empty CFAs. Tier 2/3 applied to CWE-416 produces ~40% failure rate and weak pairs. Never change tier assignments without re-running Stage 3 and re-verifying `cfa_quality_report.json` thresholds.
8. **SARD skip in Stage 3** — If the `source.startswith('sard')` check is missing, Stage 3 overwrites SARD's native `pair_id` linkage with new LLM-generated CFAs. The original NIST-validated `good()` functions become orphaned singletons. L_CFA trains on LLM quality rather than ground-truth quality.
9. **`pair_id` saved as HDF5 attribute in Stage 6** — If `grp.attrs['pair_id']` is missing, `CFAAwareBatchSampler` treats every sample as a singleton. L_CFA trains on random pairs. The entire CFA research contribution appears not to work.

---

## Reference Papers

All implementation decisions trace to these papers:

| System | Venue | What We Take From It |
|--------|-------|---------------------|
| Vul-LMGNN (Tang et al.) | arXiv Apr 2024 | CodeBERT+GGNN joint training; implicit-explicit fusion; 10% F1 gain over 17 baselines |
| VISION (Egea et al.) | AIES 2025 | CFA generation via LLM; paired GNN training; contrastive accuracy metric; Illuminati explainer |
| CFExplainer (Zhang et al.) | ISSTA 2024 | Counterfactual explanation; minimal graph perturbation; what-if analysis for GNN predictions |
| AUG-PDG (Zou et al.) | Scientific Reports 2025 | Augmented PDG + CodeBERT; context slicing (2-hop); outperforms GCN/BGRU/GAT |
| VulnSC / Inter-proc | ICSE 2024 | Inter-procedural vulnerability detection; callee summary injection |
| VulPatrol | ACM DASP 2024 | LLVM-IR inter-procedural CPG; +12% F1 over SOTA |
| Taint+BiGGNN | IoTML 2024 | Static taint path + bidirectional GGNN + attention |
| ICSE 2025 (Ding et al.) | ICSE 2025 | Data quality issues in vulnerability datasets; high duplication rates |

---

## Ask Before Assuming

If any specification is unclear, ask before implementing. Common ambiguity points:

- **Which Joern version?** → Use latest stable (4.x). The API changed significantly from 1.x. `joern-parse` in 4.x includes DFG by default; older versions require `run.ossdataflow` in REPL mode.
- **Which CodeBERT checkpoint?** → `microsoft/codebert-base` (not graphcodebert, not codet5)
- **CFA ratio target?** → 2 CFA pairs per vulnerable sample minimum; 3 is target. CWE-416 may achieve only 1.5 pairs/sample — this is acceptable.
- **Max CPG nodes?** → 200 after context slicing. Graph with > 200 nodes gets sliced.
- **Training batch size?** → 8 graphs per batch (GPU memory), gradient accumulation × 4 = effective batch 32
- **Which CFA tier for a given CWE?** → See `CWE_TIER_MAP` in `stage3_cfa.py`. Do NOT change assignments without re-running Stage 3 + verifying `cfa_quality_report.json`.
- **Do SARD samples get LLM CFA generation?** → NO. SARD already has native CFA pairs from `process_sard.py`. Stage 3 skips any sample where `source.startswith('sard')` or `cfa_type == 'native'`.
- **What is `cfa_tier` field?** → Integer 1-5. 1 = AST rule, 2 = zero-shot, 3 = CoT, 4 = few-shot, 5 = critique-refine. Written to JSONL by Stage 3. Stored in HDF5 attrs by Stage 6. Used for diagnostics and optional loss weighting.

---

*Last updated: March 2026 | StreamGuard v1.0*
