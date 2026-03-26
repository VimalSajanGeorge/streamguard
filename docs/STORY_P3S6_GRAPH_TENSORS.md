# Story P3-S6: Stage 6 Graph Tensor Assembly (Phase 3 Extension)

**Status**: COMPLETE
**Date**: 2026-03-27
**Tests**: tests/test_story_p3s6.py (14/14 PASS) + tests/test_story6.py (68/68 PASS, 0 regressions) + tests/test_story7.py (80/80 PASS, 0 downstream regressions)

---

## Summary

Extended the Phase 1 Stage 6 implementation to support the full Phase 3 dataset with CFA pair tracking, 7 explicit validation gates, per-CWE statistics, and post-write pair integrity checking.

---

## Changes Made

### Change 1: Per-Graph HDF5 Attributes (R-25 mitigation)

Each graph group in `/graphs/{idx}/` now has 4 string attributes:

| Attribute | Source | Downstream Consumer |
|-----------|--------|-------------------|
| `pair_id` | `cpg.pair_id` | Stage 7 CFAAwareSplit, CFAAwareBatchSampler |
| `cwe` | `cpg.cwe` | Stage 7 CWE diversity check |
| `source` | `cpg.source` | Analytics (sard vs cve vs repo) |
| `sample_id` | `cpg_path.stem` | Debug traceability |

The risk doc (R-25) also suggested `commit_sha`, but CPG JSON does not carry it (it's only in the collection-layer schema). Stage 7 does not use `commit_sha` for split grouping, so `sample_id` is saved instead.

### Change 2: Post-Write Pair Integrity Check

New function `check_pair_integrity(h5_path)` and CLI flag `--check-pairs`:
- Scans all graph groups for non-empty `pair_id` attrs
- Builds `{pair_id: [labels]}` map
- Flags pairs missing label=0 or label=1 as "broken"
- Broken pairs logged as WARNING (not error) -- treated as singletons in Stage 7
- Sentinel pair_id values (`""`, `"None"`, `"null"`, `"0"`) are ignored

### Change 3: Per-CWE Distribution in graph_stats.json

`graph_stats.json` now includes:
```json
{
  "cwe_distribution": {
    "CWE-121": 12345,
    "CWE-190": 8901,
    ...
  }
}
```
This lets Stage 7 verify CWE coverage before training without opening the HDF5.

### Change 4: All 7 Validation Gates Formalized

`load_sample()` return type changed from `dict | None` to `tuple[dict | None, str]`.
The second element is the gate rejection reason (empty string on success).

| Gate | Check | Reject Reason |
|------|-------|---------------|
| 1 | num_nodes < 3 | `gate1_trivial` |
| 2 | features.shape != (N, 824) | `gate2_shape` |
| 3 | NaN in features | `gate3_nan` |
| 4 | edge_type >= 4 (all edges invalid) | `gate4_edge_type` |
| 5 | edge_index.max() >= num_nodes | `gate5_oob` |
| 6 | All edges reference non-existent nodes | `gate6_dangling` |
| 7 | No valid edges after filtering | `gate7_no_edges` |

Rejected graphs are logged to `graph_rejected.log` (TSV: `sample_id\tgate`).
Gate rejection counts are included in `graph_stats.json` under `reject_reasons`.

### Change 5: Colab Notebook Updated

- Cell 6 (Stage 6 run): passes `check_pairs=True`, displays CWE distribution and pair integrity summary
- Cell 7 (save to Drive): copies `graph_rejected.log` alongside stats

---

## Risk Analysis: P3-S6 Risks

All 4 risks from the Phase 3 Risk Analysis document are mitigated:

| Risk | Severity | Status | How Mitigated |
|------|----------|--------|---------------|
| R-25: pair_id not saved as HDF5 attr | CRITICAL | MITIGATED | `g.attrs['pair_id']` written for every graph; test `test_pair_id_written_and_readable` verifies |
| R-26: edge_index not remapped to 0..N-1 | HIGH | MITIGATED | `node_id_to_idx` mapping + Gate 5 OOB check; test `test_edge_index_max_less_than_num_nodes` verifies with Joern-style IDs (100, 200, 300, 400) |
| R-27: CDG edge_type=4 in HDF5 | HIGH | MITIGATED | Gate 4 filters edges with type not in {0,1,2,3}; test `test_all_edges_bad_type_rejects_graph` verifies |
| R-28: HDF5 corruption on crash | HIGH | MITIGATED | Atomic write via `.h5.tmp` + `os.replace()`; carried forward from Phase 1 |

---

## Verification Checklist Status

| Check | Status | Notes |
|-------|--------|-------|
| VC-S6-01: pytest test_story_p3s6.py | PASS (14/14) | All new tests pass |
| VC-S6-02: pair_id attr check | PASS | Test verifies attrs written and readable |
| VC-S6-03: Edge OOB check | PASS | Gate 5 implemented + tested |
| VC-S6-04: Edge type range | PASS | Gate 4 implemented + tested. **NOTE**: Risk doc verification command references `edge_type` as HDF5 dataset name, but actual implementation uses `edge_attr` for backward compatibility with Stage 7/model/dataloader. The check command should use `f[k]['edge_attr']` not `f[k]['edge_type']` |
| VC-S6-05: Atomic write | PASS | `.h5.tmp` + `os.replace()` from Phase 1 |
| VC-S6-06: --check-pairs | PASS | `check_pair_integrity()` implemented + CLI flag |
| VC-S6-07: Sample 5 graphs | PASS | `verify_h5()` updated to print pair_id, cwe, source |

---

## Known Discrepancy: edge_attr vs edge_type HDF5 Dataset Name

The risk doc (VC-S6-04) and schema section (7.1) refer to the HDF5 dataset as `edge_type`. However, the Phase 1 implementation and all downstream consumers use `edge_attr`:

- `stage7_split.py` line 545: `dst_g.create_dataset("edge_attr", ...)`
- `cfa_dataloader.py` line 84: `g['edge_attr'][:]`
- `model.py` line 134: parameter named `edge_attr`

Renaming to `edge_type` would require changes across 6+ files with no functional benefit. We keep `edge_attr` as the HDF5 dataset name. The risk doc verification command VC-S6-04 should be adjusted accordingly when running production checks.

---

## Files Modified

| File | Change |
|------|--------|
| `training/scripts/preprocessing/stage6_graphs.py` | Added per-graph attrs, pair integrity check, CWE distribution, 7 gates with rejection logging, `--check-pairs` CLI flag |
| `tests/test_story6.py` | Updated 10 `load_sample` call sites for new tuple return; added `check_pair_integrity` import |
| `StreamGuard_Stage5_Stage6_Colab.ipynb` | Cell 6: `check_pairs=True`, CWE/pair display; Cell 7: copies `graph_rejected.log` |

## Files Created

| File | Purpose |
|------|---------|
| `tests/test_story_p3s6.py` | 14 tests covering all 4 changes + gate coverage |
| `docs/STORY_P3S6_GRAPH_TENSORS.md` | This document |

---

## HDF5 Schema (Final)

```
/metadata/
  attrs: feature_dim=824, num_graphs=N, edge_types=4
  datasets: sample_ids, labels, cwes, pair_ids

/graphs/{idx}/
  x          (num_nodes, 824)  float32  [gzip compressed]
  edge_index (2, num_edges)    int64
  edge_attr  (num_edges,)      int64    [values in {0,1,2,3}]
  y          (1,)              int64

  attrs:
    pair_id    string  (CFA pair UUID, empty if unpaired)
    cwe        string  (e.g. "CWE-121")
    source     string  (e.g. "sard", "cve", "sard_cfa")
    sample_id  string  (UUID from schema)
```
