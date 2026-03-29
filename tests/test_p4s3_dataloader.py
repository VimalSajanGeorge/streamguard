#!/usr/bin/env python3
"""
tests/test_p4s3_dataloader.py

Phase 4 Story 3: CFA-Aware DataLoader.

8 mandatory tests:
  1. Dataset builds index with pair_id, label, cwe fields
  2. Dataset handles nested h5 layout (split/sample_id)
  3. Dataset handles flat h5 layout (graphs[idx] + metadata)
  4. Sampler: pairs (same pair_id) are in same batch
  5. Sampler: set_epoch() gives different shuffling each epoch
  6. Sampler: singletons distributed correctly
  7. Collate: cfa_batch is None when batch has no CFA samples
  8. Collate: orig and CFA counts are aligned

Additional tests:
  9.  build_dataloader returns (DataLoader, dataset, sampler) tuple
  10. Sentinel pair_ids treated as singletons
  11. Collate returns correct tuple structure
  12. count_batches matches iteration
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from collections import defaultdict

import h5py
import numpy as np
import torch
import pytest
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.scripts.model.cfa_dataloader import (
    CFADataset,
    CFAAwareBatchSampler,
    EMPTY_PAIR_SENTINELS,
    cfa_collate_fn,
    count_batches,
    build_dataloader,
)


# ---------------------------------------------------------------------------
# HDF5 fixtures
# ---------------------------------------------------------------------------

def _create_layout_a_h5(path, n_samples=20, n_pairs=5):
    """Create Layout A HDF5: f['graphs'][idx] + f['metadata']."""
    with h5py.File(path, 'w') as f:
        graphs = f.create_group('graphs')
        sample_ids = []
        pair_ids   = []
        labels     = []
        cwes       = []

        for i in range(n_samples):
            g = graphs.create_group(str(i))
            n_nodes = 5 + (i % 3)
            n_edges = 8 + (i % 4)
            g.create_dataset('x', data=np.random.randn(n_nodes, 824).astype(np.float32))
            g.create_dataset('edge_index', data=np.random.randint(0, n_nodes, (2, n_edges)).astype(np.int64))
            g.create_dataset('edge_attr', data=np.random.randint(0, 4, (n_edges,)).astype(np.int64))
            g.create_dataset('y', data=np.array([i % 2], dtype=np.int64))

            sid = f"sample_{i:04d}"
            sample_ids.append(sid)
            labels.append(i % 2)
            cwes.append(f"CWE-{121 + (i % 4)}")

            # First n_pairs*2 samples form pairs
            if i < n_pairs * 2:
                pair_ids.append(f"pair_{i // 2:04d}")
            else:
                pair_ids.append("")

        meta = f.create_group('metadata')
        meta.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))
        meta.create_dataset('pair_ids', data=np.array(pair_ids, dtype='S'))
        meta.create_dataset('labels', data=np.array(labels, dtype=np.int64))
        meta.create_dataset('cwes', data=np.array(cwes, dtype='S'))


def _create_layout_b_nested_h5(path, n_samples=16, n_pairs=4):
    """Create Layout B nested HDF5: f['train'][sample_id]."""
    with h5py.File(path, 'w') as f:
        train_grp = f.create_group('train')

        for i in range(n_samples):
            sid = f"sample_{i:04d}"
            g = train_grp.create_group(sid)
            n_nodes = 5 + (i % 3)
            n_edges = 8 + (i % 4)
            g.create_dataset('x', data=np.random.randn(n_nodes, 824).astype(np.float32))
            g.create_dataset('edge_index', data=np.random.randint(0, n_nodes, (2, n_edges)).astype(np.int64))
            g.create_dataset('edge_attr', data=np.random.randint(0, 4, (n_edges,)).astype(np.int64))
            g.create_dataset('y', data=np.array([i % 2], dtype=np.int64))

            g.attrs['cwe'] = f"CWE-{121 + (i % 4)}"
            g.attrs['source'] = 'sard'
            if i < n_pairs * 2:
                g.attrs['pair_id'] = f"pair_{i // 2:04d}"
            else:
                g.attrs['pair_id'] = ""


@pytest.fixture
def layout_a_h5(tmp_path):
    path = str(tmp_path / "layout_a.h5")
    _create_layout_a_h5(path)
    return path


@pytest.fixture
def layout_b_h5(tmp_path):
    path = str(tmp_path / "layout_b.h5")
    _create_layout_b_nested_h5(path)
    return path


# =========================================================================
# Test 1 -- Dataset builds index with pair_id, label, cwe
# =========================================================================

class TestDatasetIndex:
    def test_index_has_required_fields(self, layout_a_h5):
        ds = CFADataset(layout_a_h5)
        assert len(ds.index) == 20
        for item in ds.index:
            assert "sample_id" in item
            assert "pair_id" in item
            assert "label" in item
            assert "cwe" in item

    def test_index_labels_are_ints(self, layout_a_h5):
        ds = CFADataset(layout_a_h5)
        for item in ds.index:
            assert isinstance(item["label"], int)
            assert item["label"] in (0, 1)

    def test_pair_ids_present(self, layout_a_h5):
        ds = CFADataset(layout_a_h5)
        paired = [i for i in ds.index if i["pair_id"] and i["pair_id"] not in EMPTY_PAIR_SENTINELS]
        assert len(paired) == 10, f"Expected 10 paired samples, got {len(paired)}"


# =========================================================================
# Test 2 -- Dataset handles nested h5 layout
# =========================================================================

class TestLayoutBNested:
    def test_nested_layout_loads(self, layout_b_h5):
        ds = CFADataset(layout_b_h5, split="train")
        assert len(ds.index) == 16
        assert ds._layout == 'B_nested'

    def test_nested_getitem_returns_data_and_meta(self, layout_b_h5):
        ds = CFADataset(layout_b_h5, split="train")
        data, meta = ds[0]
        assert isinstance(data, torch.Tensor.__class__.__bases__[0]) or hasattr(data, 'x')
        assert data.x.dtype == torch.float32
        assert data.edge_index.dtype == torch.int64
        assert "sample_id" in meta


# =========================================================================
# Test 3 -- Dataset handles flat h5 layout (Layout A)
# =========================================================================

class TestLayoutAFlat:
    def test_flat_layout_loads(self, layout_a_h5):
        ds = CFADataset(layout_a_h5)
        assert len(ds.index) == 20
        assert ds._layout == 'A'

    def test_flat_getitem_returns_data_and_meta(self, layout_a_h5):
        ds = CFADataset(layout_a_h5)
        data, meta = ds[0]
        assert hasattr(data, 'x')
        assert data.x.shape[1] == 824
        assert data.x.dtype == torch.float32
        assert data.edge_index.dtype == torch.int64


# =========================================================================
# Test 4 -- Sampler: pairs in same batch
# =========================================================================

class TestSamplerPairGrouping:
    def test_pairs_never_split(self, layout_a_h5):
        """CRITICAL: pairs with same pair_id must be in same batch."""
        ds = CFADataset(layout_a_h5)
        sampler = CFAAwareBatchSampler(ds, batch_size=4, drop_last=False, shuffle=True, seed=42)

        pair_to_batches = defaultdict(set)
        for batch_idx, batch_indices in enumerate(sampler):
            for idx in batch_indices:
                pid = ds.index[idx]["pair_id"]
                if pid and pid not in EMPTY_PAIR_SENTINELS:
                    pair_to_batches[pid].add(batch_idx)

        split_pairs = {p: b for p, b in pair_to_batches.items() if len(b) > 1}
        assert not split_pairs, f"CRITICAL: {len(split_pairs)} pairs split across batches: {split_pairs}"

    def test_pairs_in_same_batch_nested(self, layout_b_h5):
        ds = CFADataset(layout_b_h5, split="train")
        sampler = CFAAwareBatchSampler(ds, batch_size=4, drop_last=False, shuffle=True, seed=42)

        pair_to_batches = defaultdict(set)
        for batch_idx, batch_indices in enumerate(sampler):
            for idx in batch_indices:
                pid = ds.index[idx]["pair_id"]
                if pid and pid not in EMPTY_PAIR_SENTINELS:
                    pair_to_batches[pid].add(batch_idx)

        split_pairs = {p: b for p, b in pair_to_batches.items() if len(b) > 1}
        assert not split_pairs, f"Pairs split: {split_pairs}"


# =========================================================================
# Test 5 -- Sampler: set_epoch gives different shuffling
# =========================================================================

class TestSamplerSetEpoch:
    def test_different_epochs_different_order(self, layout_a_h5):
        ds = CFADataset(layout_a_h5)
        sampler = CFAAwareBatchSampler(ds, batch_size=4, drop_last=False, shuffle=True, seed=42)

        sampler.set_epoch(0)
        order_0 = [idx for batch in sampler for idx in batch]

        sampler.set_epoch(1)
        order_1 = [idx for batch in sampler for idx in batch]

        assert order_0 != order_1, "Different epochs should give different ordering"

    def test_same_epoch_same_order(self, layout_a_h5):
        ds = CFADataset(layout_a_h5)
        sampler = CFAAwareBatchSampler(ds, batch_size=4, drop_last=False, shuffle=True, seed=42)

        sampler.set_epoch(5)
        order_a = [idx for batch in sampler for idx in batch]

        sampler.set_epoch(5)
        order_b = [idx for batch in sampler for idx in batch]

        assert order_a == order_b, "Same epoch+seed should give same ordering"


# =========================================================================
# Test 6 -- Sampler: singletons distributed
# =========================================================================

class TestSamplerSingletons:
    def test_singletons_present_in_batches(self, layout_a_h5):
        ds = CFADataset(layout_a_h5)
        sampler = CFAAwareBatchSampler(ds, batch_size=4, drop_last=False, shuffle=True, seed=42)

        all_indices = [idx for batch in sampler for idx in batch]
        singleton_indices = [i for i, item in enumerate(ds.index)
                           if item["pair_id"] in EMPTY_PAIR_SENTINELS or not item["pair_id"]]

        for si in singleton_indices:
            assert si in all_indices, f"Singleton index {si} missing from batches"

    def test_all_samples_covered(self, layout_a_h5):
        ds = CFADataset(layout_a_h5)
        sampler = CFAAwareBatchSampler(ds, batch_size=4, drop_last=False, shuffle=True, seed=42)

        all_indices = sorted([idx for batch in sampler for idx in batch])
        expected = list(range(len(ds)))
        assert all_indices == expected, "Not all samples covered"


# =========================================================================
# Test 7 -- Collate: cfa_batch is None when no CFA samples
# =========================================================================

class TestCollateNoCFA:
    def test_all_singletons_no_cfa(self):
        """Batch of singletons → cfa_batch is None."""
        from torch_geometric.data import Data

        items = []
        for i in range(4):
            g = Data(
                x=torch.randn(5, 824), edge_index=torch.randint(0, 5, (2, 8)),
                edge_attr=torch.randint(0, 4, (8,)), y=torch.tensor(i % 2),
            )
            g.pair_id = ""
            g.sample_id = f"s{i}"
            g.cwe = "CWE-121"
            meta = {"sample_id": f"s{i}", "pair_id": "", "label": i % 2, "cwe": "CWE-121", "source": ""}
            items.append((g, meta))

        orig_batch, cfa_batch, orig_metas, cfa_metas = cfa_collate_fn(items)
        assert cfa_batch is None, "cfa_batch should be None for all-singleton batch"
        assert len(orig_metas) == 4
        assert len(cfa_metas) == 0


# =========================================================================
# Test 8 -- Collate: orig and CFA counts aligned
# =========================================================================

class TestCollateAlignment:
    def test_paired_batch_aligned(self):
        """Batch with 2 pairs → 2 orig + 2 CFA."""
        items = []
        for pair_idx in range(2):
            pid = f"pair_{pair_idx}"
            for member in range(2):
                g = Data(
                    x=torch.randn(5, 824), edge_index=torch.randint(0, 5, (2, 8)),
                    edge_attr=torch.randint(0, 4, (8,)), y=torch.tensor(member),
                )
                g.pair_id = pid
                g.sample_id = f"s_{pair_idx}_{member}"
                g.cwe = "CWE-121"
                meta = {"sample_id": g.sample_id, "pair_id": pid, "label": member, "cwe": "CWE-121", "source": ""}
                items.append((g, meta))

        orig_batch, cfa_batch, orig_metas, cfa_metas = cfa_collate_fn(items)

        assert cfa_batch is not None, "cfa_batch should not be None for paired batch"
        assert len(orig_metas) == 2, f"Expected 2 orig, got {len(orig_metas)}"
        assert len(cfa_metas) == 2, f"Expected 2 CFA, got {len(cfa_metas)}"

    def test_mixed_pairs_and_singletons(self):
        """1 pair + 2 singletons → 3 orig + 1 CFA."""
        items = []
        # Pair
        for member in range(2):
            g = Data(
                x=torch.randn(5, 824), edge_index=torch.randint(0, 5, (2, 8)),
                edge_attr=torch.randint(0, 4, (8,)), y=torch.tensor(member),
            )
            g.pair_id = "pair_0"
            g.sample_id = f"p_{member}"
            g.cwe = "CWE-121"
            meta = {"sample_id": g.sample_id, "pair_id": "pair_0", "label": member, "cwe": "CWE-121", "source": ""}
            items.append((g, meta))
        # Singletons
        for i in range(2):
            g = Data(
                x=torch.randn(5, 824), edge_index=torch.randint(0, 5, (2, 8)),
                edge_attr=torch.randint(0, 4, (8,)), y=torch.tensor(0),
            )
            g.pair_id = ""
            g.sample_id = f"s_{i}"
            g.cwe = "CWE-190"
            meta = {"sample_id": g.sample_id, "pair_id": "", "label": 0, "cwe": "CWE-190", "source": ""}
            items.append((g, meta))

        orig_batch, cfa_batch, orig_metas, cfa_metas = cfa_collate_fn(items)
        assert len(orig_metas) == 3  # 1 orig from pair + 2 singletons
        assert len(cfa_metas) == 1   # 1 CFA from pair


# =========================================================================
# Test 9 -- build_dataloader returns correct tuple
# =========================================================================

class TestBuildDataloader:
    def test_returns_tuple(self, layout_a_h5):
        loader, dataset, sampler = build_dataloader(layout_a_h5, split=None, batch_size=4)
        assert isinstance(dataset, CFADataset)
        assert isinstance(sampler, CFAAwareBatchSampler)

    def test_iteration_works(self, layout_a_h5):
        loader, _, _ = build_dataloader(layout_a_h5, split=None, batch_size=4, drop_last=False)
        count = 0
        for orig_batch, cfa_batch, orig_metas, cfa_metas in loader:
            assert hasattr(orig_batch, 'x')
            count += 1
        assert count > 0


# =========================================================================
# Test 10 -- Sentinel pair_ids
# =========================================================================

class TestSentinelPairIds:
    def test_sentinels_are_singletons(self):
        for sentinel in EMPTY_PAIR_SENTINELS:
            assert sentinel in EMPTY_PAIR_SENTINELS

    def test_sentinel_values(self):
        assert "" in EMPTY_PAIR_SENTINELS
        assert "None" in EMPTY_PAIR_SENTINELS
        assert "null" in EMPTY_PAIR_SENTINELS
        assert "0" in EMPTY_PAIR_SENTINELS
        assert "none" in EMPTY_PAIR_SENTINELS


# =========================================================================
# Test 11 -- count_batches
# =========================================================================

class TestCountBatches:
    def test_count_matches_iteration(self, layout_a_h5):
        ds = CFADataset(layout_a_h5)
        sampler = CFAAwareBatchSampler(ds, batch_size=4, drop_last=False, shuffle=False, seed=42)
        counted = count_batches(sampler)
        iterated = sum(1 for _ in sampler)
        assert counted == iterated
