# training/scripts/model/cfa_dataloader.py
#
# Phase 4 CFA-Aware DataLoader for StreamGuard.
#
# CRITICAL INVARIANT (Rule 4 — CFA Pairs Are Sacred):
#   Standard PyG DataLoader shuffles samples randomly → CFA pairs land in
#   different batches → L_CFA trains on random pairs → contributes zero.
#   CFAAwareBatchSampler groups by pair_id and shuffles GROUPS, guaranteeing
#   both members of every pair always land in the same batch.
#
# Supports TWO HDF5 layouts:
#   Layout A (Phase 1 / Stage 7 flat): f['graphs'][str(idx)], f['metadata']
#   Layout B (Phase 4 nested):         f[split][sample_id] with per-graph groups
#
# R-04 mitigation: pairs never split across batches.
# R-18 mitigation: pair_id read from HDF5 attrs; sentinel handling.

import random
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.data import Data, Batch

# Sentinel pair_id values treated as "no pair" (singletons)
EMPTY_PAIR_SENTINELS = frozenset({"", "None", "null", "0", "none"})


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CFADataset(Dataset):
    """
    Loads PyG Data objects from an HDF5 split file.

    Builds an in-memory index of (graph_idx/sample_id, pair_id, label, cwe)
    on init, then loads graph tensors lazily from disk on __getitem__.

    Supports two HDF5 layouts:
      - Layout A: f['graphs'][str(idx)], f['metadata'] with parallel arrays
      - Layout B: f[sample_id] (flat) or f[split][sample_id] (nested) with attrs
    """

    def __init__(self, h5_path: str, split: str | None = None):
        self.h5_path = h5_path
        self.split = split
        self.index = []
        self._layout = None  # 'A' or 'B'
        self._build_index()

    def _build_index(self):
        with h5py.File(self.h5_path, 'r') as f:
            if 'metadata' in f and 'graphs' in f:
                self._layout = 'A'
                self._build_index_layout_a(f)
            elif self.split and self.split in f:
                self._layout = 'B_nested'
                self._build_index_layout_b_nested(f)
            else:
                self._layout = 'B_flat'
                self._build_index_layout_b_flat(f)

    def _build_index_layout_a(self, f):
        """Layout A: f['metadata'] parallel arrays + f['graphs'][idx]."""
        meta = f['metadata']
        sample_ids = meta['sample_ids'][:]
        pair_ids   = meta['pair_ids'][:]
        labels     = meta['labels'][:]
        cwes       = meta['cwes'][:]

        for i in range(len(sample_ids)):
            self.index.append({
                "graph_idx": i,
                "sample_id": _decode(sample_ids[i]),
                "pair_id":   _decode(pair_ids[i]),
                "label":     int(labels[i]),
                "cwe":       _decode(cwes[i]),
                "source":    "",
            })

    def _build_index_layout_b_nested(self, f):
        """Layout B nested: f[split][sample_id] with per-graph groups."""
        grp = f[self.split]
        for sample_id in grp.keys():
            sg = grp[sample_id]
            self.index.append({
                "graph_idx": None,
                "sample_id": sample_id,
                "pair_id":   str(sg.attrs.get("pair_id", "")),
                "label":     int(sg["y"][0]) if "y" in sg else 0,
                "cwe":       str(sg.attrs.get("cwe", "")),
                "source":    str(sg.attrs.get("source", "")),
            })

    def _build_index_layout_b_flat(self, f):
        """Layout B flat: f[sample_id] at root, optional split attr."""
        for sample_id in f.keys():
            sg = f[sample_id]
            if self.split:
                if sg.attrs.get("split", "") != self.split:
                    continue
            self.index.append({
                "graph_idx": None,
                "sample_id": sample_id,
                "pair_id":   str(sg.attrs.get("pair_id", "")),
                "label":     int(sg["y"][0]) if "y" in sg else 0,
                "cwe":       str(sg.attrs.get("cwe", "")),
                "source":    str(sg.attrs.get("source", "")),
            })

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[Data, dict]:
        """Load graph tensors from HDF5 and return (PyG Data, metadata dict)."""
        item = self.index[idx]

        with h5py.File(self.h5_path, 'r') as f:
            if self._layout == 'A':
                g = f['graphs'][str(item["graph_idx"])]
            elif self._layout == 'B_nested':
                g = f[self.split][item["sample_id"]]
            else:
                g = f[item["sample_id"]]

            x          = torch.from_numpy(g['x'][:]).float()
            edge_index = torch.from_numpy(g['edge_index'][:]).long()
            ea_key     = "edge_type" if "edge_type" in g else "edge_attr"
            edge_attr  = torch.from_numpy(g[ea_key][:]).long()

            y_val = item["label"]
            if "y" in g:
                y_val = int(g["y"][0])

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            y=torch.tensor(y_val, dtype=torch.long),
        )
        data.pair_id   = item["pair_id"]
        data.sample_id = item["sample_id"]
        data.cwe       = item["cwe"]
        return data, item


# ─────────────────────────────────────────────────────────────────────────────
# CFA-Aware Batch Sampler
# ─────────────────────────────────────────────────────────────────────────────

class CFAAwareBatchSampler(Sampler):
    """
    Keeps CFA pair groups in the same batch.

    Strategy:
      1. Group all sample indices by pair_id (singletons form size-1 groups).
      2. Shuffle GROUPS (not individual indices) with set_epoch() for reproducibility.
      3. Fill batches greedily. If adding a group would exceed batch_size,
         yield current batch first, then start new batch with that group.
      4. Oversized groups (> batch_size) are never split.

    The invariant — both members of a pair_id always land in the same batch —
    is never violated.
    """

    def __init__(
        self,
        dataset: CFADataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.shuffle    = shuffle
        self.seed       = seed
        self._epoch     = 0

        # Group dataset indices by pair_id
        pair_groups: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(dataset.index):
            pid = item["pair_id"]
            if pid in EMPTY_PAIR_SENTINELS:
                key = f"__singleton_{i}"
            else:
                key = pid
            pair_groups[key].append(i)

        self.groups = list(pair_groups.values())

        # Stats for diagnostics
        self._n_groups  = len(self.groups)
        self._n_samples = sum(len(g) for g in self.groups)
        self._max_group = max(len(g) for g in self.groups) if self.groups else 0

    def set_epoch(self, epoch: int):
        """Call at start of each epoch for deterministic, different shuffling."""
        self._epoch = epoch

    def __iter__(self):
        groups = [g[:] for g in self.groups]
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(groups)

        current_batch = []
        for group in groups:
            if current_batch and len(current_batch) + len(group) > self.batch_size:
                yield current_batch
                current_batch = []

            current_batch.extend(group)

            if len(current_batch) >= self.batch_size:
                yield current_batch
                current_batch = []

        if current_batch and not self.drop_last:
            yield current_batch

    def __len__(self) -> int:
        if self.drop_last:
            return self._n_samples // self.batch_size
        return (self._n_samples + self.batch_size - 1) // self.batch_size


# ─────────────────────────────────────────────────────────────────────────────
# Collate + DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def cfa_collate_fn(batch_items: list[tuple[Data, dict]]):
    """
    Separate original samples and CFA counterparts into two aligned sub-batches.

    Logic:
    - Samples with empty pair_id → always "original"
    - First occurrence of a pair_id → "original"
    - Second occurrence of same pair_id → "CFA counterpart"

    Returns: (orig_batch, cfa_batch, orig_metas, cfa_metas)
    cfa_batch is None if no CFA samples in this batch (all singletons).
    """
    graphs, metas = zip(*batch_items)

    orig_graphs, cfa_graphs = [], []
    orig_metas,  cfa_metas  = [], []
    seen_pair_ids = set()

    for g, m in zip(graphs, metas):
        pid = m["pair_id"]

        if pid in EMPTY_PAIR_SENTINELS or pid not in seen_pair_ids:
            if pid not in EMPTY_PAIR_SENTINELS:
                seen_pair_ids.add(pid)
            orig_graphs.append(g)
            orig_metas.append(m)
        else:
            cfa_graphs.append(g)
            cfa_metas.append(m)

    orig_batch = Batch.from_data_list(orig_graphs)
    cfa_batch  = Batch.from_data_list(cfa_graphs) if cfa_graphs else None

    return orig_batch, cfa_batch, orig_metas, cfa_metas


def count_batches(sampler: CFAAwareBatchSampler) -> int:
    """Count actual batches by iterating the sampler once (for LR scheduler)."""
    return sum(1 for _ in sampler)


def build_dataloader(
    h5_path: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool | None = None,
) -> tuple[DataLoader, CFADataset, CFAAwareBatchSampler]:
    """
    Build a CFA-aware DataLoader from an HDF5 split file.

    Args:
        h5_path:     path to train.h5 / val.h5 / test.h5
        split:       split name (used for nested HDF5 layout)
        batch_size:  target batch size
        num_workers: 0 for Windows (h5py can't pickle file handles)
        shuffle:     shuffle groups each epoch
        seed:        reproducibility seed
        drop_last:   drop final incomplete batch (default: True for train)

    Returns:
        (DataLoader, CFADataset, CFAAwareBatchSampler)
    """
    if drop_last is None:
        drop_last = (split == "train")

    dataset = CFADataset(h5_path, split=split)
    sampler = CFAAwareBatchSampler(
        dataset, batch_size, drop_last=drop_last, shuffle=shuffle, seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=cfa_collate_fn,
    )
    return loader, dataset, sampler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode(val) -> str:
    """Decode bytes to str, pass str through."""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.bytes_):
        return val.decode("utf-8")
    return str(val)
