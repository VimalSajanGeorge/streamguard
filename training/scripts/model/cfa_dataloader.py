# training/scripts/model/cfa_dataloader.py
#
# CFA-Aware DataLoader for StreamGuard.
#
# CRITICAL INVARIANT (Rule 4 — CFA Pairs Are Sacred):
#   Standard PyG DataLoader shuffles samples randomly → CFA pairs land in
#   different batches → L_CFA trains on random pairs → contributes zero.
#   CFAAwareBatchSampler groups by pair_id and shuffles GROUPS, guaranteeing
#   both members of every pair always land in the same batch.
#
# HDF5 layout (from stage7_split):
#   f['graphs'][str(idx)] → {x, edge_index, edge_attr, y}  (no per-graph attrs)
#   f['metadata'] → {sample_ids, pair_ids, labels, cwes}    (parallel arrays, idx 0..N-1)
#   Each split is its own file: train.h5, val.h5, test.h5   (no sub-groups)

import random
from collections import defaultdict

import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.data import Data, Batch


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CFADataset(Dataset):
    """
    Loads PyG Data objects from an HDF5 split file.

    Builds an in-memory index of (graph_idx, pair_id, label, cwe, sample_id)
    on init, then loads graph tensors lazily from disk on __getitem__.

    The index is public (self.index) so that the batch sampler can group by
    pair_id without loading any tensors.
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.index = self._build_index()

    def _build_index(self) -> list[dict]:
        """Read metadata arrays once; build list aligned with graph keys."""
        index = []
        with h5py.File(self.h5_path, 'r') as f:
            meta = f['metadata']
            sample_ids = meta['sample_ids'][:]
            pair_ids   = meta['pair_ids'][:]
            labels     = meta['labels'][:]
            cwes       = meta['cwes'][:]

            for i in range(len(sample_ids)):
                pid_raw = pair_ids[i]
                pid = pid_raw.decode() if isinstance(pid_raw, bytes) else str(pid_raw)
                cwe_raw = cwes[i]
                cwe = cwe_raw.decode() if isinstance(cwe_raw, bytes) else str(cwe_raw)
                sid_raw = sample_ids[i]
                sid = sid_raw.decode() if isinstance(sid_raw, bytes) else str(sid_raw)

                index.append({
                    "graph_idx": i,
                    "sample_id": sid,
                    "pair_id":   pid,
                    "label":     int(labels[i]),
                    "cwe":       cwe,
                })
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Data:
        """Load graph tensors from HDF5 and return a PyG Data object."""
        item = self.index[idx]
        graph_key = str(item["graph_idx"])

        with h5py.File(self.h5_path, 'r') as f:
            g = f['graphs'][graph_key]
            data = Data(
                x          = torch.from_numpy(g['x'][:]).float(),
                edge_index = torch.from_numpy(g['edge_index'][:]).long(),
                edge_attr  = torch.from_numpy(g['edge_attr'][:]).long(),
                y          = torch.tensor(item["label"], dtype=torch.long),
            )

        # String attrs for CFA loss and diagnostics (PyG Batch will
        # collate these into lists automatically)
        data.pair_id   = item["pair_id"]
        data.sample_id = item["sample_id"]
        data.cwe       = item["cwe"]
        return data


# ─────────────────────────────────────────────────────────────────────────────
# CFA-Aware Batch Sampler
# ─────────────────────────────────────────────────────────────────────────────

class CFAAwareBatchSampler(Sampler):
    """
    Keeps CFA pair groups in the same batch.

    Strategy:
      1. Group all sample indices by pair_id (singletons form size-1 groups).
      2. Shuffle GROUPS (not individual indices).
      3. Fill batches greedily from the group queue.
      4. If a group would exceed batch_size, yield the current batch first,
         then start a new batch with that group — even if the group itself
         is larger than batch_size (oversized groups are never split).

    The invariant — both (all) members of a pair_id always land in the same
    batch — is never violated.
    """

    def __init__(
        self,
        dataset: CFADataset,
        batch_size: int,
        drop_last: bool = True,
        source_weights=None,  # M2 stub: None in M1 → uniform sampling
        cwe_order=None,       # M2 stub: None in M1 → random group order
        pair_first: bool = True,  # Always True — kept for API compat
    ):
        self.batch_size = batch_size
        self.drop_last  = drop_last

        # Group dataset indices by pair_id
        pair_groups: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(dataset.index):
            pid = item["pair_id"]
            # Treat empty / sentinel pair_ids as singletons
            if pid in ("", "None", "null", "0"):
                key = f"__singleton_{i}"
            else:
                key = pid
            pair_groups[key].append(i)

        self.groups = list(pair_groups.values())

        # Stats for diagnostics
        self._n_groups    = len(self.groups)
        self._n_samples   = sum(len(g) for g in self.groups)
        self._max_group   = max(len(g) for g in self.groups) if self.groups else 0

    def __iter__(self):
        # Shuffle groups each epoch (not individual samples)
        groups = [g[:] for g in self.groups]  # shallow copy so shuffle is safe
        random.shuffle(groups)

        current_batch = []
        for group in groups:
            # If adding this group would overflow, yield first
            if current_batch and len(current_batch) + len(group) > self.batch_size:
                yield current_batch
                current_batch = []

            current_batch.extend(group)

            # If batch is exactly full (or oversized group), yield
            if len(current_batch) >= self.batch_size:
                yield current_batch
                current_batch = []

        # Remainder
        if current_batch and not self.drop_last:
            yield current_batch

    def __len__(self) -> int:
        # Estimate — exact count depends on group packing order, but this
        # gives the training loop a progress bar denominator.
        if self.drop_last:
            return self._n_samples // self.batch_size
        return (self._n_samples + self.batch_size - 1) // self.batch_size


# ─────────────────────────────────────────────────────────────────────────────
# Collate + DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def cfa_collate_fn(data_list: list[Data]):
    """
    Standard PyG batching via Batch.from_data_list.

    String attributes (pair_id, sample_id, cwe) are automatically
    collated into lists by PyG. No need to split orig/cfa here — the
    loss function handles pair extraction directly from the batch.
    """
    return Batch.from_data_list(data_list)


def count_batches(sampler: CFAAwareBatchSampler) -> int:
    """
    Count actual batches by iterating the sampler once.

    Use this for LR scheduler total_steps instead of len(sampler), which is
    an estimate (total_samples // batch_size). The real count can differ by
    ~3% due to pair-group packing. A 3% mismatch in warmup_steps is small
    but unnecessary.

    Call once at training start:
        true_steps = count_batches(sampler) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, true_steps)
    """
    return sum(1 for _ in sampler)


def build_dataloader(
    h5_path: str,
    batch_size: int = 32,
    drop_last: bool = True,
    num_workers: int = 0,
    **sampler_kwargs,
) -> tuple[DataLoader, CFADataset, CFAAwareBatchSampler]:
    """
    Build a CFA-aware DataLoader from an HDF5 split file.

    Args:
        h5_path:     path to train.h5 / val.h5 / test.h5
        batch_size:  target batch size (groups may cause slight overshoot)
        drop_last:   drop the final incomplete batch (default True for train)
        num_workers: 0 for Windows (h5py file handles can't be pickled across
                     processes). On Linux with SWMR or per-worker open, set >0.
        **sampler_kwargs: forwarded to CFAAwareBatchSampler (source_weights, etc.)

    Returns:
        (DataLoader, CFADataset, CFAAwareBatchSampler)
        Returns all three so the caller can:
          - Reuse dataset across epochs (avoids re-scanning HDF5 metadata)
          - Use count_batches(sampler) for accurate LR scheduler step counts
    """
    dataset = CFADataset(h5_path)
    sampler = CFAAwareBatchSampler(
        dataset, batch_size, drop_last=drop_last, **sampler_kwargs
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=cfa_collate_fn,
    )
    return loader, dataset, sampler
