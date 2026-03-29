# StreamGuard Stage 5 + 6: Google Colab GPU Guide

## Why Colab?

Stage 5 runs CodeBERT inference on every node statement across ~35K CPG files (~1.4M nodes total). On a local CPU this takes **~87 hours** (0.11 samples/s). On a Colab T4 GPU: **~2-4 hours** (batch_size=64, AMP fp16).

Stage 6 is CPU-only and fast (~42 graphs/s, ~15 min total), but runs in the same notebook for convenience.

---

## Prerequisites

| Item | Details |
|------|---------|
| Google account | With Google Drive (15 GB free is enough) |
| Google Colab | Free tier works; Pro gives longer runtimes |
| CPG data | `training/data/processed/cpg/` directory (34K+ JSON files, ~857 MB) |
| Notebook | `StreamGuard_Stage5_Stage6_Colab.ipynb` (in repo root) |

---

## Step-by-Step Process

### 1. Prepare the CPG Zip Locally

On your Windows machine, open a terminal in the project root:

```bash
cd "C:\Users\Vimal Sajan\streamguard"
python -c "import shutil; shutil.make_archive('cpg_data', 'zip', '.', 'training/data/processed/cpg')"
```

This creates `cpg_data.zip` (~200-400 MB compressed) in the project root. The zip preserves the shard directory structure (`00/`, `01/`, ..., `ff/`).

**Verify before uploading:**
```bash
python -c "import zipfile; z=zipfile.ZipFile('cpg_data.zip'); print(f'{len(z.namelist())} entries, {sum(i.file_size for i in z.infolist())/1024/1024:.0f} MB uncompressed')"
```

### 2. Upload to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Create folder: `My Drive` > `StreamGuard`
3. Upload `cpg_data.zip` into the `StreamGuard` folder
4. Wait for upload to complete (progress bar in bottom-right)

Expected Drive path: `My Drive/StreamGuard/cpg_data.zip`

### 3. Upload the Notebook to Colab

Option A (recommended):
1. Go to [Google Colab](https://colab.research.google.com)
2. File > Upload notebook
3. Select `StreamGuard_Stage5_Stage6_Colab.ipynb` from your local repo

Option B:
1. Upload the `.ipynb` file to `My Drive/StreamGuard/`
2. Double-click it in Drive — it opens in Colab automatically

### 4. Set GPU Runtime

1. In Colab: Runtime > Change runtime type
2. Hardware accelerator: **GPU** (select T4 if given a choice)
3. Click Save
4. The runtime restarts — this is expected

### 5. Run the Notebook

Run cells **in order** (Shift+Enter or Runtime > Run all):

| Cell | What it does | Expected time |
|------|-------------|---------------|
| Cell 1 | Verify GPU is available | instant |
| Cell 2 | Mount Google Drive | ~10s (auth popup) |
| Cell 3 | Clone repo from GitHub + install deps | 2-3 min |
| Cell 4 | Unzip CPG data to Colab's local SSD | 2-5 min |
| Cell 5a | Dry run (verify setup, 5 samples) | ~30s |
| Cell 5b | **Full Stage 5 run (all samples)** | **2-4 hours** |
| Cell 5c | Verify embeddings | instant |
| Cell 6a | Run Stage 6 (graph tensor assembly) | 5-10 min |
| Cell 6b | Verify HDF5 | instant |
| Cell 7 | Zip outputs + copy to Drive | 5-10 min |
| Cell 8 | Disk usage summary | instant |

**Total wall time: ~3-5 hours**

### 6. Monitor Progress

Stage 5 prints progress every 50 samples:
```
[Stage5] 50/34691 (0.1%) | 3.2 samples/s | ETA: 2h 58m
[Stage5] 100/34691 (0.3%) | 3.4 samples/s | ETA: 2h 49m
...
```

If the Colab runtime disconnects mid-run:
1. Reconnect (the "Reconnect" button appears automatically)
2. Re-run Cell 2 (re-mount Drive)
3. Re-run Cell 5b — it **automatically skips** already-embedded samples (checkpoint/resume via existing `.npz` files)

### 7. Download Results from Drive

After the notebook completes, your Drive `StreamGuard` folder contains:

| File | Size (est.) | Description |
|------|-------------|-------------|
| `embedded_data.zip` | 3,149.9 MB | All `.npz` embedding files |
| `all_graphs.h5` | 3,744.1 MB | Final HDF5 with graph tensors |
| `embed_stats.json` | <1 KB | Stage 5 statistics |
| `graph_stats.json` | <1 KB | Stage 6 statistics |

Download these to your local machine.

### 8. Integrate Results Locally

Embeddings extracted to: `training/data/processed/embedded_data/embedded/`
HDF5 copied to: `training/data/graphs/all_graphs.h5`

### 9. Verify Locally

```bash
# Verify embeddings
python -c "from training.scripts.preprocessing.stage5_embed import verify_embeddings; verify_embeddings('training/data/processed/embedded/', 5)"

# Verify HDF5
python -c "from training.scripts.preprocessing.stage6_graphs import verify_h5; verify_h5('training/data/graphs/all_graphs.h5', 5)"
```

Expected output:
```
Embedding file: <id>.npz
  features shape: (N, 824)   # N = number of nodes
  NaN check: OK
...
HDF5 file: all_graphs.h5
  Total graphs: ~34000+
  feature_dim: 824
  Sample graph: nodes=47, edges=222, label=1
```

---

## Key Points

### Runtime Limits
- **Free Colab**: ~12 hours max, may disconnect after ~90 min idle. Keep the browser tab active.
- **Colab Pro**: ~24 hours, less disconnection risk.
- **Checkpoint/resume**: Stage 5 skips existing `.npz` files, so disconnections don't lose progress. Just re-run the cell.

### Space Consumption on Colab
| Data | Approx Size |
|------|-------------|
| CPG zip (on Drive) | 200-400 MB |
| CPG extracted (local SSD) | ~857 MB |
| Embeddings (local SSD) | ~3.8 GB |
| HDF5 (local SSD) | ~3-4 GB |
| **Total local disk** | **~8 GB** |
| **Total Drive** | **~6-7 GB** |

Free Colab provides ~107 GB local disk and 15 GB Drive. This fits comfortably.

### GPU Memory
- CodeBERT base model: ~500 MB VRAM
- Batch of 64 tokenized sequences: ~200 MB VRAM
- **Peak VRAM: ~1-2 GB** (well within T4's 16 GB)
- AMP (fp16) is enabled automatically on GPU for faster inference

### What Gets Cloned from GitHub
The notebook clones the full repo from `https://github.com/VimalSajanGeorge/streamguard.git`. This provides `stage5_embed.py` and `stage6_graphs.py` with all dependencies. The CPG data is NOT in the repo — it comes from your Drive zip.

### Troubleshooting

| Issue | Fix |
|-------|-----|
| "No GPU detected" | Runtime > Change runtime type > GPU |
| "CPG zip not found" | Check Drive path: `My Drive/StreamGuard/cpg_data.zip` |
| Runtime disconnects | Re-mount Drive (Cell 2), re-run Stage 5 (auto-resumes) |
| Out of disk space | Free tier: clear `/content/` with `!rm -rf /content/streamguard` and re-run |
| Import errors | Re-run Cell 3 (install deps) |
| Slow speed (<1 sample/s) | Check GPU is active: `!nvidia-smi` — if "No devices found", change runtime to GPU |

---

## After Colab: Next Steps

With `embedded_data/` and `all_graphs.h5` on your local machine:

1. **Stage 7**: CFA-Aware Split (train/val/test partitioning)
2. **Story 8**: Model Training (GNN + Transformer)

---

## Files Reference

| File | Location | Purpose |
|------|----------|---------|
| `stage5_embed.py` | `training/scripts/preprocessing/` | Node embedding (824-d features) |
| `stage6_graphs.py` | `training/scripts/preprocessing/` | Graph tensor assembly (HDF5) |
| `StreamGuard_Stage5_Stage6_Colab.ipynb` | repo root | Colab notebook |
| `test_story6.py` | `tests/` | 68 unit/integration tests |
| `STORY6_STAGE5_STAGE6.md` | `docs/` | Story 6 completion report |
| This file | `docs/STORY6_COLAB_GUIDE.md` | Colab process guide |
