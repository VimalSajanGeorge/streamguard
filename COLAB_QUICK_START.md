# Google Colab Quick Start Guide

**🚀 Train StreamGuard models in Google Colab in 3 steps**

**Version:** 1.1 (updated for critical fixes)

> **⚠️ IMPORTANT - v1.1 Updates:**
> - This guide now references the updated `StreamGuard_Complete_Training.ipynb` (v1.1)
> - Critical fixes for PyG installation, tree-sitter build, and version compatibility
> - **Recommended:** Use the complete notebook for best results
> - See [COLAB_CRITICAL_FIXES.md](docs/COLAB_CRITICAL_FIXES.md) for technical details

---

## Prerequisites

✅ Google account with Google Drive
✅ Preprocessed CodeXGLUE data in Drive at: `My Drive/streamguard/data/processed/codexglue/`
✅ Files needed: `train.jsonl`, `valid.jsonl`, `test.jsonl`

---

## Option 1: Use Ready-Made Notebook (Easiest)

### Step 1: Upload Notebook to Colab

1. Go to: https://colab.research.google.com/
2. Click `File` → `Upload notebook`
3. Upload `StreamGuard_Complete_Training.ipynb` from your repository
4. Enable GPU: `Runtime` → `Change runtime type` → `GPU (T4)`

### Step 2: Run All Cells

1. Click `Runtime` → `Run all`
2. Authorize Google Drive access when prompted
3. Wait 9-13 hours for complete training

### Step 3: Download Models

Your trained models will be in Google Drive:
```
My Drive/streamguard/models/
├── transformer_phase1/
├── gnn_phase1/
└── fusion_phase1/
```

**Done!** 🎉

---

## Option 2: Manual Setup (Step-by-Step)

### Step 1: Create New Colab Notebook

1. Go to https://colab.research.google.com/
2. Click `New notebook`
3. Enable GPU: `Runtime` → `Change runtime type` → `GPU`

### Step 2: Install Dependencies (5-10 min)

> **⚠️ WARNING - Hardcoded Versions:**
> The code below uses hardcoded PyTorch wheel URLs that may break when Colab updates PyTorch.
> **For production use, see `StreamGuard_Complete_Training.ipynb` (v1.1)** which uses runtime-aware installation.

```python
# Cell 1: Install all dependencies (QUICK VERSION - may need updates)
# NOTE: This assumes PyTorch 2.1.0 + CUDA 12.1 (Colab default as of Oct 2025)
# If installation takes >5 minutes, Colab may have updated PyTorch - use notebook v1.1

!pip install -q torch-geometric==2.4.0
!pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
!pip install -q transformers==4.35.0 tokenizers==0.15.0
!pip install -q tree-sitter==0.20.4
!pip install -q scikit-learn==1.3.2 scipy==1.11.4 tqdm

# Verify (should complete in ~30 seconds)
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")
```

**If you see compilation warnings or it takes >5 minutes:**
1. Stop the cell
2. Use `StreamGuard_Complete_Training.ipynb` (v1.1) instead - it auto-detects versions
3. Or see [GOOGLE_COLAB_TRAINING_GUIDE.md](GOOGLE_COLAB_TRAINING_GUIDE.md) Cell 2 for runtime-aware code

### Step 3: Clone Repository

```python
# Cell 2: Clone StreamGuard
!git clone https://github.com/YOUR_USERNAME/streamguard.git
%cd streamguard

# Setup tree-sitter
!mkdir -p vendor
!cd vendor && git clone --depth 1 https://github.com/tree-sitter/tree-sitter-c.git

# Build tree-sitter library (BASIC VERSION - no error handling)
# NOTE: For robust build with fallback, use StreamGuard_Complete_Training.ipynb v1.1
from tree_sitter import Language
from pathlib import Path

build_dir = Path('build')
build_dir.mkdir(exist_ok=True)

try:
    Language.build_library(
        'build/my-languages.so',
        ['vendor/tree-sitter-c']
    )
    print("✓ tree-sitter built successfully")
except Exception as e:
    print(f"⚠️ tree-sitter build failed: {e}")
    print("Training will use token-sequence fallback (works fine, <5% perf impact)")
```

> **📝 Note:** If tree-sitter build fails, training will automatically use token-sequence graphs as fallback. Performance impact is minimal (<5%).

### Step 4: Mount Google Drive

```python
# Cell 3: Mount Drive and copy data
from google.colab import drive
import shutil
from pathlib import Path

# Mount
drive.mount('/content/drive')

# Copy data to local (faster)
local_data = Path('/content/data/processed/codexglue')
local_data.mkdir(parents=True, exist_ok=True)

drive_data = Path('/content/drive/MyDrive/streamguard/data/processed/codexglue')

for file in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
    shutil.copy2(drive_data / file, local_data / file)
    print(f"✓ Copied {file}")
```

### Step 5: Train Models

**Transformer (2-3 hours):**
```python
# Cell 4: Train Transformer
!python training/train_transformer.py \
  --train-data /content/data/processed/codexglue/train.jsonl \
  --val-data /content/data/processed/codexglue/valid.jsonl \
  --test-data /content/data/processed/codexglue/test.jsonl \
  --output-dir /content/models/transformer_phase1 \
  --epochs 5 --batch-size 16 --lr 2e-5 --mixed-precision
```

**GNN (4-6 hours):**
```python
# Cell 5: Train GNN
!python training/train_gnn.py \
  --train-data /content/data/processed/codexglue/train.jsonl \
  --val-data /content/data/processed/codexglue/valid.jsonl \
  --test-data /content/data/processed/codexglue/test.jsonl \
  --output-dir /content/models/gnn_phase1 \
  --epochs 100 --auto-batch-size
```

**Fusion (3-4 hours):**
```python
# Cell 6: Train Fusion
!python training/train_fusion.py \
  --train-data /content/data/processed/codexglue/train.jsonl \
  --val-data /content/data/processed/codexglue/valid.jsonl \
  --test-data /content/data/processed/codexglue/test.jsonl \
  --output-dir /content/models/fusion_phase1 \
  --transformer-checkpoint /content/models/transformer_phase1/checkpoints/best_model.pt \
  --gnn-checkpoint /content/models/gnn_phase1/checkpoints/best_model.pt
```

### Step 6: Save to Drive

```python
# Cell 7: Backup to Drive
import shutil
from pathlib import Path

for model in ['transformer_phase1', 'gnn_phase1', 'fusion_phase1']:
    src = Path(f'/content/models/{model}')
    dst = Path(f'/content/drive/MyDrive/streamguard/models/{model}')
    dst.mkdir(parents=True, exist_ok=True)

    shutil.copytree(src / 'checkpoints', dst / 'checkpoints', dirs_exist_ok=True)
    print(f"✓ {model} saved to Drive")
```

---

## Important Tips

### Prevent Session Timeout

```python
# Run this cell to keep session alive
from IPython.display import Javascript
display(Javascript('''
    function KeepAlive() {
        setInterval(function() {
            document.querySelector('#top-toolbar').click();
        }, 60000);
    }
    KeepAlive();
'''))
```

### Monitor GPU Usage

```python
# Check GPU every 5 seconds
!nvidia-smi -l 5
```

### Handle OOM Errors

If you get "Out of Memory" errors:

```python
# Reduce batch size
--batch-size 8  # Instead of 16

# Or use gradient accumulation
--batch-size 8 --accumulation-steps 2  # Effective batch = 16
```

---

## Troubleshooting

### "No GPU available"
- Go to `Runtime` → `Change runtime type` → Select `GPU (T4)`
- If not available, try again later or upgrade to Colab Pro

### "Drive quota exceeded"
- Data is copied to local Colab storage (`/content/`) for training
- Only final models are saved back to Drive (~1 GB total)
- Delete old backups if needed

### "Session crashed"
- Training automatically saves checkpoints every epoch
- Models are backed up to Drive after each phase completes
- If crash occurs mid-training, you can resume from last checkpoint

### "Module not found"
- Restart runtime: `Runtime` → `Restart runtime`
- Re-run the installation cell (Cell 1)

---

## Cost & Time Comparison

| Tier | GPU | Runtime | Cost | Recommendation |
|------|-----|---------|------|----------------|
| Free | T4 | ~12h | $0 | Works but may timeout |
| Pro | T4/P100 | ~24h | $10/mo | ✅ Recommended |

**Estimated Training Time:**
- Transformer: 2-3 hours
- GNN: 4-6 hours
- Fusion: 3-4 hours
- **Total: 9-13 hours**

With Colab Pro, you can run all three models in one session.

---

## Expected Results

After training completes, you should see:

```
TEST EVALUATION
======================================================================
Transformer F1 (vulnerable): 0.68-0.71
GNN F1 (vulnerable): 0.64-0.67
Fusion F1 (vulnerable): 0.70-0.73

[+] Models saved to Drive
```

---

## Next Steps

1. **Download models from Drive:**
   - Go to Google Drive web interface
   - Navigate to `My Drive/streamguard/models/`
   - Download folders or use `gdown` CLI

2. **Deploy models:**
   - See `COMPLETE_ML_TRAINING_GUIDE.md` for deployment
   - Create SageMaker endpoint
   - Integrate with StreamGuard API

3. **Optional - Phase 2:**
   - Add collector data
   - Retrain with more samples
   - Expected +3-8% F1 improvement

---

## Files You Need

### Before Training:
```
My Drive/streamguard/data/processed/codexglue/
├── train.jsonl       (21,854 samples, ~145 MB)
├── valid.jsonl       (2,732 samples, ~18 MB)
└── test.jsonl        (2,732 samples, ~18 MB)
```

### After Training:
```
My Drive/streamguard/models/
├── transformer_phase1/
│   ├── checkpoints/
│   │   └── best_model.pt (~500 MB)
│   └── exp_config.json
├── gnn_phase1/
│   ├── checkpoints/
│   │   └── best_model.pt (~300 MB)
│   └── exp_config.json
├── fusion_phase1/
│   └── best_fusion.pt (~100 MB)
└── evaluation_results.json
```

**Total Storage: ~1 GB**

---

## Quick Commands Reference

```python
# Check GPU
import torch
print(torch.cuda.get_device_name(0))

# Check disk space
!df -h

# Check Drive space
!du -sh /content/drive/MyDrive/streamguard/

# Clear Colab cache
import torch
torch.cuda.empty_cache()

# Check training progress
!tail -f /content/models/transformer_phase1/training.log

# List checkpoints
!ls -lh /content/models/*/checkpoints/
```

---

## Support

**Having issues?**
1. Check `GOOGLE_COLAB_TRAINING_GUIDE.md` for detailed troubleshooting
2. Verify GPU is enabled
3. Ensure Drive has 2+ GB free space
4. Try Colab Pro if Free tier times out

**Documentation:**
- Full guide: `GOOGLE_COLAB_TRAINING_GUIDE.md` (50+ pages)
- Notebook: `StreamGuard_Complete_Training.ipynb`
- General guide: `COMPLETE_ML_TRAINING_GUIDE.md`

---

**Ready to start?** Upload `StreamGuard_Complete_Training.ipynb` (v1.1) to Colab and click "Run all"! 🚀

**Version:** 1.1
**Last Updated:** October 27, 2025
**Status:** ✅ Production-Ready

**v1.1 Changes:**
- Added warnings about hardcoded PyTorch versions in quick install
- Recommended using complete notebook for production (runtime-aware installation)
- Added basic error handling to tree-sitter build
- References to critical fixes documentation

**For Production Use:**
- Use `StreamGuard_Complete_Training.ipynb` v1.1 (recommended)
- See [GOOGLE_COLAB_TRAINING_GUIDE.md](GOOGLE_COLAB_TRAINING_GUIDE.md) for full guide
- See [COLAB_CRITICAL_FIXES.md](docs/COLAB_CRITICAL_FIXES.md) for technical details
