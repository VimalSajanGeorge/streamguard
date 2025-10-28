# StreamGuard - Production Fixes & Best Practices

**Date:** October 14, 2025
**Status:** ✅ Production Ready with Safety Features

---

## Table of Contents

1. [Critical Fixes Applied](#critical-fixes-applied)
2. [Graceful Shutdown](#graceful-shutdown)
3. [Dashboard Crash Recovery](#dashboard-crash-recovery)
4. [Windows Multiprocessing](#windows-multiprocessing)
5. [Environment Variables](#environment-variables)
6. [Testing Guide](#testing-guide)
7. [Troubleshooting](#troubleshooting)

---

## Critical Fixes Applied

### ✅ Fix 1: Environment Variable Loading in Subprocesses

**Problem:** `.env` files don't auto-load in `multiprocessing.Process` subprocesses

**Solution:** Triple redundancy approach
- `load_dotenv()` in parent process ([run_full_collection.py:219](../training/scripts/collection/run_full_collection.py#L219))
- `load_dotenv()` in each subprocess worker ([master_orchestrator.py:88](../training/scripts/collection/master_orchestrator.py#L88))
- Token passed explicitly as parameter to collectors

```python
# Parent process
load_dotenv()  # Line 219

# Subprocess worker
from dotenv import load_dotenv
load_dotenv()  # Line 88
```

### ✅ Fix 2: Windows Multiprocessing Guard

**Problem:** Windows spawns new processes that can cause recursive imports

**Solution:** `if __name__ == '__main__'` guard already present
```python
if __name__ == '__main__':
    sys.exit(main())  # Line 290
```

### ✅ Fix 3: Graceful Shutdown with Signal Handling

**Problem:** Ctrl+C during long runs loses all progress

**Solution:** SIGINT/SIGTERM handlers that save partial progress
```python
signal.signal(signal.SIGINT, graceful_shutdown)   # Line 222
signal.signal(signal.SIGTERM, graceful_shutdown)  # Line 224
```

### ✅ Fix 4: Partial Progress Checkpointing

**Problem:** Crashes lose hours of collection work

**Solution:** `save_partial_results()` method saves:
- Partial collection statistics
- What data files were created
- Progress status for each collector
- Timestamped checkpoint files

---

## Graceful Shutdown

### How It Works

When you press **Ctrl+C** during collection:

1. **Signal Handler Triggered**
   - `SIGINT` caught by `graceful_shutdown()` function
   - Prevents immediate termination

2. **Save Partial Progress**
   - Calls `orchestrator.save_partial_results()`
   - Creates timestamped checkpoint: `collection_partial_YYYYMMDD_HHMMSS.json`
   - Lists all data files created so far

3. **Clean Termination**
   - Prints summary of what was collected
   - Exits with code 130 (standard Ctrl+C code)

### Example Shutdown Output

```
======================================================================
⚠ SHUTDOWN SIGNAL RECEIVED
======================================================================

Attempting graceful shutdown...
Saving partial progress and cleaning up...

Saving partial results...
✓ Partial results saved to: data/raw/collection_partial_20251014_153045.json

Data files created:
  cve: 3 file(s), 45.23 MB
  github: 2 file(s), 28.15 MB
  repo: 1 file(s), 12.50 MB
  synthetic: 1 file(s), 5.00 MB

Partial Collection Summary:
======================================================================
Total Duration: 3245.2s
Mode: Parallel

Collectors: 2/4 successful
Total Samples: 12,500/50,000 (25.0%)
...
======================================================================
Graceful shutdown complete.
======================================================================

Partial results saved. You can:
  1. Resume collection with the same parameters
  2. Run with --no-dashboard if VS Code crashed
  3. Check data/raw/ for collected samples
```

### Testing Graceful Shutdown

```bash
# Start collection
python training/scripts/collection/run_full_collection.py --quick-test

# Wait 30 seconds, then press Ctrl+C

# Verify partial results saved
ls data/raw/collection_partial_*.json
```

---

## Dashboard Crash Recovery

### Problem: Rich Dashboard Can Crash VS Code

The Rich library's live dashboard uses terminal control sequences that can sometimes crash VS Code's integrated terminal, especially during long-running collections.

### Solution 1: Disable Dashboard

If VS Code crashes during collection:

```bash
# Resume with dashboard disabled
python training/scripts/collection/run_full_collection.py --no-dashboard
```

**Benefits:**
- ✅ More stable on VS Code
- ✅ Lower memory usage
- ✅ Can redirect to log files
- ✅ Works in any terminal

**Output:**
```
Starting parallel collection...

✓ Starting cve collector...
✓ Starting github collector...
✓ Starting repo collector...
✓ Starting synthetic collector...

Monitoring progress...

[10:15:30] cve: Initializing CVE collector...
[10:15:31] cve: CVE collector started
[10:15:32] github: Initializing GitHub collector...
...
```

### Solution 2: Run in External Terminal

```bash
# Open PowerShell/CMD outside VS Code
cd "C:\Users\Vimal Sajan\streamguard"
python training/scripts/collection/run_full_collection.py
```

### Solution 3: Sequential Mode

If parallel mode causes issues:

```bash
python training/scripts/collection/run_full_collection.py --sequential
```

**Trade-offs:**
- ✅ More stable
- ✅ Lower memory (2GB vs 4GB)
- ❌ Slower (15-24 hours vs 6-10 hours)

### Solution 4: Use tmux/screen (Linux/Mac)

```bash
# Start tmux session
tmux new -s streamguard

# Run collection
python training/scripts/collection/run_full_collection.py

# Detach: Ctrl+B, then D
# Re-attach later: tmux attach -t streamguard
```

---

## Windows Multiprocessing

### Why `if __name__ == '__main__'` Matters

On Windows, Python uses **spawn** instead of **fork** for multiprocessing:

```python
# Without guard (BAD - recursive spawning!)
orchestrator = MasterOrchestrator(...)
orchestrator.run_collection()

# With guard (GOOD)
if __name__ == '__main__':
    orchestrator = MasterOrchestrator(...)
    orchestrator.run_collection()
```

**What happens without the guard:**
1. Parent process imports module
2. Spawns child process
3. Child process imports module AGAIN
4. Child spawns another child
5. **Infinite recursion!**

### Verification

Your code already has the guard:

**File:** [run_full_collection.py:290](../training/scripts/collection/run_full_collection.py#L290)
```python
if __name__ == '__main__':
    sys.exit(main())
```

**File:** [master_orchestrator.py:574](../training/scripts/collection/master_orchestrator.py#L574)
```python
if __name__ == '__main__':
    # Example usage
    ...
```

✅ **All entry points are protected!**

---

## Environment Variables

### The Multiprocessing Environment Variable Problem

**Issue:** Subprocesses on different platforms handle environment variables differently:

| Platform | Behavior | `.env` Inherited? |
|----------|----------|-------------------|
| Windows | Parent loads `.env`, children inherit `os.environ` | ✅ Usually Yes |
| Linux (fork) | Children inherit parent's entire memory | ✅ Yes |
| Linux (spawn) | Fresh interpreter, no inheritance | ❌ No |
| macOS | Depends on Python version and spawn method | ⚠️ Maybe |

### Our Solution: Defense in Depth

**Layer 1:** Load in parent process
```python
# run_full_collection.py:219
load_dotenv()
```

**Layer 2:** Load in each subprocess
```python
# master_orchestrator.py:88 (in _collect_worker)
from dotenv import load_dotenv
load_dotenv()
```

**Layer 3:** Pass explicitly as parameter
```python
# master_orchestrator.py:110
collector = collector_class(
    output_dir=output_dir,
    cache_enabled=config.get('cache_enabled', True),
    github_token=config.get('github_token')  # Explicit parameter
)
```

**Layer 4:** Fallback to environment
```python
# github_advisory_collector_enhanced.py:64
self.github_token = github_token or os.getenv("GITHUB_TOKEN")
```

### Testing Environment Loading

Run the test script:
```bash
cd "C:\Users\Vimal Sajan\streamguard"
python training/scripts/collection/test_env_loading.py
```

Expected output:
```
[Test 1] Parent process BEFORE load_dotenv():
  GITHUB_TOKEN present: False

[Test 2] Parent process AFTER load_dotenv():
  GITHUB_TOKEN present: True
  Token preview: ghp_ygkzzz...

[Test 3] Subprocess WITHOUT load_dotenv():
  GITHUB_TOKEN present: True  # Windows inherits from parent

[Test 4] Subprocess WITH load_dotenv():
  GITHUB_TOKEN present: True

[SUCCESS] Environment variables loaded correctly!
```

---

## Testing Guide

### Test 1: Quick Smoke Test

```bash
# Test all systems (100 samples each, ~10 minutes)
python training/scripts/collection/run_full_collection.py --quick-test
```

**What to verify:**
- ✅ All 4 collectors start
- ✅ Progress dashboard appears
- ✅ No errors in output
- ✅ Data files created in `data/raw/`

### Test 2: Graceful Shutdown

```bash
# Start quick test
python training/scripts/collection/run_full_collection.py --quick-test

# After 30 seconds, press Ctrl+C

# Verify:
ls data/raw/collection_partial_*.json  # Checkpoint created
cat data/raw/collection_partial_*.json  # Valid JSON
```

### Test 3: No Dashboard Mode

```bash
# Run without Rich dashboard
python training/scripts/collection/run_full_collection.py --quick-test --no-dashboard
```

**Expected:**
- ✅ Simple text output
- ✅ No terminal control codes
- ✅ More stable in VS Code

### Test 4: Sequential Mode

```bash
# Run collectors one at a time
python training/scripts/collection/run_full_collection.py --quick-test --sequential
```

**Expected:**
- ✅ One collector at a time
- ✅ Lower memory usage
- ✅ Easier to debug

### Test 5: Individual Collector

```bash
# Test just synthetic (fastest)
python training/scripts/collection/run_full_collection.py --collectors synthetic

# Test just CVE and GitHub
python training/scripts/collection/run_full_collection.py --collectors cve github --quick-test
```

---

## Troubleshooting

### Issue: "Can't pickle local object"

**Cause:** Windows multiprocessing can't serialize local functions

**Solution:** ✅ Fixed - all worker functions are module-level or static methods

---

### Issue: "GITHUB_TOKEN environment variable is required"

**Cause:** `.env` not loaded or token missing

**Solution:**
```bash
# Check .env file exists
ls .env

# Check token is set
type .env | findstr GITHUB_TOKEN

# Test loading
python training/scripts/collection/test_env_loading.py
```

---

### Issue: VS Code terminal crashes during collection

**Cause:** Rich dashboard terminal codes overwhelm VS Code

**Solution:**
```bash
# Option 1: Disable dashboard
python training/scripts/collection/run_full_collection.py --no-dashboard

# Option 2: Run in external terminal
# Open PowerShell/CMD outside VS Code
cd "C:\Users\Vimal Sajan\streamguard"
python training/scripts/collection/run_full_collection.py
```

---

### Issue: Collection hangs indefinitely

**Cause:** One collector blocked on API rate limit or network

**Debug:**
```bash
# Check which collectors are running
tasklist | findstr python

# Check network connections
netstat -ano | findstr :443

# Kill and restart with sequential mode
python training/scripts/collection/run_full_collection.py --sequential --no-dashboard
```

---

### Issue: Out of memory

**Cause:** 4 parallel collectors + Rich dashboard = ~4-6GB RAM

**Solution:**
```bash
# Option 1: Sequential mode (lower memory)
python training/scripts/collection/run_full_collection.py --sequential

# Option 2: Reduce sample targets
python training/scripts/collection/run_full_collection.py \
    --cve-samples 5000 \
    --github-samples 3000 \
    --repo-samples 10000 \
    --synthetic-samples 2000
```

---

### Issue: Partial data after crash

**Recovery:**
```bash
# Check what was collected
dir data\raw\cves\*.jsonl
dir data\raw\github\*.jsonl
dir data\raw\opensource\*.jsonl
dir data\raw\synthetic\*.jsonl

# Check partial results file
type data\raw\collection_partial_*.json

# Resume collection (collectors skip existing data via caching)
python training/scripts/collection/run_full_collection.py
```

---

## Best Practices for Long Runs

### 1. Start with Quick Test

```bash
# Always test first!
python training/scripts/collection/run_full_collection.py --quick-test
```

### 2. Use External Terminal for Full Collection

Don't run 6-10 hour collections in VS Code integrated terminal.

```bash
# PowerShell or CMD outside VS Code
cd "C:\Users\Vimal Sajan\streamguard"
python training/scripts/collection/run_full_collection.py
```

### 3. Consider Disabling Dashboard for Long Runs

```bash
# More stable for overnight runs
python training/scripts/collection/run_full_collection.py --no-dashboard > collection.log 2>&1
```

### 4. Monitor Progress from Another Terminal

```bash
# Terminal 1: Run collection
python training/scripts/collection/run_full_collection.py --no-dashboard

# Terminal 2: Watch progress
dir data\raw\cves\*.jsonl
dir data\raw\github\*.jsonl
# Repeat every 10 minutes
```

### 5. Keep System Awake

```powershell
# PowerShell: Prevent sleep during collection
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 30

# After collection completes:
powercfg /change standby-timeout-ac 30
powercfg /change monitor-timeout-ac 10
```

---

## Summary Checklist

Before running full collection (50K samples, 6-10 hours):

- [ ] ✅ `.env` file has `GITHUB_TOKEN=your_token`
- [ ] ✅ Run quick test first (`--quick-test`)
- [ ] ✅ Quick test succeeded
- [ ] ✅ Enough disk space (~2-5GB)
- [ ] ✅ Enough RAM (~4-6GB for parallel, ~2GB for sequential)
- [ ] ✅ Running in external terminal (not VS Code)
- [ ] ✅ System set to not sleep
- [ ] ✅ Know how to press Ctrl+C for graceful shutdown

**Ready to go!** 🚀

```bash
python training/scripts/collection/run_full_collection.py
```

---

**Version:** 2.0
**Last Updated:** October 14, 2025
**Status:** ✅ Production Ready with Safety Features
