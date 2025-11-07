"""
Pre-Flight Validation for StreamGuard A100 Production Training

Runs all critical checks before starting expensive GPU training:
✅ GPU detection
✅ Safety utilities test suite
✅ Tiny-overfit smoke tests
✅ Single-batch memory tests
✅ Data availability
✅ LR cache readiness

Run this script BEFORE cell_51-53 to catch issues early.

Usage:
    python training/scripts/pre_flight_validation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import warnings
from datetime import datetime


def check_gpu():
    """Check GPU availability and specs."""
    print("\n" + "=" * 80)
    print("1. GPU DETECTION")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        print("   → Training will run on CPU (VERY SLOW)")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"✅ GPU detected: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f} GB")

    # Check if A100
    if "A100" in gpu_name:
        print("   ✨ A100 detected - optimal configuration will be used")
    else:
        print(f"   ℹ️  Not an A100 - adaptive config will adjust batch sizes")

    return True


def check_safety_utilities():
    """Run safety utilities test suite."""
    print("\n" + "=" * 80)
    print("2. SAFETY UTILITIES TEST SUITE")
    print("=" * 80)

    try:
        from training.tests import test_safety_utilities

        print("[*] Running test suite...")
        exit_code = test_safety_utilities.run_all_tests()

        if exit_code == 0:
            print("✅ All safety utilities tests passed")
            return True
        else:
            print("❌ Some safety tests failed")
            print("   → Review test output above")
            return False

    except Exception as e:
        print(f"❌ Failed to run safety tests: {str(e)}")
        return False


def check_smoke_tests():
    """Run tiny-overfit smoke tests."""
    print("\n" + "=" * 80)
    print("3. TINY-OVERFIT SMOKE TESTS")
    print("=" * 80)

    try:
        from training.tests import test_overfit_smoke

        print("[*] Running smoke tests...")
        exit_code = test_overfit_smoke.run_smoke_tests()

        if exit_code == 0:
            print("✅ All smoke tests passed")
            return True
        else:
            print("❌ Smoke tests failed")
            print("   → Models cannot overfit on tiny data")
            print("   → DO NOT proceed to production training")
            return False

    except Exception as e:
        print(f"❌ Failed to run smoke tests: {str(e)}")
        return False


def check_data_availability():
    """Check that training data exists."""
    print("\n" + "=" * 80)
    print("4. DATA AVAILABILITY")
    print("=" * 80)

    checks = []

    # Transformer data
    train_jsonl = Path("data/processed/codexglue/train.jsonl")
    val_jsonl = Path("data/processed/codexglue/val.jsonl")

    if train_jsonl.exists():
        size_mb = train_jsonl.stat().st_size / 1024**2
        print(f"✅ Transformer train data: {train_jsonl} ({size_mb:.0f} MB)")
        checks.append(True)
    else:
        print(f"❌ Transformer train data missing: {train_jsonl}")
        checks.append(False)

    if val_jsonl.exists():
        size_mb = val_jsonl.stat().st_size / 1024**2
        print(f"✅ Transformer val data: {val_jsonl} ({size_mb:.0f} MB)")
        checks.append(True)
    else:
        print(f"❌ Transformer val data missing: {val_jsonl}")
        checks.append(False)

    # Graph data
    train_graphs = Path("data/processed/graphs/train")
    val_graphs = Path("data/processed/graphs/val")

    if train_graphs.exists():
        num_files = len(list(train_graphs.glob("*.pt")))
        print(f"✅ GNN train graphs: {train_graphs} ({num_files} files)")
        checks.append(True)
    else:
        print(f"❌ GNN train graphs missing: {train_graphs}")
        print(f"   → Run: python training/preprocessing/create_simple_graph_data.py")
        checks.append(False)

    if val_graphs.exists():
        num_files = len(list(val_graphs.glob("*.pt")))
        print(f"✅ GNN val graphs: {val_graphs} ({num_files} files)")
        checks.append(True)
    else:
        print(f"❌ GNN val graphs missing: {val_graphs}")
        print(f"   → Run: python training/preprocessing/create_simple_graph_data.py")
        checks.append(False)

    return all(checks)


def check_output_directories():
    """Create output directories if needed."""
    print("\n" + "=" * 80)
    print("5. OUTPUT DIRECTORIES")
    print("=" * 80)

    dirs = [
        Path("training/outputs/transformer_v17_production"),
        Path("training/outputs/gnn_v17_production"),
        Path("training/outputs/fusion_v17_production"),
        Path("training/outputs/production_summary")
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"✅ {d}")

    return True


def check_lr_cache():
    """Check LR cache configuration."""
    print("\n" + "=" * 80)
    print("6. LR FINDER CACHE")
    print("=" * 80)

    cache_dirs = [
        Path("models/transformer/.lr_cache"),
        Path("models/gnn/.lr_cache"),
        Path("models/fusion/.lr_cache")
    ]

    for cache_dir in cache_dirs:
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing cache
        cache_files = list(cache_dir.glob("*.json"))
        if cache_files:
            print(f"ℹ️  {cache_dir}: {len(cache_files)} cached LR(s)")
        else:
            print(f"✅ {cache_dir}: Ready (no cache yet)")

    print("\n   LR Finder will cache results for 168 hours (7 days)")
    return True


def main():
    """Run all pre-flight checks."""
    print("\n" + "=" * 80)
    print("STREAMGUARD PRE-FLIGHT VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("\nRunning critical checks before A100 production training...")

    results = {
        "gpu": False,
        "safety_tests": False,
        "smoke_tests": False,
        "data": False,
        "directories": False,
        "lr_cache": False
    }

    # Run all checks
    results["gpu"] = check_gpu()
    results["safety_tests"] = check_safety_utilities()
    results["smoke_tests"] = check_smoke_tests()
    results["data"] = check_data_availability()
    results["directories"] = check_output_directories()
    results["lr_cache"] = check_lr_cache()

    # Summary
    print("\n" + "=" * 80)
    print("PRE-FLIGHT VALIDATION SUMMARY")
    print("=" * 80)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:9} - {check.replace('_', ' ').title()}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)

    if all_passed:
        print("🚀 ALL CHECKS PASSED - READY FOR PRODUCTION TRAINING!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Run: python training/scripts/cell_51_transformer_production.py")
        print("  2. Run: python training/scripts/cell_52_gnn_production.py")
        print("  3. Run: python training/scripts/cell_53_fusion_production.py")
        print("  4. Run: python training/scripts/cell_54_metrics_aggregator.py")
        print("\nOr add these to your Jupyter notebook as cells 51-54.\n")
        return 0

    else:
        print("❌ PRE-FLIGHT CHECKS FAILED!")
        print("=" * 80)
        print("\nDO NOT proceed with production training until all checks pass.")
        print("Review the failures above and fix them first.\n")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n[!] Pre-flight validation crashed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
