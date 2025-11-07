"""
Test Suite for Production Safety Utilities

Tests all critical safety components before production training:
- JSON serialization safety
- Adaptive GPU configuration
- Collapse detection
- AMP-safe gradient clipping

Run: python training/tests/test_safety_utilities.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest
import tempfile
import json
import numpy as np

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[!] PyTorch not available. Some tests will be skipped.")

# Import safety utilities
from training.utils.json_safety import (
    safe_jsonify, atomic_write_json, validate_json_safe, load_json_safe
)
from training.utils.adaptive_config import (
    detect_gpu, load_adaptive_config, adjust_batch_size_for_memory
)
from training.utils.collapse_detector import CollapseDetector
from training.utils.amp_utils import (
    clip_gradients_amp_safe, check_gradients_health, safe_backward
)


class TestJSONSafety(unittest.TestCase):
    """Test JSON serialization utilities."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        # Clean up temp files
        for f in self.temp_dir.glob("*.json"):
            f.unlink()
        self.temp_dir.rmdir()

    def test_safe_jsonify_primitives(self):
        """Test safe_jsonify with primitive types."""
        data = {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "bool": True,
            "none": None
        }
        result = safe_jsonify(data)
        self.assertEqual(result, data)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_safe_jsonify_tensors(self):
        """Test safe_jsonify with PyTorch tensors."""
        data = {
            "scalar": torch.tensor(0.5),
            "vector": torch.tensor([1.0, 2.0, 3.0]),
            "matrix": torch.tensor([[1, 2], [3, 4]])
        }
        result = safe_jsonify(data)

        self.assertAlmostEqual(result["scalar"], 0.5)
        self.assertEqual(result["vector"], [1.0, 2.0, 3.0])
        self.assertEqual(result["matrix"], [[1, 2], [3, 4]])

    def test_safe_jsonify_numpy(self):
        """Test safe_jsonify with NumPy arrays."""
        data = {
            "scalar": np.array(0.5),
            "vector": np.array([1, 2, 3]),
            "matrix": np.array([[1, 2], [3, 4]])
        }
        result = safe_jsonify(data)

        self.assertAlmostEqual(result["scalar"], 0.5)
        self.assertEqual(result["vector"], [1, 2, 3])
        self.assertEqual(result["matrix"], [[1, 2], [3, 4]])

    def test_safe_jsonify_path(self):
        """Test safe_jsonify with Path objects."""
        data = {"path": Path("/tmp/model.pt")}
        result = safe_jsonify(data)
        self.assertEqual(result["path"], "/tmp/model.pt")

    def test_atomic_write_json(self):
        """Test atomic JSON writing."""
        data = {
            "epoch": 10,
            "loss": 0.5
        }
        if TORCH_AVAILABLE:
            data["tensor"] = torch.tensor(0.3)

        output_file = self.temp_dir / "test_metadata.json"
        atomic_write_json(data, output_file)

        # Verify file exists
        self.assertTrue(output_file.exists())

        # Load and verify
        loaded = load_json_safe(output_file)
        self.assertEqual(loaded["epoch"], 10)
        self.assertAlmostEqual(loaded["loss"], 0.5)
        if TORCH_AVAILABLE:
            self.assertAlmostEqual(loaded["tensor"], 0.3)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_validate_json_safe(self):
        """Test JSON validation."""
        # Valid data
        valid = {"epoch": 10, "loss": 0.5}
        errors = validate_json_safe(valid)
        self.assertEqual(len(errors), 0)

        # Invalid data (tensor)
        invalid = {"tensor": torch.tensor(0.5)}
        errors = validate_json_safe(invalid)
        self.assertGreater(len(errors), 0)


class TestAdaptiveConfig(unittest.TestCase):
    """Test adaptive GPU configuration."""

    def test_detect_gpu(self):
        """Test GPU detection."""
        gpu_info = detect_gpu()

        self.assertIn("name", gpu_info)
        self.assertIn("memory_gb", gpu_info)
        self.assertIn("device", gpu_info)
        self.assertIn("count", gpu_info)

        # Should be either CPU or CUDA
        self.assertIn(gpu_info["device"], ["cpu", "cuda"])

    def test_load_adaptive_config_transformer(self):
        """Test loading transformer config."""
        config = load_adaptive_config(model_type="transformer")

        # Check required keys
        self.assertIn("batch_size", config)
        self.assertIn("mixed_precision", config)
        self.assertIn("gpu_info", config)
        self.assertIn("model_type", config)

        self.assertEqual(config["model_type"], "transformer")
        self.assertIsInstance(config["batch_size"], int)
        self.assertGreater(config["batch_size"], 0)

    def test_load_adaptive_config_gnn(self):
        """Test loading GNN config (smaller batch size)."""
        config = load_adaptive_config(model_type="gnn")

        # GNN should have smaller batch size than transformer
        transformer_config = load_adaptive_config(model_type="transformer")
        self.assertLessEqual(config["batch_size"], transformer_config["batch_size"])

    def test_load_adaptive_config_fusion(self):
        """Test loading fusion config (smallest batch size)."""
        config = load_adaptive_config(model_type="fusion")

        # Fusion should have smallest batch size
        self.assertLessEqual(config["batch_size"], 32)

    def test_adjust_batch_size_for_memory(self):
        """Test memory-based batch size adjustment."""
        # Large memory - should keep base size
        adjusted = adjust_batch_size_for_memory(64, 40.0)
        self.assertEqual(adjusted, 64)

        # Small memory - should reduce
        adjusted = adjust_batch_size_for_memory(64, 8.0)
        self.assertLess(adjusted, 64)

    def test_config_override(self):
        """Test configuration override."""
        override = {"batch_size": 128, "custom_param": "test"}
        config = load_adaptive_config(model_type="transformer", override=override)

        self.assertEqual(config["batch_size"], 128)
        self.assertEqual(config["custom_param"], "test")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestCollapseDetector(unittest.TestCase):
    """Test collapse detection system."""

    def setUp(self):
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        # Clean up
        for f in self.temp_dir.glob("*.json"):
            f.unlink()
        self.temp_dir.rmdir()

    def test_collapse_detector_init(self):
        """Test collapse detector initialization."""
        detector = CollapseDetector(
            window_size=5,
            collapse_threshold=3,
            enable_auto_stop=True
        )

        self.assertEqual(detector.window_size, 5)
        self.assertEqual(detector.collapse_threshold, 3)
        self.assertTrue(detector.enable_auto_stop)
        self.assertFalse(detector.collapsed)

    def test_normal_training_no_collapse(self):
        """Test that normal training doesn't trigger collapse."""
        detector = CollapseDetector(collapse_threshold=3)

        # Simulate normal training
        for step in range(10):
            x = torch.randn(32, 10)
            y = torch.randint(0, 2, (32,))
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()

            result = detector.step(self.model, loss.item(), logits, y, step)

            self.assertFalse(result["collapse_detected"])
            self.assertFalse(result["should_stop"])

            self.model.zero_grad()

    def test_zero_gradient_collapse(self):
        """Test detection of zero gradients."""
        detector = CollapseDetector(
            collapse_threshold=2,
            grad_norm_epsilon=1e-7
        )

        # Zero all gradients manually
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param)

        x = torch.randn(32, 10)
        logits = self.model(x)

        # Should detect collapse
        result = detector.step(self.model, 0.5, logits)
        self.assertTrue(result["collapse_detected"])

    def test_collapse_auto_stop(self):
        """Test that auto-stop triggers after threshold."""
        detector = CollapseDetector(
            collapse_threshold=3,
            enable_auto_stop=True,
            grad_norm_epsilon=1e-7
        )

        # Trigger collapse multiple times
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param)

        x = torch.randn(32, 10)
        logits = self.model(x)

        for step in range(5):
            result = detector.step(self.model, 0.5, logits, step=step)

            if step >= 2:  # After 3 consecutive collapses
                self.assertTrue(result["should_stop"])
                break

    def test_collapse_report_save(self):
        """Test that collapse report is saved."""
        report_path = self.temp_dir / "collapse_report.json"
        detector = CollapseDetector(
            collapse_threshold=2,
            enable_auto_stop=True,
            report_path=report_path
        )

        # Trigger collapse
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param)

        x = torch.randn(32, 10)
        logits = self.model(x)

        for step in range(3):
            detector.step(self.model, 0.5, logits, step=step)

        # Should have triggered auto-stop and saved report
        if detector.collapsed:
            self.assertTrue(report_path.exists())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAMPUtils(unittest.TestCase):
    """Test AMP-safe utilities."""

    def setUp(self):
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def test_clip_gradients_without_amp(self):
        """Test gradient clipping without AMP."""
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()

        stats = clip_gradients_amp_safe(
            self.model,
            max_grad_norm=1.0,
            scaler=None
        )

        self.assertIn("total_norm", stats)
        self.assertIn("clipped", stats)
        self.assertIsInstance(stats["total_norm"], float)
        self.assertIsInstance(stats["clipped"], bool)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_clip_gradients_with_amp(self):
        """Test gradient clipping with AMP."""
        device = torch.device("cuda")
        model = self.model.to(device)
        scaler = GradScaler()

        x = torch.randn(32, 10).to(device)
        y = torch.randint(0, 2, (32,)).to(device)

        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

        scaler.scale(loss).backward()

        # Must store optimizer in model for unscaling
        model.optimizer = self.optimizer

        stats = clip_gradients_amp_safe(
            model,
            max_grad_norm=1.0,
            scaler=scaler
        )

        self.assertIn("scale", stats)
        self.assertGreater(stats["scale"], 0)

    def test_check_gradients_health(self):
        """Test gradient health checking."""
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()

        health = check_gradients_health(self.model)

        self.assertIn("has_nan", health)
        self.assertIn("has_inf", health)
        self.assertIn("has_zero_grad", health)
        self.assertIn("total_params", health)

        self.assertFalse(health["has_nan"])
        self.assertFalse(health["has_inf"])

    def test_safe_backward(self):
        """Test safe backward pass."""
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)

        success = safe_backward(loss)
        self.assertTrue(success)

        # Test with NaN loss
        nan_loss = torch.tensor(float('nan'))
        success = safe_backward(nan_loss)
        self.assertFalse(success)


def run_all_tests():
    """Run all test suites."""
    print("\n" + "=" * 80)
    print("STREAMGUARD SAFETY UTILITIES TEST SUITE")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestJSONSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveConfig))

    if TORCH_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestCollapseDetector))
        suite.addTests(loader.loadTestsFromTestCase(TestAMPUtils))
    else:
        print("\n[!] PyTorch not available. Skipping PyTorch-dependent tests.")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
