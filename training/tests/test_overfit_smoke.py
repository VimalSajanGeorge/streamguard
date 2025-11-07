"""
Tiny-Overfit Smoke Test for StreamGuard Production Training

Verifies that training loops can overfit on a tiny dataset (32 samples, 50 steps).
This catches issues like:
- Gradient flow problems
- Loss function errors
- Data loading bugs
- Model architecture issues

Run before production A100 training to avoid wasting GPU hours.

Usage:
    pytest training/tests/test_overfit_smoke.py -v
    # OR
    python training/tests/test_overfit_smoke.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings

# Try importing models
try:
    from training.train_transformer import EnhancedSQLIntentTransformer, set_seed
    from transformers import AutoTokenizer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    warnings.warn("Transformer not available")

try:
    from training.train_gnn import EnhancedTaintFlowGNN, set_seed as gnn_set_seed
    from torch_geometric.data import Data
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    warnings.warn("GNN not available")


class TinyCodeDataset(Dataset):
    """Tiny dataset for overfitting test."""

    def __init__(self, num_samples=32, seq_length=64):
        self.num_samples = num_samples
        self.seq_length = seq_length

        # Fixed random data for reproducibility
        torch.manual_seed(42)
        self.input_ids = torch.randint(0, 1000, (num_samples, seq_length))
        self.attention_mask = torch.ones((num_samples, seq_length))
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label': self.labels[idx]
        }


def create_tiny_graph_dataset(num_samples=32, num_nodes=10):
    """Create tiny graph dataset."""
    torch.manual_seed(42)
    graphs = []

    for i in range(num_samples):
        # Random graph
        x = torch.randn(num_nodes, 768)

        # Sequential edges
        edges = []
        for j in range(num_nodes - 1):
            edges.append([j, j + 1])
            edges.append([j + 1, j])

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Label
        y = torch.tensor([i % 2], dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)

    return graphs


@unittest.skipUnless(TRANSFORMER_AVAILABLE, "Transformer not available")
class TestTransformerOverfit(unittest.TestCase):
    """Test Transformer can overfit on tiny dataset."""

    def test_transformer_overfit(self):
        """Transformer should overfit on 32 samples in 50 steps."""
        print("\n[Test] Transformer Overfit Test")
        print("=" * 60)

        # Setup
        set_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        model = EnhancedSQLIntentTransformer(
            model_name="microsoft/codebert-base",
            num_labels=2,
            hidden_dim=256,
            dropout=0.1
        ).to(device)

        # Tiny dataset
        dataset = TinyCodeDataset(num_samples=32, seq_length=64)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Training
        model.train()
        initial_loss = None
        final_loss = None

        for step in range(50):
            batch = next(iter(loader))

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            if step == 49:
                final_loss = loss.item()

            if step % 10 == 0:
                print(f"  Step {step:2d}: Loss = {loss.item():.4f}")

        print(f"\n  Initial Loss: {initial_loss:.4f}")
        print(f"  Final Loss:   {final_loss:.4f}")
        print(f"  Reduction:    {((initial_loss - final_loss) / initial_loss * 100):.1f}%")

        # Assert overfitting occurred
        self.assertLess(final_loss, 0.5,
                       f"Failed to overfit: final_loss={final_loss:.4f} (should be < 0.5)")
        self.assertLess(final_loss, initial_loss * 0.3,
                       f"Loss didn't decrease enough: {final_loss:.4f} vs {initial_loss:.4f}")

        print("  ✅ PASS - Transformer can overfit")


@unittest.skipUnless(GNN_AVAILABLE, "GNN not available")
class TestGNNOverfit(unittest.TestCase):
    """Test GNN can overfit on tiny dataset."""

    def test_gnn_overfit(self):
        """GNN should overfit on 32 graphs in 50 steps."""
        print("\n[Test] GNN Overfit Test")
        print("=" * 60)

        from torch_geometric.data import DataLoader as PyGDataLoader

        # Setup
        gnn_set_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        model = EnhancedTaintFlowGNN(
            node_feature_dim=768,
            hidden_dim=256,
            num_labels=2,
            dropout=0.1
        ).to(device)

        # Tiny dataset
        graphs = create_tiny_graph_dataset(num_samples=32, num_nodes=10)
        loader = PyGDataLoader(graphs, batch_size=8, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Training
        model.train()
        initial_loss = None
        final_loss = None

        for step in range(50):
            batch = next(iter(loader))
            batch = batch.to(device)

            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            if step == 49:
                final_loss = loss.item()

            if step % 10 == 0:
                print(f"  Step {step:2d}: Loss = {loss.item():.4f}")

        print(f"\n  Initial Loss: {initial_loss:.4f}")
        print(f"  Final Loss:   {final_loss:.4f}")
        print(f"  Reduction:    {((initial_loss - final_loss) / initial_loss * 100):.1f}%")

        # Assert overfitting occurred
        self.assertLess(final_loss, 0.5,
                       f"Failed to overfit: final_loss={final_loss:.4f} (should be < 0.5)")
        self.assertLess(final_loss, initial_loss * 0.3,
                       f"Loss didn't decrease enough: {final_loss:.4f} vs {initial_loss:.4f}")

        print("  ✅ PASS - GNN can overfit")


def run_smoke_tests():
    """Run all smoke tests."""
    print("\n" + "=" * 80)
    print("STREAMGUARD TINY-OVERFIT SMOKE TESTS")
    print("=" * 80)
    print("\nThese tests verify training loops work before A100 run.")
    print("Expected: Both models should overfit on 32 samples in 50 steps.\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if TRANSFORMER_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestTransformerOverfit))
    else:
        print("[!] Skipping Transformer tests (not available)")

    if GNN_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestGNNOverfit))
    else:
        print("[!] Skipping GNN tests (not available)")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL SMOKE TESTS PASSED!")
        print("   → Ready for production training on A100")
        return 0
    else:
        print("\n❌ SMOKE TESTS FAILED!")
        print("   → DO NOT run production training until fixed")
        return 1


if __name__ == "__main__":
    exit_code = run_smoke_tests()
    sys.exit(exit_code)
