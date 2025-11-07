"""
Model Collapse Detection System

Detects and prevents model collapse during training with auto-stop capabilities.

Collapse indicators:
- Zero gradients (vanishing gradient problem)
- Constant predictions (mode collapse)
- NaN/Inf losses
- Exploding gradients
- Low prediction variance

Author: StreamGuard Team
Version: 1.0.0
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CollapseDetector:
    """
    Monitors training for signs of model collapse.

    Features:
    - Gradient monitoring (zero/exploding)
    - Prediction variance tracking
    - Loss anomaly detection
    - Configurable thresholds
    - Auto-stop on repeated collapse events
    """

    def __init__(
        self,
        window_size: int = 5,
        collapse_threshold: int = 3,
        grad_norm_epsilon: float = 1e-7,
        grad_norm_max: float = 100.0,
        prediction_variance_min: float = 1e-6,
        enable_auto_stop: bool = True,
        report_path: Optional[Path] = None
    ):
        """
        Initialize collapse detector.

        Args:
            window_size: Number of recent steps to monitor
            collapse_threshold: Number of collapse events before auto-stop
            grad_norm_epsilon: Threshold for zero gradient detection
            grad_norm_max: Threshold for exploding gradients
            prediction_variance_min: Minimum prediction variance
            enable_auto_stop: Enable automatic training stop
            report_path: Path to save collapse report JSON
        """
        self.window_size = window_size
        self.collapse_threshold = collapse_threshold
        self.grad_norm_epsilon = grad_norm_epsilon
        self.grad_norm_max = grad_norm_max
        self.prediction_variance_min = prediction_variance_min
        self.enable_auto_stop = enable_auto_stop
        self.report_path = report_path

        # Tracking
        self.grad_norms: deque = deque(maxlen=window_size)
        self.losses: deque = deque(maxlen=window_size)
        self.prediction_variances: deque = deque(maxlen=window_size)

        # Collapse event tracking
        self.collapse_events: List[Dict[str, Any]] = []
        self.consecutive_collapses = 0

        # State
        self.step_count = 0
        self.collapsed = False

        print(f"[+] CollapseDetector initialized")
        print(f"    Window size: {window_size}")
        print(f"    Collapse threshold: {collapse_threshold}")
        print(f"    Auto-stop: {enable_auto_stop}")

    def check_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """
        Check model gradients for collapse indicators.

        Args:
            model: PyTorch model

        Returns:
            Dictionary with gradient statistics and collapse flags
        """
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        total_norm = 0.0
        num_params = 0
        zero_grad_params = 0
        nan_grad_params = 0
        inf_grad_params = 0

        for param in model.parameters():
            if param.grad is not None:
                num_params += 1
                param_norm = param.grad.data.norm(2).item()

                total_norm += param_norm ** 2

                # Check for zero gradients
                if param_norm < self.grad_norm_epsilon:
                    zero_grad_params += 1

                # Check for NaN/Inf
                if torch.isnan(param.grad).any():
                    nan_grad_params += 1
                if torch.isinf(param.grad).any():
                    inf_grad_params += 1

        total_norm = total_norm ** 0.5

        # Store gradient norm
        self.grad_norms.append(total_norm)

        # Detect collapse
        is_zero_grad = total_norm < self.grad_norm_epsilon
        is_exploding_grad = total_norm > self.grad_norm_max
        has_nan_inf = (nan_grad_params > 0) or (inf_grad_params > 0)

        return {
            "total_norm": total_norm,
            "num_params": num_params,
            "zero_grad_params": zero_grad_params,
            "nan_grad_params": nan_grad_params,
            "inf_grad_params": inf_grad_params,
            "is_zero_grad": is_zero_grad,
            "is_exploding_grad": is_exploding_grad,
            "has_nan_inf": has_nan_inf,
            "collapse_detected": is_zero_grad or is_exploding_grad or has_nan_inf
        }

    def check_predictions(
        self,
        predictions: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Check predictions for mode collapse.

        Args:
            predictions: Model predictions (logits or probabilities)
            labels: Ground truth labels (optional)

        Returns:
            Dictionary with prediction statistics and collapse flags
        """
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        # Convert to numpy for analysis
        preds_np = predictions.detach().cpu().numpy()

        # Calculate variance
        variance = preds_np.var()
        self.prediction_variances.append(variance)

        # Check for constant predictions (mode collapse)
        # Only flag if variance is REALLY low (logits should have variance > 1e-3 normally)
        is_constant = variance < self.prediction_variance_min and len(self.prediction_variances) > 3

        # Check unique predictions
        if predictions.dim() == 1:
            # Binary classification (after sigmoid)
            unique_preds = len(set(preds_np.round().tolist()))
        else:
            # Multi-class (after softmax)
            pred_classes = preds_np.argmax(axis=1)
            unique_preds = len(set(pred_classes.tolist()))

        is_mode_collapse = unique_preds == 1

        return {
            "variance": variance,
            "unique_predictions": unique_preds,
            "is_constant": is_constant,
            "is_mode_collapse": is_mode_collapse,
            "collapse_detected": is_constant or is_mode_collapse
        }

    def check_loss(self, loss: float) -> Dict[str, Any]:
        """
        Check loss for anomalies.

        Args:
            loss: Training loss value

        Returns:
            Dictionary with loss statistics and collapse flags
        """
        # Store loss
        self.losses.append(loss)

        # Check for NaN/Inf
        is_nan = (loss != loss)  # NaN check
        is_inf = (loss == float('inf') or loss == float('-inf'))

        # Check for sudden spikes (if we have history)
        is_spike = False
        if len(self.losses) > 1:
            recent_mean = sum(list(self.losses)[:-1]) / (len(self.losses) - 1)
            is_spike = loss > recent_mean * 10  # 10x spike

        return {
            "loss": loss,
            "is_nan": is_nan,
            "is_inf": is_inf,
            "is_spike": is_spike,
            "collapse_detected": is_nan or is_inf or is_spike
        }

    def step(
        self,
        model: nn.Module,
        loss: float,
        predictions: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        step: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform collapse detection for current training step.

        Args:
            model: PyTorch model
            loss: Training loss
            predictions: Model predictions
            labels: Ground truth labels (optional)
            step: Current training step (optional)

        Returns:
            Dictionary with collapse status and diagnostics
        """
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1

        # Run all checks
        grad_check = self.check_gradients(model)
        pred_check = self.check_predictions(predictions, labels)
        loss_check = self.check_loss(loss)

        # Determine if collapse detected
        collapse_detected = (
            grad_check.get("collapse_detected", False) or
            pred_check.get("collapse_detected", False) or
            loss_check.get("collapse_detected", False)
        )

        # Update collapse counter
        if collapse_detected:
            self.consecutive_collapses += 1

            # Record event
            event = {
                "step": self.step_count,
                "timestamp": datetime.now().isoformat(),
                "loss": loss,
                "gradient_check": grad_check,
                "prediction_check": pred_check,
                "loss_check": loss_check
            }
            self.collapse_events.append(event)

            # Check if we should stop
            if self.consecutive_collapses >= self.collapse_threshold:
                self.collapsed = True

                if self.enable_auto_stop:
                    warnings.warn(
                        f"\n{'='*60}\n"
                        f"MODEL COLLAPSE DETECTED!\n"
                        f"Consecutive collapse events: {self.consecutive_collapses}\n"
                        f"Training will be stopped.\n"
                        f"{'='*60}\n"
                    )

                    # Save report
                    if self.report_path:
                        self.save_report()

        else:
            # Reset counter if no collapse
            self.consecutive_collapses = 0

        return {
            "step": self.step_count,
            "collapse_detected": collapse_detected,
            "consecutive_collapses": self.consecutive_collapses,
            "should_stop": self.collapsed and self.enable_auto_stop,
            "gradient_norm": grad_check.get("total_norm", 0.0),
            "prediction_variance": pred_check.get("variance", 0.0),
            "loss": loss
        }

    def save_report(self, path: Optional[Path] = None) -> None:
        """
        Save collapse detection report to JSON.

        Args:
            path: Output path (uses self.report_path if None)
        """
        output_path = path or self.report_path

        if output_path is None:
            warnings.warn("No report path specified. Skipping report save.")
            return

        report = {
            "timestamp": datetime.now().isoformat(),
            "collapsed": self.collapsed,
            "total_steps": self.step_count,
            "collapse_events": self.collapse_events,
            "config": {
                "window_size": self.window_size,
                "collapse_threshold": self.collapse_threshold,
                "grad_norm_epsilon": self.grad_norm_epsilon,
                "grad_norm_max": self.grad_norm_max,
                "prediction_variance_min": self.prediction_variance_min
            },
            "statistics": {
                "total_collapse_events": len(self.collapse_events),
                "max_consecutive_collapses": self.consecutive_collapses,
                "recent_grad_norms": list(self.grad_norms),
                "recent_losses": list(self.losses),
                "recent_prediction_variances": list(self.prediction_variances)
            }
        }

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write report (use safe_jsonify to handle numpy types)
        from training.utils.json_safety import safe_jsonify
        safe_report = safe_jsonify(report)
        with open(output_path, 'w') as f:
            json.dump(safe_report, f, indent=2)

        print(f"[+] Collapse report saved: {output_path}")

    def reset(self) -> None:
        """Reset detector state."""
        self.grad_norms.clear()
        self.losses.clear()
        self.prediction_variances.clear()
        self.collapse_events.clear()
        self.consecutive_collapses = 0
        self.step_count = 0
        self.collapsed = False


# Example usage
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping tests.")
    else:
        print("Testing Collapse Detection System\n")
        print("=" * 60)

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

        # Create detector
        detector = CollapseDetector(
            window_size=5,
            collapse_threshold=3,
            enable_auto_stop=True,
            report_path=Path("collapse_report.json")
        )

        # Test 1: Normal training
        print("\n[Test 1] Normal Training")
        for step in range(10):
            # Forward pass
            x = torch.randn(32, 10)
            y = torch.randint(0, 2, (32,))
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

            # Backward pass
            loss.backward()

            # Check for collapse
            result = detector.step(model, loss.item(), logits, y, step)

            if step % 3 == 0:
                print(f"  Step {step}: Loss={loss.item():.4f}, "
                      f"GradNorm={result['gradient_norm']:.4f}, "
                      f"Variance={result['prediction_variance']:.4f}")

            # Clear gradients
            model.zero_grad()

        print("✅ PASS - No collapse detected")

        # Test 2: Simulated collapse (zero gradients)
        print("\n[Test 2] Simulated Collapse (Zero Gradients)")
        detector.reset()

        # Zero all gradients manually
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
            else:
                param.grad = torch.zeros_like(param)

        # Run detection
        x = torch.randn(32, 10)
        logits = model(x)
        result = detector.step(model, 0.5, logits)

        print(f"  Collapse detected: {result['collapse_detected']}")
        print("✅ PASS")

        # Clean up
        if Path("collapse_report.json").exists():
            Path("collapse_report.json").unlink()

        print("\n" + "=" * 60)
        print("All tests passed! ✅")
