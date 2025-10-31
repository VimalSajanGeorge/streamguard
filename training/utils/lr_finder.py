"""
Learning Rate Finder - Leslie Smith's Method

Reference:
Leslie N. Smith, "Cyclical Learning Rates for Training Neural Networks" (2017)
https://arxiv.org/abs/1506.01186

Automatically finds optimal learning rate by:
1. Starting with very small LR (e.g., 1e-8)
2. Exponentially increasing LR each batch
3. Tracking loss at each LR
4. Finding LR with steepest descent (before divergence)
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional, List, Dict, Any
import math
import numpy as np


class LRFinder:
    """
    Learning Rate Finder using Leslie Smith's method.

    Usage:
        >>> lr_finder = LRFinder(model, optimizer, criterion, device)
        >>> lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=1.0, num_iter=100)
        >>> suggested_lr, lr_loss_history = lr_finder.get_best_lr()
        >>> print(f"Suggested LR: {suggested_lr:.2e}")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ):
        """
        Initialize LR Finder.

        Args:
            model: Model to train
            optimizer: Optimizer instance
            criterion: Loss criterion
            device: Device to run on
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Save initial state to restore later
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

        # Results storage
        self.lr_history = []
        self.loss_history = []
        self.best_lr = None

    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0
    ) -> Tuple[List[float], List[float]]:
        """
        Perform LR range test.

        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate (default: 1e-7)
            end_lr: Ending learning rate (default: 1.0)
            num_iter: Number of iterations (default: 100)
            smooth_f: Smoothing factor for loss (default: 0.05)
            diverge_th: Divergence threshold (stop if loss > diverge_th * min_loss)

        Returns:
            Tuple of (lr_history, loss_history)
        """
        print(f"\n[*] Running LR Finder...")
        print(f"    Start LR: {start_lr:.2e}")
        print(f"    End LR: {end_lr:.2e}")
        print(f"    Iterations: {num_iter}")

        # Set model to training mode
        self.model.train()

        # Calculate LR multiplier per iteration
        # LR grows exponentially: lr = start_lr * (end_lr / start_lr) ** (iter / num_iter)
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)

        # Initialize
        lr = start_lr
        best_loss = float('inf')
        smoothed_loss = 0.0
        iteration = 0

        # Iterator over training data
        train_iter = iter(train_loader)

        while iteration < num_iter:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart iterator if we run out of data
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Update learning rate for all param groups
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Get code features if available
            code_features = batch.get('code_features')
            if code_features is not None:
                code_features = code_features.to(self.device)

            # Compute loss
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, code_features)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track loss
            loss_value = loss.item()

            # Exponential moving average for smoothing
            if iteration == 0:
                smoothed_loss = loss_value
            else:
                smoothed_loss = smooth_f * loss_value + (1 - smooth_f) * smoothed_loss

            # Store results
            self.lr_history.append(lr)
            self.loss_history.append(smoothed_loss)

            # Check for divergence
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            if smoothed_loss > diverge_th * best_loss:
                print(f"[!] Loss diverged at LR={lr:.2e}, stopping early")
                break

            # Progress update
            if (iteration + 1) % 10 == 0:
                print(f"    Iter {iteration + 1}/{num_iter}: LR={lr:.2e}, Loss={smoothed_loss:.4f}")

            # Update LR for next iteration
            lr *= lr_mult
            iteration += 1

        print(f"[+] LR Finder complete. Tested {len(self.lr_history)} learning rates")

        # Restore model and optimizer to initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        return self.lr_history, self.loss_history

    def get_best_lr(
        self,
        skip_start: int = 10,
        skip_end: int = 5
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Find the best learning rate from the range test.

        Uses the "steepest descent" heuristic:
        - Find the LR where loss decreases fastest
        - Skip initial and final iterations (noisy regions)

        Args:
            skip_start: Skip first N iterations (default: 10)
            skip_end: Skip last N iterations (default: 5)

        Returns:
            Tuple of (best_lr, lr_loss_history)
        """
        if not self.lr_history:
            raise ValueError("Must run range_test() first!")

        # Skip noisy regions
        start_idx = skip_start
        end_idx = len(self.loss_history) - skip_end

        if end_idx <= start_idx:
            # Not enough data, use all
            start_idx = 0
            end_idx = len(self.loss_history)

        # Find steepest gradient (most negative derivative)
        # gradient[i] ≈ (loss[i+1] - loss[i]) / (lr[i+1] - lr[i])
        max_gradient = 0.0
        best_lr_idx = start_idx

        for i in range(start_idx, end_idx - 1):
            # Compute gradient in log space (since LR grows exponentially)
            delta_loss = self.loss_history[i + 1] - self.loss_history[i]
            delta_log_lr = math.log(self.lr_history[i + 1]) - math.log(self.lr_history[i])

            if delta_log_lr == 0:
                continue

            gradient = delta_loss / delta_log_lr

            # We want the steepest descent (most negative gradient)
            if gradient < max_gradient:
                max_gradient = gradient
                best_lr_idx = i

        # Best LR is where loss decreases fastest
        suggested_lr = self.lr_history[best_lr_idx]

        # Alternative heuristic: Use LR 10x smaller than steepest descent
        # (more conservative, often works better)
        # suggested_lr = suggested_lr / 10.0

        print(f"\n[*] LR Finder Results:")
        print(f"    Steepest descent at LR: {suggested_lr:.2e}")
        print(f"    Gradient: {max_gradient:.4f}")
        print(f"    Loss at that point: {self.loss_history[best_lr_idx]:.4f}")

        # Return history as list of (lr, loss) tuples
        lr_loss_history = list(zip(self.lr_history, self.loss_history))

        self.best_lr = suggested_lr
        return suggested_lr, lr_loss_history

    def plot(
        self,
        output_path: str = 'lr_finder_plot.png',
        skip_start: int = 10,
        skip_end: int = 5
    ):
        """
        Plot the LR finder results.

        Args:
            output_path: Path to save plot
            skip_start: Skip first N points
            skip_end: Skip last N points
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[!] matplotlib not available, skipping plot")
            return

        if not self.lr_history:
            print("[!] No data to plot. Run range_test() first.")
            return

        # Skip noisy regions
        start_idx = skip_start
        end_idx = len(self.loss_history) - skip_end

        if end_idx <= start_idx:
            start_idx = 0
            end_idx = len(self.loss_history)

        lrs = self.lr_history[start_idx:end_idx]
        losses = self.loss_history[start_idx:end_idx]

        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, marker='o', markersize=3)
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss (smoothed)')
        plt.title('LR Finder: Loss vs Learning Rate')
        plt.grid(True, alpha=0.3)

        # Mark the suggested LR
        if self.best_lr:
            plt.axvline(self.best_lr, color='red', linestyle='--', label=f'Suggested LR: {self.best_lr:.2e}')
            plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"[+] LR Finder plot saved to: {output_path}")
        plt.close()


def analyze_lr_loss_curve(
    lrs: List[float],
    losses: List[float],
    min_slope_threshold: float = 1e-4
) -> Dict[str, Any]:
    """
    Analyze LR-loss curve quality and assign confidence score.

    Uses signal analysis to determine if the curve shows a clear learning rate
    sweet spot, or if it's too flat/noisy/divergent to trust.

    Args:
        lrs: List of learning rates tested
        losses: List of corresponding losses (smoothed)
        min_slope_threshold: Minimum slope magnitude to consider significant

    Returns:
        Dictionary with analysis results:
        {
            'suggested_lr': float,      # LR at steepest descent
            'confidence': str,           # 'high' / 'medium' / 'low'
            'slope_mag': float,          # Magnitude of steepest slope
            'snr': float,                # Signal-to-noise ratio
            'diverged': bool,            # Whether loss diverged after minimum
            'reason': List[str],         # Reasons for low confidence
            'idx': int                   # Index of suggested LR
        }

    Example:
        >>> lrs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        >>> losses = [1.0, 0.8, 0.5, 0.4, 2.0]  # Clear descent then divergence
        >>> analysis = analyze_lr_loss_curve(lrs, losses)
        >>> print(f"Confidence: {analysis['confidence']}")
        >>> print(f"Suggested LR: {analysis['suggested_lr']:.2e}")
    """
    # Convert to numpy for easier operations
    lrs = np.asarray(lrs)
    losses = np.asarray(losses)

    if len(lrs) < 10:
        # Too few points for reliable analysis
        return {
            'suggested_lr': float(lrs[len(lrs) // 2]),
            'confidence': 'low',
            'slope_mag': 0.0,
            'snr': 0.0,
            'diverged': False,
            'reason': ['insufficient_data'],
            'idx': len(lrs) // 2
        }

    # Smooth losses with simple moving average to reduce noise
    window = max(1, min(5, len(losses) // 20))
    if window > 1:
        kernel = np.ones(window) / window
        losses_smooth = np.convolve(losses, kernel, mode='same')
    else:
        losses_smooth = losses

    # Compute derivative w.r.t log(lr)
    log_lrs = np.log10(lrs + 1e-30)
    # Use numpy gradient for central differences
    deriv = np.gradient(losses_smooth, log_lrs)

    # Find index of steepest negative slope (most negative derivative)
    idx = int(np.argmin(deriv))
    suggested_lr = float(lrs[idx])

    # Signal quality metrics
    slope_mag = float(abs(deriv[idx]))
    deriv_std = float(np.std(deriv))
    snr = slope_mag / (deriv_std + 1e-12)

    # Heuristics to assign confidence
    if slope_mag > 1e-3 and snr > 2.0:
        confidence = "high"
    elif slope_mag > 1e-4 and snr > 1.0:
        confidence = "medium"
    else:
        confidence = "low"

    # Check for catastrophic divergence after minimum
    min_loss = float(losses_smooth.min())
    loss_at_end = float(losses_smooth[-1])
    diverged = (loss_at_end > min_loss * 2.0)  # Loss doubled from minimum

    # Collect reasons for low confidence
    reason = []
    if diverged:
        reason.append("divergence_after_min")
    if confidence == "low":
        if slope_mag < min_slope_threshold:
            reason.append("flat_curve")
        if snr < 1.0:
            reason.append("noisy_curve")

    return {
        "suggested_lr": suggested_lr,
        "confidence": confidence,
        "slope_mag": slope_mag,
        "snr": snr,
        "diverged": diverged,
        "reason": reason,
        "idx": idx
    }


def validate_and_cap_lr(
    suggested_lr: float,
    analysis: Dict[str, Any],
    cap: float = 5e-4,
    conservative_fallback: float = 1e-5
) -> Dict[str, Any]:
    """
    Validate and cap learning rate with safety checks.

    Applies conservative safety cap and falls back to a safe LR if the
    curve analysis indicates poor quality (flat, noisy, or divergent).

    Args:
        suggested_lr: LR suggested by steepest descent heuristic
        analysis: Output from analyze_lr_loss_curve()
        cap: Maximum safe LR to apply (default: 5e-4)
        conservative_fallback: Safe LR to use if validation fails (default: 1e-5)

    Returns:
        Dictionary with validation results:
        {
            'lr': float,              # Final validated LR
            'used_fallback': bool,    # Whether fallback was applied
            'note': str               # Description of action taken
        }

    Example:
        >>> analysis = {'confidence': 'low', 'diverged': True}
        >>> result = validate_and_cap_lr(1e-3, analysis)
        >>> print(f"Final LR: {result['lr']:.2e} ({result['note']})")
    """
    final_lr = suggested_lr
    note = "accepted"
    used_fallback = False

    # Apply safety cap
    if suggested_lr > cap:
        final_lr = cap
        note = f"capped_to_{cap:.1e}"

    # If analysis indicates poor curve quality -> fallback
    if analysis.get("confidence") == "low" or analysis.get("diverged", False):
        final_lr = conservative_fallback
        used_fallback = True
        reasons = analysis.get("reason", [])
        note = f"fallback_due_to_{','.join(reasons)}" if reasons else "fallback_due_to_poor_curve"

    return {
        "lr": float(final_lr),
        "used_fallback": used_fallback,
        "note": note
    }


# Example usage
if __name__ == '__main__':
    print("LR Finder - Example Usage:")
    print("""
    from training.utils.lr_finder import LRFinder

    # Initialize
    lr_finder = LRFinder(model, optimizer, criterion, device)

    # Run range test
    lr_finder.range_test(
        train_loader,
        start_lr=1e-7,
        end_lr=1.0,
        num_iter=100
    )

    # Get best LR
    best_lr, history = lr_finder.get_best_lr()
    print(f"Suggested LR: {best_lr:.2e}")

    # Plot results
    lr_finder.plot('lr_finder_plot.png')

    # Use the suggested LR for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr)
    """)
