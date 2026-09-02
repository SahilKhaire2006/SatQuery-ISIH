"""
Confidence Calibration Module for Text-Guided Grounding

Implements empirical Temperature Scaling to convert raw vision-language logit scores
into calibrated probability confidence values using validation set outcomes.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

class TemperatureScaler:
    """
    Temperature Scaling for calibrating grounding confidence scores.
    Optimizes temperature T > 0 on validation logits such that sigmoid(logits / T)
    accurately reflects empirical precision (IoU >= 0.5).
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits: List[float], labels: List[int]) -> float:
        """
        Fit temperature parameter T on validation set logits and binary correctness labels (IoU >= 0.5).

        Args:
            logits: List of raw model score outputs
            labels: Binary correctness (1 if IoU >= 0.5 else 0)

        Returns:
            Optimized temperature scalar T
        """
        if not logits or len(logits) < 2:
            return self.temperature

        logits_arr = np.array(logits, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.float32)

        def nll_loss(t_val: float) -> float:
            t = max(1e-4, float(t_val))
            # Sigmoid with temperature scaling
            scaled_logits = logits_arr / t
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -15, 15)))
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            # Binary Cross Entropy Loss
            bce = -np.mean(labels_arr * np.log(probs) + (1.0 - labels_arr) * np.log(1.0 - probs))
            return float(bce)

        # Pure numpy grid search over temperature space T in [0.1, 5.0]
        t_candidates = np.linspace(0.1, 5.0, 50)
        losses = [nll_loss(t) for t in t_candidates]
        best_t = float(t_candidates[np.argmin(losses)])
        self.temperature = best_t

        return self.temperature

    def calibrate(self, raw_score: float) -> float:
        """Calibrate a single raw score logit using learned temperature parameter T."""
        scaled_score = float(raw_score) / max(1e-4, self.temperature)
        prob = 1.0 / (1.0 + np.exp(-np.clip(scaled_score, -15, 15)))
        return round(float(prob), 4)

    def compute_ece(self, probs: List[float], labels: List[int], num_bins: int = 5) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        if not probs or len(probs) == 0:
            return 0.0

        probs_arr = np.array(probs)
        labels_arr = np.array(labels)

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (probs_arr >= bin_lower) & (probs_arr < bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels_arr[in_bin])
                avg_confidence_in_bin = np.mean(probs_arr[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return round(float(ece), 4)
