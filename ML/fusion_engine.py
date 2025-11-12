import numpy as np

class FusionEngine:
    """
    Combines multi-modal behavioral signals into a single trust score.
    Each input is weighted according to signal reliability.
    """
    def __init__(self):
        # Tunable weights (can be personalized later)
        self.weights = {
            "keystroke": 0.4,
            "gaze": 0.2,
            "pose": 0.2,
            "emotion": 0.2
        }

    def compute_trust_score(self, signals: dict) -> float:
        """Combine normalized signals into one trust score."""
        total = sum(signals.get(k, 0.0) * w for k, w in self.weights.items())
        return round(float(np.clip(total, 0.0, 1.0)), 3)

    def detect_anomaly(self, trust_score: float, threshold: float = 0.6):
        """Detect anomalies based on a trust threshold."""
        if trust_score < threshold:
            return True, f"⚠️ Low trust detected ({trust_score:.2f})"
        return False, None
