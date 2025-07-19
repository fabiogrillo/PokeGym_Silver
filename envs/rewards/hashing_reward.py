import imagehash
from PIL import Image
import numpy as np

class HashingReward:
    def __init__(self, threshold=5):
        self.hashes = []
        self.threshold = threshold
        self.last_hash = None

    def compute_reward(self, frame):
        # Rimuove dimensione extra (1, H, W) → (H, W)
        if frame.ndim == 3 and frame.shape[0] == 1:
            frame_2d = np.squeeze(frame, axis=0)
        else:
            frame_2d = frame

        if frame_2d.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {frame_2d.shape}")

        h = imagehash.phash(Image.fromarray(frame_2d))

        if self.last_hash == h:
            reward, status = -0.2, "Same"
        elif any(abs(h - old) < self.threshold for old in self.hashes):
            reward, status = -0.05, "Similar"
        else:
            # Più frame unici trovi, meno bonus (es: 1.0 → 0.5)
            novelty_bonus = 1.0 / (1 + len(self.hashes) * 0.05)
            reward, status = novelty_bonus, f"New (bonus {novelty_bonus:.2f})"
            self.hashes.append(h)

        self.last_hash = h
        print(f"HashingReward: {status} | Reward: {reward:.2f}")
        return reward
