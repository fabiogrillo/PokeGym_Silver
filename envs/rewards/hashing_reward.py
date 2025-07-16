import imagehash
from PIL import Image
import numpy as np

class HashingReward:
    def __init__(self, threshold=5):
        self.hashes = []
        self.threshold = threshold
        self.last_hash = None

    def compute_reward(self, frame):
        # Se frame è (1, H, W), rimuovi la dimensione extra
        if frame.ndim == 3 and frame.shape[0] == 1:
            frame_2d = np.squeeze(frame, axis=0)
        else:
            frame_2d = frame

        # Verifica che ora sia 2D
        if frame_2d.ndim != 2:
            raise ValueError(f"compute_reward: expected 2D array, got shape {frame_2d.shape}")

        h = imagehash.phash(Image.fromarray(frame_2d))

        if self.last_hash is not None and abs(h - self.last_hash) == 0:
            reward = -0.05
            status = "Identical frame (no change)"
        else:
            for existing in self.hashes:
                if abs(h - existing) < self.threshold:
                    reward = 0.1
                    status = "Similar to known frame"
                    break
            else:
                reward = 1.0
                status = "New frame discovered"
                self.hashes.append(h)

        self.last_hash = h
        print(f"HashingReward: {status} | Reward: {reward}")
        return reward
