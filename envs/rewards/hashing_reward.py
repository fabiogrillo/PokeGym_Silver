import imagehash
from PIL import Image

class HashingReward:
    def __init__(self, threshold=5):
        self.hashes = []
        self.threshold = threshold
        self.last_hash = None

    def compute_reward(self, frame):
        h = imagehash.phash(Image.fromarray(frame))
        
        # Penalize if same as last hash
        if self.last_hash is not None and abs(h - self.last_hash) == 0:
            reward = -0.05
            status = "Identical frame (no change)"
        else:
            # Check if similar to any previous hash
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
