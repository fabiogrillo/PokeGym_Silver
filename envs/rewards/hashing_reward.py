import imagehash
from PIL import Image

class HashingReward:
    def __init__(self, threshold=5):
        self.hashes = []
        self.threshold = threshold

    def compute_reward(self, frame):
        h = imagehash.phash(Image.fromarray(frame))
        for existing in self.hashes:
            if abs(h-existing) < self.threshold:
                return 0.0
        self.hashes.append(h)
        return 1.0