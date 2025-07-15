class PositionReward:
    def __init__(self, pyboy):
        self.visited = set()
        self.pyboy = pyboy

    def compute_reward(self, frame):
        x = self.pyboy.get_memory_value(0xD20D)
        y = self.pyboy.get_memory_value(0xD20E)
        pos = (x, y)
        if pos in self.visited:
            return 0.0
        self.visited.add(pos)
        return 1.0
