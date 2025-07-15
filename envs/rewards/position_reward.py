class PositionReward:
    def __init__(self, pyboy):
        self.visited = set()
        self.pyboy = pyboy
        self.last_pos = None

    def compute_reward(self):
        x_address = 0xDA02 # Correct val DA02
        y_address = 0xDA03 # Correct val DA03

        x = self.pyboy.memory[x_address]
        y = self.pyboy.memory[y_address]
        pos = (x, y)

        if pos == self.last_pos:
            reward = -0.05 # Penalize not moving
            status = "Same as last"
        elif pos in self.visited:
            reward = 0.1 # Small reward
            status = "Already Visited"
        else:
            reward = 1.0
            self.visited.add(pos)
            status = "New position"

        self.last_pos = pos

        print(f"Player pos: {pos} | {status} | Reward: {reward}")

        return reward
