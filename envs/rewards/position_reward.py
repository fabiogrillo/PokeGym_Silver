from math import dist

class PositionReward:
    def __init__(self, pyboy):
        self.visited = set()
        self.pyboy = pyboy
        self.last_pos = None

    def compute_reward(self, observation=None):
        x = self.pyboy.memory[0xDA02]
        y = self.pyboy.memory[0xDA03]
        pos = (x, y)

        if self.last_pos is None:
            self.last_pos = pos
            self.visited.add(pos)
            return 1.0  # Prima posizione

        if pos == self.last_pos:
            reward, status = -0.2, "Stuck"
        elif pos in self.visited:
            reward, status = -0.05, "Already visited"
        else:
            # Posizione nuova con bonus in base alla distanza dallo step precedente
            base_reward = 1.0
            distance_bonus = 0.01 * dist(pos, self.last_pos)
            reward = base_reward + distance_bonus
            status = f"New pos, dist bonus {distance_bonus:.2f}"
            self.visited.add(pos)

        self.last_pos = pos
        print(f"Player pos: {pos} | {status} | Reward: {reward:.2f}")
        return reward
