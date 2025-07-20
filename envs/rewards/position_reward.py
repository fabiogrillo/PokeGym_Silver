from math import dist

class PositionReward:
    def __init__(self, pyboy, explore_weight=1.0, reward_scale=1.0,
                 stuck_penalty=-0.2, revisit_penalty=-0.05,
                 decay_rate=0.001, map_change_bonus=0.5):
        self.pyboy = pyboy
        self.explore_weight = explore_weight
        self.reward_scale = reward_scale
        self.stuck_penalty = stuck_penalty
        self.revisit_penalty = revisit_penalty
        self.decay_rate = decay_rate
        self.map_change_bonus = map_change_bonus

        self.visited = set()
        self.visit_count = {}  # Penalità su revisit
        self.last_pos = None
        self.last_map = None
        self.last_reward = 0.0
        self.global_explore = set()  # Global exploration memory

    def compute_reward(self, observation=None):
        # Lettura posizione e mappa dalla memoria
        map_id = self.pyboy.memory[0xDA01]
        x = self.pyboy.memory[0xDA02]
        y = self.pyboy.memory[0xDA03]
        in_battle = self.pyboy.memory[0xD116] != 0

        if in_battle:
            return 0.0

        pos = (map_id, x, y)

        # Penalità se bloccato
        if self.last_pos == pos:
            reward = self.stuck_penalty
            status = "Stuck"
        
        # Penalità se rivisitato (con decay nel tempo)
        elif pos in self.visited:
            count = self.visit_count.get(pos, 1)
            reward = self.revisit_penalty * count
            status = f"Revisit {count:.2f} times"
            self.visit_count[pos] = count + 1
        else:
            # Nuova posizione esplorata
            base_reward = self.reward_scale * self.explore_weight * 0.1
            distance_bonus = 0.01 * (dist(self.last_pos, pos) if self.last_pos else 0)
            reward = base_reward + distance_bonus
            status = f"New pos, dist bonus {distance_bonus:.3f}"
            self.visited.add(pos)
            self.visit_count[pos] = 1
            self.global_explore.add(pos)

        # Bonus se ha cambiato mappa (es. è entrato in un edificio)
        if self.last_map is not None and map_id != self.last_map:
            reward += self.map_change_bonus
            status += f" | Map changed (+{self.map_change_bonus})"

        self.last_pos = pos
        self.last_map = map_id

        # Ricompensa globale cumulativa (più celle esplorate → più reward)
        global_reward = 0.01 * len(self.global_explore)
        total_reward = reward + global_reward

        # Decay sul visit count per permettere esplorazioni future
        for visited_pos in list(self.visit_count.keys()):
            self.visit_count[visited_pos] = max(1.0, self.visit_count[visited_pos] - self.decay_rate)

        # Debug logging
        print(f"[PositionReward] {status} | Map: {map_id} Pos: ({x},{y}) | In battle: {in_battle} |"
              f"Step: {reward:.3f} | Global: {global_reward:.2f} | Total: {total_reward:.3f}")

        return total_reward
