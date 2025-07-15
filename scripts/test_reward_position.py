import sys
import numpy as np
from envs.pokemon_silver_env import PokemonSilver

env = PokemonSilver(
    rom_path="roms/Pokemon_Silver.gbc",
    render_mode="human",
    save_frames=False,
    reward_strategy="position"
)

obs = env.reset()

for i in range(20):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Step {i}: Reward {reward}")
