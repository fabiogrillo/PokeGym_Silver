import sys
import numpy as np
from envs.pokemon_silver_env import PokemonSilver

env = PokemonSilver(
    rom_path="roms/Pokemon_Silver.gbc",
    render_mode="human",
    save_frames=False,
    reward_strategy="hashing"
)

obs = env.reset()
cumulative_reward = 0.0

for step in range(20):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    cumulative_reward += reward
    
    print(f"[Step {step}] Reward this step: {reward:.2f} | Cumulative: {cumulative_reward:.2f}")

input("\n[INFO] Press Enter to close...")
env.close()