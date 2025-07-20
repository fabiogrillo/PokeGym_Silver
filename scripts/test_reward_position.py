import numpy as np
from envs.pokemon_silver_env import PokemonSilver

env = PokemonSilver(
    rom_path="roms/Pokemon_Silver.gbc",
    render_mode="human",
    save_frames=False,
    reward_strategy="position"
)

obs = env.reset()
cumulative_reward = 0.0

print("\n[INFO] Starting test with PositionReward...\n")

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward

    print(f"[Step {step}] Reward this step: {reward:.2f} | Cumulative: {cumulative_reward:.2f}")
    
input("\n[INFO] Press Enter to close...")
env.close()
