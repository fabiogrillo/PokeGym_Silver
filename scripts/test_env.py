import time
from envs.pokemon_silver_env import PokemonSilver

env = PokemonSilver('roms/Pokemon_Silver.gbc')

obs = env.reset()
print("Reset done. Observation shape:", obs.shape)

# Try 20 random steps
for i in range(20):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Step {i} - Action: {action} - Reward: {reward} - Done: {done}")
    time.sleep(1)

env.close()