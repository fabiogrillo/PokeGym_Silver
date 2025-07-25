import time
from envs.pokemon_silver_env import PokemonSilver

ROM_PATH = "roms/Pokemon_Silver.gbc"


env = PokemonSilver(rom_path=ROM_PATH)
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(reward, info)

env.close()