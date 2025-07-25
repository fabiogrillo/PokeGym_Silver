from stable_baselines3.common import env_checker
from envs.pokemon_silver_env import PokemonSilver

env = PokemonSilver(
    rom_path="roms/Pokemon_Silver.gbc",
    render_mode="human"
)

obs = env.reset()
print(f"Reset done. Shape: {type(obs), obs}")

env_checker.check_env(env)
