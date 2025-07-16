from stable_baselines3.common import env_checker
from envs.pokemon_silver_env import PokemonSilver

env = PokemonSilver(
    rom_path="roms/Pokemon_Silver.gbc",
    render_mode="headless",
    reward_strategy="position"
)

env_checker.check_env(env)
