from envs.pokemon_silver_env import PokemonSilver

env = PokemonSilver("roms/Pokemon_Silver.gbc", render_mode="human")
env.reset()

input("\nPress enter to close...")
env.close()
