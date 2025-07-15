import gymnasium 
from envs.pokemon_silver_env import PokemonSilver
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

def main():
    rom_path = "roms/Pokemon_Silver.gbc"
    env = PokemonSilver(rom_path, render_mode='headless')

    # Callback to save checkpoint
    checkpoint_dir = "outputs/models"
    os.makedirs(checkpoint_dir, exist=True)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir, name_prefix="poke_model")

    # Create PPO model
    model = PPO("CnnPolicy", env, verbose=1)

    # Training
    model.learn(total_timesteps=500000, callback=checkpoint_callback)

    # Save final model
    model.save(os.path.join(checkpoint_dir, "final_model"))

    env.close()

if __name__ == "__main__":
    main()