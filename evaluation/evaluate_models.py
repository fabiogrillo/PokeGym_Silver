# evaluation/evaluate_models.py
import os
import time
from stable_baselines3 import PPO
from envs.pokemon_silver_env import PokemonSilver 
from gymnasium.wrappers import RecordEpisodeStatistics

ROM_PATH = "roms/Pokemon_Silver.gbc"
MAX_STEPS = 1000

def evaluate_model(model_path, reward_strategy):
    env = PokemonSilver(
        rom_path=ROM_PATH,
        render_mode="human",
        save_frames=True,
        reward_strategy=reward_strategy,
        max_steps=MAX_STEPS
    )
    
    env = RecordEpisodeStatistics(env)

    # Carica il modello PPO
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0

    print(f"\n*** Starting PPO evaluation with reward strategy: '{reward_strategy}' ***\n")
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step += 1

    print(f"\n*** Episode terminated in {step} steps ***\n")
    env.close()

    # Video Name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = f"{reward_strategy}_{step}steps_PPO_{timestamp}.mp4"
    video_path = os.path.join("outputs/videos_trials", video_filename)

    # Create Folder
    os.makedirs("outputs/video_trials", exist_ok=True)

    # Recall make_video.sh
    cmd = f"scripts/make_video.sh {video_path}"
    print(f"Executing: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    models = [
        ("ppo_pokemon_hashing.zip", "hashing"),
        ("ppo_pokemon_position.zip", "position")
    ]

    for model_path, reward_strategy in models:
        evaluate_model(model_path, reward_strategy)
