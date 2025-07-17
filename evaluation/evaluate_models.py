import os
import time
import shutil
from stable_baselines3 import PPO
from envs.pokemon_silver_env import PokemonSilver 
from gymnasium.wrappers import RecordEpisodeStatistics

ROM_PATH = "roms/Pokemon_Silver.gbc"
MAX_STEPS = 200

def evaluate_model(model_path, reward_strategy):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    version_folder = f"PPO_{timestamp}"
    
    model_save_path = os.path.join("outputs", "models_evaluated", reward_strategy, version_folder)
    video_save_path = os.path.join("outputs", "video_trials", reward_strategy, version_folder)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)

    env = PokemonSilver(
        rom_path=ROM_PATH,
        render_mode="human",
        save_frames=True,
        reward_strategy=reward_strategy,
        max_steps=MAX_STEPS
    )
    env = RecordEpisodeStatistics(env)

    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0
    cumulative_reward = 0

    print(f"\n*** Starting PPO evaluation with reward strategy: '{reward_strategy}' ***\n")
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        cumulative_reward += reward
        step += 1

    print(f"\n*** Episode terminated in {step} steps | Total reward: {cumulative_reward:.2f} ***\n")
    env.close()

    video_filename = f"video.mp4"
    video_output_path = os.path.join(video_save_path, video_filename)

    cmd = f"./scripts/make_video.sh {video_output_path}"
    print(f"Executing: {cmd}")
    os.system(cmd)

    shutil.copy(model_path, os.path.join(model_save_path, "model.zip"))

    log_path = os.path.join(model_save_path, "log.txt")
    with open(log_path, "w") as f:
        f.write(f"Model evaluated: {model_path}\n")
        f.write(f"Reward strategy: {reward_strategy}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Steps taken: {step}\n")
        f.write(f"Total reward: {cumulative_reward:.2f}\n")

if __name__ == "__main__":
    models = [
        ("ppo_pokemon_hashing_v1.zip", "hashing"),
        ("ppo_pokemon_position_v1.zip", "position")
    ]

    for model_path, reward_strategy in models:
        evaluate_model(model_path, reward_strategy)
