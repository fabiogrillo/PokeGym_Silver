import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from envs.pokemon_silver_env import PokemonSilver

# CONFIG
CHECKPOINT_DIR = "trained_agents/position_reward/v1"
ROM_PATH = "roms/Pokemon_Silver.gbc"
N_EVAL_EPISODES = 2
RENDER_MODE = "headless"  # Cambia in "human" se vuoi vedere visivamente

def evaluate_all_checkpoints():
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_*.zip")))
    results = []

    for path in checkpoints:
        print(f"📦 Evaluating {path}")
        model = PPO.load(path)

        env = PokemonSilver(rom_path=ROM_PATH, render_mode=RENDER_MODE)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES, return_episode_rewards=False)
        results.append((path, mean_reward, std_reward))
        env.close()

    return results

def save_results(results):
    output_csv = os.path.join(CHECKPOINT_DIR, "checkpoint_eval.csv")
    with open(output_csv, "w") as f:
        f.write("checkpoint,mean_reward,std_reward\n")
        for path, mean, std in results:
            f.write(f"{os.path.basename(path)},{mean:.2f},{std:.2f}\n")
    print(f"✅ Risultati salvati in {output_csv}")

def plot_results(results):
    steps = [int(p.split("_")[-1].split(".")[0]) for p, _, _ in results]
    rewards = [r for _, r, _ in results]

    plt.figure(figsize=(10,6))
    plt.plot(steps, rewards, marker="o")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Performance del modello per checkpoint")
    plt.grid()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "checkpoint_eval_plot.png"))
    print("📈 Graph saved.")

if __name__ == "__main__":
    results = evaluate_all_checkpoints()
    save_results(results)
    plot_results(results)
