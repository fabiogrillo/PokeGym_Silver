import os
import sys
from pathlib import Path
import numpy as np
from termcolor import cprint

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from envs.pokemon_silver_env import PokemonSilver

BASE_DIR = Path(__file__).parent.parent
ROM_PATH = BASE_DIR / "roms/Pokemon_Silver.gbc"

def quick_evaluate(model_path, episodes=1, render=True):
    """Valutazione veloce di un checkpoint"""
    
    if not os.path.exists(model_path):
        cprint(f"❌ Model not found: {model_path}", "red")
        return
    
    cprint(f"📂 Loading model: {model_path}", "cyan")
    model = PPO.load(model_path)
    
    env = PokemonSilver(
        rom_path=str(ROM_PATH),
        render_mode="human" if render else "headless",
        max_steps=5000  # Limite per test veloce
    )
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        cprint(f"\n🎮 Episode {ep + 1}", "yellow")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"\rStep {steps}: Reward={total_reward:.2f}, "
                      f"Tiles={info.get('unique_tiles', 0)}, "
                      f"Badges={info.get('badges', 0)}", end="", flush=True)
            
            done = terminated or truncated
        
        print()  # New line
        cprint(f"✅ Episode complete:", "green")
        cprint(f"   Total Reward: {total_reward:.2f}", "blue")
        cprint(f"   Steps: {steps}", "blue")
        cprint(f"   Unique Tiles: {info.get('unique_tiles', 0)}", "blue")
        cprint(f"   Badges: {info.get('badges', 0)}", "blue")
        
    env.close()

if __name__ == "__main__":
    # Trova il miglior modello disponibile
    model_dirs = [
        BASE_DIR / "trained_agents/exploration_v2/best_model/best_model.zip",
        BASE_DIR / "trained_agents/exploration_v2/final_model.zip",
    ]
    
    # Trova anche gli ultimi checkpoint
    checkpoint_dir = BASE_DIR / "trained_agents/exploration_v2"
    if checkpoint_dir.exists():
        checkpoints = sorted([f for f in checkpoint_dir.glob("ppo_pokemon_silver_*.zip")])
        if checkpoints:
            model_dirs.insert(0, checkpoints[-1])  # Ultimo checkpoint
    
    # Prova a trovare un modello
    model_path = None
    for path in model_dirs:
        if path.exists():
            model_path = path
            break
    
    if model_path:
        quick_evaluate(str(model_path), episodes=1, render=True)
    else:
        cprint("❌ No model found to evaluate!", "red")
        cprint("Available paths checked:", "yellow")
        for path in model_dirs:
            print(f"  - {path}")