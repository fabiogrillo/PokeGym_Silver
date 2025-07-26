import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.pokemon_silver_env_v3_simplified import PokemonSilverV3Simplified
from termcolor import cprint
import json
from pathlib import Path

ROM_PATH = "roms/Pokemon_Silver.gbc"
MODEL_DIR = "trained_agents/exploration_v3"

def evaluate_model(model_path, num_episodes=5, render=True, save_stats=True):
    """Evaluate the saved model"""
    
    cprint(f"📂 Loading model from {model_path}", "cyan")
    model = PPO.load(model_path)
    
    # Stats collection
    all_stats = {
        "rewards": [],
        "steps": [],
        "party_size": [],
        "badges": [],
        "level_sum": [],
        "hp_fraction": []
    }
    
    for episode in range(num_episodes):
        cprint(f"\n🎮 Episode {episode + 1}/{num_episodes}", "yellow")
        
        env = PokemonSilverV3Simplified(
            rom_path=ROM_PATH, 
            render_mode="human" if render else "headless",
            max_steps=5000,  # Shorter for evaluation
            start_state="post_starter.state"
        )
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        # Track progress
        step_rewards = []
        party_sizes = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Collect step data
            step_rewards.append(total_reward)
            party_sizes.append(info.get("party_size", 1))
            
            # Print progress every 500 steps
            if step_count % 500 == 0:
                cprint(f"  Step {step_count}: Reward={total_reward:.2f}, "
                      f"Party={info.get('party_size', 0)}, "
                      f"Badges={info.get('badges', 0)}", "green")
            
            done = terminated or truncated
        
        # Collect episode stats
        final_info = info # type: ignore
        all_stats["rewards"].append(total_reward)
        all_stats["steps"].append(step_count)
        all_stats["party_size"].append(final_info.get("party_size", 1))
        all_stats["badges"].append(final_info.get("badges", 0))
        all_stats["level_sum"].append(final_info.get("level_sum", 5))
        all_stats["hp_fraction"].append(final_info.get("hp_fraction", 1.0))
        
        cprint(f"\n✅ Episode {episode + 1} Complete:", "green")
        cprint(f"  Total Reward: {total_reward:.2f}", "blue")
        cprint(f"  Steps: {step_count}", "blue")
        cprint(f"  Party Size: {final_info.get('party_size', 1)}", "blue")
        cprint(f"  Badges: {final_info.get('badges', 0)}", "blue")
        cprint(f"  Level Sum: {final_info.get('level_sum', 5)}", "blue")
        
        # Save progress plot
        if save_stats:
            save_dir = Path(MODEL_DIR) / "evaluation_results"
            save_dir.mkdir(exist_ok=True)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(step_rewards)
            plt.xlabel('Steps')
            plt.ylabel('Cumulative Reward')
            plt.title('Reward Progress')
            
            plt.subplot(1, 2, 2)
            plt.plot(party_sizes)
            plt.xlabel('Steps')
            plt.ylabel('Party Size')
            plt.title('Party Growth')
            
            plt.tight_layout()
            plt.savefig(save_dir / f"progress_ep{episode + 1}.png")
            plt.close()
        
        env.close()
    
    # Print summary
    cprint("\n" + "="*50, "yellow")
    cprint("📊 EVALUATION SUMMARY", "yellow")
    cprint("="*50, "yellow")
    
    cprint(f"\nAverage Reward: {np.mean(all_stats['rewards']):.2f} ± {np.std(all_stats['rewards']):.2f}", "green")
    cprint(f"Average Steps: {np.mean(all_stats['steps']):.0f}", "green")
    cprint(f"Average Party Size: {np.mean(all_stats['party_size']):.1f}", "green")
    cprint(f"Average Badges: {np.mean(all_stats['badges']):.1f}", "green")
    cprint(f"Max Badges: {max(all_stats['badges'])}", "cyan")
    
    return all_stats

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Pokemon Silver PPO Agent')
    parser.add_argument('--model', type=str, default=None, help='Path to specific model checkpoint')
    parser.add_argument('--episodes', type=int, default=3, help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    
    args = parser.parse_args()
    
    # Use specific model or final
    if args.model:
        model_path = args.model
    elif os.path.exists(os.path.join(MODEL_DIR, "best_model", "best_model.zip")):
        model_path = os.path.join(MODEL_DIR, "best_model", "best_model.zip")
        cprint("📁 Using best model", "green")
    else:
        model_path = os.path.join(MODEL_DIR, "final_model.zip")
        cprint("📁 Using final model", "green")
    
    evaluate_model(
        model_path, 
        num_episodes=args.episodes,
        render=not args.no_render,
        save_stats=True
    )

if __name__ == "__main__":
    main()