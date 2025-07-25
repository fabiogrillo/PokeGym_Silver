import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.pokemon_silver_env_v2 import PokemonSilverV2
from termcolor import cprint
import json
from pathlib import Path

ROM_PATH = "roms/Pokemon_Silver.gbc"
MODEL_DIR = "trained_agents/exploration_v4"

def evaluate_model(model_path, num_episodes=5, render=True, save_stats=True):
    """Evaluate the saved model"""
    
    cprint(f"📂 Loading model from {model_path}", "cyan")
    model = PPO.load(model_path)
    
    # Stats collection
    all_stats = {
        "rewards": [],
        "steps": [],
        "unique_tiles": [],
        "map_transitions": [],
        "badges": [],
        "final_maps": [],
        "exploration_maps": []
    }
    
    for episode in range(num_episodes):
        cprint(f"\n🎮 Episode {episode + 1}/{num_episodes}", "yellow")
        
        env = PokemonSilverV2(
            rom_path=ROM_PATH, 
            render_mode="human" if render else "headless",
            max_steps=20000  # Longer episodes for evaluation
        )
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        # For tracking progress during episode
        step_rewards = []
        unique_tiles_progress = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Collect step data
            step_rewards.append(total_reward)
            unique_tiles_progress.append(info.get("unique_tiles", 0))
            
            # Print progress every 500 steps
            if step_count % 500 == 0:
                cprint(f"  Step {step_count}: Reward={total_reward:.2f}, "
                      f"Tiles={info.get('unique_tiles', 0)}, "
                      f"Badges={info.get('badges', 0)}", "green")
            
            done = terminated or truncated
        
        # Collect episode stats
        final_info = info # type: ignore
        all_stats["rewards"].append(total_reward)
        all_stats["steps"].append(step_count)
        all_stats["unique_tiles"].append(final_info.get("unique_tiles", 0))
        all_stats["map_transitions"].append(final_info.get("map_transitions", 0))
        all_stats["badges"].append(final_info.get("badges", 0))
        all_stats["exploration_maps"].append(env.explore_map.copy())
        
        cprint(f"\n✅ Episode {episode + 1} Complete:", "green")
        cprint(f"  Total Reward: {total_reward:.2f}", "blue")
        cprint(f"  Steps: {step_count}", "blue")
        cprint(f"  Unique Tiles: {final_info.get('unique_tiles', 0)}", "blue")
        cprint(f"  Map Transitions: {final_info.get('map_transitions', 0)}", "blue")
        cprint(f"  Badges: {final_info.get('badges', 0)}", "blue")
        
        # Save exploration map
        if save_stats:
            save_dir = Path(MODEL_DIR) / "evaluation_results"
            save_dir.mkdir(exist_ok=True)
            
            # Save exploration heatmap
            plt.figure(figsize=(10, 10))
            plt.imshow(env.explore_map, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Visit Count')
            plt.title(f'Exploration Map - Episode {episode + 1}')
            plt.savefig(save_dir / f"exploration_map_ep{episode + 1}.png")
            plt.close()
            
            # Plot reward progress
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(step_rewards)
            plt.xlabel('Steps')
            plt.ylabel('Cumulative Reward')
            plt.title('Reward Progress')
            
            plt.subplot(1, 2, 2)
            plt.plot(unique_tiles_progress)
            plt.xlabel('Steps')
            plt.ylabel('Unique Tiles Visited')
            plt.title('Exploration Progress')
            
            plt.tight_layout()
            plt.savefig(save_dir / f"progress_ep{episode + 1}.png")
            plt.close()
        
        env.close()
    
    # Print summary statistics
    cprint("\n" + "="*50, "yellow")
    cprint("📊 EVALUATION SUMMARY", "yellow")
    cprint("="*50, "yellow")
    
    cprint(f"\nAverage Reward: {np.mean(all_stats['rewards']):.2f} ± {np.std(all_stats['rewards']):.2f}", "green")
    cprint(f"Average Steps: {np.mean(all_stats['steps']):.0f} ± {np.std(all_stats['steps']):.0f}", "green")
    cprint(f"Average Unique Tiles: {np.mean(all_stats['unique_tiles']):.0f} ± {np.std(all_stats['unique_tiles']):.0f}", "green")
    cprint(f"Average Map Transitions: {np.mean(all_stats['map_transitions']):.0f} ± {np.std(all_stats['map_transitions']):.0f}", "green")
    cprint(f"Average Badges: {np.mean(all_stats['badges']):.1f} ± {np.std(all_stats['badges']):.1f}", "green")
    cprint(f"Max Badges Achieved: {max(all_stats['badges'])}", "cyan")
    
    # Save summary stats
    if save_stats:
        save_dir = Path(MODEL_DIR) / "evaluation_results"
        save_dir.mkdir(exist_ok=True)
        
        summary = {
            "num_episodes": num_episodes,
            "avg_reward": float(np.mean(all_stats['rewards'])),
            "std_reward": float(np.std(all_stats['rewards'])),
            "avg_steps": float(np.mean(all_stats['steps'])),
            "avg_unique_tiles": float(np.mean(all_stats['unique_tiles'])),
            "avg_map_transitions": float(np.mean(all_stats['map_transitions'])),
            "avg_badges": float(np.mean(all_stats['badges'])),
            "max_badges": int(max(all_stats['badges'])),
            "all_rewards": all_stats['rewards'],
            "all_unique_tiles": all_stats['unique_tiles']
        }
        
        with open(save_dir / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Create combined exploration heatmap
        combined_map = np.zeros_like(all_stats['exploration_maps'][0], dtype=np.float32)
        for exp_map in all_stats['exploration_maps']:
            combined_map += (exp_map > 0).astype(np.float32)
        combined_map /= num_episodes
        
        plt.figure(figsize=(12, 12))
        plt.imshow(combined_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Visit Frequency')
        plt.title('Combined Exploration Heatmap (All Episodes)')
        plt.savefig(save_dir / "combined_exploration_map.png", dpi=150)
        plt.close()
        
        cprint(f"\n💾 Results saved to {save_dir}", "green")
    
    return all_stats

def compare_checkpoints(model_dir, num_episodes=3):
    """Compare different model checkpoints"""
    
    checkpoints = sorted([
        f for f in os.listdir(model_dir) 
        if f.startswith("ppo_pokemon_silver_") and f.endswith(".zip")
    ])
    
    if not checkpoints:
        cprint("❌ No checkpoints found!", "red")
        return
    
    cprint(f"📊 Found {len(checkpoints)} checkpoints to evaluate", "cyan")
    
    results = {}
    
    for checkpoint in checkpoints[-5:]:  # Evaluate only last 5
        checkpoint_path = os.path.join(model_dir, checkpoint)
        cprint(f"\n🔍 Evaluating {checkpoint}...", "yellow")
        
        stats = evaluate_model(checkpoint_path, num_episodes=num_episodes, render=False, save_stats=False)
        
        results[checkpoint] = {
            "avg_reward": np.mean(stats['rewards']),
            "avg_tiles": np.mean(stats['unique_tiles']),
            "avg_badges": np.mean(stats['badges'])
        }
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    checkpoints_names = list(results.keys())
    avg_rewards = [results[c]["avg_reward"] for c in checkpoints_names]
    avg_tiles = [results[c]["avg_tiles"] for c in checkpoints_names]
    avg_badges = [results[c]["avg_badges"] for c in checkpoints_names]
    
    plt.subplot(1, 3, 1)
    plt.bar(range(len(checkpoints_names)), avg_rewards)
    plt.xlabel('Checkpoint')
    plt.ylabel('Average Reward')
    plt.title('Reward Progression')
    plt.xticks(range(len(checkpoints_names)), [c.split('_')[-1].split('.')[0] for c in checkpoints_names], rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.bar(range(len(checkpoints_names)), avg_tiles)
    plt.xlabel('Checkpoint')
    plt.ylabel('Average Unique Tiles')
    plt.title('Exploration Progression')
    plt.xticks(range(len(checkpoints_names)), [c.split('_')[-1].split('.')[0] for c in checkpoints_names], rotation=45)
    
    plt.subplot(1, 3, 3)
    plt.bar(range(len(checkpoints_names)), avg_badges)
    plt.xlabel('Checkpoint')
    plt.ylabel('Average Badges')
    plt.title('Badge Progression')
    plt.xticks(range(len(checkpoints_names)), [c.split('_')[-1].split('.')[0] for c in checkpoints_names], rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "checkpoint_comparison.png"))
    plt.show()
    
    # Best checkpoint
    best_checkpoint = max(results.items(), key=lambda x: x[1]["avg_reward"])[0]
    cprint(f"\n🏆 Best checkpoint: {best_checkpoint}", "green")
    cprint(f"   Avg Reward: {results[best_checkpoint]['avg_reward']:.2f}", "blue")
    cprint(f"   Avg Tiles: {results[best_checkpoint]['avg_tiles']:.0f}", "blue")
    cprint(f"   Avg Badges: {results[best_checkpoint]['avg_badges']:.1f}", "blue")

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Pokemon Silver PPO Agent')
    parser.add_argument('--model', type=str, default=None, help='Path to specific model checkpoint')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--compare', action='store_true', help='Compare multiple checkpoints')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_checkpoints(MODEL_DIR, num_episodes=3)
    else:
        # Use specific model or best/final
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