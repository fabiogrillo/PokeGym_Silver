import os
import sys
from pathlib import Path
import numpy as np
from termcolor import cprint
import matplotlib.pyplot as plt

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from envs.pokemon_silver_env_v2 import PokemonSilverV2

BASE_DIR = Path(__file__).parent.parent
ROM_PATH = BASE_DIR / "roms/Pokemon_Silver.gbc"

def evaluate_with_video(model_path, num_episodes=3, max_steps=10000, save_best_only=True):
    """Evaluate model and record videos of gameplay"""
    
    if not os.path.exists(model_path):
        cprint(f"❌ Model not found: {model_path}", "red")
        return
    
    cprint(f"📂 Loading model: {model_path}", "cyan")
    model = PPO.load(model_path)
    
    # Create video directory
    video_dir = BASE_DIR / "evaluation_videos"
    video_dir.mkdir(exist_ok=True)
    
    # Stats for all episodes
    all_stats = {
        "rewards": [],
        "steps": [],
        "unique_tiles": [],
        "badges": [],
        "levels": [],
        "events": [],
    }
    
    best_reward = -float('inf')
    best_episode = -1
    
    for episode in range(num_episodes):
        cprint(f"\n🎮 Episode {episode + 1}/{num_episodes}", "yellow")
        
        # Create environment with video recording
        env = PokemonSilverV2(
            rom_path=str(ROM_PATH),
            render_mode="human",
            max_steps=max_steps,
            save_video=True,
            video_dir=video_dir / f"episode_{episode}" # type: ignore
        )
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        # Track progress
        step_rewards = []
        step_tiles = []
        step_levels = []
        step_events = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Collect step data
            step_rewards.append(total_reward)
            step_tiles.append(info.get("unique_tiles", 0))
            step_levels.append(info.get("level_sum", 0))
            step_events.append(info.get("events", 0))
            
            # Progress update
            if step_count % 500 == 0:
                cprint(f"  Step {step_count}: Reward={total_reward:.2f}, "
                      f"Tiles={info.get('unique_tiles', 0)}, "
                      f"Badges={info.get('badges', 0)}, "
                      f"Level={info.get('level_sum', 0)}, "
                      f"Events={info.get('events', 0)}", "green")
            
            done = terminated or truncated
        
        # Episode complete
        final_info = info # type: ignore
        episode_stats = {
            "reward": total_reward,
            "steps": step_count,
            "unique_tiles": final_info.get("unique_tiles", 0),
            "badges": final_info.get("badges", 0),
            "level_sum": final_info.get("level_sum", 0),
            "events": final_info.get("events", 0),
            "hp_fraction": final_info.get("hp_fraction", 0),
        }
        
        # Track best episode
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
        
        # Collect stats
        all_stats["rewards"].append(total_reward)
        all_stats["steps"].append(step_count)
        all_stats["unique_tiles"].append(episode_stats["unique_tiles"])
        all_stats["badges"].append(episode_stats["badges"])
        all_stats["levels"].append(episode_stats["level_sum"])
        all_stats["events"].append(episode_stats["events"])
        
        cprint(f"\n✅ Episode {episode + 1} Complete:", "green")
        cprint(f"  Total Reward: {total_reward:.2f}", "blue")
        cprint(f"  Steps: {step_count}", "blue")
        cprint(f"  Unique Tiles: {episode_stats['unique_tiles']}", "blue")
        cprint(f"  Badges: {episode_stats['badges']}", "blue")
        cprint(f"  Level Sum: {episode_stats['level_sum']}", "blue")
        cprint(f"  Events: {episode_stats['events']}", "blue")
        cprint(f"  HP: {episode_stats['hp_fraction']:.2%}", "blue")
        
        # Save episode plot
        save_episode_plot(
            video_dir / f"episode_{episode}",
            step_rewards, step_tiles, step_levels, step_events,
            episode_stats
        )
        
        # Save exploration heatmap
        plt.figure(figsize=(10, 10))
        plt.imshow(env.explore_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Visited')
        plt.title(f'Exploration Map - Episode {episode + 1}')
        plt.savefig(video_dir / f"episode_{episode}" / "exploration_map.png")
        plt.close()
        
        env.close()
        
        # Delete non-best videos if requested
        if save_best_only and episode != best_episode and best_episode >= 0:
            # Remove previous non-best episode videos
            for old_ep in range(episode):
                if old_ep != best_episode:
                    old_dir = video_dir / f"episode_{old_ep}"
                    if old_dir.exists():
                        for f in old_dir.glob("*.mp4"):
                            f.unlink()
                        cprint(f"🗑️ Removed videos from episode {old_ep + 1}", "yellow")
    
    # Summary
    cprint("\n" + "="*50, "yellow")
    cprint("📊 EVALUATION SUMMARY", "yellow")
    cprint("="*50, "yellow")
    
    cprint(f"\n🏆 Best Episode: {best_episode + 1} (Reward: {best_reward:.2f})", "green")
    cprint(f"\nAverage Performance:", "cyan")
    cprint(f"  Reward: {np.mean(all_stats['rewards']):.2f} ± {np.std(all_stats['rewards']):.2f}", "blue")
    cprint(f"  Unique Tiles: {np.mean(all_stats['unique_tiles']):.0f} ± {np.std(all_stats['unique_tiles']):.0f}", "blue")
    cprint(f"  Badges: {np.mean(all_stats['badges']):.1f} ± {np.std(all_stats['badges']):.1f}", "blue")
    cprint(f"  Level Sum: {np.mean(all_stats['levels']):.1f} ± {np.std(all_stats['levels']):.1f}", "blue")
    cprint(f"  Events: {np.mean(all_stats['events']):.1f} ± {np.std(all_stats['events']):.1f}", "blue")
    
    # Save summary plot
    save_summary_plot(video_dir, all_stats)
    
    return all_stats

def save_episode_plot(save_dir, rewards, tiles, levels, events, final_stats):
    """Save episode progress plots"""
    save_dir.mkdir(exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward progress
    ax1.plot(rewards)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Reward Progress')
    ax1.grid(True)
    
    # Exploration progress
    ax2.plot(tiles)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Unique Tiles')
    ax2.set_title('Exploration Progress')
    ax2.grid(True)
    
    # Level progress
    ax3.plot(levels)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Total Level')
    ax3.set_title('Pokemon Level Progress')
    ax3.grid(True)
    
    # Event progress
    ax4.plot(events)
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Events Completed')
    ax4.set_title('Story Progress')
    ax4.grid(True)
    
    # Add final stats as text
    stats_text = (
        f"Final Stats:\n"
        f"Reward: {final_stats['reward']:.2f}\n"
        f"Tiles: {final_stats['unique_tiles']}\n"
        f"Badges: {final_stats['badges']}\n"
        f"Level: {final_stats['level_sum']}\n"
        f"Events: {final_stats['events']}\n"
        f"HP: {final_stats['hp_fraction']:.2%}"
    )
    
    plt.figtext(0.98, 0.02, stats_text, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_dir / "progress.png", dpi=150)
    plt.close()

def save_summary_plot(save_dir, all_stats):
    """Save summary comparison plot"""
    episodes = range(1, len(all_stats["rewards"]) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards
    ax1.bar(episodes, all_stats["rewards"])
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Rewards per Episode')
    ax1.axhline(y=np.mean(all_stats["rewards"]), color='r', linestyle='--', label='Average')
    ax1.legend()
    
    # Exploration
    ax2.bar(episodes, all_stats["unique_tiles"])
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Unique Tiles')
    ax2.set_title('Exploration per Episode')
    ax2.axhline(y=np.mean(all_stats["unique_tiles"]), color='r', linestyle='--', label='Average')
    ax2.legend()
    
    # Levels
    ax3.bar(episodes, all_stats["levels"])
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Level')
    ax3.set_title('Pokemon Levels per Episode')
    ax3.axhline(y=np.mean(all_stats["levels"]), color='r', linestyle='--', label='Average')
    ax3.legend()
    
    # Events
    ax4.bar(episodes, all_stats["events"])
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Events Completed')
    ax4.set_title('Story Progress per Episode')
    ax4.axhline(y=np.mean(all_stats["events"]), color='r', linestyle='--', label='Average')
    ax4.legend()
    
    plt.suptitle('Evaluation Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / "summary.png", dpi=150)
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Pokemon Silver Agent with Video Recording')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to evaluate')
    parser.add_argument('--max-steps', type=int, default=10000, help='Max steps per episode')
    parser.add_argument('--save-all', action='store_true', help='Save all episode videos (default: best only)')
    
    args = parser.parse_args()
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        # Try to find best model
        candidates = [
            BASE_DIR / "trained_agents/exploration_v3/best_model/best_model.zip",
            BASE_DIR / "trained_agents/exploration_v2/best_model/best_model.zip",
            BASE_DIR / "trained_agents/exploration_v4/best_model/best_model.zip",
        ]
        
        model_path = None
        for candidate in candidates:
            if candidate.exists():
                model_path = str(candidate)
                break
        
        if not model_path:
            cprint("❌ No model found! Please specify with --model", "red")
            return
    
    cprint(f"🎬 Starting evaluation with video recording", "cyan")
    cprint(f"   Model: {model_path}", "blue")
    cprint(f"   Episodes: {args.episodes}", "blue")
    cprint(f"   Max steps: {args.max_steps}", "blue")
    cprint(f"   Save all videos: {args.save_all}", "blue")
    
    evaluate_with_video(
        model_path,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        save_best_only=not args.save_all
    )

if __name__ == "__main__":
    main()