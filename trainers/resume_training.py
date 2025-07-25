import os
import sys
from pathlib import Path
from termcolor import cprint
import shutil

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = Path(__file__).parent.parent
OLD_PATH = BASE_DIR / "trained_agents/exploration_v2"
NEW_PATH = BASE_DIR / "trained_agents/exploration_v3"

def find_latest_checkpoint():
    """Find the latest saved checkpoint"""
    if not OLD_PATH.exists():
        return None, 0
    
    checkpoints = sorted([f for f in OLD_PATH.glob("ppo_pokemon_silver_*.zip")])
    if not checkpoints:
        return None, 0
    
    latest = checkpoints[-1]
    # Extract timesteps from filename
    # Handle both formats: ppo_pokemon_silver_3700000.zip and ppo_pokemon_silver_3700000_steps.zip
    stem_parts = latest.stem.split('_')
    
    # Find the numeric part
    timesteps = 0
    for part in stem_parts:
        try:
            timesteps = int(part)
            break
        except ValueError:
            continue
    
    return latest, timesteps

def copy_best_model():
    """Copy the best model if it exists"""
    old_best = OLD_PATH / "best_model/best_model.zip"
    if old_best.exists():
        new_best_dir = NEW_PATH / "best_model"
        new_best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_best, new_best_dir / "best_model.zip")
        cprint("✅ Copied best model", "green")

def main():
    cprint("🔍 Looking for existing checkpoints...", "cyan")
    
    checkpoint, timesteps = find_latest_checkpoint()
    
    if checkpoint:
        cprint(f"✅ Found checkpoint: {checkpoint.name}", "green")
        cprint(f"   Timesteps completed: {timesteps:,}", "blue")
        
        # Create new directory
        NEW_PATH.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint as starting point
        start_model = NEW_PATH / "start_model.zip"
        shutil.copy2(checkpoint, start_model)
        cprint(f"📁 Copied to: {start_model}", "green")
        
        # Copy best model if exists
        copy_best_model()
        
        # Show command to resume
        cprint("\n📝 To resume training with optimized settings, run:", "yellow")
        cprint(f"   python trainers/train_agent_optimized.py --resume {start_model} --timesteps {timesteps}", "cyan")
        
        return str(start_model), timesteps
    else:
        cprint("❌ No checkpoints found!", "red")
        return None, 0

if __name__ == "__main__":
    main()