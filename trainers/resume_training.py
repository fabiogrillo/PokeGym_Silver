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
    """Trova l'ultimo checkpoint salvato"""
    if not OLD_PATH.exists():
        return None, 0
    
    checkpoints = []
    for f in OLD_PATH.glob("ppo_pokemon_silver_*.zip"):
        # Estrai il numero dal nome del file
        parts = f.stem.split('_')
        for part in reversed(parts):
            try:
                timesteps = int(part)
                checkpoints.append((f, timesteps))
                break
            except ValueError:
                continue
    
    if not checkpoints:
        # Prova a cercare altri pattern
        for f in OLD_PATH.glob("*.zip"):
            if f.name not in ["best_model.zip", "final_model.zip"]:
                # Default a 0 se non riusciamo a estrarre timesteps
                checkpoints.append((f, 0))
    
    if not checkpoints:
        return None, 0
    
    # Ordina per timesteps
    checkpoints.sort(key=lambda x: x[1])
    latest, timesteps = checkpoints[-1]
    
    return latest, timesteps

def copy_best_model():
    """Copia il best model se esiste"""
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
        
        # Crea nuova directory
        NEW_PATH.mkdir(parents=True, exist_ok=True)
        
        # Copia checkpoint come starting point
        start_model = NEW_PATH / "start_model.zip"
        shutil.copy2(checkpoint, start_model)
        cprint(f"📁 Copied to: {start_model}", "green")
        
        # Copia best model se esiste
        copy_best_model()
        
        # Mostra comando per riprendere
        cprint("\n📝 To resume training with optimized settings, run:", "yellow")
        cprint(f"   python trainers/train_agent_optimized.py --resume {start_model} --timesteps {timesteps}", "cyan")
        
        return str(start_model), timesteps
    else:
        cprint("❌ No checkpoints found!", "red")
        return None, 0

if __name__ == "__main__":
    main()