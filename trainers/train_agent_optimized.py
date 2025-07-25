import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm
from termcolor import cprint
import torch

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from envs.pokemon_silver_env import PokemonSilver

# Paths assoluti per evitare problemi
BASE_DIR = Path(__file__).parent.parent
ROM_PATH = BASE_DIR / "roms/Pokemon_Silver.gbc"
SAVE_PATH = BASE_DIR / "trained_agents/exploration_v3"
TENSORBOARD_LOG = BASE_DIR / "tensorboard/exploration_v3"

# Configurazione ottimizzata per RTX 5080
TOTAL_TIMESTEPS = 10_000_000
CHECKPOINT_INTERVAL = 100_000  # Checkpoint più frequenti
EVAL_FREQ = 500_000  # Evaluation meno frequente (causa delle pause!)
N_ENVS = 32  # Più environments per saturare GPU
BATCH_SIZE = 8192  # Batch molto più grande per GPU
N_STEPS = 256  # Steps ridotti per update più frequenti
LEARNING_RATE = 3e-4
ENT_COEF = 0.01
CLIP_RANGE = 0.2
GAE_LAMBDA = 0.95
GAMMA = 0.99
N_EPOCHS = 4
MINIBATCH_SIZE = 256  # Minibatch per ottimizzare GPU usage

# Crea directories
SAVE_PATH.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG.mkdir(parents=True, exist_ok=True)

class OptimizedProgressCallback(BaseCallback):
    """Callback ottimizzato con logging meno frequente"""
    def __init__(self, log_interval=5000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        
    def _on_step(self) -> bool:
        # Log solo ogni log_interval steps per ridurre overhead
        if self.n_calls % self.log_interval == 0:
            infos = self.locals.get("infos", [])
            if infos:
                # Prendi solo il primo env per evitare spam
                info = infos[0]
                if "total_reward" in info:
                    self.logger.record("game/total_reward", info["total_reward"])
                if "unique_tiles" in info:
                    self.logger.record("game/unique_tiles", info["unique_tiles"])
                if "map_transitions" in info:
                    self.logger.record("game/map_transitions", info["map_transitions"])
                if "badges" in info:
                    self.logger.record("game/badges", info["badges"])
        return True

def make_env(rank, seed=0):
    """Crea environment con configurazione ottimizzata"""
    def _init():
        env = PokemonSilver(
            rom_path=str(ROM_PATH),
            render_mode="headless",
            max_steps=1024  # Episodi MOLTO più corti per training veloce
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def linear_schedule(initial_value):
    """Learning rate schedule"""
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def main(resume_path=None, start_timesteps=0):
    cprint("🚀 Training Pokemon Silver Agent (Optimized for RTX 5080)...", "cyan")
    cprint(f"📁 Base directory: {BASE_DIR}", "blue")
    cprint(f"📊 TensorBoard directory: {TENSORBOARD_LOG}", "blue")
    cprint(f"💾 Save directory: {SAVE_PATH}", "blue")
    cprint(f"📁 Base directory: {BASE_DIR}", "blue")
    cprint(f"📊 TensorBoard directory: {TENSORBOARD_LOG}", "blue")
    cprint(f"💾 Save directory: {SAVE_PATH}", "blue")
    
    # Verifica CUDA
    if torch.cuda.is_available():
        cprint(f"✅ CUDA available! Device: {torch.cuda.get_device_name(0)}", "green")
        cprint(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", "green")
    else:
        cprint("⚠️ CUDA not available, using CPU", "yellow")
    
    # Setup environments
    cprint(f"🎮 Creating {N_ENVS} parallel environments...", "cyan")
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    
    # Eval env con meno episodi
    eval_env = SubprocVecEnv([make_env(N_ENVS + i) for i in range(2)])  # Solo 2 eval envs
    
    # Setup logger con path assoluto
    new_logger = configure(str(TENSORBOARD_LOG), ["tensorboard", "stdout"])
    
    # Crea o carica modello
    if resume_path and os.path.exists(resume_path):
        cprint(f"📂 Resuming from checkpoint: {resume_path}", "yellow")
        cprint(f"   Starting from timestep: {start_timesteps:,}", "yellow")
        
        model = PPO.load(
            resume_path,
            env=env,
            tensorboard_log=str(TENSORBOARD_LOG),
            device="cuda",
        )
        
        # Reset num_timesteps se necessario
        if start_timesteps > 0:
            model.num_timesteps = start_timesteps
        
        # Aggiorna learning rate schedule
        model.learning_rate = linear_schedule(LEARNING_RATE)
        
        cprint("✅ Model loaded successfully!", "green")
    else:
        cprint("🆕 Creating new model...", "cyan")
        
        # Crea modello PPO con configurazione ottimizzata per GPU
        policy_kwargs = {
            "net_arch": dict(
                pi=[512, 512, 256],  # Network più profondo
                vf=[512, 512, 256]
            ),
            "activation_fn": torch.nn.ReLU,
            "normalize_images": True,
            "share_features_extractor": False,  # Reti separate per policy e value
        }
        
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=linear_schedule(LEARNING_RATE),
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            clip_range_vf=None,
            ent_coef=ENT_COEF,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(TENSORBOARD_LOG),
            verbose=1,
            device="cuda",
            seed=42
        )
    
    model.set_logger(new_logger)
    
    # Callbacks ottimizzati
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_INTERVAL // N_ENVS,
        save_path=str(SAVE_PATH),
        name_prefix="ppo_pokemon_silver",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Eval callback con meno episodi e meno frequente
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(SAVE_PATH / "best_model"),
        log_path=str(SAVE_PATH / "eval_logs"),
        eval_freq=EVAL_FREQ // N_ENVS,  # Molto meno frequente!
        deterministic=True,
        render=False,
        n_eval_episodes=2,  # Solo 2 episodi invece di 5
        warn=False,
    )
    
    progress_callback = OptimizedProgressCallback(log_interval=5000)
    
    callbacks = [checkpoint_callback, eval_callback, progress_callback]
    
    # Training con gestione interruzioni
    try:
        cprint("🎮 Starting optimized training...", "green")
        cprint(f"   Environments: {N_ENVS}", "blue")
        cprint(f"   Batch size: {BATCH_SIZE}", "blue")
        cprint(f"   Minibatch size: {MINIBATCH_SIZE}", "blue")
        cprint(f"   Update frequency: every {N_STEPS * N_ENVS} steps", "blue")
        
        # Calcola timesteps rimanenti
        remaining_timesteps = TOTAL_TIMESTEPS - start_timesteps
        if remaining_timesteps <= 0:
            cprint("⚠️ Training already completed!", "yellow")
            return
        
        cprint(f"   Remaining timesteps: {remaining_timesteps:,}", "green")
        
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            log_interval=1,  # Log ogni update
            progress_bar=True,
            reset_num_timesteps=False,  # Importante per resume!
        )
        
    except KeyboardInterrupt:
        cprint("\n⏹️ Training interrupted by user", "red")
        
    finally:
        # Salva modello corrente
        interrupted_path = SAVE_PATH / f"interrupted_{model.num_timesteps}"
        model.save(str(interrupted_path))
        cprint(f"💾 Current model saved to {interrupted_path}", "yellow")
        
        # Salva anche come final model
        final_model_path = SAVE_PATH / "final_model"
        model.save(str(final_model_path))
        cprint(f"💾 Final model saved to {final_model_path}", "green")
        
        # Salva metadata
        import json
        metadata = {
            "total_timesteps_completed": int(model.num_timesteps),
            "total_timesteps_target": TOTAL_TIMESTEPS,
            "n_envs": N_ENVS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "best_mean_reward": getattr(model, 'best_mean_reward', 'N/A'),
        }
        
        with open(SAVE_PATH / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Chiudi environments
        env.close()
        eval_env.close()
        
        cprint("✅ Training completed/interrupted gracefully!", "green")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    parser.add_argument("--timesteps", type=int, default=0, help="Timesteps already completed")
    args = parser.parse_args()
    
    # Check if TensorBoard is already running
    tb_port = 6006
    tb_running = False
    
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', tb_port))
        s.close()
        tb_running = True
        cprint(f"📊 TensorBoard already running at http://localhost:{tb_port}", "yellow")
    except:
        pass
    
    if not tb_running:
        # Lancia TensorBoard con path assoluto
        tb_process = subprocess.Popen(
            ["tensorboard", "--logdir", str(TENSORBOARD_LOG), "--port", str(tb_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        cprint(f"📊 TensorBoard launched at http://localhost:{tb_port}", "blue")
        cprint(f"   Monitoring: {TENSORBOARD_LOG}", "blue")
    
    try:
        main(resume_path=args.resume, start_timesteps=args.timesteps)
    finally:
        if not tb_running and 'tb_process' in locals():
            tb_process.terminate()