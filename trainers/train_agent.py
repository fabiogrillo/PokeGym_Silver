import os
import subprocess
from tqdm import tqdm
from termcolor import cprint
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from envs.pokemon_silver_env import PokemonSilver

ROM_PATH = "roms/Pokemon_Silver.gbc"
SAVE_PATH = "trained_agents/exploration_v2"
TENSORBOARD_LOG = "./tensorboard/exploration_v2"

# Configurazione training
TOTAL_TIMESTEPS = 10_000_000  # 10M steps
CHECKPOINT_INTERVAL = 50_000
EVAL_FREQ = 100_000
N_ENVS = 16  # Usa più env paralleli con la tua hardware
BATCH_SIZE = 2048
N_STEPS = 512  # Steps per environment prima di update
LEARNING_RATE = 2.5e-4
ENT_COEF = 0.01
CLIP_RANGE = 0.2
GAE_LAMBDA = 0.95
GAMMA = 0.99

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(TENSORBOARD_LOG, exist_ok=True)

class ProgressCallback(BaseCallback):
    """Callback per logging dettagliato"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log custom metrics ogni 1000 steps
        if self.n_calls % 1000 == 0:
            infos = self.locals.get("infos", [])
            for info in infos:
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
    """Crea environment con seed unico"""
    def _init():
        env = PokemonSilver(
            rom_path=ROM_PATH, 
            render_mode="headless",
            max_steps=2048*40  # Episodi più corti per training più veloce
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

def main():
    cprint("🚀 Training Pokemon Silver Agent...", "cyan")
    cprint(f"Hardware: RTX 5080 (16GB), AMD 9800X3D, 64GB RAM", "blue")
    
    # Setup environments
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    
    # Test environment per eval
    eval_env = SubprocVecEnv([make_env(N_ENVS + i) for i in range(4)])
    
    # Setup logger
    new_logger = configure(TENSORBOARD_LOG, ["tensorboard"])
    
    # Crea modello PPO ottimizzato
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=linear_schedule(LEARNING_RATE),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=4,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        clip_range_vf=None,
        ent_coef=ENT_COEF,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,  # Non usare SDE per questo task
        sde_sample_freq=-1,
        policy_kwargs={
            "net_arch": dict(
                pi=[512, 512],  # Policy network più grande
                vf=[512, 512]   # Value network più grande
            ),
            "activation_fn": torch.nn.ReLU,
            "normalize_images": True,
        },
        tensorboard_log=TENSORBOARD_LOG,
        verbose=1,
        device="cuda",  # Usa GPU
        seed=42
    )
    
    model.set_logger(new_logger)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_INTERVAL // N_ENVS,
        save_path=SAVE_PATH,
        name_prefix="ppo_pokemon_silver",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_PATH, "best_model"),
        log_path=os.path.join(SAVE_PATH, "eval_logs"),
        eval_freq=EVAL_FREQ // N_ENVS,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        warn=False,
    )
    
    progress_callback = ProgressCallback()
    
    callbacks = [checkpoint_callback, eval_callback, progress_callback]
    
    # Training
    try:
        cprint("🎮 Starting training...", "green")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False,
        )
        
    except KeyboardInterrupt:
        cprint("\n⏹️ Training interrupted by user", "red")
        
    finally:
        # Salva modello finale
        final_model_path = os.path.join(SAVE_PATH, "final_model")
        model.save(final_model_path)
        cprint(f"💾 Model saved to {final_model_path}", "green")
        
        # Salva metadata
        import json
        metadata = {
            "total_timesteps": model.num_timesteps,
            "n_envs": N_ENVS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "architecture": "PPO with MultiInputPolicy",
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
        }
        
        with open(os.path.join(SAVE_PATH, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Chiudi environments
        env.close()
        eval_env.close()
        
        cprint("✅ Training completed!", "green")

if __name__ == "__main__":
    # Lancia TensorBoard in background
    tb_process = subprocess.Popen(
        ["tensorboard", "--logdir", TENSORBOARD_LOG, "--port", "6006"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    cprint("📊 TensorBoard running at http://localhost:6006", "blue")
    
    try:
        main()
    finally:
        # Termina TensorBoard
        tb_process.terminate()