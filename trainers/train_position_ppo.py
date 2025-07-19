from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.pokemon_silver_env import PokemonSilver

def make_env():
    def _init():
        env = PokemonSilver(
            rom_path="roms/Pokemon_Silver.gbc",
            render_mode="headless",
            reward_strategy="position"
        )
        return env
    return _init

if __name__ == "__main__":
    num_envs = 10
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/position")

    # Checkpoint every 100,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path="outputs/models_position/",
        name_prefix="ppo_pokemon_position"
    )

    total_timesteps = 10000 * 500
    model.learn(total_timesteps=total_timesteps,
                callback=checkpoint_callback)

    model.save("ppo_pokemon_position")
    env.close()
