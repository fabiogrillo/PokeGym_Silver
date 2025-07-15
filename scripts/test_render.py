from envs.pokemon_silver_env import PokemonSilver

def test_env(render_mode, save_frames):
    print(f"Testing {render_mode} mode with save_frames={save_frames}...")
    env = PokemonSilver("roms/Pokemon_Silver.gbc", render_mode=render_mode, save_frames=save_frames)
    obs = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()
    print(f"{render_mode} mode with save_frames={save_frames} completed.\n")

if __name__ == "__main__":
    test_env("headless", save_frames=False)
    test_env("headless", save_frames=True)
    test_env("human-fast", save_frames=False)
    test_env("human", save_frames=True)
    print("All tests completed.")
