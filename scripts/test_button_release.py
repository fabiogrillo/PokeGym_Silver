from envs.pokemon_silver_env_v3_simplified import PokemonSilverV3Simplified
import numpy as np
from termcolor import cprint
import time

def test_buttons():
    """Test button press and release"""
    env = PokemonSilverV3Simplified(
        rom_path="roms/Pokemon_Silver.gbc",
        render_mode="human",
        max_steps=1000,
        start_state="post_starter.state"
    )
    
    obs, _ = env.reset()
    
    cprint("🎮 Testing button press/release...", "cyan")
    
    # Test sequence
    test_actions = [
        (3, "UP", 5),      # Move up 5 times
        (4, "A", 3),       # Press A 3 times
        (0, "DOWN", 5),    # Move down 5 times
        (4, "A", 3),       # Press A again
        (1, "LEFT", 5),    # Move left
        (2, "RIGHT", 5),   # Move right
    ]
    
    for action_id, action_name, repeats in test_actions:
        cprint(f"\n🔹 Testing {action_name} x{repeats}", "yellow")
        
        for i in range(repeats):
            obs, reward, _, _, info = env.step(action_id)
            
            if reward != 0:
                cprint(f"   Step {i+1}: Reward = {reward:.2f}", "green")
            
            time.sleep(0.1)  # Slow down to see actions
        
        # Pause between different actions
        time.sleep(0.5)
    
    cprint("\n✅ If character moved in all directions and didn't get stuck, button release is working!", "green")
    
    env.close()

if __name__ == "__main__":
    test_buttons()