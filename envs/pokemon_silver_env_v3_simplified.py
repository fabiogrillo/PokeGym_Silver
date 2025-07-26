# envs/pokemon_silver_env_v3_simplified.py
import gymnasium
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import os
from skimage.transform import downscale_local_mean
from pathlib import Path
import mediapy as media

class PokemonSilverV3Simplified(gymnasium.Env):
    """
    Simplified Environment without map/event observations.
    Optimized for learning basic movement and battles.
    """

    def __init__(self, rom_path, render_mode="headless", max_steps=2048, 
                 save_video=False, video_dir="rollouts", start_state="post_starter.state"):
        super().__init__()

        self.rom_path = rom_path
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.act_freq = 24
        self.frame_stacks = 3
        self.reset_count = 0
        self.save_video = save_video
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(exist_ok=True)
        self.start_state = start_state  # Use post-starter state
        
        # Video writers
        self.full_frame_writer = None
        self.model_frame_writer = None

        # PyBoy setup
        if render_mode == "human":
            self.pyboy = PyBoy(rom_path, window="SDL2")
            self.pyboy.set_emulation_speed(3)
        elif render_mode == "human-fast":
            self.pyboy = PyBoy(rom_path, window="SDL2")
            self.pyboy.set_emulation_speed(6)
        elif render_mode == "headless":
            self.pyboy = PyBoy(rom_path, window="null")
            self.pyboy.set_emulation_speed(0)
        else:
            raise ValueError(f"Unknown render mode: {render_mode}")
        
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        self.output_shape = (72, 80, self.frame_stacks)
        self.enc_freqs = 8

        # Action space
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # Simplified observation space
        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks),
                "badges": spaces.MultiBinary(8),  # Just Johto badges
                "party_size": spaces.Box(low=0, high=6, shape=(1,), dtype=np.uint8),
                "health": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,), dtype=np.float32),
            }
        )

    def reset(self, *, seed=None, options=None):
        self.seed = seed

        # Load initial state
        with open(self.start_state, "rb") as f:
            self.pyboy.load_state(f)

        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        # Tracking variables
        self.step_count = 0
        self.total_reward = 0
        
        # Game state tracking
        self.last_party_size = self.get_party_size()
        self.last_badges = self.get_badges_sum()
        self.last_level_sum = self.get_levels_sum()
        self.last_health = self.read_hp_fraction()
        self.last_position = self.get_position()
        
        # Movement tracking
        self.positions_history = []
        self.stuck_counter = 0
        
        # Battle tracking
        self.battles_won = 0
        self.wild_pokemon_seen = 0
        
        self.reset_count += 1
        
        if self.save_video:
            self.start_video()
        
        return self._get_obs(), {}

    def render(self, reduce_res=True):
        game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1]
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1))
            ).astype(np.uint8)
        return game_pixels_render
    
    def _get_obs(self):
        screen = self.render()
        self.update_recent_screens(screen)

        level_sum = 0.02 * self.get_levels_sum()

        observation = {
            "screens": self.recent_screens,
            "recent_actions": self.recent_actions.copy(),
            "badges": self.get_badges_array()[:8],  # Only Johto
            "party_size": np.array([self.get_party_size()], dtype=np.uint8),
            "health": np.array([self.read_hp_fraction()], dtype=np.float32),
            "level": self.fourier_encode(level_sum),
        }

        return observation
    
    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs)).astype(np.float32)
    
    def get_position(self):
        x = self.read_m(0xDA02)
        y = self.read_m(0xDA03)
        map_id = self.read_m(0xDA01)
        return (x, y, map_id)
    
    def get_party_size(self):
        return self.read_m(0xDA22)
    
    def get_badges_sum(self):
        return bin(self.read_m(0xD57C)).count('1')
    
    def get_badges_array(self):
        badges = self.read_m(0xD57C)
        return np.array([int(x) for x in format(badges, '08b')[::-1]], dtype=np.int8)
    
    def get_levels_sum(self):
        return sum([self.read_m(0xDA49 + i*0x30) for i in range(6)])
    
    def read_hp_fraction(self):
        hp_sum = 0
        max_hp_sum = 0
        
        for i in range(6):
            curr_hp = self.read_m(0xDA4C + i*0x30) * 256 + self.read_m(0xDA4D + i*0x30)
            max_hp = self.read_m(0xDA4E + i*0x30) * 256 + self.read_m(0xDA4F + i*0x30)
            hp_sum += curr_hp
            max_hp_sum += max_hp
        
        return hp_sum / max(max_hp_sum, 1)
    
    def read_m(self, address):
        return self.pyboy.memory[address]
    
    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:,:,0] = cur_screen[:,:,0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action
    
    def check_if_stuck(self):
        """Check if player is stuck in same position"""
        current_pos = self.get_position()
        self.positions_history.append(current_pos)
        
        # Keep last 50 positions
        if len(self.positions_history) > 50:
            self.positions_history.pop(0)
        
        # Check if stuck in same spot
        if len(self.positions_history) >= 20:
            unique_positions = len(set(self.positions_history[-20:]))
            if unique_positions <= 2:  # Stuck in 1-2 positions
                self.stuck_counter += 1
                return True
        
        self.stuck_counter = 0
        return False
    
    def calculate_reward(self):
        """Simplified reward focused on progression"""
        reward = 0
        
        # 1. Party growth reward (catching Pokemon)
        party_size = self.get_party_size()
        if party_size > self.last_party_size:
            reward += 50 * (party_size - self.last_party_size)
            print(f"🎉 Caught Pokemon! Party size: {party_size}")
        self.last_party_size = party_size
        
        # 2. Badge reward
        badges = self.get_badges_sum()
        if badges > self.last_badges:
            reward += 100 * (badges - self.last_badges)
            print(f"🏅 Got badge! Total: {badges}")
        self.last_badges = badges
        
        # 3. Level up reward
        level_sum = self.get_levels_sum()
        if level_sum > self.last_level_sum:
            reward += 5 * (level_sum - self.last_level_sum)
        self.last_level_sum = level_sum
        
        # 4. Exploration reward (position changes)
        current_pos = self.get_position()
        if current_pos != self.last_position:
            reward += 0.1  # Small reward for movement
        self.last_position = current_pos
        
        # 5. Battle reward (HP changes)
        current_hp = self.read_hp_fraction()
        in_battle = self.read_m(0xD116) != 0
        
        if in_battle:
            self.wild_pokemon_seen += 0.01  # Small reward for encountering Pokemon
            reward += 0.1
        
        # 6. Stuck penalty
        if self.check_if_stuck():
            reward -= 0.5 * self.stuck_counter
        
        # 7. Time penalty (small)
        reward -= 0.01
        
        return reward
    
    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()
            
        self.run_action_on_emulator(action)
        self.update_recent_actions(action)

        reward = self.calculate_reward()
        self.total_reward += reward
        
        terminated = False
        truncated = self.step_count >= self.max_steps - 1
        
        obs = self._get_obs()
        
        self.step_count += 1
        
        if self.save_video:
            self.add_video_frame()
        
        info = {
            "step": self.step_count,
            "total_reward": self.total_reward,
            "party_size": self.get_party_size(),
            "badges": self.get_badges_sum(),
            "level_sum": self.get_levels_sum(),
            "hp_fraction": self.read_hp_fraction(),
        }
        
        return obs, reward, terminated, truncated, info

    def run_action_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        
        render_screen = self.save_video or self.render_mode != "headless"
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)
    
    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        
        base_name = f"episode_{self.reset_count}_reward_{self.total_reward:.0f}"
        
        full_path = self.video_dir / f"full_{base_name}.mp4"
        self.full_frame_writer = media.VideoWriter(
            str(full_path), (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        
        model_path = self.video_dir / f"model_{base_name}.mp4"
        self.model_frame_writer = media.VideoWriter(
            str(model_path), self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
    
    def add_video_frame(self):
        if self.full_frame_writer:
            self.full_frame_writer.add_image(self.render(reduce_res=False)[:,:,0])
        if self.model_frame_writer:
            self.model_frame_writer.add_image(self.render(reduce_res=True)[:,:,0])
    
    def close(self):
        if self.save_video:
            if self.full_frame_writer:
                self.full_frame_writer.close()
            if self.model_frame_writer:
                self.model_frame_writer.close()
        
        self.pyboy.stop()