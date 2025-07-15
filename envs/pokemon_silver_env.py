import gymnasium
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import cv2
import os

from .rewards.hashing_reward import HashingReward
from .rewards.position_reward import PositionReward

class PokemonSilver(gymnasium.Env):
    """
    Gym Environment that uses PyBoy to emulate Pokemon Silver
    """

    def __init__(self, rom_path, render_mode="headless", save_frames=False, reward_strategy='hashing'):
        super().__init__()

        self.rom_path = rom_path
        self.frame_counter = 0 # for saving purposes
        self.save_frames = save_frames
        self.render_mode = render_mode

        # Mode configuration
        if render_mode == 'human':
            self.pyboy = PyBoy(rom_path, window="SDL2")
            self.pyboy.set_emulation_speed(1) # normal speed
        elif render_mode == 'human-fast':
            self.pyboy = PyBoy(rom_path, window="SDL2")
            self.pyboy.set_emulation_speed(0) # max speed
        elif render_mode == 'headless':
            self.pyboy = PyBoy(rom_path, window="null")
            self.pyboy.set_emulation_speed(0) # max speed
        else:
            raise ValueError(f"Unknown render mode: {render_mode}")

        # Set reward startegy
        if reward_strategy == "hashing":
            self.reward_strategy = HashingReward()
        elif reward_strategy == "position":
            self.reward_strategy = PositionReward(self.pyboy)
        else:
            raise ValueError("Invalid reward strategy")

        # Action space
        """
        0: no input
        1: A
        2: B
        3: Start
        4: Up
        5: Down
        6: Left
        7: Right
        """
        self.action_space = spaces.Discrete(8)

        # Observation Space will be a grey image 160x144  (GameBoy resolution)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(160,144),
            dtype=np.uint8
        )

    def reset(self):
        # Restart the rom
        self.pyboy.stop()
        
        if self.render_mode == 'human':
            self.pyboy = PyBoy(self.rom_path, window="SDL2")
            self.pyboy.set_emulation_speed(1)
        elif self.render_mode == 'human-fast':
            self.pyboy = PyBoy(self.rom_path, window="SDL2")
            self.pyboy.set_emulation_speed(0)
        elif self.render_mode == 'headless':
            self.pyboy = PyBoy(self.rom_path, window="null")
            self.pyboy.set_emulation_speed(0)

        # Load state file to skip intro
        try:
            with open("start_of_game.state", "rb") as f:
                self.pyboy.load_state(f)
        except FileNotFoundError:
            print("[WARNING] State file `start_of_game.state` not found. Starting from fresh ROM.")
        
        # TICK to stabilize
        for _ in range(50):
            self.pyboy.tick()
        
        obs = self._get_observation()
        return obs
    
    def step(self, action):
        # Clean input
        for event in [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT
        ]:
            self.pyboy.send_input(event)

        # Action map
        action_event = None
        if action == 1:
            action_event = WindowEvent.PRESS_BUTTON_A
        elif action == 2:
            action_event = WindowEvent.PRESS_BUTTON_B
        elif action == 3:
            action_event = WindowEvent.PRESS_BUTTON_START
        elif action == 4:
            action_event = WindowEvent.PRESS_ARROW_UP
        elif action == 5:
            action_event = WindowEvent.PRESS_ARROW_DOWN
        elif action == 6:
            action_event = WindowEvent.PRESS_ARROW_LEFT
        elif action == 7:
            action_event = WindowEvent.PRESS_ARROW_RIGHT

        # Hold the action during all ticks
        if action_event is not None:
            self.pyboy.send_input(action_event)

        # Advance emulator
        for _ in range(10):
            self.pyboy.tick()

        # Important: release the action after ticking
        if action_event is not None:
            release_event = {
                WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
                WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
                WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
                WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
                WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
                WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
                WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
            }[action_event]
            self.pyboy.send_input(release_event)

        obs = self._get_observation()
        reward = self.reward_strategy.compute_reward(obs)
        done = False
        info = {}

        return obs, reward, done, info

    
    def render(self, mode="human"):
        pass # windows already visible

    def close(self):
        self.pyboy.stop()

    def _get_observation(self):
        pil_image = self.pyboy.screen.image
        # Convert in numpy array (RGB)
        rgb_array = np.array(pil_image)
        # convert in gray scale
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        return gray


