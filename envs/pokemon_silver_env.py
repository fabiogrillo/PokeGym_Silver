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
    Gym Environment that uses PyBoy to emulate Pokemon Silver.
    """

    def __init__(self, rom_path, render_mode="headless", save_frames=False, reward_strategy="hashing", hashing_threshold=5, max_steps=2000):
        super().__init__()

        self.rom_path = rom_path
        self.save_frames = save_frames
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.step_counter = 0

        # Create frames if not exists
        if save_frames and not os.path.exists("outputs/frames"):
            os.makedirs("outputs/frames", exist_ok=True)

        # Mode configuration
        if render_mode == "human":
            self.pyboy = PyBoy(rom_path, window="SDL2")
            self.pyboy.set_emulation_speed(1)
        elif render_mode == "human-fast":
            self.pyboy = PyBoy(rom_path, window="SDL2")
            self.pyboy.set_emulation_speed(0)
        elif render_mode == "headless":
            self.pyboy = PyBoy(rom_path, window="null")
            self.pyboy.set_emulation_speed(0)
        else:
            raise ValueError(f"Unknown render mode: {render_mode}")

        # Reward strategy
        if reward_strategy == "hashing":
            self.reward_strategy = HashingReward(threshold=hashing_threshold)
        elif reward_strategy == "position":
            self.reward_strategy = PositionReward(self.pyboy)
        else:
            raise ValueError("Invalid reward strategy")

        # Action space
        self.action_space = spaces.Discrete(8)

        # Observation space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, 144, 160),
            dtype=np.uint8
        )

    def reset(self, *, seed=None, options=None):
        self.step_counter = 0

        try:
            with open("start_of_game.state", "rb") as f:
                self.pyboy.load_state(f)
        except FileNotFoundError:
            print("[WARNING] State file `start_of_game.state` not found. Starting from fresh ROM.")

        for _ in range(50):
            self.pyboy.tick()

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.step_counter += 1

        # Release all buttons
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

        # Map action
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

        if action_event is not None:
            self.pyboy.send_input(action_event)

        for _ in range(10):
            self.pyboy.tick()

        if action_event is not None:
            release_event = {
                WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
                WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
                WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
                WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
                WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
                WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
                WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
            }.get(action_event)

            if release_event:
                self.pyboy.send_input(release_event)

        obs = self._get_observation()
        reward = self.reward_strategy.compute_reward(obs)

        # Distinzione terminazione
        terminated = False  # Qui puoi mettere logica se serve un "game over"
        truncated = self.step_counter >= self.max_steps

        info = {"steps": self.step_counter}

        return obs, reward, terminated, truncated, info


    def render(self, mode="human"):
        pass

    def close(self):
        self.pyboy.stop()

    def _get_observation(self):
        pil_image = self.pyboy.screen.image
        rgb_array = np.array(pil_image)
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        gray = np.expand_dims(gray, axis=0)  # shape becomes (1, 160, 144)
        return gray
