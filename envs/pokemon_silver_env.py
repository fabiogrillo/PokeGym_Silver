import gymnasium
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import cv2

class PokemonSilver(gymnasium.Env):
    """
    Gym Environment that uses PyBoy to emulate Pokemon Silver
    """

    def __init__(self, rom_path):
        super().__init__()

        self.rom_path = rom_path
        self.frame_counter = 0 # for saving purposes
        # Starting Pyboy no window
        self.pyboy = PyBoy(rom_path, window="null")
        self.pyboy.set_emulation_speed(0)

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
        self.pyboy = PyBoy(self.rom_path, window="null")
        self.pyboy.set_emulation_speed(0)

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
        if action == 1:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        if action == 2:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
        if action == 3:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        if action == 4:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
        if action == 5:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
        if action == 6:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
        if action == 7:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)

        # 0 -> no input

        # move 10 frames on to see result
        for i in range(10):
            self.pyboy.tick()
            
            if i == 9:
                frame = np.array(self.pyboy.screen.image)
                frame_path = f"frames/frame_{self.frame_counter:05d}.png"
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.frame_counter += 1
            

        obs = self._get_observation()
        reward = 0 # custom reward, 0 for now
        done = False # define episode's end criteria
        
        info = {}

        return obs, reward, done, info
    
    def render(self, mode="human"):
        # show window
        if mode == 'human':
            pyboy = PyBoy(self.rom_path, window="SDL2")
            
            while not pyboy._stopped:
                pyboy.tick()

    def close(self):
        self.pyboy.stop()

    def _get_observation(self):
        pil_image = self.pyboy.screen.image
        # Convert in numpy array (RGB)
        rgb_array = np.array(pil_image)
        # convert in gray scale
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        return gray


