import gymnasium
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import os
from skimage.transform import downscale_local_mean
from collections import deque
import json

# Carica la mappa dati
with open("map_data.json", "r") as f:
    MAP_DATA = json.load(f)

# Crea dizionario per lookup veloce
MAP_TO_GLOBAL_OFFSET = {}
for region in MAP_DATA["regions"]:
    map_id = int(region["id"])
    coords = region["coordinates"]
    MAP_TO_GLOBAL_OFFSET[map_id] = (coords[1], coords[0])  # (row, col)

PAD = 20
GLOBAL_MAP_SHAPE = (180 + PAD * 2, 180 + PAD * 2)  # Aumentato per sicurezza

class PokemonSilver(gymnasium.Env):
    """
    Gym Environment per Pokemon Silver con reward basato su esplorazione.
    """

    def __init__(self, rom_path, render_mode="headless", max_steps=2048*80, save_video=False):
        super().__init__()

        self.rom_path = rom_path
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.act_freq = 24
        self.frame_stacks = 3
        self.reset_count = 0
        self.save_video = save_video

        # Mappa essenziale per progressi (corretta per Silver)
        self.essential_map_locations = {
            v: i for i, v in enumerate([
                1,   # New Bark Town
                3,   # Cherrygrove City  
                6,   # Violet City
                12,  # Azalea Town
                16,  # Goldenrod City
                22,  # Ecruteak City
                27,  # Olivine City
                33,  # Cianwood City
                36,  # Mahogany Town
                41,  # Blackthorn City
                90,  # Indigo Plateau
                88,  # Victory Road
                93,  # Tohjo Falls
                47,  # Pallet Town (post-game)
                49,  # Viridian City (post-game)
            ])
        }

        # PyBoy setup
        if render_mode == "human":
            self.pyboy = PyBoy(rom_path, window="SDL2")
            self.pyboy.set_emulation_speed(5)
        elif render_mode == "human-fast":
            self.pyboy = PyBoy(rom_path, window="SDL2")
            self.pyboy.set_emulation_speed(0)
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
        self.coords_pad = 12

        # Action space
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # Observation space corretto
        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad*4, self.coords_pad*4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks),
                "badges": spaces.MultiBinary(16),  # 8 Johto + 8 Kanto
                "party_size": spaces.Box(low=0, high=6, shape=(1,), dtype=np.uint8),
            }
        )

    def reset(self, *, seed=None, options=None):
        self.seed = seed

        # Carica lo stato iniziale
        with open("start_of_game.state", "rb") as f:
            self.pyboy.load_state(f)
        
        self.init_map_mem()

        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.uint8)
        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.visited_coords_count = {}
        self.unique_tiles_visited = set()
        self.total_steps_in_grass = 0
        self.battles_encountered = 0
        
        self.step_count = 0
        self.last_map_id = None
        self.map_transition_count = 0
        
        # Per tracking rewards
        self.last_reward_components = {}
        self.total_reward = 0
        
        self.reset_count += 1
        
        return self._get_obs(), {}
    
    def init_map_mem(self):
        self.seen_coords = {}
        self.seen_maps = set()

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

        observation = {
            "screens": self.recent_screens,
            "map": self.get_explore_map()[:,:,None],
            "recent_actions": self.recent_actions.copy(),
            "badges": self.get_badges_array(),
            "party_size": np.array([self.get_party_size()], dtype=np.uint8),
        }

        return observation
    
    def get_badges_array(self):
        """Ottieni array di badge (8 Johto + 8 Kanto)"""
        johto_badges = self.read_m(0xD57C)
        kanto_badges = self.read_m(0xD57D)
        
        badges = np.zeros(16, dtype=np.int8)
        for i in range(8):
            badges[i] = (johto_badges >> i) & 1
            badges[i+8] = (kanto_badges >> i) & 1
        
        return badges
    
    def get_party_size(self):
        """Ottieni numero di Pokemon nel party"""
        return self.read_m(0xDA22)
    
    def get_explore_map(self):
        """Ottieni la porzione di mappa visibile centrata sul giocatore"""
        c = self.get_global_coords()
        if c is None:
            return np.zeros((self.coords_pad*4, self.coords_pad*4), dtype=np.uint8)
        
        x, y = c[0] + PAD, c[1] + PAD
        
        # Estrai la porzione visibile con padding
        x0 = max(0, x - self.coords_pad*2)
        x1 = min(self.explore_map.shape[0], x + self.coords_pad*2)
        y0 = max(0, y - self.coords_pad*2)
        y1 = min(self.explore_map.shape[1], y + self.coords_pad*2)
        
        cropped = self.explore_map[x0:x1, y0:y1]
        
        # Pad a dimensione fissa
        padded = np.zeros((self.coords_pad*4, self.coords_pad*4), dtype=np.uint8)
        h, w = cropped.shape
        cx = (self.coords_pad*4 - h) // 2
        cy = (self.coords_pad*4 - w) // 2
        padded[cx:cx+h, cy:cy+w] = cropped
        
        return padded
    
    def local_to_global(self, r, c, map_n):
        """Converti coordinate locali in globali usando map_data.json"""
        if map_n not in MAP_TO_GLOBAL_OFFSET:
            return None
        
        offset = MAP_TO_GLOBAL_OFFSET[map_n]
        return (offset[0] + r, offset[1] + c)
    
    def get_global_coords(self):
        """Ottieni coordinate globali del giocatore"""
        x_pos, y_pos, map_n = self.get_game_coords()
        return self.local_to_global(y_pos, x_pos, map_n)
    
    def get_game_coords(self):
        """Leggi coordinate dal gioco (corrette per Silver)"""
        map_n = self.read_m(0xDA00)  # Map bank
        map_id = self.read_m(0xDA01)  # Map number
        x_pos = self.read_m(0xDA02)  # X coordinate
        y_pos = self.read_m(0xDA03)  # Y coordinate
        
        # In Silver, spesso serve combinare bank e ID
        full_map_id = map_id  # Per ora usiamo solo map_id
        
        return (x_pos, y_pos, full_map_id)
    
    def read_m(self, address):
        """Leggi byte dalla memoria"""
        return self.pyboy.memory[address]
    
    def update_recent_screens(self, cur_screen):
        """Aggiorna stack di frame recenti"""
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:,:,0] = cur_screen[:,:,0]

    def update_recent_actions(self, action):
        """Aggiorna azioni recenti"""
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action
    
    def step(self, action):
        self.run_action_on_emulator(action)
        self.update_recent_actions(action)

        # Update exploration
        self.update_seen_coords()
        self.update_explore_map()
        
        # Check battle status
        in_battle = self.read_m(0xD116) != 0
        if in_battle:
            self.battles_encountered += 1

        # Calcola reward
        reward = self.calculate_reward()
        
        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps - 1
        
        obs = self._get_obs()
        
        self.step_count += 1
        
        info = {
            "step": self.step_count,
            "total_reward": self.total_reward,
            "unique_tiles": len(self.unique_tiles_visited),
            "map_transitions": self.map_transition_count,
            "battles": self.battles_encountered,
            "badges": sum(self.get_badges_array()),
        }
        
        return obs, reward, terminated, truncated, info

    def run_action_on_emulator(self, action):
        """Esegui azione sull'emulatore"""
        self.pyboy.send_input(self.valid_actions[action])
        
        render_screen = self.save_video or self.render_mode != "headless"
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)

    def update_seen_coords(self):
        """Aggiorna coordinate visitate"""
        if self.read_m(0xD116) == 0:  # Not in battle
            x_pos, y_pos, map_n = self.get_game_coords()
            
            # Track mappa corrente
            if map_n != self.last_map_id:
                self.map_transition_count += 1
                self.last_map_id = map_n
                self.seen_maps.add(map_n)
            
            # Track coordinate
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string not in self.visited_coords_count:
                self.visited_coords_count[coord_string] = 0
            self.visited_coords_count[coord_string] += 1
            
            # Track unique global tiles
            global_coord = self.get_global_coords()
            if global_coord:
                self.unique_tiles_visited.add(global_coord)

    def update_explore_map(self):
        """Aggiorna mappa di esplorazione"""
        c = self.get_global_coords()
        if c is None:
            return
            
        x, y = c[0] + PAD, c[1] + PAD
        if 0 <= x < self.explore_map.shape[0] and 0 <= y < self.explore_map.shape[1]:
            self.explore_map[x, y] = 255

    def calculate_reward(self):
        """Calcola reward multi-componente"""
        rewards = {}
        
        # 1. Exploration reward
        unique_tiles = len(self.unique_tiles_visited)
        rewards['exploration'] = unique_tiles * 0.02
        
        # 2. Map progression reward
        current_map = self.get_game_coords()[2]
        if current_map in self.essential_map_locations:
            map_progress = self.essential_map_locations[current_map]
            rewards['map_progress'] = map_progress * 2.0
        else:
            rewards['map_progress'] = 0
        
        # 3. Badge reward
        total_badges = sum(self.get_badges_array())
        rewards['badges'] = total_badges * 5.0
        
        # 4. Party size reward
        party_size = self.get_party_size()
        rewards['party'] = party_size * 0.5
        
        # 5. Penalty for staying in same spot
        current_coord = f"x:{self.get_game_coords()[0]} y:{self.get_game_coords()[1]} m:{self.get_game_coords()[2]}"
        visit_count = self.visited_coords_count.get(current_coord, 0)
        if visit_count > 100:
            rewards['stuck_penalty'] = -0.1 * (visit_count - 100) / 100
        else:
            rewards['stuck_penalty'] = 0
        
        # 6. Map diversity reward
        rewards['map_diversity'] = len(self.seen_maps) * 0.5
        
        # Calcola reward totale come differenza
        total_current = sum(rewards.values())
        last_total = sum(self.last_reward_components.values()) if self.last_reward_components else 0
        
        step_reward = total_current - last_total
        self.last_reward_components = rewards.copy()
        self.total_reward = total_current
        
        return step_reward

    def close(self):
        self.pyboy.stop()