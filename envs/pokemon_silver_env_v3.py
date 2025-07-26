import gymnasium
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import os
from skimage.transform import downscale_local_mean
from collections import deque
import json
from pathlib import Path
import mediapy as media

# Load map data
with open("map_data.json", "r") as f:
    MAP_DATA = json.load(f)

# Load events data
with open("events.json", "r") as f:
    EVENT_DATA = json.load(f)

# Create dictionary for fast lookup
MAP_TO_GLOBAL_OFFSET = {}
for region in MAP_DATA["regions"]:
    map_id = int(region["id"])
    coords = region["coordinates"]
    MAP_TO_GLOBAL_OFFSET[map_id] = (coords[1], coords[0])  # (row, col)

PAD = 20
GLOBAL_MAP_SHAPE = (180 + PAD * 2, 180 + PAD * 2)

# Event flags memory range
EVENT_FLAGS_START = 0xD7B7
EVENT_FLAGS_END = 0xD8B6

# Important locations for interactions
IMPORTANT_LOCATIONS = {
    206: "Elm's Lab",  # Professor Elm's Lab
    207: "Player's House",
    163: "Oak's Lab",
    1: "New Bark Town",
    3: "Cherrygrove City",
}

# Critical events for progression
CRITICAL_EVENTS = {
    "0xD7BA-2": "GOT_STARTER_POKEMON",
    "0xD857-5": "PLAYER_HAS_POKEDEX",
    "0xD859-0": "PLAYER_HAS_POKEMON",
}

# Useful memory addresses for Pokemon Silver
MENU_FLAGS = {
    'start_menu': 0xCF6B,      # Start menu open
    'text_box': 0xCF6C,        # Text box active
    'game_mode': 0xCF5F,       # Current game mode
    'waiting_input': 0xCF63,   # Waiting for button press
    'text_delay': 0xCFC3,      # Text printing delay
    'battle_mode': 0xD116,     # In battle
    'map_bank': 0xDA00,        # Current map bank
    'map_id': 0xDA01,          # Current map number
    'x_pos': 0xDA02,           # X position
    'y_pos': 0xDA03,           # Y position
}

class PokemonSilverV3(gymnasium.Env):
    """
    Enhanced Environment with better reward system for interactions and progression.
    """

    def __init__(self, rom_path, render_mode="headless", max_steps=2048*80, save_video=False, video_dir="rollouts"):
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
        
        # Video writers
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None

        # Essential map progression with weights
        self.essential_map_locations = {
            1: 0.5,    # New Bark Town - start
            206: 3.0,  # Elm's Lab - CRITICAL
            3: 1.0,    # Cherrygrove City
            6: 2.0,    # Violet City
            12: 3.0,   # Azalea Town
            16: 4.0,   # Goldenrod City
            22: 5.0,   # Ecruteak City
            27: 6.0,   # Olivine City
            33: 7.0,   # Cianwood City
            36: 8.0,   # Mahogany Town
            41: 9.0,   # Blackthorn City
            90: 15.0,  # Indigo Plateau
        }

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
        self.coords_pad = 12
        self.enc_freqs = 8  # For Fourier encoding

        # Action space
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # Enhanced observation space
        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad*4, self.coords_pad*4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks),
                "badges": spaces.MultiBinary(16),  # 8 Johto + 8 Kanto
                "party_size": spaces.Box(low=0, high=6, shape=(1,), dtype=np.uint8),
                "health": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,), dtype=np.float32),
                "events": spaces.MultiBinary((EVENT_FLAGS_END - EVENT_FLAGS_START + 1) * 8),
            }
        )

        # Progress tracking
        self.has_starter = False
        self.entered_elms_lab = False
        self.talked_to_elm = False
        self.menu_interactions = 0
        self.successful_interactions = 0
        self.last_action = None
        self.stuck_penalty_multiplier = 1.0

    def reset(self, *, seed=None, options=None):
        self.seed = seed

        # Load initial state
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
        
        # Health and level tracking
        self.last_health = 1.0
        self.max_level_sum = 0
        self.total_healing_reward = 0
        self.died_count = 0
        
        # Event tracking
        self.base_event_flags = self.count_event_flags()
        self.max_event_flags = 0
        self.current_event_flags_set = {}
        
        # Interaction tracking
        self.visited_important_locations = set()
        self.last_menu_seen = 0
        self.dialogue_seen = 0
        self.buttons_pressed = {
            WindowEvent.PRESS_BUTTON_A: 0,
            WindowEvent.PRESS_BUTTON_B: 0,
            WindowEvent.PRESS_BUTTON_START: 0,
        }
        
        # Movement tracking
        self.last_coords = (0, 0, 0)
        self.steps_without_movement = 0
        self.total_distance_traveled = 0
        
        # Progress tracking reset
        self.has_starter = False
        self.entered_elms_lab = False
        self.talked_to_elm = False
        self.menu_interactions = 0
        self.successful_interactions = 0
        self.last_action = None
        self.stuck_penalty_multiplier = 1.0
        
        # For reward tracking
        self.last_reward_components = {}
        self.total_reward = 0
        
        self.reset_count += 1
        
        # Start video if enabled
        if self.save_video and self.step_count == 0:
            self.start_video()
        
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

        # Get level sum for Fourier encoding
        level_sum = 0.02 * sum([
            self.read_m(a) for a in [
                0xDA49,  # Pokemon 1 level
                0xDA79,  # Pokemon 2 level
                0xDAA9,  # Pokemon 3 level
                0xDAD9,  # Pokemon 4 level
                0xDB09,  # Pokemon 5 level
                0xDB39,  # Pokemon 6 level
            ]
        ])

        observation = {
            "screens": self.recent_screens,
            "map": self.get_explore_map()[:,:,None],
            "recent_actions": self.recent_actions.copy(),
            "badges": self.get_badges_array(),
            "party_size": np.array([self.get_party_size()], dtype=np.uint8),
            "health": np.array([self.read_hp_fraction()], dtype=np.float32),
            "level": self.fourier_encode(level_sum),
            "events": self.read_event_bits(),
        }

        return observation
    
    def fourier_encode(self, val):
        """Fourier encoding for continuous values"""
        return np.sin(val * 2 ** np.arange(self.enc_freqs)).astype(np.float32)
    
    def read_hp_fraction(self):
        """Read total HP fraction of all Pokemon in party"""
        hp_sum = 0
        max_hp_sum = 0
        
        # Party Pokemon HP addresses
        hp_addrs = [
            (0xDA4C, 0xDA4E),  # Pokemon 1 current/max HP
            (0xDA7C, 0xDA7E),  # Pokemon 2
            (0xDAAC, 0xDAAE),  # Pokemon 3
            (0xDADC, 0xDADE),  # Pokemon 4
            (0xDB0C, 0xDB0E),  # Pokemon 5
            (0xDB3C, 0xDB3E),  # Pokemon 6
        ]
        
        for curr_addr, max_addr in hp_addrs:
            curr_hp = self.read_m(curr_addr) * 256 + self.read_m(curr_addr + 1)
            max_hp = self.read_m(max_addr) * 256 + self.read_m(max_addr + 1)
            hp_sum += curr_hp
            max_hp_sum += max_hp
        
        return hp_sum / max(max_hp_sum, 1)
    
    def read_event_bits(self):
        """Read all event flag bits"""
        bits = []
        for addr in range(EVENT_FLAGS_START, EVENT_FLAGS_END + 1):
            byte_val = self.read_m(addr)
            for i in range(8):
                bits.append((byte_val >> i) & 1)
        return np.array(bits, dtype=np.int8)
    
    def count_event_flags(self):
        """Count total number of set event flags"""
        return sum(self.read_event_bits())
    
    def get_badges_array(self):
        """Get badge array (8 Johto + 8 Kanto)"""
        johto_badges = self.read_m(0xD57C)
        kanto_badges = self.read_m(0xD57D)
        
        badges = np.zeros(16, dtype=np.int8)
        for i in range(8):
            badges[i] = (johto_badges >> i) & 1
            badges[i+8] = (kanto_badges >> i) & 1
        
        return badges
    
    def get_party_size(self):
        """Get number of Pokemon in party"""
        return self.read_m(0xDA22)
    
    def get_explore_map(self):
        """Get visible portion of map centered on player"""
        c = self.get_global_coords()
        if c is None:
            return np.zeros((self.coords_pad*4, self.coords_pad*4), dtype=np.uint8)
        
        x, y = c[0] + PAD, c[1] + PAD
        
        # Extract visible portion with padding
        x0 = max(0, x - self.coords_pad*2)
        x1 = min(self.explore_map.shape[0], x + self.coords_pad*2)
        y0 = max(0, y - self.coords_pad*2)
        y1 = min(self.explore_map.shape[1], y + self.coords_pad*2)
        
        cropped = self.explore_map[x0:x1, y0:y1]
        
        # Pad to fixed size
        padded = np.zeros((self.coords_pad*4, self.coords_pad*4), dtype=np.uint8)
        h, w = cropped.shape
        cx = (self.coords_pad*4 - h) // 2
        cy = (self.coords_pad*4 - w) // 2
        padded[cx:cx+h, cy:cy+w] = cropped
        
        return padded
    
    def local_to_global(self, r, c, map_n):
        """Convert local to global coordinates using map_data.json"""
        if map_n not in MAP_TO_GLOBAL_OFFSET:
            return None
        
        offset = MAP_TO_GLOBAL_OFFSET[map_n]
        return (offset[0] + r, offset[1] + c)
    
    def get_global_coords(self):
        """Get player's global coordinates"""
        x_pos, y_pos, map_n = self.get_game_coords()
        return self.local_to_global(y_pos, x_pos, map_n)
    
    def get_game_coords(self):
        """Read coordinates from game (using datacrystal addresses)"""
        map_bank = self.read_m(0xDA00)  # Map bank
        map_number = self.read_m(0xDA01)  # Map number
        x_pos = self.read_m(0xDA02)  # X coordinate
        y_pos = self.read_m(0xDA03)  # Y coordinate
        
        return (x_pos, y_pos, map_number)
    
    def read_m(self, address):
        """Read byte from memory"""
        return self.pyboy.memory[address]
    
    def update_recent_screens(self, cur_screen):
        """Update stack of recent frames"""
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:,:,0] = cur_screen[:,:,0]

    def update_recent_actions(self, action):
        """Update recent actions"""
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action
    
    def check_if_menu_open(self):
        """Check if any menu is open"""
        # Multiple checks for different menu types in Pokemon Silver
        start_menu = self.read_m(0xCF6B)  # Start menu flag
        text_box = self.read_m(0xCF6C)    # Text box flag  
        battle_menu = self.read_m(0xD116)  # Battle flag (already used elsewhere)
        
        # Also check if we're in a special mode
        game_mode = self.read_m(0xCF5F)   # Game mode/state
        
        return start_menu != 0 or text_box != 0 or game_mode > 0
    
    def check_if_dialogue_active(self):
        """Check if dialogue box is active"""
        # Text engine state addresses for Pokemon Silver
        text_state = self.read_m(0xCF6C)      # Text box state
        text_delay = self.read_m(0xCFC3)      # Text delay counter
        waiting_for_input = self.read_m(0xCF63)  # Waiting for A/B press
        
        return text_state != 0 or waiting_for_input != 0
    
    def check_game_progress(self):
        """Check major game milestones"""
        # Check if we have a starter Pokemon
        party_size = self.get_party_size()
        if party_size > 0 and not self.has_starter:
            self.has_starter = True
            print("🎉 Got starter Pokemon!")
            return 100.0  # Huge reward
        
        # Check if we entered Elm's lab (map 206)
        current_map = self.get_game_coords()[2]
        if current_map == 206 and not self.entered_elms_lab:
            self.entered_elms_lab = True
            print("📍 Entered Elm's Lab!")
            return 20.0
        
        # Check for successful menu interactions
        if self.check_if_menu_open() or self.check_if_dialogue_active():
            self.menu_interactions += 1
            # If we pressed A while dialogue was active, it's a successful interaction
            if self.last_action == 4:  # A button
                self.successful_interactions += 1
                return 1.0
        
        return 0.0
    
    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()
        
        # Store last action
        self.last_action = action
            
        # Track button presses
        if self.valid_actions[action] in self.buttons_pressed:
            self.buttons_pressed[self.valid_actions[action]] += 1
            
        self.run_action_on_emulator(action)
        self.update_recent_actions(action)

        # Update exploration
        self.update_seen_coords()
        self.update_explore_map()
        
        # Check interactions
        if self.check_if_menu_open():
            self.last_menu_seen = self.step_count
        if self.check_if_dialogue_active():
            self.dialogue_seen += 1
        
        # Check battle status
        in_battle = self.read_m(0xD116) != 0
        if in_battle:
            self.battles_encountered += 1

        # Update health tracking
        self.update_heal_reward()
        
        # Movement tracking
        current_coords = self.get_game_coords()
        if current_coords == self.last_coords:
            self.steps_without_movement += 1
        else:
            self.steps_without_movement = 0
            # Calculate distance traveled
            if self.last_coords[2] == current_coords[2]:  # Same map
                dist = abs(current_coords[0] - self.last_coords[0]) + abs(current_coords[1] - self.last_coords[1])
                self.total_distance_traveled += dist
        self.last_coords = current_coords
        
        # Check if agent is stuck in a loop
        if self.steps_without_movement > 200:
            # Force random exploration
            self.stuck_penalty_multiplier = min(self.steps_without_movement / 100, 5.0)
        else:
            self.stuck_penalty_multiplier = 1.0

        # Calculate reward
        reward = self.calculate_reward()
        
        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps - 1
        
        obs = self._get_obs()
        
        # Update event tracking
        if self.step_count % 100 == 0:
            self.update_event_tracking()
        
        self.step_count += 1
        
        # Save video frame
        if self.save_video:
            self.add_video_frame()
        
        info = {
            "step": self.step_count,
            "total_reward": self.total_reward,
            "unique_tiles": len(self.unique_tiles_visited),
            "map_transitions": self.map_transition_count,
            "battles": self.battles_encountered,
            "badges": sum(self.get_badges_array()),
            "hp_fraction": self.read_hp_fraction(),
            "level_sum": self.get_levels_sum(),
            "events": self.max_event_flags - self.base_event_flags,
            "healing_reward": self.total_healing_reward,
            "important_locations": len(self.visited_important_locations),
            "dialogue_seen": self.dialogue_seen,
            "distance_traveled": self.total_distance_traveled,
            "party_size": self.get_party_size(),
        }
        
        return obs, reward, terminated, truncated, info

    def run_action_on_emulator(self, action):
        """Execute action on emulator"""
        self.pyboy.send_input(self.valid_actions[action])
        
        render_screen = self.save_video or self.render_mode != "headless"
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)

    def update_seen_coords(self):
        """Update visited coordinates"""
        if self.read_m(0xD116) == 0:  # Not in battle
            x_pos, y_pos, map_n = self.get_game_coords()
            
            # Track current map
            if map_n != self.last_map_id:
                self.map_transition_count += 1
                self.last_map_id = map_n
                self.seen_maps.add(map_n)
                
                # Check if important location
                if map_n in IMPORTANT_LOCATIONS and map_n not in self.visited_important_locations:
                    self.visited_important_locations.add(map_n)
                    print(f"Important location visited: {IMPORTANT_LOCATIONS[map_n]}")
            
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
        """Update exploration map"""
        c = self.get_global_coords()
        if c is None:
            return
            
        x, y = c[0] + PAD, c[1] + PAD
        if 0 <= x < self.explore_map.shape[0] and 0 <= y < self.explore_map.shape[1]:
            self.explore_map[x, y] = 255
    
    def get_levels_sum(self):
        """Get sum of all Pokemon levels"""
        levels = [
            self.read_m(0xDA49),  # Pokemon 1
            self.read_m(0xDA79),  # Pokemon 2
            self.read_m(0xDAA9),  # Pokemon 3
            self.read_m(0xDAD9),  # Pokemon 4
            self.read_m(0xDB09),  # Pokemon 5
            self.read_m(0xDB39),  # Pokemon 6
        ]
        return sum(levels)
    
    def update_heal_reward(self):
        """Track healing actions"""
        cur_health = self.read_hp_fraction()
        party_size = self.get_party_size()
        
        # If health increased and party size didn't change
        if cur_health > self.last_health and party_size > 0:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_reward += heal_amount * heal_amount * 10
            else:
                # Revived from fainted
                self.died_count += 1
                self.total_healing_reward += 5
        
        self.last_health = cur_health
    
    def update_event_tracking(self):
        """Update event flag tracking"""
        current_flags = self.count_event_flags()
        self.max_event_flags = max(self.max_event_flags, current_flags)
        
        # Track which specific events are set
        event_bits = self.read_event_bits()
        
        for addr in range(EVENT_FLAGS_START, EVENT_FLAGS_END + 1):
            for bit in range(8):
                idx = (addr - EVENT_FLAGS_START) * 8 + bit
                if idx < len(event_bits) and event_bits[idx]:
                    key = f"0x{addr:X}-{bit}"
                    # Only print if it's a new event AND it's in our known events
                    if key in EVENT_DATA and key not in self.current_event_flags_set:
                        self.current_event_flags_set[key] = EVENT_DATA[key]
                        print(f"Event unlocked: {EVENT_DATA[key]}")
                        
                        # Special handling for critical events
                        if key in CRITICAL_EVENTS:
                            print(f"🎉 CRITICAL EVENT: {CRITICAL_EVENTS[key]}")

    def calculate_reward(self):
        """Calculate multi-component reward with enhanced interaction rewards"""
        rewards = {}
        
        # Check game progress first
        progress_reward = self.check_game_progress()
        if progress_reward > 0:
            rewards['progress'] = progress_reward
        
        # 1. Exploration reward (reduced to prevent just wandering)
        unique_tiles = len(self.unique_tiles_visited)
        rewards['exploration'] = unique_tiles * 0.01  # Reduced from 0.02
        
        # 2. Map progression reward (increased for important locations)
        current_map = self.get_game_coords()[2]
        if current_map in self.essential_map_locations:
            map_progress = self.essential_map_locations[current_map]
            rewards['map_progress'] = map_progress
        else:
            rewards['map_progress'] = 0
        
        # 3. Badge reward
        total_badges = sum(self.get_badges_array())
        rewards['badges'] = total_badges * 10.0  # Increased
        
        # 4. Party size reward (much higher to incentivize getting starter)
        party_size = self.get_party_size()
        if party_size == 0:
            rewards['party'] = 0
        elif party_size == 1:
            rewards['party'] = 20.0  # Big bonus for getting first Pokemon!
        else:
            rewards['party'] = 20.0 + (party_size - 1) * 5.0
        
        # 5. Level reward
        level_sum = self.get_levels_sum()
        if level_sum > self.max_level_sum:
            rewards['level'] = (level_sum - self.max_level_sum) * 2.0
            self.max_level_sum = level_sum
        else:
            rewards['level'] = 0
        
        # 6. Event reward (much higher for critical events)
        current_events = self.count_event_flags() - self.base_event_flags
        rewards['events'] = current_events * 1.0
        
        # Special bonus for critical events
        for key in CRITICAL_EVENTS:
            if key in self.current_event_flags_set:
                rewards[f'critical_event_{key}'] = 50.0
        
        # 7. Healing reward
        rewards['healing'] = self.total_healing_reward
        
        # 8. Interaction rewards
        rewards['important_locations'] = len(self.visited_important_locations) * 5.0
        rewards['dialogue'] = min(self.dialogue_seen * 0.5, 10.0)  # Cap at 10
        
        # 9. Button usage reward (encourage trying different buttons)
        button_diversity = min(self.buttons_pressed[WindowEvent.PRESS_BUTTON_A], 50) * 0.1
        button_diversity += min(self.buttons_pressed[WindowEvent.PRESS_BUTTON_START], 20) * 0.1
        rewards['button_usage'] = button_diversity
        
        # 10. Movement reward (encourage actual movement)
        rewards['distance'] = min(self.total_distance_traveled * 0.01, 20.0)
        
        # 11. Penalty for staying in same spot (stronger with multiplier)
        current_coord = f"x:{self.get_game_coords()[0]} y:{self.get_game_coords()[1]} m:{self.get_game_coords()[2]}"
        visit_count = self.visited_coords_count.get(current_coord, 0)
        if visit_count > 50:  # Lower threshold
            rewards['stuck_penalty'] = -0.2 * self.stuck_penalty_multiplier * (visit_count - 50) / 50
        else:
            rewards['stuck_penalty'] = 0
        
        # 12. Penalty for not moving at all
        if self.steps_without_movement > 100:
            rewards['no_movement_penalty'] = -0.5 * (self.steps_without_movement - 100) / 100
        else:
            rewards['no_movement_penalty'] = 0
        
        # 13. Map diversity reward
        rewards['map_diversity'] = len(self.seen_maps) * 1.0
        
        # 14. Death penalty
        rewards['death_penalty'] = -self.died_count * 10.0
        
        # 15. Battle reward (small to encourage some battles)
        rewards['battles'] = min(self.battles_encountered * 0.5, 10.0)
        
        # Calculate total reward as difference
        total_current = sum(rewards.values())
        last_total = sum(self.last_reward_components.values()) if self.last_reward_components else 0
        
        step_reward = total_current - last_total
        self.last_reward_components = rewards.copy()
        self.total_reward = total_current
        
        # Debug print for significant rewards
        if abs(step_reward) > 1.0:
            print(f"Step {self.step_count}: Reward {step_reward:.2f}")
            for k, v in rewards.items():
                if k in self.last_reward_components:
                    diff = v - self.last_reward_components.get(k, 0)
                    if abs(diff) > 0.1:
                        print(f"  {k}: {diff:+.2f}")
        
        return step_reward
    
    def start_video(self):
        """Start video recording"""
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()
        
        base_name = f"episode_{self.reset_count}_reward_{self.total_reward:.0f}"
        
        # Full resolution video
        full_path = self.video_dir / f"full_{base_name}.mp4"
        self.full_frame_writer = media.VideoWriter(
            str(full_path), (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        
        # Model input video
        model_path = self.video_dir / f"model_{base_name}.mp4"
        self.model_frame_writer = media.VideoWriter(
            str(model_path), self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        
        # Exploration map video
        map_path = self.video_dir / f"map_{base_name}.mp4"
        self.map_frame_writer = media.VideoWriter(
            str(map_path), (self.coords_pad*4, self.coords_pad*4), fps=60, input_format="gray"
        )
        self.map_frame_writer.__enter__()
    
    def add_video_frame(self):
        """Add frame to video"""
        if self.full_frame_writer:
            self.full_frame_writer.add_image(self.render(reduce_res=False)[:,:,0])
        if self.model_frame_writer:
            self.model_frame_writer.add_image(self.render(reduce_res=True)[:,:,0])
        if self.map_frame_writer:
            self.map_frame_writer.add_image(self.get_explore_map()[:,:,0])
    
    def close(self):
        """Close environment and save video"""
        if self.save_video:
            if self.full_frame_writer:
                self.full_frame_writer.close()
            if self.model_frame_writer:
                self.model_frame_writer.close()
            if self.map_frame_writer:
                self.map_frame_writer.close()
        
        self.pyboy.stop()