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

class PokemonSilverV2(gymnasium.Env):
    """
    Enhanced Gym Environment for Pokemon Silver with health, level, events tracking and video recording.
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

        # Essential map progression
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
        """Read coordinates from game (corrected for Silver)"""
        map_n = self.read_m(0xDA00)  # Map bank
        map_id = self.read_m(0xDA01)  # Map number
        x_pos = self.read_m(0xDA02)  # X coordinate
        y_pos = self.read_m(0xDA03)  # Y coordinate
        
        # In Silver, often need to combine bank and ID
        full_map_id = map_id  # For now just use map_id
        
        return (x_pos, y_pos, full_map_id)
    
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
    
    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()
            
        self.run_action_on_emulator(action)
        self.update_recent_actions(action)

        # Update exploration
        self.update_seen_coords()
        self.update_explore_map()
        
        # Check battle status
        in_battle = self.read_m(0xD116) != 0
        if in_battle:
            self.battles_encountered += 1

        # Update health tracking
        self.update_heal_reward()

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
        addr_offset = 0
        
        for addr in range(EVENT_FLAGS_START, EVENT_FLAGS_END + 1):
            for bit in range(8):
                if event_bits[addr_offset * 8 + bit]:
                    key = f"0x{addr:X}-{bit}"
                    if key in EVENT_DATA and key not in self.current_event_flags_set:
                        self.current_event_flags_set[key] = EVENT_DATA[key]
                        print(f"Event unlocked: {EVENT_DATA[key]}")
            addr_offset += 1

    def calculate_reward(self):
        """Calculate multi-component reward"""
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
        
        # 5. Level reward
        level_sum = self.get_levels_sum()
        if level_sum > self.max_level_sum:
            rewards['level'] = (level_sum - self.max_level_sum) * 1.0
            self.max_level_sum = level_sum
        else:
            rewards['level'] = 0
        
        # 6. Event reward
        current_events = self.count_event_flags() - self.base_event_flags
        rewards['events'] = current_events * 0.5
        
        # 7. Healing reward
        rewards['healing'] = self.total_healing_reward
        
        # 8. Penalty for staying in same spot
        current_coord = f"x:{self.get_game_coords()[0]} y:{self.get_game_coords()[1]} m:{self.get_game_coords()[2]}"
        visit_count = self.visited_coords_count.get(current_coord, 0)
        if visit_count > 100:
            rewards['stuck_penalty'] = -0.1 * (visit_count - 100) / 100
        else:
            rewards['stuck_penalty'] = 0
        
        # 9. Map diversity reward
        rewards['map_diversity'] = len(self.seen_maps) * 0.5
        
        # 10. Death penalty
        rewards['death_penalty'] = -self.died_count * 5.0
        
        # Calculate total reward as difference
        total_current = sum(rewards.values())
        last_total = sum(self.last_reward_components.values()) if self.last_reward_components else 0
        
        step_reward = total_current - last_total
        self.last_reward_components = rewards.copy()
        self.total_reward = total_current
        
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