# PokeGym Silver - Pokemon Silver RL Environment

A Gymnasium-based reinforcement learning environment for Pokemon Silver, designed to train agents to explore and progress through the game using PPO (Proximal Policy Optimization).

## 🎮 Overview

This project implements a custom Gymnasium environment that interfaces with Pokemon Silver through the PyBoy Game Boy emulator. The agent learns to play the game by receiving rewards for exploration, progression, and achieving game objectives like collecting badges and visiting new locations.

### Key Features

- **Multi-modal observations**: Combines visual input (game frames), spatial memory (exploration map), and game state (badges, party size)
- **Sophisticated reward system**: Encourages exploration while preventing getting stuck
- **Optimized training**: Leverages modern GPUs (tested on RTX 5080) with parallel environments
- **Comprehensive evaluation tools**: Track progress, visualize exploration patterns, and compare checkpoints

## 📁 Project Structure

```
PokeGym_Silver/
├── envs/
│   └── pokemon_silver_env.py      # Main Gymnasium environment
├── trainers/
│   ├── train_agent.py             # Basic training script
│   ├── train_agent_optimized.py   # GPU-optimized training
│   └── resume_training.py         # Resume from checkpoint
├── evaluation/
│   ├── evaluate_model.py          # Full evaluation suite
│   ├── evaluate_checkpoints.py    # Compare multiple checkpoints
│   └── quick_eval.py              # Quick single-episode test
├── scripts/
│   ├── create_state.py            # Create game save states
│   ├── test_env.py                # Environment testing
│   └── landmark_offsets.py        # Map coordinate helpers
├── roms/
│   └── Pokemon_Silver.gbc         # Game ROM (not included)
├── trained_agents/                # Saved models
├── tensorboard/                   # Training logs
├── map_data.json                  # Map coordinate database
├── start_of_game.state           # Initial game state
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with virtual environment
2. **CUDA-capable GPU** (optional but recommended)
3. **Pokemon Silver ROM** (USA version)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/PokeGym_Silver.git
cd PokeGym_Silver

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your Pokemon Silver ROM in roms/
cp /path/to/Pokemon_Silver.gbc roms/
```

### Training

```bash
# Start new training (optimized for GPU)
python trainers/train_agent_optimized.py

# Resume from checkpoint
python trainers/resume_training.py  # Shows available checkpoints
python trainers/train_agent_optimized.py --resume path/to/checkpoint.zip --timesteps 1000000
```

### Evaluation

```bash
# Quick test of latest model
python evaluation/quick_eval.py

# Full evaluation with statistics
python evaluation/evaluate_model.py --episodes 10

# Compare checkpoints
python evaluation/evaluate_checkpoints.py
```

## 🧠 Environment Design

### Observation Space

The environment provides a dictionary observation with multiple modalities:

1. **screens** (72×80×3): Stack of 3 recent game frames (downscaled 2x)
   - Captures motion and animations
   - Grayscale for efficiency

2. **map** (48×48×1): Local view of exploration map
   - 255 = visited, 0 = unvisited
   - Centered on player position
   - Helps agent remember where it's been

3. **recent_actions** (3,): Last 3 actions taken
   - Prevents action loops
   - Provides temporal context

4. **badges** (16,): Binary array of gym badges
   - 8 Johto + 8 Kanto badges
   - Major progression indicator

5. **party_size** (1,): Number of Pokemon in party
   - Proxy for game progress

### Action Space

7 discrete actions corresponding to Game Boy buttons:
- 0-3: Arrow keys (Down, Left, Right, Up)
- 4: A button
- 5: B button  
- 6: Start button

### Reward System

Multi-component reward designed to encourage exploration and progression:

```python
rewards = {
    'exploration': unique_tiles * 0.02,      # Explore new areas
    'map_progress': map_progress * 2.0,      # Reach important cities
    'badges': total_badges * 5.0,            # Collect gym badges
    'party': party_size * 0.5,               # Build Pokemon team
    'stuck_penalty': -0.1 * excess_visits,   # Don't stay in one place
    'map_diversity': len(seen_maps) * 0.5    # Visit different maps
}
```

**Key Design Principles:**
- **Differential rewards**: Only the change from previous step is given
- **Multi-objective**: Balance exploration with progression
- **Anti-stalling**: Penalties increase quadratically after 100 visits to same spot

### Map System

The environment tracks player position globally across all game maps:
- Local coordinates from game RAM are converted to global coordinates
- `map_data.json` contains offsets for all 400+ game locations
- Exploration is tracked in a 220×220 global map

## 🏋️ Training Architecture

### PPO Algorithm

We use Proximal Policy Optimization (PPO) because:
- **Stable**: Clips policy updates to prevent destructive changes
- **Sample efficient**: Reuses collected data multiple times
- **General purpose**: Works well without extensive tuning

### Hyperparameters (Optimized for RTX 5080)

```python
N_ENVS = 32              # Parallel game instances
BATCH_SIZE = 8192        # Large batch to utilize GPU
N_STEPS = 256            # Steps before policy update
LEARNING_RATE = 3e-4     # With linear decay
ENT_COEF = 0.01          # Exploration bonus
CLIP_RANGE = 0.2         # PPO clipping parameter
GAE_LAMBDA = 0.95        # Advantage estimation
GAMMA = 0.99             # Discount factor
```

### Neural Network Architecture

- **Separate networks** for policy (actions) and value (reward prediction)
- **3 hidden layers** of 512, 512, 256 units with ReLU activation
- **Image normalization** built-in
- **No shared features** between policy and value heads

### Training Optimizations

1. **Parallel environments**: 32 games running simultaneously
2. **Large batches**: 8192 samples per update (vs typical 64-2048)
3. **Reduced evaluation frequency**: Every 500k steps instead of 100k
4. **Shorter episodes**: 1024 steps for faster iteration

## 📊 Monitoring Progress

### TensorBoard

Monitor training in real-time:
```bash
tensorboard --logdir tensorboard/exploration_v3
# Open http://localhost:6006
```

Key metrics:
- `rollout/ep_rew_mean`: Average episode reward
- `game/unique_tiles`: Exploration progress
- `game/badges`: Gym badges collected
- `game/map_transitions`: Movement between areas

### Evaluation Metrics

The evaluation scripts track:
- Total reward achieved
- Unique tiles explored
- Map transitions (movement between areas)
- Badges collected
- Exploration heatmaps

## 🔧 Advanced Usage

### Custom Reward Functions

Modify `calculate_reward()` in `pokemon_silver_env.py`:

```python
def calculate_reward(self):
    rewards = {}
    # Add your custom reward components
    rewards['my_metric'] = self.calculate_my_metric() * weight
    # ...
```

### Different Starting Points

Create new save states:
```python
# In scripts/create_state.py
pyboy = PyBoy('roms/Pokemon_Silver.gbc')
# Play to desired starting point
pyboy.save_state('custom_start.state')
```

### Hyperparameter Tuning

Key parameters to adjust in `train_agent_optimized.py`:
- `N_ENVS`: More = faster but more RAM
- `BATCH_SIZE`: Larger = more stable but slower
- `ENT_COEF`: Higher = more exploration
- `LEARNING_RATE`: Lower = more stable convergence

## 🐛 Troubleshooting

### GPU Not Being Used
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure `device="cuda"` in PPO initialization
- Monitor GPU usage with `nvidia-smi`

### TensorBoard Empty
- Check correct directory: `tensorboard --logdir tensorboard/exploration_v3`
- Wait for first logging interval (5000 steps)
- Verify files exist: `ls tensorboard/exploration_v3/PPO*/events*`

### Training Too Slow
- Reduce `EVAL_FREQ` to avoid evaluation pauses
- Decrease episode length with `max_steps`
- Use more parallel environments (`N_ENVS`)

### Out of Memory
- Reduce `BATCH_SIZE` or `N_ENVS`
- Use gradient accumulation
- Clear PyBoy states between episodes

## 📈 Performance Expectations

With default settings on RTX 5080:
- **Training speed**: ~2000-3000 steps/second
- **First badge**: Usually within 500k-1M steps
- **Multiple badges**: 2-5M steps
- **GPU usage**: 60-80% utilization
- **VRAM usage**: 8-12GB

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional reward components (items, Pokemon levels, etc.)
- Better action space (macro actions, action repeat)
- Curriculum learning (progressive difficulty)
- Multi-agent training
- Integration with other Pokemon games

## 📄 License

This project is for educational and research purposes only. Pokemon is a trademark of Nintendo/Game Freak.

## 🙏 Acknowledgments

- [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - RL environment framework
- Inspired by [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)

## 📚 Technical Details

### RAM Addresses (Pokemon Silver)

Key memory locations used:
```
0xDA00: Current map bank
0xDA01: Current map number
0xDA02: X coordinate
0xDA03: Y coordinate
0xD57C: Johto badges
0xD57D: Kanto badges
0xDA22: Party Pokemon count
0xD116: Battle status (0 = not in battle)
```

### Coordinate System

- **Local**: Each map has coordinates (0,0) to (width,height)
- **Global**: All maps placed in unified coordinate system
- **Conversion**: `global = local + map_offset` (from map_data.json)

### State Management

- Game state saved/loaded via PyBoy's state system
- Always starts from `start_of_game.state` for consistency
- States are ~32KB each (Game Boy RAM snapshot)

### Performance Considerations

- Frame skip: 24 frames per action (~0.4 seconds game time)
- Downscaling: 144×160 → 72×80 (4x reduction)
- Grayscale: RGB not needed, saves computation
- Memory efficiency: Reuse NumPy arrays where possible