# 🎮 PokeGym Silver

Reinforcement Learning environment for Pokémon Silver using PyBoy and Gymnasium.

---

## 📂 Project Structure

- `envs/`: Custom Gym environment
- `roms/`: Pokémon Silver ROM
- `outputs/`: Frames, videos, and trained models
- `scripts/`: Utility scripts (rendering, env checks, etc.)
- `trainers/`: Training scripts (one per reward strategy)
- `venv/`: Python virtual environment

---

## ✅ Completed Tasks

- [x] PyBoy Gym environment
- [x] Rendering modes (headless, human, human-fast)
- [x] Frame saving and video generation
- [x] .gitignore for outputs + logs
- [x] Render testing script
- [x] Organized project folders
- [x] Exploration-based reward (position and hashing)
- [x] PPO training on GPU
- [x] Multi-environment parallel training
- [x] TensorBoard logging
- [x] Parallel training script for both strategies

---

## 🚀 Next Steps

- [ ] Advanced rewards (battles, dialogues)
- [ ] Observation stacking (frame history)
- [ ] Model checkpointing (save during training)
- [ ] Evaluation/inference scripts
- [ ] Custom CNN feature extractors
- [ ] Experimentation with alternative policies (e.g. SAC, A2C)

---

## ▶️ Quickstart

```bash
pip install -r requirements.txt
```
 --- 


## Refernces & Online Resources

https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Gold_and_Silver/RAM_map#Miscellaneous 

Link above for RAM mapping

---

## Start Training

```bash
python -m trainers.train_both
```

## Launch Tensorboard

```bash
tensorboard --logdir ./tensorboard_logs --port 6060
```

then open browser at 'http://localhost:6060'
