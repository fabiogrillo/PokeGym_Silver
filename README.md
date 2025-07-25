# 🎮 PokéGym Silver - Reinforcement Learning per Pokémon Silver

Un ambiente Gymnasium personalizzato per addestrare agenti di Reinforcement Learning a giocare a Pokémon Silver usando PyBoy e Stable Baselines3.

## 📋 Indice

- [Overview](#overview)
- [Struttura del Progetto](#struttura-del-progetto)
- [Installazione](#installazione)
- [Quick Start](#quick-start)
- [Environment Design](#environment-design)
- [Training System](#training-system)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

Questo progetto implementa un ambiente di RL per Pokémon Silver con l'obiettivo di addestrare un agente PPO (Proximal Policy Optimization) a completare il gioco. L'agente impara attraverso:

- **Esplorazione del mondo di gioco**: Reward per visitare nuove aree
- **Progressione nella storia**: Reward maggiori per raggiungere città chiave
- **Ottenimento di medaglie**: Forte incentivo per progressi concreti
- **Cattura di Pokémon**: Reward per espandere il team

### Caratteristiche Principali

- 🚀 **Training parallelo** su 32+ environments
- 🎮 **Emulazione Game Boy** tramite PyBoy
- 📊 **Monitoring real-time** con TensorBoard
- 🏃 **Ottimizzato per GPU** (testato su RTX 5080)
- 💾 **Checkpoint automatici** e resume del training
- 📈 **Sistema di reward multi-componente**

## 📁 Struttura del Progetto

```
PokeGym_Silver/
│
├── envs/                          # Environment Gymnasium
│   └── pokemon_silver_env.py      # Classe principale PokemonSilver
│
├── trainers/                      # Script di training
│   ├── train_agent.py            # Training base
│   ├── train_agent_optimized.py  # Training ottimizzato GPU
│   └── resume_training.py        # Utility per riprendere training
│
├── evaluation/                    # Script di valutazione
│   ├── evaluate_model.py         # Valutazione completa
│   ├── evaluate_checkpoints.py   # Confronto checkpoint
│   └── quick_eval.py            # Test veloce modelli
│
├── scripts/                       # Utility e test
│   ├── create_state.py           # Crea save state iniziale
│   ├── test_env.py              # Test environment
│   ├── env_check.py             # Verifica compatibilità
│   └── landmark_offsets.py      # Costanti mappe (deprecato)
│
├── roms/                         # ROM e save files
│   ├── Pokemon_Silver.gbc       # ROM del gioco (non inclusa)
│   └── *.state                  # Save states PyBoy
│
├── trained_agents/               # Modelli salvati
│   └── exploration_v*/          # Versioni training
│       ├── checkpoints/         # Checkpoint periodici
│       ├── best_model/          # Miglior modello
│       └── eval_logs/           # Log valutazioni
│
├── tensorboard/                  # Log TensorBoard
│   └── exploration_v*/          # Metriche training
│
├── map_data.json                # Database coordinate mappe
├── start_of_game.state         # Save state iniziale
├── requirements.txt            # Dipendenze Python
└── README.md                   # Questo file
```

## 🛠️ Installazione

### Prerequisiti

- Python 3.8+
- CUDA 11.8+ (per GPU support)
- 16GB+ RAM
- ROM di Pokémon Silver (versione USA)

### Setup

```bash
# Clone repository
git clone https://github.com/username/PokeGym_Silver.git
cd PokeGym_Silver

# Crea virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Installa dipendenze
pip install -r requirements.txt

# Posiziona la ROM
cp /path/to/Pokemon_Silver.gbc roms/
```

### Verifica Installazione

```bash
# Test environment
python scripts/test_env.py

# Verifica CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### 1. Training da Zero

```bash
# Training ottimizzato per GPU
python trainers/train_agent_optimized.py

# Training base (più lento)
python trainers/train_agent.py
```

### 2. Resume Training

```bash
# Trova checkpoint esistenti
python trainers/resume_training.py

# Riprendi da checkpoint specifico
python trainers/train_agent_optimized.py --resume path/to/checkpoint.zip --timesteps 1000000
```

### 3. Valutazione

```bash
# Test veloce dell'ultimo modello
python evaluation/quick_eval.py

# Valutazione completa
python evaluation/evaluate_model.py --episodes 10

# Confronta checkpoint
python evaluation/evaluate_checkpoints.py
```

### 4. Monitoring

```bash
# Avvia TensorBoard
tensorboard --logdir tensorboard/exploration_v3

# Apri browser su http://localhost:6006
```

## 🎮 Environment Design

### Observation Space

L'environment fornisce un dizionario di osservazioni multi-modali:

```python
{
    "screens": Box(72, 80, 3),      # Stack di 3 frame consecutivi
    "map": Box(48, 48, 1),          # Mappa esplorazione locale
    "recent_actions": MultiDiscrete([7]*3),  # Ultime 3 azioni
    "badges": MultiBinary(16),       # 8 Johto + 8 Kanto badges
    "party_size": Box(1,)           # Numero Pokémon nel team
}
```

#### Screens (Visual Input)
- **Risoluzione**: 72×80 (downscaled 2x dall'originale 144×160)
- **Frame Stack**: 3 frame consecutivi per catturare movimento
- **Preprocessing**: Grayscale, normalizzato 0-255

#### Map (Exploration Memory)
- **Dimensione**: 48×48 tiles centrati sul giocatore
- **Encoding**: 255 = visitato, 0 = non visitato
- **Scopo**: Memoria a lungo termine dell'esplorazione

#### Game State
- **Recent Actions**: Previene loop di azioni
- **Badges**: Tracking progressi principali
- **Party Size**: Proxy per progressi generali

### Action Space

7 azioni discrete mappate sui controlli Game Boy:

```python
0: DOWN
1: LEFT  
2: RIGHT
3: UP
4: A (conferma/interagisci)
5: B (annulla/corri)
6: START (menu)
```

### Reward System

Sistema multi-componente per guidare comportamenti complessi:

```python
{
    'exploration': unique_tiles * 0.02,      # Incentiva movimento
    'map_progress': location_idx * 2.0,      # Città story-critical
    'badges': total_badges * 5.0,            # Progressi maggiori
    'party': party_size * 0.5,               # Cattura Pokémon
    'stuck_penalty': -0.1 * excess_visits,   # Anti-stalling
    'map_diversity': unique_maps * 0.5       # Varietà esplorazione
}
```

#### Design Rationale

1. **Exploration (0.02/tile)**: 
   - Reward piccolo ma costante
   - Incentiva scoperta continua
   - Previene stagnazione iniziale

2. **Map Progress (2.0/location)**:
   - Boost significativi per location critiche
   - Guida verso obiettivi principali
   - 15 location ordinate per progressione

3. **Badges (5.0/badge)**:
   - Reward maggiore del gioco
   - Obiettivi chiari e misurabili
   - 16 totali (Johto + Kanto)

4. **Stuck Penalty**:
   - Attiva dopo 100 visite stesso tile
   - Cresce quadraticamente
   - Forza esplorazione nuova

5. **Reward Differenziale**:
   - `step_reward = total_current - total_previous`
   - Evita reward inflation
   - Focus su miglioramenti incrementali

### Memory Mapping

Indirizzi RAM critici per Pokémon Silver:

```python
0xDA00: Map bank
0xDA01: Map ID  
0xDA02: X position
0xDA03: Y position
0xD57C: Johto badges
0xD57D: Kanto badges
0xDA22: Party size
0xD116: Battle status (0 = overworld)
```

## 🧠 Training System

### Algoritmo: PPO (Proximal Policy Optimization)

PPO scelto per:
- **Stabilità**: Non diverge facilmente
- **Sample efficiency**: Riusa dati multiple volte
- **Generalità**: Funziona bene senza tuning eccessivo

### Hyperparameters Ottimizzati

```python
# Parallelizzazione
N_ENVS = 32                 # Environments paralleli
BATCH_SIZE = 8192          # Batch per GPU update
N_STEPS = 256              # Steps prima di learning

# Learning
LEARNING_RATE = 3e-4       # Con decay lineare
N_EPOCHS = 4               # Epochs per update
CLIP_RANGE = 0.2          # PPO clipping
ENT_COEF = 0.01           # Exploration bonus

# Reward
GAMMA = 0.99              # Discount factor
GAE_LAMBDA = 0.95         # Advantage estimation
```

### Neural Network Architecture

```python
policy_kwargs = {
    "net_arch": {
        "pi": [512, 512, 256],  # Policy network
        "vf": [512, 512, 256]   # Value network
    },
    "activation_fn": ReLU,
    "normalize_images": True,
    "share_features_extractor": False
}
```

- **Reti separate**: Policy e Value indipendenti
- **3 hidden layers**: Capacità per pattern complessi
- **512-512-256 neuroni**: Bilanciamento capacità/velocità

### Training Pipeline

1. **Raccolta Dati** (N_ENVS × N_STEPS = 8192 samples)
2. **Calcolo Advantages** (GAE con λ=0.95)
3. **Ottimizzazione** (4 epochs, 32 minibatch)
4. **Logging** (TensorBoard ogni update)
5. **Checkpoint** (ogni 100k steps)
6. **Evaluation** (ogni 500k steps)

### GPU Optimization

- **Batch size 8192**: Satura GPU memory bandwidth
- **Minibatch 256**: Ottimale per RTX 5080
- **Mixed precision**: Non usato (stabilità > velocità)
- **Pin memory**: Automatico in PyTorch

## 📊 Evaluation

### Metriche Principali

1. **Total Reward**: Somma cumulativa componenti
2. **Unique Tiles**: Tiles globali uniche visitate
3. **Map Transitions**: Cambi di mappa/zona
4. **Badges**: Medaglie ottenute (max 16)
5. **Episode Length**: Steps prima di truncation

### Visualizzazioni

- **Exploration Heatmap**: Mappa di calore visite
- **Progress Plots**: Reward/tiles nel tempo
- **Checkpoint Comparison**: Performance nel training
- **Combined Heatmap**: Media multi-episodio

### Best Practices Evaluation

```bash
# 1. Test rapido per sanity check
python evaluation/quick_eval.py --episodes 1

# 2. Valutazione statistica (10+ episodi)
python evaluation/evaluate_model.py --episodes 20 --no-render

# 3. Confronto checkpoint per overfitting
python evaluation/evaluate_checkpoints.py

# 4. Test con rendering per debugging
python evaluation/evaluate_model.py --episodes 1 --render
```

## 🐛 Troubleshooting

### TensorBoard Vuoto

```bash
# Verifica directory corretta
ls -la tensorboard/

# Riavvia con path giusto
tensorboard --logdir tensorboard/exploration_v3 --reload_interval 30
```

### Training Lento

1. **Verifica GPU usage**:
   ```bash
   nvidia-smi -l 1  # Monitor ogni secondo
   ```

2. **Reduce evaluation frequency**:
   ```python
   EVAL_FREQ = 1_000_000  # Da 500k
   ```

3. **Aumenta batch size** (se VRAM permette):
   ```python
   BATCH_SIZE = 16384  # Da 8192
   ```

### Out of Memory

```python
# Riduci environments paralleli
N_ENVS = 16  # Da 32

# O riduci batch size
BATCH_SIZE = 4096  # Da 8192
```

### Checkpoint Corrotti

```bash
# Trova checkpoint validi
python -c "
from stable_baselines3 import PPO
import glob
for ckpt in glob.glob('trained_agents/**/*.zip', recursive=True):
    try:
        PPO.load(ckpt, device='cpu')
        print(f'✓ {ckpt}')
    except:
        print(f'✗ {ckpt}')
"
```

## 🎯 Obiettivi e Sfide

### Obiettivi Raggiunti
- ✅ Esplorazione efficiente del mondo
- ✅ Navigazione tra città
- ✅ Interazione con NPC/oggetti
- ✅ Gestione menu base

### Work in Progress
- 🔄 Combattimenti strategici
- 🔄 Cattura Pokémon mirata
- 🔄 Gestione inventario
- 🔄 Team building

### Sfide Tecniche
- **Sparse rewards**: Medaglie molto distanziate
- **Partial observability**: Non tutta la info è su schermo
- **Long horizons**: Obiettivi a 10k+ steps
- **Stochasticity**: RNG battaglie/catture

## 📚 Riferimenti

- [PyBoy Documentation](https://github.com/Baekalfen/PyBoy)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Pokémon Silver RAM Map](https://datacrystal.romhacking.net/wiki/Pokémon_Gold_and_Silver:RAM_map)
- [Gymnasium](https://gymnasium.farama.org/)

## 📄 License

Questo progetto è per scopi educativi e di ricerca. Pokémon è un marchio registrato di Nintendo/Game Freak.

---

**Autore**: Fabio Grillo
**Ultimo aggiornamento**: Gennaio 2025