import os
import cv2
from envs.pokemon_silver_env import PokemonSilver

# Crea la cartella se non esiste
if not os.path.exists("frames"):
    os.makedirs("frames")
    
# Inizializza l'ambiente
env = PokemonSilver("roms/Pokemon_Silver.gbc")
obs = env.reset()

# Loop di 100 passi
for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    # Salva 1 frame ogni 10
    if i % 10 == 0:
        frame_path = os.path.join("frames", f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, obs)

env.close()
