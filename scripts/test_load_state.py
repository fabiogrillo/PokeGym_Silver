from pyboy import PyBoy

pyboy = PyBoy("roms/Pokemon_Silver.gbc", window="SDL2")
pyboy.set_emulation_speed(1)

# Carica il save state
with open("start_of_game.state", "rb") as f:
    pyboy.load_state(f)

# Tick iniziali per far stabilizzare lo stato
for _ in range(30):
    pyboy.tick()

print("State loaded. Use arrows to move. Press CTRL+C to exit.\n")

try:
    while True:
        # Tick normale
        for _ in range(5):
            pyboy.tick()
        
        # Leggi e stampa la posizione
        x = pyboy.memory[0xDA02]
        y = pyboy.memory[0xDA03]
        print(f"Current position: X={x}, Y={y}")
        
except KeyboardInterrupt:
    print("\nExiting...")
    pyboy.stop()
