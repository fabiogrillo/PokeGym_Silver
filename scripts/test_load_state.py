from pyboy import PyBoy

pyboy = PyBoy("roms/Pokemon_Silver.gbc", window="SDL2")
pyboy.set_emulation_speed(1)

with open("start_of_game.state", "rb") as f:
    pyboy.load_state(f)

for _ in range(30):
    pyboy.tick()

print("State loaded. Use arrows to move. Press CTRL+C to exit.")

try:
    while True:
        pyboy.tick()
except KeyboardInterrupt:
    pyboy.stop()
