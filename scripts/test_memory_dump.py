# This script aims to get the right RAM values for X and Y position
from pyboy import PyBoy
import time

rom_path = "roms/Pokemon_Silver.gbc"

# Start pyboy
pyboy = PyBoy(rom_path, window="SDL2")
pyboy.set_emulation_speed(1)

# Range of addresses
x_address = 0xDA02 # Correct val DA02
y_address = 0xDA03 # Correct val DA03

print("Move the player with arrow keys and observe changing addresses...\nPress CTRL+C to stop.\n")

try:
    while True:
        for _ in range(5):
            pyboy.tick()

        for addr in range(x_address, y_address + 1):
            value = pyboy.memory[addr]
            print(f"0x{addr:04X}: {value}")
        
        print("-"*40)
        time.sleep(0.2)
except KeyboardInterrupt:
    print("Stopped.")
finally:
    pyboy.stop()