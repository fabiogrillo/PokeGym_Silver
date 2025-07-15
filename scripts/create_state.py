from pyboy import PyBoy

# Avvia la finestra
pyboy = PyBoy("roms/Pokemon_Silver.gbc", window="SDL2")
pyboy.set_emulation_speed(1)

print("\n[INFO] Start game, press Start, choose name, get in overworld.")
print("[INFO] Once ready, type 's' + Enter to save.\n")

while True:
    pyboy.tick()
    # Check if user press 's' on console
    try:
        import sys
        import select
        # check if there are inputs on stdin (no blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line.strip().lower() == 's':
                print("\n[INFO] Saving state...")
                with open("start_of_game.state", "wb") as f:
                    pyboy.save_state(f)
                print("[INFO] Saved as 'start_of_game.state'. Exiting.")
                break
    except Exception as e:
        print("[ERROR]", e)
        break

pyboy.stop()
