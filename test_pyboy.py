from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Starting emulator
pyboy = PyBoy("roms/Pokemon_Silver.gbc", window="SDL2")

while not pyboy.tick():
    # Press A each 6 seconds for testing
    if pyboy.get_frame_count() % 60 == 0:
        pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    elif pyboy.get_frame_count() % 60 == 30:
        pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

# Close
pyboy.stop()
