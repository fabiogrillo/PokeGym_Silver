# scripts/create_post_starter_state.py
from pyboy import PyBoy
from termcolor import cprint
import sys
import select

def create_post_starter_save():
    """Create save state after getting starter Pokemon"""
    
    cprint("🎮 Creating post-starter save state...", "cyan")
    
    pyboy = PyBoy("roms/Pokemon_Silver.gbc", window="SDL2")
    pyboy.set_emulation_speed(5)
    
    cprint("\n📝 Instructions:", "yellow")
    cprint("1. Load your game or start new", "blue")
    cprint("2. Get your starter from Elm", "blue")
    cprint("3. Exit Elm's lab", "blue")
    cprint("4. Stand in New Bark Town (outside)", "blue")
    cprint("5. Type 's' + Enter to save", "blue")
    cprint("6. Type 'v' + Enter to verify", "blue")
    cprint("7. Type 'q' + Enter to quit", "blue")
    
    while True:
        pyboy.tick()
        
        try:
            if sys.platform != 'win32' and select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip().lower()
                
                if line == 's':
                    cprint("\n💾 Saving state...", "yellow")
                    with open("post_starter.state", "wb") as f:
                        pyboy.save_state(f)
                    cprint("✅ Saved to post_starter.state", "green")
                    
                elif line == 'v':
                    # Verify state
                    party_size = pyboy.memory[0xDA22]
                    map_id = pyboy.memory[0xDA01]
                    badges = pyboy.memory[0xD57C]
                    
                    cprint("\n🔍 State check:", "cyan")
                    cprint(f"   Party size: {party_size} {'✅' if party_size > 0 else '❌'}", 
                           "green" if party_size > 0 else "red")
                    cprint(f"   Map: {map_id} (should be 1 for New Bark)", "blue")
                    cprint(f"   Badges: {badges} {'✅' if badges == 0 else '❌'}", 
                           "green" if badges == 0 else "red")
                    
                    if party_size > 0:
                        pokemon_id = pyboy.memory[0xDA23]
                        level = pyboy.memory[0xDA49]
                        cprint(f"   Starter: Pokemon #{pokemon_id}, Level {level}", "green")
                    
                elif line == 'q':
                    break
                    
        except KeyboardInterrupt:
            break
    
    pyboy.stop()

if __name__ == "__main__":
    create_post_starter_save()