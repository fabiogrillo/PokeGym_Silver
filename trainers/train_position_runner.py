import subprocess
import os
import threading
import time

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
WHITE = "\033[97m"
BLUE = "\033[94m"
RESET = "\033[0m"
CYAN = "\033[96m"

def stream_output(process, log_file):
    for line in iter(process.stdout.readline, b""):
        decoded = line.decode("utf-8").rstrip()

        # Colori per reward (grezzi, cerca "Reward:" nelle righe)
        if "Reward:" in decoded or "reward" in decoded.lower():
            try:
                # Estrai il primo numero decimale trovato
                reward_str = [word for word in decoded.split() if word.replace('.', '', 1).replace('-', '', 1).isdigit()]
                reward_val = float(reward_str[0]) if reward_str else 0.0
                if reward_val > 0:
                    color = GREEN
                elif reward_val < 0:
                    color = RED
                else:
                    color = WHITE
            except:
                color = WHITE
        elif "TRAINING COMPLETED" in decoded:
            color = BLUE
        else:
            color = WHITE

        print(f"{color}{decoded}{RESET}")
        log_file.write(decoded + "\n")
        log_file.flush()

if __name__ == "__main__":
    os.makedirs("outputs/models_position", exist_ok=True)

    # Avvia TensorBoard
    tb_proc = subprocess.Popen(
        ["tensorboard", "--logdir", "./tensorboard_logs", "--port", "6060"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    print(f"{CYAN}TensorBoard started at http://localhost:6060{RESET}")
    time.sleep(3)

    # Avvia lo script di training attuale
    position_proc = subprocess.Popen(
        ["python", "-u", "-m", "trainers.train_position_ppo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    log_file = open("outputs/models_position/train_position.log", "w")
    thread = threading.Thread(target=stream_output, args=(position_proc, log_file), daemon=True)
    thread.start()

    position_proc.wait()
    thread.join()
    log_file.close()

    print(f"{BLUE}\n=== TRAINING COMPLETED ==={RESET}\n")

    tb_proc.terminate()
