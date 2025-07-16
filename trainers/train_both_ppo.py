import subprocess
import os
import threading
import time

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
CYAN = "\033[96m"

def stream_output(process, color, log_file):
    for line in iter(process.stdout.readline, b""):
        decoded_line = line.decode("utf-8").rstrip()
        print(f"{color}{decoded_line}{RESET}")
        log_file.write(decoded_line + "\n")
        log_file.flush()

if __name__ == "__main__":
    os.makedirs("outputs/models_hashing", exist_ok=True)
    os.makedirs("outputs/models_position", exist_ok=True)

    # Launch TensorBoard
    tb_proc = subprocess.Popen(
        [
            "tensorboard",
            "--logdir",
            "./tensorboard_logs",
            "--port",
            "6060"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    print(f"{CYAN}TensorBoard started on http://localhost:6060{RESET}")

    # Wait a bit so TensorBoard initializes
    time.sleep(3)

    # Launch both trainings
    hashing_proc = subprocess.Popen(
        ["python", "-u", "-m", "trainers.train_hashing_ppo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    position_proc = subprocess.Popen(
        ["python", "-u", "-m", "trainers.train_position_ppo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    hash_log = open("outputs/models_hashing/train_hashing.log", "w")
    pos_log = open("outputs/models_position/train_position.log", "w")

    print("Training started for both models...")

    hash_thread = threading.Thread(
        target=stream_output,
        args=(hashing_proc, RED, hash_log),
        daemon=True
    )
    pos_thread = threading.Thread(
        target=stream_output,
        args=(position_proc, GREEN, pos_log),
        daemon=True
    )

    hash_thread.start()
    pos_thread.start()

    # Wait for trainings
    hashing_proc.wait()
    position_proc.wait()

    hash_thread.join()
    pos_thread.join()

    hash_log.close()
    pos_log.close()

    print("Training completed.")

    # Optionally: kill TensorBoard when done
    tb_proc.terminate()
