import subprocess
import signal
import sys
import os
import psutil
import time

# Add a global variable to track if Ctrl+C has been pressed
ctrl_c_pressed = False

def start_process(cmd):
    try:
        process = subprocess.Popen(cmd)
    except subprocess.SubprocessError as e:
        print(f'Failed to start {cmd}: {e}')
        sys.exit(1)
    p = psutil.Process(process.pid)
    # p.cpu_affinity(cpu)
    return process

# Start the scripts
process1 = start_process(['python', 'main.py', '-c', 'config/config1.ini', '-s', 'rtsp://192.168.0.157:8554/chute'])
process2 = start_process(['python', 'main.py', '-c', 'config/config2.ini', '-s', 'rtsp://192.168.0.157:8554/chute'])
process3 = start_process(['python', 'main.py', '-c', 'config/config3.ini', '-s', 'rtsp://192.168.0.157:8554/chute'])

def signal_handler(sig, frame):
    global ctrl_c_pressed
    ctrl_c_pressed = True
    print('Ctrl+C detected. Terminating scripts...')
    os.kill(process1.pid, signal.SIGTERM)
    os.kill(process2.pid, signal.SIGTERM)
    os.kill(process3.pid, signal.SIGTERM)
    process1.wait()
    process2.wait()
    process3.wait()
    print('All processes have been terminated.')
    sys.exit(0)
# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

while True:
    time.sleep(1)
