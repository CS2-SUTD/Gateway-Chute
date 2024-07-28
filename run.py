import subprocess
import signal
import sys
import os
import psutil
import time

# Add a global variable to track if Ctrl+C has been pressed
ctrl_c_pressed = False

def start_process(cmd, cpu):
    """Start a process with the given command and CPU affinity.
    
    Args:
        cmd (list): The command to run.
        cpu (list): The CPU cores to run the process on.
    
    Returns:
        The started process.
    """
    try:
        process = subprocess.Popen(cmd)
        p = psutil.Process(process.pid)
        p.cpu_affinity(cpu)
        return process
    except subprocess.SubprocessError as e:
        print(f'Failed to start {cmd}: {e}')
        sys.exit(1)

# Start the scripts
process1 = start_process(['python', 'main.py', '-c', 'config/config1.ini', '-s', 'data/sample.mp4'], [0,1])
process2 = start_process(['python', 'main.py', '-c', 'config/config2.ini', '-s', 'data/sample.mp4'], [2,3])
process3 = start_process(['python', 'main.py', '-c', 'config/config3.ini', '-s', 'data/sample.mp4'], [4,5])

def signal_handler(sig, frame):
    """Signal handler for Ctrl+C to stop processes.
    
    Args:
        sig (int): The signal number.
        frame (frame): The current stack frame.
    """

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
