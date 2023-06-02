import os
import subprocess
import threading
import sys
from threading import Event
import signal
class TrainingThread(threading.Thread):
    def __init__(self, stdout_redirect):
        super().__init__()
        self.stdout_redirect = stdout_redirect
        self.process = None
        self.terminate_event = Event()

    def run(self):
        command = ['start_kohya.bat']

        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        for line in self.process.stdout:
            if self.terminate_event.is_set():
                break
            self.stdout_redirect.write(line)

        self.process.wait()

    def terminate_process(self):
        if self.process and self.process.poll() is None:
            self.terminate_event.set()
            self.process.terminate()

            # Send a specific signal to force termination
            if sys.platform == 'win32':
                # For Windows
                os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
            else:
                # For Unix-like systems
                os.kill(self.process.pid, signal.SIGTERM)
            self.process.wait()