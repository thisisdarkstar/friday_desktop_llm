"""Keep-Awake Watchdog for Windows.

Prevents Windows from entering Sleep / Modern Standby while the GPU training process is running.
Allows display to turn off / lock while keeping CPU & GPU running at full performance.
Automatically releases power lock when the target training process finishes.
"""

import ctypes
import time
import sys
import psutil

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def keep_awake_until_exit(target_pid: int):
    # Tell Windows power manager that system is performing background work
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    )
    print(f"[*] Windows Sleep Lock active for PID {target_pid}. System will NOT sleep.")
    sys.stdout.flush()

    try:
        while True:
            if not psutil.pid_exists(target_pid):
                print(f"[*] Target PID {target_pid} has completed. Releasing sleep lock.")
                break
            time.sleep(15)
    finally:
        # Restore normal power state
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("[*] Normal Windows power state restored.")


if __name__ == "__main__":
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 14792
    keep_awake_until_exit(pid)
