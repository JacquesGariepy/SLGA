"""System monitoring utilities for training."""

from __future__ import annotations
import os
import subprocess
import threading
import time
from typing import Optional


def _run_command(cmd: list[str]) -> str:
    """Run a shell command and return its output or error message."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        return f"[command failed] {' '.join(cmd)}\n{exc}\n"


def log_system_info(tag: str = "SNAPSHOT") -> None:
    """
    Emit a system snapshot including GPU/RAM/FS information for logging.

    This collects various system metrics that are useful for monitoring
    training stability and resource usage over time.

    Args:
        tag: Tag to identify this snapshot
    """
    commands = {
        "date": ["date"],
        "uname": ["uname", "-a"],
        "uptime": ["uptime"],
        "nvidia-smi": ["nvidia-smi"],
        "lscpu": ["bash", "-lc", "lscpu | head -n 20"],
        "free": ["bash", "-lc", "free -h"],
        "df": ["bash", "-lc", "df -h"],
        "top": ["bash", "-lc", "top -b -n1 | head -n 5"],
    }

    print(f"\n===== SYSTEM SNAPSHOT ({tag}) =====", flush=True)
    for label, cmd in commands.items():
        print(f"$ {' '.join(cmd)}  # {label}", flush=True)
        output = _run_command(cmd)
        print(output.rstrip(), flush=True)

    print("-- Python system summary --", flush=True)
    try:
        import platform
        cpu_count = os.cpu_count()
        platform_str = platform.platform()
        print(f"platform: {platform_str}", flush=True)
        print(f"cpu_count: {cpu_count}", flush=True)

        try:
            import psutil  # type: ignore

            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=None)
            load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()

            if cpu_freq is not None:
                print(f"cpu_freq: current={cpu_freq.current:.1f}MHz max={cpu_freq.max:.1f}MHz", flush=True)
            print(f"cpu_percent: {cpu_percent:.1f}%", flush=True)
            if load_avg is not None:
                print(f"load_avg (1m,5m,15m): {load_avg}", flush=True)
            print(
                "memory: used={:.2f}GB free={:.2f}GB total={:.2f}GB".format(
                    vm.used / 1e9, vm.available / 1e9, vm.total / 1e9
                ),
                flush=True,
            )
            print(
                "swap: used={:.2f}GB free={:.2f}GB total={:.2f}GB".format(
                    swap.used / 1e9, swap.free / 1e9, swap.total / 1e9
                ),
                flush=True,
            )
        except ImportError:
            print("psutil not available; install it for detailed CPU/memory stats", flush=True)
    except Exception as exc:
        print(f"[python system summary failed] {exc}", flush=True)

    print("===== END SYSTEM SNAPSHOT =====\n", flush=True)


class SystemMonitor:
    """
    Background system monitoring thread.

    Periodically logs system information during training.
    """

    def __init__(self, interval_seconds: int = 120):
        """
        Initialize system monitor.

        Args:
            interval_seconds: Seconds between system snapshots
        """
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the monitoring thread."""
        if self.thread is not None:
            return

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._monitor_loop,
            name="SystemMonitor",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        """Stop the monitoring thread."""
        if self.thread is None:
            return

        self.stop_event.set()
        self.thread.join(timeout=5.0)
        self.thread = None

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_event.wait(self.interval_seconds):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_system_info(f"PERIODIC {timestamp}")
