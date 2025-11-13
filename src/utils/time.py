"""
Time formatting and estimation utilities.

This module provides functions for formatting time durations
and estimating training time remaining.
"""

from typing import Dict


def format_time(seconds: float) -> str:
    """
    Format a time duration in seconds to a human-readable string.

    Args:
        seconds: Time duration in seconds.

    Returns:
        formatted: Formatted time string (e.g., "1h 23m 45s", "5m 30s", "42s").

    Example:
        >>> format_time(7265)
        '2h 1m 5s'
        >>> format_time(90)
        '1m 30s'
        >>> format_time(42)
        '42s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def estimate_training_time(
    total_steps: int,
    current_step: int,
    elapsed_time: float,
) -> Dict[str, any]:
    """
    Estimate remaining training time based on current progress.

    This function calculates the expected time to completion based on
    the average speed of completed steps.

    Args:
        total_steps: Total number of training steps planned.
        current_step: Current step number (steps completed so far).
        elapsed_time: Time elapsed since training started (in seconds).

    Returns:
        estimates: Dictionary containing:
            - 'remaining_time': Estimated seconds remaining (float)
            - 'eta': Human-readable ETA string (e.g., "2h 15m 30s")
            - 'steps_per_sec': Average training speed (steps/second)

    Example:
        >>> # After 100 steps taking 50 seconds out of 1000 total
        >>> est = estimate_training_time(
        ...     total_steps=1000,
        ...     current_step=100,
        ...     elapsed_time=50.0
        ... )
        >>> print(f"Speed: {est['steps_per_sec']:.2f} steps/sec")
        >>> print(f"ETA: {est['eta']}")
        Speed: 2.00 steps/sec
        ETA: 7m 30s

    Note:
        Returns default values if current_step is 0 (no progress yet):
        - remaining_time: 0
        - eta: "Unknown"
        - steps_per_sec: 0
    """
    if current_step == 0:
        return {
            "remaining_time": 0.0,
            "eta": "Unknown",
            "steps_per_sec": 0.0,
        }

    # Calculate average training speed
    steps_per_sec = current_step / elapsed_time

    # Calculate remaining steps and estimated time
    remaining_steps = total_steps - current_step
    remaining_time = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0

    return {
        "remaining_time": remaining_time,
        "eta": format_time(remaining_time),
        "steps_per_sec": steps_per_sec,
    }
