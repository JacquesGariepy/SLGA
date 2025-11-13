"""Fixtures for performance tests."""

import pytest
import time

@pytest.fixture
def performance_timer():
    """Timer fixture for performance benchmarks."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.end_time is None:
                return time.time() - self.start_time
            return self.end_time - self.start_time

    return Timer()
