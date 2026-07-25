"""
utils/timer.py

Utility class for measuring execution time and FPS.
"""

import time


class Timer:
    """
    Simple timer utility.
    """

    def __init__(self):
        self.start_time = None

    def start(self):
        """
        Start the timer.
        """
        self.start_time = time.perf_counter()

    def stop(self):
        """
        Stop the timer and return elapsed time.
        """
        if self.start_time is None:
            return 0.0

        return time.perf_counter() - self.start_time


class FPSCounter:
    """
    Calculates Frames Per Second.
    """

    def __init__(self):
        self.previous_time = time.perf_counter()
        self.fps = 0.0

    def update(self):
        """
        Update FPS value.
        """
        current_time = time.perf_counter()

        delta = current_time - self.previous_time

        if delta > 0:
            self.fps = 1.0 / delta

        self.previous_time = current_time

        return self.fps