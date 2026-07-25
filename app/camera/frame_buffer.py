"""
app/camera/frame_buffer.py

Frame buffer for ExamGuardian AI.
"""

from collections import deque


class FrameBuffer:
    """
    FIFO buffer for camera frames.
    """

    def __init__(self, max_size=30):
        self.buffer = deque(maxlen=max_size)

    def push(self, frame):
        """
        Add a frame to the buffer.
        """
        self.buffer.append(frame)

    def pop(self):
        """
        Get the oldest frame from the buffer.
        """
        if self.is_empty():
            return None

        return self.buffer.popleft()

    def is_empty(self):
        """
        Check whether the buffer is empty.
        """
        return len(self.buffer) == 0

    def clear(self):
        """
        Remove all frames.
        """
        self.buffer.clear()

    def size(self):
        """
        Current number of frames.
        """
        return len(self.buffer)