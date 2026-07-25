"""
app/camera/exceptions.py

Custom exceptions for the camera module.
"""


class CameraError(Exception):
    """Base exception for camera-related errors."""
    pass


class CameraOpenError(CameraError):
    """Raised when the camera cannot be opened."""
    pass


class FrameReadError(CameraError):
    """Raised when a frame cannot be read."""
    pass