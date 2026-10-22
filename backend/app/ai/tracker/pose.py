# Re-export for compatibility - canonical is head_pose.py
from app.ai.tracker.head_pose import HeadPoseEstimator, HeadPose

__all__ = ["HeadPoseEstimator", "HeadPose"]
