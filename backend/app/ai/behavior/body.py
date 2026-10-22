"""
Body pose interface - Phase 5 Foundation
Expected landmarks: nose, shoulders, elbows, wrists, hips, knees, ankles
No verified model currently available - returns unavailable.
Do NOT fabricate landmarks.
"""
from typing import Optional

import numpy as np

from app.ai.behavior.schemas import BodyPoseObservation
from app.ai.tracker.roi import ROI


class BodyPoseAnalyzer:
    """
    Body pose estimation interface - now integrated with YOLO26n-Pose.

    Uses YOLO26n-Pose (7.8MB, 17 COCO keypoints, 40ms full-frame) as primary body pose.
    Shares GPU with YOLO person detector, runs periodically (every 60 frames) per pose scheduler.
    Preserves actual keypoint confidence from model.
    """

    def __init__(self, model_name: str = "yolo26n-pose", device: str = "cuda", conf: float = 0.3, iou: float = 0.65, imgsz: int = 1280):
        self.model_name = model_name
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO26n-Pose if available - check installed version and local checkpoint."""
        import pathlib

        # Check ultralytics version
        try:
            import ultralytics
            ver = ultralytics.__version__
        except Exception:
            ver = "unknown"

        # Check for yolo26n-pose.pt locally
        local_paths = [
            pathlib.Path(__file__).resolve().parents[2] / "models" / "yolo26n-pose.pt",
            pathlib.Path("yolo26n-pose.pt"),
            pathlib.Path(r"D:\clg project\ExamInvigilationAI\yolo26n-pose.pt"),
            pathlib.Path(r"D:\clg project\ExamInvigilationAI\backend\yolo26n-pose.pt"),
            pathlib.Path(r"D:\clg project\ExamInvigilationAI\backend\models\yolo26n-pose.pt"),
        ]
        found = None
        for p in local_paths:
            if p.exists():
                found = p
                break

        if found is None:
            # Try to load via YOLO('yolo26n-pose.pt') which will auto-download if not found
            try:
                from ultralytics import YOLO

                self.model = YOLO("yolo26n-pose.pt")
                self.model_name = f"yolo26n-pose (auto-download, ultralytics {ver})"
                return
            except Exception as e:
                self.model = None
                self.model_name = f"yolo26n-pose not found: {e}"
                return
        else:
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(found))
                self.model_name = f"yolo26n-pose ({found.name}, ultralytics {ver})"
                return
            except Exception as e:
                self.model = None
                self.model_name = f"yolo26n-pose load failed: {e}"
                return

    def from_detection(
        self,
        detection,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> BodyPoseObservation:
        """
        Create BodyPoseObservation directly from a YOLO pose Detection that already contains keypoints.
        This is the efficient path when YOLO pose is the primary detector (one forward pass gives both bbox + pose).
        No second inference needed.

        Args:
            detection: Detection with metadata["keypoints"] from YOLO pose (17x3)
            frame_id, timestamp: for observation

        Returns:
            BodyPoseObservation with actual keypoint confidences, not fake.
        """
        obs = BodyPoseObservation(frame_id=frame_id, timestamp=timestamp)
        kpts = None
        if hasattr(detection, "metadata") and isinstance(detection.metadata, dict):
            kpts = detection.metadata.get("keypoints")
        if kpts is None:
            obs.available = False
            return obs
        # kpts is (17,3) np array or tensor
        try:
            if hasattr(kpts, "cpu"):
                kpts = kpts.cpu().numpy()
            # Ensure it's numpy
            kpts = np.array(kpts)
            if kpts.shape != (17, 3):
                # Try to handle different shape
                if len(kpts) == 17:
                    pass
                else:
                    obs.available = False
                    return obs
            obs.available = True
            obs.keypoints = {}
            names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                     "left_wrist", "right_wrist", "left_hip", "right_hip",
                     "left_knee", "right_knee", "left_ankle", "right_ankle"]
            for idx, name in enumerate(names):
                if idx < len(kpts):
                    x, y, c = float(kpts[idx, 0]), float(kpts[idx, 1]), float(kpts[idx, 2])
                    obs.keypoints[name] = (x, y, c)
                    if hasattr(obs, name):
                        setattr(obs, name, (x, y, c))
            # Confidence as mean of all keypoint confs
            if obs.keypoints:
                obs.confidence = float(sum(v[2] for v in obs.keypoints.values()) / len(obs.keypoints))
                obs.quality = obs.confidence
            else:
                obs.confidence = None
                obs.quality = None
            return obs
        except Exception:
            obs.available = False
            return obs

    def analyze(
        self,
        image: np.ndarray,
        roi: ROI,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> BodyPoseObservation:
        """
        Estimate body pose for ROI - kept for compatibility, but for YOLO pose as primary,
        use from_detection() or analyze_frame() instead to avoid second inference.
        This per-ROI path will run YOLO pose on the ROI crop which is not optimal.
        Returns unavailable to encourage using from_detection when YOLO pose is primary.
        """
        # For YOLO pose as primary detector, per-ROI should use from_detection, not this
        # To avoid second full-frame inference per ROI, return unavailable and let caller use from_detection
        obs = BodyPoseObservation(frame_id=frame_id, timestamp=timestamp)
        obs.available = False
        obs.quality = None
        # Note: use from_detection() with the Detection that already has keypoints
        return obs

    def analyze_frame(
        self,
        image: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> list[BodyPoseObservation]:
        """
        Full-frame YOLO pose - preferred for 50-60 students.
        Runs YOLO26n-Pose once per frame, returns list of BodyPoseObservations (one per person).

        This is the efficient path: 1 inference per frame vs 60 per-ROI inferences.
        """
        if self.model is None:
            return []

        try:
            results = self.model.predict(
                image, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, verbose=False
            )
            observations = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                if result.keypoints is None or result.keypoints.data is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'cpu') else result.boxes.xyxy
                kpts = result.keypoints.data.cpu().numpy() if hasattr(result.keypoints.data, 'cpu') else result.keypoints.data
                confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, 'cpu') else result.boxes.conf
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    # Build observation with actual keypoint confidences
                    obs = BodyPoseObservation(frame_id=frame_id, timestamp=timestamp, available=True)
                    # COCO 17
                    names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                             "left_wrist", "right_wrist", "left_hip", "right_hip",
                             "left_knee", "right_knee", "left_ankle", "right_ankle"]
                    obs.keypoints = {}
                    for idx, name in enumerate(names):
                        if idx < kpts.shape[1]:
                            x, y, c = float(kpts[i, idx, 0]), float(kpts[i, idx, 1]), float(kpts[i, idx, 2])
                            obs.keypoints[name] = (x, y, c)
                            # Set specific fields
                            if hasattr(obs, name):
                                setattr(obs, name, (x, y, c))
                    # Also handle left/right variants
                    # Already set via names, but ensure the body pose fields are correctly mapped
                    # The above loop already sets left_shoulder etc
                    obs.confidence = float(confs[i]) if i < len(confs) else 0.5
                    # Quality as mean keypoint conf
                    if obs.keypoints:
                        avg_conf = float(sum(v[2] for v in obs.keypoints.values()) / len(obs.keypoints))
                        obs.quality = avg_conf
                    else:
                        obs.quality = 0.0
                    observations.append(obs)
            return observations
        except Exception:
            return []

    def get_info(self) -> dict:
        # Check if model actually loaded
        if self.model is not None:
            try:
                import ultralytics
                return {
                    "model_name": self.model_name,
                    "available": True,
                    "weights": str(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else "yolo26n-pose.pt",
                    "ultralytics": ultralytics.__version__,
                    "input_size": self.imgsz,
                    "note": "YOLO26n-Pose full-frame, 17 keypoints, actual confidence preserved, not fake",
                }
            except Exception:
                pass
        return {
            "model_name": self.model_name,
            "available": self.model is not None,
            "reason": "No verified body pose model" if self.model is None else "Loaded",
            "required_model": "yolo26n-pose.pt via ultralytics YOLO('yolo26n-pose.pt')",
        }
