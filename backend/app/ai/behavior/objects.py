"""
Object perception - Phase 5 Foundation
Expected: mobile phone, laptop, tablet, book, paper, calculator, earphones
Must be independent from person detection - do not modify YOLO person detector.
"""
from typing import Optional, List

import numpy as np

from app.ai.behavior.schemas import ObjectsObservation
from app.ai.tracker.roi import ROI


class ObjectsAnalyzer:
    """
    Prohibited object detection per student ROI.

    Future: Use separate lightweight YOLO (e.g., yolo26n 640) for objects,
    or YOLO with multiple classes (person + phone) but currently person-only yolo.py:51.

    Currently: NOT IMPLEMENTED - returns available=False
    Do NOT modify YOLODetector PERSON_CLASS behavior in this task.
    """

    # COCO class IDs for reference (YOLO 80 classes)
    COCO_OBJECTS = {
        "mobile_phone": 67,  # cell phone
        "laptop": 63,
        "tablet": 72,  # not in COCO 80, would be laptop
        "book": 73,
        "paper": 73,  # book
        "calculator": 72,  # no direct calculator in COCO
        "earphones": 72,  # no direct earphones
    }

    def __init__(self):
        self.model_name = "not_implemented"
        self.model = None
        # Future integration points:
        # - yolo26n 640 for objects (3ms) vs person 1536 (47ms)
        # - Run periodically (every 60 frames) or conditionally (only when hand near desk)
        # - On relevant ROIs (lower 30% for phone below desk) at lower resolution
        # For now, no model

    def analyze(
        self,
        image: np.ndarray,
        roi: ROI,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> ObjectsObservation:
        """
        Detect prohibited objects for ROI.

        Args:
            image: full frame
            roi: person ROI (or lower-body ROI for phone)
            frame_id, timestamp: for observation

        Returns:
            ObjectsObservation with available=False until real model integrated
        """
        obs = ObjectsObservation(frame_id=frame_id, timestamp=timestamp)

        if not roi.valid:
            obs.available = False
            return obs

        # No verified object model available - explicitly unavailable
        # Do NOT run second full-resolution YOLO 1536 per student per frame (would be 60*47ms = 2.8s)
        # Future: separate object YOLO 640 3ms * 0.66/frame (every 90) = 2ms avg
        obs.available = False
        obs.objects = []
        obs.phone_detected = False
        obs.laptop_detected = False
        obs.tablet_detected = False
        obs.book_detected = False
        obs.paper_detected = False
        obs.calculator_detected = False
        obs.earphones_detected = False

        return obs

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "available": False,
            "reason": "No verified object model - YOLO currently person-only (yolo.py:51 PERSON_CLASS 0), need separate yolo26n 640 for phone/laptop/book",
            "note": "Object detection must be independent, periodic/conditional/ROI-based, not second 1536 YOLO per student per frame",
        }
