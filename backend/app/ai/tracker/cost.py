import numpy as np

from app.ai.detector.models import BoundingBox, Detection
from app.ai.tracker.models import Track


class CostCalculator:
    """
    Computes the association cost matrix between tracks and detections.
    """

    def __init__(
        self,
        iou_weight: float = 0.7,
        distance_weight: float = 0.3,
    ):
        self.iou_weight = iou_weight
        self.distance_weight = distance_weight

    def build_cost_matrix(
        self,
        tracks: list[Track],
        detections: list[Detection],
    ) -> np.ndarray:

        cost_matrix = np.zeros(
            (
                len(tracks),
                len(detections),
            ),
            dtype=np.float32,
        )

        for i, track in enumerate(tracks):

            track_box = track.motion.predicted_bbox

            if track_box is None:
                track_box = track.motion.bbox

            for j, detection in enumerate(detections):

                detection_box = detection.bbox

                iou_cost = 1.0 - self.iou(
                    track_box,
                    detection_box,
                )

                distance_cost = self.center_distance(
                    track_box,
                    detection_box,
                )

                cost = (
                    self.iou_weight * iou_cost
                    +
                    self.distance_weight * distance_cost
                )

                cost_matrix[i, j] = cost

        return cost_matrix

    @staticmethod
    def center_distance(
        box1: BoundingBox,
        box2: BoundingBox,
    ) -> float:

        c1x = (box1.x1 + box1.x2) / 2
        c1y = (box1.y1 + box1.y2) / 2

        c2x = (box2.x1 + box2.x2) / 2
        c2y = (box2.y1 + box2.y2) / 2

        distance = np.sqrt(
            (c1x - c2x) ** 2 +
            (c1y - c2y) ** 2
        )

        return distance / 1000.0

    @staticmethod
    def iou(
        box1: BoundingBox,
        box2: BoundingBox,
    ) -> float:

        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = box1.width * box1.height
        area2 = box2.width * box2.height

        union = area1 + area2 - inter

        if union <= 0:
            return 0.0

        return inter / union