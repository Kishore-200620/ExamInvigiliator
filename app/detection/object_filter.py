"""
app/detection/object_filter.py

Detection filtering for ExamGuardian AI.
"""

from config.constants import (
    PERSON_CLASS_ID,
    CELL_PHONE_CLASS_ID,
    BOOK_CLASS_ID,
)

from config.settings import DETECTION_CONFIDENCE


class ObjectFilter:
    """
    Filters YOLO detections.
    """

    VALID_CLASSES = {
        PERSON_CLASS_ID,
        CELL_PHONE_CLASS_ID,
        BOOK_CLASS_ID,
    }

    @staticmethod
    def filter(results):
        """
        Filter YOLO detections.
        """

        detections = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence < DETECTION_CONFIDENCE:
                    continue

                if class_id not in ObjectFilter.VALID_CLASSES:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])

                detections.append(
                    {
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    }
                )

        return detections