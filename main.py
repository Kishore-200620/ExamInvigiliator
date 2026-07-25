"""
main.py

ExamGuardian AI
Application Entry Point
"""

import cv2

from app.camera.camera import Camera
from app.detection.detector import Detector
from app.utils.logger import logger
CLASS_NAMES = {
    0: "Person",
    67: "Phone",
    73: "Book",
}
def main():
    """
    Main application.
    """

    logger.info("Starting ExamGuardian AI...")

    camera = Camera()

    detector = Detector()

    while True:

        success, frame = camera.read()

        if not success:
            logger.warning("Failed to read frame.")
            break

        detections = detector.detect(frame)

        for detection in detections:

            x1, y1, x2, y2 = map(int, detection["bbox"])

            confidence = detection["confidence"]

            label = CLASS_NAMES.get(
    detection["class_id"],
    "Unknown"
)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1,
            )

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        display_frame = cv2.resize(
            frame,
            (1280, 720)
        )

        cv2.imshow(
            "ExamGuardian AI",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()

    cv2.destroyAllWindows()

    logger.info("ExamGuardian AI stopped.")


if __name__ == "__main__":
    main()