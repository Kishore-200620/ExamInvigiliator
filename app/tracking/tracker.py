import numpy as np
from types import SimpleNamespace

from .byte_tracker import BYTETracker


class Tracker:

    def __init__(self):

        args = SimpleNamespace(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            mot20=False,
        )

        self.tracker = BYTETracker(args)

    def update(self, detections, frame):

        outputs = []

        for detection in detections:

            if detection["class_id"] != 0:
                continue

            x1, y1, x2, y2 = detection["bbox"]

            score = detection["confidence"]

            outputs.append(
                [x1, y1, x2, y2, score]
            )

        if len(outputs) == 0:

            outputs = np.empty((0, 5), dtype=float)

        else:

            outputs = np.asarray(outputs, dtype=float)

        h, w = frame.shape[:2]

        tracks = self.tracker.update(
            outputs,
            (h, w),
            (h, w)
        )

        return tracks