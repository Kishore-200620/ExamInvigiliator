from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.ai.detector.models import Detection
from app.ai.tracker.models import Track


@dataclass(slots=True)
class MatchResult:
    """
    Output of the Hungarian matcher.
    """

    matches: list[tuple[int, int]]

    unmatched_tracks: list[int]

    unmatched_detections: list[int]


class HungarianMatcher:
    """
    Assign detections to tracks using a cost matrix.
    """

    def __init__(self, max_cost: float = 0.85):
        self.max_cost = max_cost

    def match(
        self,
        tracks: list[Track],
        detections: list[Detection],
        cost_matrix: np.ndarray,
    ) -> MatchResult:

        if len(tracks) == 0:
            return MatchResult(
                matches=[],
                unmatched_tracks=[],
                unmatched_detections=list(range(len(detections))),
            )

        if len(detections) == 0:
            return MatchResult(
                matches=[],
                unmatched_tracks=list(range(len(tracks))),
                unmatched_detections=[],
            )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []

        matched_tracks = set()

        matched_detections = set()

        for track_idx, detection_idx in zip(row_ind, col_ind):

            if cost_matrix[track_idx, detection_idx] > self.max_cost:
                continue

            matches.append((track_idx, detection_idx))

            matched_tracks.add(track_idx)

            matched_detections.add(detection_idx)

        unmatched_tracks = [
            i
            for i in range(len(tracks))
            if i not in matched_tracks
        ]

        unmatched_detections = [
            i
            for i in range(len(detections))
            if i not in matched_detections
        ]

        return MatchResult(
            matches=matches,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
        )