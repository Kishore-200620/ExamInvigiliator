"""
Student ROI Manager - Phase 5 Foundation
Centralizes ROI calculations for all behavioral perception modules.
Safe against image boundaries, invalid boxes, very small boxes.
Follows existing head-pose convention: upper 40% of person bbox is head/face.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.ai.detector.models import BoundingBox
from app.ai.tracker.models import Track


@dataclass(slots=True)
class ROI:
    """Explicit ROI object - not raw tuple."""

    bbox: BoundingBox
    valid: bool
    reason: str = ""
    clipped: bool = False
    original_bbox: BoundingBox | None = None


@dataclass(slots=True)
class ROISet:
    """Per-student ROIs for one frame."""

    person: ROI
    head: ROI  # upper 40% of person (face region)
    upper_body: ROI  # upper 60%
    body: ROI  # full person (clipped)


class ROIManager:
    """
    Reusable ROI manager for StudentTrack.

    Input: Track + frame dimensions
    Output: ROISet with 4 ROIs, all clipped to image boundaries.

    Safe handling:
    - negative coordinates
    - outside image
    - x1 >= x2, y1 >= y2
    - very small boxes
    - empty crops
    """

    def __init__(
        self,
        min_person_height: int = 40,
        min_person_width: int = 20,
        min_head_height: int = 20,
        min_head_width: int = 20,
        head_ratio: float = 0.4,
        upper_body_ratio: float = 0.6,
    ):
        self.min_person_height = min_person_height
        self.min_person_width = min_person_width
        self.min_head_height = min_head_height
        self.min_head_width = min_head_width
        self.head_ratio = head_ratio
        self.upper_body_ratio = upper_body_ratio

    def _clip_bbox(
        self, bbox: BoundingBox, image_width: int, image_height: int
    ) -> tuple[BoundingBox, bool]:
        """Clip bbox to image boundaries, return (clipped_bbox, was_clipped)."""
        x1 = int(bbox.x1)
        y1 = int(bbox.y1)
        x2 = int(bbox.x2)
        y2 = int(bbox.y2)

        # Check invalid before clipping
        if x2 <= x1 or y2 <= y1:
            return bbox, False

        was_clipped = False
        orig_x1, orig_y1, orig_x2, orig_y2 = x1, y1, x2, y2

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        if x1 != orig_x1 or y1 != orig_y1 or x2 != orig_x2 or y2 != orig_y2:
            was_clipped = True

        # After clipping, check again
        if x2 <= x1 or y2 <= y1:
            # Still invalid after clipping (e.g., completely outside)
            return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), was_clipped

        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), was_clipped

    def _is_valid_roi(
        self, bbox: BoundingBox, min_w: int, min_h: int, image_width: int, image_height: int
    ) -> tuple[bool, str]:
        """Check if ROI is valid for processing."""
        if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
            return False, "invalid: x2<=x1 or y2<=y1"
        if bbox.x1 < 0 or bbox.y1 < 0 or bbox.x2 > image_width or bbox.y2 > image_height:
            # Should have been clipped, but if still outside, invalid
            if bbox.x1 < -100 or bbox.y1 < -100:
                return False, "outside image (large negative)"
        w = bbox.width
        h = bbox.height
        if w < min_w or h < min_h:
            return False, f"too small: {w}x{h} < {min_w}x{min_h}"
        if w == 0 or h == 0:
            return False, "empty crop"
        return True, "valid"

    def get_rois(
        self, track: Track, image_width: int, image_height: int
    ) -> ROISet:
        """
        Derive 4 ROIs from StudentTrack.

        Args:
            track: StudentTrack with motion.bbox
            image_width, image_height: frame dimensions

        Returns:
            ROISet with person, head, upper_body, body
        """
        person_bbox = track.motion.bbox

        # Person ROI - full bbox clipped
        clipped_person, was_clipped_p = self._clip_bbox(
            person_bbox, image_width, image_height
        )
        valid_p, reason_p = self._is_valid_roi(
            clipped_person, self.min_person_width, self.min_person_height, image_width, image_height
        )
        person_roi = ROI(
            bbox=clipped_person,
            valid=valid_p,
            reason=reason_p,
            clipped=was_clipped_p,
            original_bbox=person_bbox,
        )

        # Head ROI - upper 40% of person bbox (existing convention from head_pose.py:107)
        # Use ORIGINAL bbox for ratio, then clip
        x1, y1, x2, y2 = int(person_bbox.x1), int(person_bbox.y1), int(person_bbox.x2), int(person_bbox.y2)
        head_h = int((y2 - y1) * self.head_ratio)
        head_y2 = y1 + head_h
        # Head is same width as person, but upper portion, and slightly narrower (0.8*width centered) for better face focus
        # For ROI manager, keep full width for compatibility, but also provide centered version as alternative
        # Here we use full width upper 40% as defined in spec: "upper approximately 40% of person bounding box"
        head_bbox_raw = BoundingBox(x1=x1, y1=y1, x2=x2, y2=head_y2)
        clipped_head, was_clipped_h = self._clip_bbox(
            head_bbox_raw, image_width, image_height
        )
        valid_h, reason_h = self._is_valid_roi(
            clipped_head, self.min_head_width, self.min_head_height, image_width, image_height
        )
        # Additional check: if original person bbox was invalid, head is also invalid
        if not valid_p:
            valid_h = False
            reason_h = f"person invalid: {reason_p}"
        head_roi = ROI(
            bbox=clipped_head,
            valid=valid_h,
            reason=reason_h,
            clipped=was_clipped_h,
            original_bbox=head_bbox_raw,
        )

        # Upper body ROI - upper 60% of person bbox
        upper_h = int((y2 - y1) * self.upper_body_ratio)
        upper_y2 = y1 + upper_h
        upper_bbox_raw = BoundingBox(x1=x1, y1=y1, x2=x2, y2=upper_y2)
        clipped_upper, was_clipped_u = self._clip_bbox(
            upper_bbox_raw, image_width, image_height
        )
        valid_u, reason_u = self._is_valid_roi(
            clipped_upper, self.min_person_width, self.min_person_height, image_width, image_height
        )
        if not valid_p:
            valid_u = False
            reason_u = f"person invalid: {reason_p}"
        upper_body_roi = ROI(
            bbox=clipped_upper,
            valid=valid_u,
            reason=reason_u,
            clipped=was_clipped_u,
            original_bbox=upper_bbox_raw,
        )

        # Body ROI - full person clipped (same as person, but explicit for body pose)
        body_roi = ROI(
            bbox=clipped_person,
            valid=valid_p,
            reason=reason_p,
            clipped=was_clipped_p,
            original_bbox=person_bbox,
        )

        return ROISet(
            person=person_roi,
            head=head_roi,
            upper_body=upper_body_roi,
            body=body_roi,
        )

    def crop_image(
        self, image: np.ndarray, roi: ROI
    ) -> tuple[np.ndarray | None, str]:
        """
        Safely crop image using ROI.

        Returns:
            (cropped_image or None, reason)
        """
        if not roi.valid:
            return None, f"ROI invalid: {roi.reason}"

        bbox = roi.bbox
        h, w = image.shape[:2]

        # Double-check bounds
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))

        if x2 <= x1 or y2 <= y1:
            return None, "invalid after bounds check"

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None, "empty crop"

        return crop, "valid"
