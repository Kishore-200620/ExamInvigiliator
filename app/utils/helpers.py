"""
utils/helpers.py

Common helper functions for ExamGuardian AI.
"""

from math import sqrt


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    """
    x1, y1 = point1
    x2, y2 = point2

    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_bbox_center(bbox):
    """
    Return the center point of a bounding box.

    bbox format:
    (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return (center_x, center_y)


def bbox_width(bbox):
    """
    Return bounding box width.
    """
    return bbox[2] - bbox[0]


def bbox_height(bbox):
    """
    Return bounding box height.
    """
    return bbox[3] - bbox[1]


def clamp(value, minimum, maximum):
    """
    Restrict a value between minimum and maximum.
    """
    return max(minimum, min(value, maximum))


def point_inside_bbox(point, bbox):
    """
    Check whether a point lies inside a bounding box.
    """
    px, py = point
    x1, y1, x2, y2 = bbox

    return x1 <= px <= x2 and y1 <= py <= y2


def bbox_area(bbox):
    """
    Calculate bounding box area.
    """
    return bbox_width(bbox) * bbox_height(bbox)