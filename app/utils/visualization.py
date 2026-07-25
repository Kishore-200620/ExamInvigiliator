"""
utils/visualization.py

Drawing utilities for ExamGuardian AI.
"""

import cv2

from config.constants import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    COLOR_BLUE,
    COLOR_WHITE,
)


def draw_bbox(frame, bbox, color=COLOR_GREEN, thickness=2):
    """
    Draw a bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_text(frame, text, position, color=COLOR_WHITE):
    """
    Draw text.
    """
    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_track_id(frame, track_id, bbox):
    """
    Draw tracking ID above bounding box.
    """
    x1, y1, _, _ = map(int, bbox)

    draw_text(
        frame,
        f"ID : {track_id}",
        (x1, y1 - 10),
        COLOR_BLUE,
    )


def draw_seat_id(frame, seat_id, position):
    """
    Draw seat ID.
    """
    draw_text(
        frame,
        f"Seat : {seat_id}",
        position,
        COLOR_GREEN,
    )


def draw_polygon(frame, points, color=COLOR_YELLOW, thickness=2):
    """
    Draw polygon.
    """
    cv2.polylines(
        frame,
        [points],
        True,
        color,
        thickness,
    )


def draw_circle(frame, point, color=COLOR_RED, radius=4):
    """
    Draw a point.
    """
    cv2.circle(
        frame,
        tuple(map(int, point)),
        radius,
        color,
        -1,
    )


def draw_line(frame, point1, point2, color=COLOR_BLUE, thickness=2):
    """
    Draw line.
    """
    cv2.line(
        frame,
        tuple(map(int, point1)),
        tuple(map(int, point2)),
        color,
        thickness,
    )


def draw_fps(frame, fps):
    """
    Draw FPS.
    """
    draw_text(
        frame,
        f"FPS : {fps:.1f}",
        (20, 30),
        COLOR_GREEN,
    )


def draw_risk(frame, score, level):
    """
    Draw current risk score.
    """
    draw_text(
        frame,
        f"Risk : {score} ({level})",
        (20, 60),
        COLOR_RED,
    )