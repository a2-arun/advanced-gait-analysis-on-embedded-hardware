"""Live OpenCV preview shared by enrollment and identification."""

import os
import sys

import cv2

from src.pose.pose_extraction import GaitFeatureExtractor

GREEN, RED, ORANGE, WHITE = (0, 200, 0), (0, 0, 255), (0, 165, 255), (255, 255, 255)
_BONES = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26),
          (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32)]


def display_available() -> bool:
    return not sys.platform.startswith("linux") or bool(os.environ.get("DISPLAY"))


def _text(frame, text, org, scale, color, thickness):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_preview(frame, landmarks, window, info_lines, banner="", banner_color=GREEN):
    """Draw the gait skeleton, a tracking-status border, info_lines at the
    top-left and an optional banner at the bottom, then show it."""
    h, w = frame.shape[:2]
    tracking = landmarks is not None
    color = GREEN if tracking else RED

    if tracking:
        pts = {i: (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
               for i in GaitFeatureExtractor.GAIT_LANDMARKS}
        for a, b in _BONES:
            cv2.line(frame, pts[a], pts[b], GREEN, 2)
        for p in pts.values():
            cv2.circle(frame, p, 4, WHITE, -1)

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)
    lines = list(info_lines) + ["TRACKING" if tracking else "NO POSE - get full body in view"]
    for i, text in enumerate(lines):
        _text(frame, text, (12, 30 + i * 28), 0.7, color if i == len(lines) - 1 else WHITE, 2)

    if banner:
        _text(frame, banner, (12, h - 20), 1.0, banner_color, 2)

    cv2.imshow(window, frame)
