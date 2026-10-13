"""Live OpenCV preview shared by enrollment and identification."""

import os
import sys

import cv2
import numpy as np

from src.pose.pose_extraction import GaitFeatureExtractor

GREEN, RED, ORANGE, WHITE = (0, 200, 0), (0, 0, 255), (0, 165, 255), (255, 255, 255)
_BONES = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26),
          (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32)]


_fullscreen = False
_fullscreen_windows = set()


def display_available() -> bool:
    return not sys.platform.startswith("linux") or bool(os.environ.get("DISPLAY"))


def set_fullscreen(enabled: bool) -> None:
    """Make every preview window fill the screen (e.g. the chassis LCD)."""
    global _fullscreen
    _fullscreen = enabled


def _show(window, frame):
    if _fullscreen and window not in _fullscreen_windows:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        _fullscreen_windows.add(window)
    cv2.imshow(window, frame)


def _text(frame, text, org, scale, color, thickness):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = {i: (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
           for i in GaitFeatureExtractor.GAIT_LANDMARKS}
    for a, b in _BONES:
        cv2.line(frame, pts[a], pts[b], GREEN, 2)
    for p in pts.values():
        cv2.circle(frame, p, 4, WHITE, -1)


def draw_preview(frame, landmarks, window, info_lines, banner="", banner_color=GREEN):
    """Draw the gait skeleton, a tracking-status border, info_lines at the
    top-left and an optional banner at the bottom, then show it."""
    h, w = frame.shape[:2]
    tracking = landmarks is not None
    color = GREEN if tracking else RED

    if tracking:
        _draw_skeleton(frame, landmarks)

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)
    lines = list(info_lines) + ["TRACKING" if tracking else "NO POSE - get full body in view"]
    for i, text in enumerate(lines):
        _text(frame, text, (12, 30 + i * 28), 0.7, color if i == len(lines) - 1 else WHITE, 2)

    if banner:
        _text(frame, banner, (12, h - 20), 1.0, banner_color, 2)

    _show(window, frame)


def show_enrolled_skeleton(pose_sequence, title, info_lines, size=(640, 480), fps=12):
    """Loop the captured skeleton on a plain background until q/Esc is
    pressed. pose_sequence is the (T, 33, 3) raw landmark array from
    CapturedSequence; fps is deliberately below camera rate so the short
    walk pass reads as a slow-motion replay."""
    w, h = size
    window = "Enrolled gait skeleton - press q to close"
    delay_ms = int(1000 / fps)
    while True:
        for landmarks in pose_sequence:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            _draw_skeleton(canvas, landmarks)
            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), GREEN, 6)
            for i, text in enumerate(info_lines):
                _text(canvas, text, (12, 30 + i * 28), 0.7, WHITE, 2)
            _text(canvas, title, (12, h - 20), 1.0, GREEN, 2)
            _show(window, canvas)
            if cv2.waitKey(delay_ms) & 0xFF in (ord("q"), 27):
                cv2.destroyWindow(window)
                return
