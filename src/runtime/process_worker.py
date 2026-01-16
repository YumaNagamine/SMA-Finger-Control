from __future__ import annotations

import math
from queue import Empty

import cv2

from observation.vision.angle_processor import AngleProcessor


def process_worker(frame_queue, log_queue, stop_event, angle_cfg: dict) -> None:
    processor = AngleProcessor(angle_cfg)
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame_id, t_frame, frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue

        try:
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            raw_markers = processor.detect_markers(lab_frame)
            modified = processor.modify_markers(raw_markers)
            angles = processor.calculate_angles(modified)
            angle_0, angle_1, angle_2 = angles
        except Exception:
            angle_0 = math.nan
            angle_1 = math.nan
            angle_2 = math.nan

        log_queue.put((frame_id, t_frame, angle_0, angle_1, angle_2))
