from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from utils.config_loader import load_config


DEFAULT_CONFIG = Path(__file__).with_name("camera_config.json")


def _resolve_backend(backend_name: str | None) -> int | None:
    if not backend_name:
        return None
    return {
        "CAP_DSHOW": cv2.CAP_DSHOW,
        "CAP_ANY": cv2.CAP_ANY,
    }.get(backend_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check camera availability.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camera_cfg = load_config(args.config)

    cam_num = int(camera_cfg.get("index", 0))
    backend = _resolve_backend(camera_cfg.get("backend"))
    cap = cv2.VideoCapture(cam_num, backend) if backend is not None else cv2.VideoCapture(cam_num)
    if not cap.isOpened():
        print(f"Camera index {cam_num} not available.")
        return 1

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"Camera index {cam_num} opened but no frames returned.")
        return 1

    height, width = frame.shape[:2]
    print(f"Camera index {cam_num} available: {width}x{height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
