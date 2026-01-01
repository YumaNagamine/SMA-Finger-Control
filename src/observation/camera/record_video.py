from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import cv2

from utils.config_loader import load_config


TEST_VIDEO_DIR = Path(__file__).with_name("test_video")
DEFAULT_CONFIG = Path(__file__).with_name("camera_config.json")


def _resolve_backend(backend_name: str | None) -> int | None:
    if not backend_name:
        return None
    return {
        "CAP_DSHOW": cv2.CAP_DSHOW,
        "CAP_ANY": cv2.CAP_ANY,
    }.get(backend_name)


def _fourcc_from_str(code: str | None) -> int | None:
    if not code or len(code) != 4:
        return None
    return cv2.VideoWriter_fourcc(*code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a test video from camera_config.json.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--duration-s", type=float, default=5.0, help="Recording duration in seconds.")
    parser.add_argument("--frame-limit", type=int, default=0, help="Stop after N frames (0 = ignore).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_cfg = load_config(args.config)

    cam_num = int(camera_cfg.get("index", 0))
    backend = _resolve_backend(camera_cfg.get("backend"))
    cap = cv2.VideoCapture(cam_num, backend) if backend is not None else cv2.VideoCapture(cam_num)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    width = int(camera_cfg.get("width", 1600))
    height = int(camera_cfg.get("height", 1200))
    target_fps = float(camera_cfg.get("target_fps", 90))
    buffersize = int(camera_cfg.get("buffersize", 0))
    auto_exposure = float(camera_cfg.get("auto_exposure", 1))
    gain = float(camera_cfg.get("gain", 0))
    exposure = float(camera_cfg.get("exposure", -11))
    capture_fourcc = _fourcc_from_str(camera_cfg.get("capture_fourcc", "MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
    cap.set(cv2.CAP_PROP_GAIN, gain)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if capture_fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, capture_fourcc)

    TEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = TEST_VIDEO_DIR / f"test_{timestamp}.mp4"
    writer_fourcc = _fourcc_from_str(camera_cfg.get("writer_fourcc", "mp4v"))
    writer = cv2.VideoWriter(str(filename), writer_fourcc, target_fps, (width, height))

    start = time.time()
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            writer.write(frame)
            frame_count += 1
            if args.frame_limit and frame_count >= args.frame_limit:
                break
            if args.duration_s > 0 and (time.time() - start) >= args.duration_s:
                break
    finally:
        cap.release()
        writer.release()

    print(f"Saved test video to {filename}")


if __name__ == "__main__":
    main()
