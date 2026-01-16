from __future__ import annotations

import time
from pathlib import Path
from queue import Full

import cv2
from PIL import Image


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


def _put_drop_oldest(queue, item) -> None:
    try:
        queue.put_nowait(item)
    except Full:
        try:
            queue.get_nowait()
        except Exception:
            pass
        try:
            queue.put_nowait(item)
        except Full:
            pass


def capture_worker(
    frame_queue,
    stop_event,
    camera_cfg: dict,
    session_dir: Path,
    enable_capture_video: bool,
    shared=None,
) -> None:
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

    writer = None
    if enable_capture_video:
        session_dir.mkdir(parents=True, exist_ok=True)
        output_path = session_dir / "raw_capture.mp4"
        writer_fourcc = _fourcc_from_str(camera_cfg.get("writer_fourcc", "mp4v"))
        writer = cv2.VideoWriter(str(output_path), writer_fourcc, target_fps, (width, height))

    frame_id = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            t_frame = time.perf_counter()
            if shared is not None:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                shared["photo"] = Image.fromarray(image)
                shared["photo_acquired_t"] = time.time()
            if writer is not None:
                writer.write(frame)
            _put_drop_oldest(frame_queue, (frame_id, t_frame, frame))
            frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
