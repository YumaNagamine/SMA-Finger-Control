from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import multiprocessing as mp

# Ensure repository src is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from controller.control_state import DutyBus
from runtime.capture_worker import capture_worker
from runtime.control_worker import control_worker
from runtime.logger_worker import logger_worker
from runtime.process_worker import process_worker
from utils.config_loader import load_config


DEFAULT_CONFIG = Path(__file__).with_name("pipeline_config.json")
DEFAULT_CAMERA_CONFIG = ROOT / "src" / "observation" / "camera" / "camera_config.json"
DEFAULT_ANGLE_CONFIG = ROOT / "src" / "observation" / "vision" / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime capture/process/log pipeline.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def _session_dir(base_dir: str | Path) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / timestamp


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    runtime_cfg = cfg.get("runtime", {})
    camera_cfg_path = Path(cfg.get("camera", {}).get("config_path", DEFAULT_CAMERA_CONFIG))
    angle_cfg_path = Path(cfg.get("angle", {}).get("config_path", DEFAULT_ANGLE_CONFIG))
    enable_capture_video = bool(cfg.get("camera", {}).get("enable_capture_video", True))

    duty_channels = int(runtime_cfg.get("duty_channels", 7))
    duty_buffer_len = int(runtime_cfg.get("duty_buffer_len", 180))
    frame_queue_len = int(runtime_cfg.get("frame_queue_len", 180))
    log_queue_len = int(runtime_cfg.get("log_queue_len", 360))
    log_dir = runtime_cfg.get("log_dir", "log/sessions")
    include_fps = bool(runtime_cfg.get("include_fps", False))

    camera_cfg = load_config(camera_cfg_path)
    angle_cfg = load_config(angle_cfg_path, required_keys=("markers",))

    session_dir = _session_dir(log_dir)

    duty_bus = DutyBus.create(channels=duty_channels, buffer_len=duty_buffer_len)
    duty_spec = duty_bus.to_spec()
    stop_event = mp.Event()
    frame_queue = mp.Queue(maxsize=frame_queue_len)
    log_queue = mp.Queue(maxsize=log_queue_len)
    manager = mp.Manager()
    shared = manager.dict()
    shared["photo"] = None
    shared["photo_acquired_t"] = 0.0
    shared["prev_photo_acquired_t"] = 0.0
    shared["exit"] = False

    proc_logger = mp.Process(
        target=logger_worker,
        name="logger",
        args=(log_queue, stop_event, duty_spec, session_dir, duty_channels, include_fps),
    )
    proc_process = mp.Process(
        target=process_worker,
        name="process",
        args=(frame_queue, log_queue, stop_event, angle_cfg),
    )
    proc_capture = mp.Process(
        target=capture_worker,
        name="capture",
        args=(frame_queue, stop_event, camera_cfg, session_dir, enable_capture_video, shared),
    )
    proc_control = mp.Process(
        target=control_worker,
        name="control",
        args=(stop_event, duty_spec, duty_channels, shared),
    )

    proc_logger.start()
    proc_process.start()
    proc_capture.start()
    proc_control.start()

    try:
        while True:
            time.sleep(0.5)
            if not any(p.is_alive() for p in (proc_logger, proc_process, proc_capture, proc_control)):
                break
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for proc in (proc_control, proc_capture, proc_process, proc_logger):
            proc.join(timeout=5)

        frame_queue.close()
        log_queue.close()
        duty_bus.close()
        manager.shutdown()


if __name__ == "__main__":
    main()
