from __future__ import annotations

import csv
from pathlib import Path
from queue import Empty

from controller.control_state import DutyBus, DutyBusSpec


def logger_worker(
    log_queue,
    stop_event,
    duty_bus_spec: DutyBusSpec,
    session_dir: Path,
    duty_channels: int,
    include_fps: bool,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    csv_path = session_dir / "duty_angle.csv"

    duty_bus = DutyBus.attach(duty_bus_spec)
    fieldnames = ["frame_id", "time"]
    fieldnames.extend([f"duty{i}" for i in range(duty_channels)])
    fieldnames.extend([f"angle{i}" for i in range(3)])
    if include_fps:
        fieldnames.append("fps")

    prev_t_frame = None
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while not stop_event.is_set() or not log_queue.empty():
            try:
                frame_id, t_frame, angle_0, angle_1, angle_2 = log_queue.get(timeout=0.1)
            except Empty:
                continue

            duty = duty_bus.get_duty_at(t_frame)
            row = {
                "frame_id": frame_id,
                "time": t_frame,
                "angle0": angle_0,
                "angle1": angle_1,
                "angle2": angle_2,
            }
            for idx in range(duty_channels):
                row[f"duty{idx}"] = float(duty[idx]) if idx < duty.size else 0.0

            if include_fps:
                if prev_t_frame is None:
                    fps = 0.0
                else:
                    dt = max(t_frame - prev_t_frame, 1e-9)
                    fps = 1.0 / dt
                row["fps"] = fps
                prev_t_frame = t_frame

            writer.writerow(row)

    duty_bus.close()
