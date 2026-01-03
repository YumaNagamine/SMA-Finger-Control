from __future__ import annotations

from pathlib import Path

from controller.control_state import DutyBus, DutyBusSpec
from controller.controller_GUI import gui_process


class DutyBusInterface:
    """Adapter to feed GUI duty values into DutyBus."""

    def __init__(self, duty_bus: DutyBus, duty_channels: int):
        self.duty_bus = duty_bus
        self.duty_channels = duty_channels

    def set(self, duty_list):
        duty = [float(v) for v in duty_list[: self.duty_channels]]
        if len(duty) < self.duty_channels:
            duty.extend([0.0] * (self.duty_channels - len(duty)))
        self.duty_bus.set_duty(duty)
        return duty


def control_worker(stop_event, duty_bus_spec: DutyBusSpec, duty_channels: int, shared=None) -> None:
    duty_bus = DutyBus.attach(duty_bus_spec)
    if shared is None:
        shared = {"photo": None, "photo_acquired_t": 0.0, "prev_photo_acquired_t": 0.0, "exit": False}
    else:
        shared.setdefault("photo", None)
        shared.setdefault("photo_acquired_t", 0.0)
        shared.setdefault("prev_photo_acquired_t", 0.0)
        shared.setdefault("exit", False)

    def _builder(_config_path, mock=False, log_path=None):
        return DutyBusInterface(duty_bus, duty_channels)

    def _on_exit():
        stop_event.set()

    try:
        config_path = Path(__file__).resolve().parents[1] / "controller" / "interface_config.json"
        gui_process(
            shared,
            config_path=config_path,
            mock=True,
            interface_builder=_builder,
            on_exit_callback=_on_exit,
            stop_event=stop_event,
        )
    finally:
        duty_bus.close()
