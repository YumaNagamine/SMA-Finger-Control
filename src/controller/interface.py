from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:
    from finger_MP import PWMGENERATOR, ctrlProcess
    DEFAULT_CHANNELS = PWMGENERATOR.CH_EVEN
except Exception:
    # Fallback for mock mode or when hardware libs are absent.
    DEFAULT_CHANNELS = [0, 2, 4, 6, 8, 10]
    ctrlProcess = None


class ActuatorBackend:
    """Hardware backend for PCA9685 over FTDI."""

    def __init__(self, ftdi_url: str | None, channels=None):
        self.channels = channels or DEFAULT_CHANNELS
        if ctrlProcess is None:
            raise RuntimeError("Hardware backend not available in this environment.")
        url = ftdi_url or os.environ.get("FTDI_DEVICE", "ftdi:///1")
        self.device = ctrlProcess(url, "ADC001")
        if not self.device:
            raise RuntimeError(f"Failed to connect actuator device via {url}")

    def set_duty(self, duty_list):
        for ch, duty in zip(self.channels, duty_list):
            self.device.setDutyRatioCH(ch, float(duty), relax=False)

    def stop(self):
        zeros = [0.0] * len(self.channels)
        self.set_duty(zeros)


class MockActuatorBackend:
    """Mock backend for testing without hardware."""

    def __init__(self, channels=None, log_path: str | None = None):
        self.channels = channels or DEFAULT_CHANNELS
        self.log_path = Path(log_path) if log_path else None
        self.log_path.parent.mkdir(parents=True, exist_ok=True) if self.log_path else None

    def set_duty(self, duty_list):
        line = f"{time.perf_counter():.6f} " + " ".join(f"{d:.3f}" for d in duty_list)
        print(f"[MOCK DUTY] {line}")
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def stop(self):
        self.set_duty([0.0] * len(self.channels))


class DutyLimiter:
    """Clamp, rate-limit, and watchdog for duty ratios."""

    def __init__(self, upper=1.0, lower=0.0, max_delta=0.1, watchdog_s=1.0):
        self.upper = upper
        self.lower = lower
        self.max_delta = max_delta
        self.watchdog_s = watchdog_s
        self.last_output = None
        self.last_command_time = None

    def apply(self, duty_list):
        now = time.perf_counter()
        if self.last_command_time is None:
            self.last_command_time = now
        dt = now - self.last_command_time
        self.last_command_time = now

        duty_list = [max(self.lower, min(self.upper, float(d))) for d in duty_list]
        if self.last_output is None:
            self.last_output = duty_list
            return duty_list

        limited = []
        max_step = self.max_delta * max(dt, 1e-3)
        for prev, new in zip(self.last_output, duty_list):
            delta = new - prev
            if delta > max_step:
                new = prev + max_step
            elif delta < -max_step:
                new = prev - max_step
            limited.append(new)

        self.last_output = limited
        return limited

    def timed_out(self):
        if self.watchdog_s <= 0:
            return False
        if self.last_command_time is None:
            return False
        return (time.perf_counter() - self.last_command_time) > self.watchdog_s


class DutyInterface:
    """High-level interface with safety features and mock support."""

    def __init__(
        self,
        backend,
        limiter: DutyLimiter,
        enforce_watchdog: bool = True,
    ):
        self.backend = backend
        self.limiter = limiter
        self.enforce_watchdog = enforce_watchdog

    def set(self, duty_list):
        if self.enforce_watchdog and self.limiter.timed_out():
            # Fail-safe to zeros if stale.
            duty_list = [0.0] * len(duty_list)
        duty_out = self.limiter.apply(duty_list)
        self.backend.set_duty(duty_out)
        return duty_out

    def stop(self):
        self.backend.stop()
        self.limiter.last_output = [0.0] * len(self.backend.channels)

    def run_sequence(self, sequence):
        """
        sequence: list of dicts [{"duration": 0.1, "duty": [1, 1, ...]}, ...]
        Timing uses perf_counter for ~0.1s granularity; OS scheduling still applies.
        """
        for step in sequence:
            duration = float(step["duration"])
            duty = step["duty"]
            start = time.perf_counter()
            self.set(duty)
            # Busy-wait + sleep hybrid for better precision
            while True:
                elapsed = time.perf_counter() - start
                remaining = duration - elapsed
                if remaining <= 0:
                    break
                if remaining > 0.01:
                    time.sleep(remaining - 0.005)
                else:
                    time.sleep(0)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_interface(config_path: str | Path, mock: bool = False, log_path: str | None = None) -> DutyInterface:
    cfg = load_config(config_path)
    hw_cfg = cfg.get("hardware", {})
    safety_cfg = cfg.get("safety", {})

    channels = hw_cfg.get("channels", DEFAULT_CHANNELS)
    if mock:
        backend = MockActuatorBackend(channels=channels, log_path=log_path)
    else:
        backend = ActuatorBackend(hw_cfg.get("ftdi_url"), channels=channels)

    limiter = DutyLimiter(
        upper=safety_cfg.get("upper", 1.0),
        lower=safety_cfg.get("lower", 0.0),
        max_delta=safety_cfg.get("max_delta_per_s", 1.0),
        watchdog_s=safety_cfg.get("watchdog_s", 1.0),
    )
    return DutyInterface(backend=backend, limiter=limiter, enforce_watchdog=safety_cfg.get("enable_watchdog", True))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Duty ratio interface with optional mock.")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("interface_config.json")))
    parser.add_argument("--mock", action="store_true", help="Use mock backend instead of hardware.")
    parser.add_argument(
        "--sequence",
        type=str,
        help='JSON string: e.g. \'[{"duration":0.1,"duty":[1,1,1,1]},{"duration":0.05,"duty":[0.05,0.05,0.05,0.05]}]\'',
    )
    parser.add_argument("--log", type=str, help="Path to log duty outputs (mock mode).")
    args = parser.parse_args()

    interface = build_interface(args.config, mock=args.mock, log_path=args.log)

    if args.sequence:
        seq = json.loads(args.sequence)
        interface.run_sequence(seq)
    else:
        print("No sequence provided; sending zeros.")
        interface.set([0.0] * len(interface.backend.channels))
        time.sleep(0.1)
        interface.stop()
