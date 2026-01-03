from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import Lock, Value, shared_memory

import numpy as np


@dataclass(frozen=True)
class DutyBusSpec:
    times_name: str
    duties_name: str
    channels: int
    buffer_len: int
    write_idx: Value
    lock: Lock


class DutyBus:
    """SharedMemory ring buffer for duty ratios synchronized by perf_counter."""

    def __init__(
        self,
        times_shm: shared_memory.SharedMemory,
        duties_shm: shared_memory.SharedMemory,
        write_idx: Value,
        channels: int,
        buffer_len: int,
        lock: Lock | None,
        owner: bool,
    ):
        self.times_shm = times_shm
        self.duties_shm = duties_shm
        self.write_idx = write_idx
        self.channels = channels
        self.buffer_len = buffer_len
        self.lock = lock
        self.owner = owner

        self.times = np.ndarray((buffer_len,), dtype=np.float64, buffer=self.times_shm.buf)
        self.duties = np.ndarray((buffer_len, channels), dtype=np.float32, buffer=self.duties_shm.buf)

    @classmethod
    def create(cls, channels: int, buffer_len: int) -> "DutyBus":
        times_size = buffer_len * np.dtype(np.float64).itemsize
        duties_size = buffer_len * channels * np.dtype(np.float32).itemsize
        times_shm = shared_memory.SharedMemory(create=True, size=times_size)
        duties_shm = shared_memory.SharedMemory(create=True, size=duties_size)
        times = np.ndarray((buffer_len,), dtype=np.float64, buffer=times_shm.buf)
        duties = np.ndarray((buffer_len, channels), dtype=np.float32, buffer=duties_shm.buf)
        times.fill(0.0)
        duties.fill(0.0)
        write_idx = Value("i", 0)
        lock = Lock()
        return cls(times_shm, duties_shm, write_idx, channels, buffer_len, lock, owner=True)

    @classmethod
    def attach(cls, spec: DutyBusSpec) -> "DutyBus":
        times_shm = shared_memory.SharedMemory(name=spec.times_name)
        duties_shm = shared_memory.SharedMemory(name=spec.duties_name)
        return cls(
            times_shm=times_shm,
            duties_shm=duties_shm,
            write_idx=spec.write_idx,
            channels=spec.channels,
            buffer_len=spec.buffer_len,
            lock=spec.lock,
            owner=False,
        )

    def to_spec(self) -> DutyBusSpec:
        return DutyBusSpec(
            times_name=self.times_shm.name,
            duties_name=self.duties_shm.name,
            channels=self.channels,
            buffer_len=self.buffer_len,
            write_idx=self.write_idx,
            lock=self.lock,
        )

    def set_duty(self, duty: list[float]) -> None:
        duty_array = np.asarray(duty, dtype=np.float32)
        if duty_array.size != self.channels:
            duty_array = np.resize(duty_array, self.channels)
        t_now = time.perf_counter()
        if self.lock:
            with self.lock:
                idx = int(self.write_idx.value) % self.buffer_len
                self.times[idx] = t_now
                self.duties[idx, :] = duty_array
                self.write_idx.value += 1
        else:
            idx = int(self.write_idx.value) % self.buffer_len
            self.times[idx] = t_now
            self.duties[idx, :] = duty_array
            self.write_idx.value += 1

    def get_duty_at(self, t_frame: float) -> np.ndarray:
        latest_idx = int(self.write_idx.value) - 1
        if latest_idx < 0:
            return np.zeros(self.channels, dtype=np.float32)

        for offset in range(self.buffer_len):
            idx = (latest_idx - offset) % self.buffer_len
            if self.times[idx] <= t_frame and self.times[idx] > 0:
                return np.array(self.duties[idx, :], dtype=np.float32)

        return np.zeros(self.channels, dtype=np.float32)

    def close(self) -> None:
        self.times_shm.close()
        self.duties_shm.close()
        if self.owner:
            self.times_shm.unlink()
            self.duties_shm.unlink()
