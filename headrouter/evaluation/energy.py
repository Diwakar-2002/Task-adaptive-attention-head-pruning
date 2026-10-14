"""GPU energy measurement via NVML (total-energy counter, or power sampling fallback)."""

import threading
import time
from typing import Optional


class EnergyMeter:
    def __init__(self, device: str = "cuda", index: int = 0):
        self.mode = "unavailable"
        self._h = None
        if not device.startswith("cuda"):
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h)
                self.mode = "counter"
            except pynvml.NVMLError:
                pynvml.nvmlDeviceGetPowerUsage(self._h)
                self.mode = "sampling"
        except Exception:
            self.mode = "unavailable"

    @property
    def available(self) -> bool:
        return self.mode != "unavailable"

    def start(self):
        if self.mode == "counter":
            self._e0 = self._nv.nvmlDeviceGetTotalEnergyConsumption(self._h)  # mJ
        elif self.mode == "sampling":
            self._joules = 0.0
            self._run = True
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()

    def _sample(self):
        last = time.perf_counter()
        while self._run:
            time.sleep(0.005)
            now = time.perf_counter()
            self._joules += self._nv.nvmlDeviceGetPowerUsage(self._h) / 1000.0 * (now - last)
            last = now

    def stop(self) -> Optional[float]:
        """Joules consumed by the whole GPU since start()."""
        if self.mode == "counter":
            return (self._nv.nvmlDeviceGetTotalEnergyConsumption(self._h) - self._e0) / 1000.0
        if self.mode == "sampling":
            self._run = False
            self._thread.join()
            return self._joules
        return None

    def power_w(self) -> Optional[float]:
        if self._h is None:
            return None
        return self._nv.nvmlDeviceGetPowerUsage(self._h) / 1000.0
