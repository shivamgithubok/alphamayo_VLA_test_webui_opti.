# SPDX-License-Identifier: Apache-2.0
"""Lightweight GPU performance sampling for inference benchmark logs."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field

import torch


def _avg_peak(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return sum(values) / len(values), max(values)


def _fmt_pct(avg: float | None, peak: float | None) -> str:
    if avg is None or peak is None:
        return "n/a"
    return f"avg/peak={avg:.0f}/{peak:.0f}%"


def _fmt_gib(bytes_val: float | None) -> str:
    if bytes_val is None:
        return "n/a"
    return f"{bytes_val / (1024 ** 3):.2f} GiB"


@dataclass
class GPUSampleBatch:
    gpu_util: list[float] = field(default_factory=list)
    sm_activity: list[float] = field(default_factory=list)
    mem_controller: list[float] = field(default_factory=list)
    sm_occupancy: list[float] = field(default_factory=list)
    tensor_activity: list[float] = field(default_factory=list)
    hmma_activity: list[float] = field(default_factory=list)
    power_w: list[float] = field(default_factory=list)


class InferenceGPUMonitor:
    """Poll GPU counters while inference runs, then print a summary."""

    def __init__(self, poll_interval: float = 0.25):
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = GPUSampleBatch()
        self._gpm_supported: bool | None = None
        self._peak_allocated: int | None = None
        self._peak_reserved: int | None = None
        self._device_total: int | None = None

    def __enter__(self) -> InferenceGPUMonitor:
        if not torch.cuda.is_available():
            return self

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        self._device_total = props.total_memory

        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        if not torch.cuda.is_available():
            return

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

        torch.cuda.synchronize()
        self._peak_allocated = torch.cuda.max_memory_allocated()
        self._peak_reserved = torch.cuda.max_memory_reserved()

    def print_summary(self) -> None:
        if not torch.cuda.is_available():
            return
        self._print_summary()

    def _poll_loop(self) -> None:
        last_dmon = 0.0
        while not self._stop.is_set():
            self._poll_query()
            now = time.monotonic()
            if now - last_dmon >= 0.35:
                self._poll_dmon()
                last_dmon = now
            time.sleep(self.poll_interval)

    def _poll_query(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            if result.returncode != 0:
                return

            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) < 2:
                return

            gpu_util, mem_util = parts[0], parts[1]
            if gpu_util not in ("", "[N/A]"):
                self._samples.gpu_util.append(float(gpu_util))
            if mem_util not in ("", "[N/A]"):
                self._samples.mem_controller.append(float(mem_util))
            if len(parts) >= 3 and parts[2] not in ("", "[N/A]"):
                self._samples.power_w.append(float(parts[2]))
        except (subprocess.SubprocessError, ValueError, OSError):
            return

    def _poll_dmon(self) -> None:
        cmd = ["nvidia-smi", "dmon", "-s", "um", "-c", "1"]
        if self._gpm_supported is not False:
            cmd = [
                "nvidia-smi",
                "dmon",
                "-s",
                "um",
                "--gpm-metrics",
                "2,3,5,7",
                "-c",
                "1",
            ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode != 0:
                self._gpm_supported = False
                return

            for line in result.stdout.splitlines():
                if not line.startswith("    0"):
                    continue

                cols = line.split()
                if len(cols) < 3:
                    continue

                sm, mem = cols[1], cols[2]
                if sm not in ("-", ""):
                    self._samples.sm_activity.append(float(sm))
                if mem not in ("-", ""):
                    self._samples.mem_controller.append(float(mem))

                if len(cols) >= 14:
                    gpm_vals = cols[10:14]
                    if any(v != "-" for v in gpm_vals):
                        self._gpm_supported = True
                    if gpm_vals[0] != "-":
                        self._samples.sm_activity.append(float(gpm_vals[0]))
                    if gpm_vals[1] != "-":
                        self._samples.sm_occupancy.append(float(gpm_vals[1]))
                    if gpm_vals[2] != "-":
                        self._samples.tensor_activity.append(float(gpm_vals[2]))
                    if gpm_vals[3] != "-":
                        self._samples.hmma_activity.append(float(gpm_vals[3]))
                elif self._gpm_supported is None:
                    self._gpm_supported = False
                break
        except (subprocess.SubprocessError, ValueError, OSError):
            return

    def _print_summary(self) -> None:
        gpu_avg, gpu_peak = _avg_peak(self._samples.gpu_util)
        sm_avg, sm_peak = _avg_peak(self._samples.sm_activity)
        mem_avg, mem_peak = _avg_peak(self._samples.mem_controller)
        occ_avg, occ_peak = _avg_peak(self._samples.sm_occupancy)
        tensor_avg, tensor_peak = _avg_peak(self._samples.tensor_activity)
        hmma_avg, hmma_peak = _avg_peak(self._samples.hmma_activity)
        pwr_avg, pwr_peak = _avg_peak(self._samples.power_w)

        print(f"GPU utilization: {_fmt_pct(gpu_avg, gpu_peak)}")
        print(f"GPU memory: peak allocated={_fmt_gib(self._peak_allocated)}, "
              f"peak reserved={_fmt_gib(self._peak_reserved)}, "
              f"device total={_fmt_gib(self._device_total)}")

        if sm_avg is not None and sm_peak > 0:
            print(f"SM activity: {_fmt_pct(sm_avg, sm_peak)}")
        else:
            print("SM activity: n/a")

        if mem_avg is not None and mem_peak > 0:
            print(f"Memory controller utilization: {_fmt_pct(mem_avg, mem_peak)}")

        if occ_avg is not None:
            print(f"SM occupancy: {_fmt_pct(occ_avg, occ_peak)}")
        else:
            print("SM occupancy: n/a (GPM metrics unavailable on this GPU)")

        if hmma_avg is not None:
            print(f"Tensor Core activity (HMMA): {_fmt_pct(hmma_avg, hmma_peak)}")
        elif tensor_avg is not None:
            print(f"Tensor Core activity: {_fmt_pct(tensor_avg, tensor_peak)}")
        else:
            print("Tensor Core activity: n/a (GPM metrics unavailable on this GPU)")

        if pwr_avg is not None:
            print(f"GPU power: avg/peak={pwr_avg:.1f}/{pwr_peak:.1f} W")
