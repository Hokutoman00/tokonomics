"""Local x86 dev measurement — validate the *pipeline* with real numbers.

We can't run Arm int8 kernels on a developer x86 box, but we CAN measure that
box's real compute ceiling (float GFLOP/s via BLAS) and real memory ceiling
(STREAM-triad GB/s via numpy), then push those through the exact same
roofline/economics path the Arm harness uses. This proves the instrument
works on genuine measurements before any silicon is rented.

The result is tagged kind="dev": it is an x86 FLOAT proxy, NOT an Arm int8
number. i8mm on/off are set equal (x86 has no i8mm) — the ablation itself is
an Arm-only, CI-measured quantity.
"""

from __future__ import annotations

import time

import numpy as np

from .schema import MachineResult, Ceilings


def measure_compute_gflops(n: int = 1024, reps: int = 5) -> float:
    """Peak float32 GFLOP/s from a dense matmul (BLAS-backed)."""
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    np.dot(a, b)  # warm up / let BLAS spin up threads
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        np.dot(a, b)
        best = min(best, time.perf_counter() - t0)
    flops = 2.0 * n ** 3
    return flops / best / 1e9


def measure_bandwidth_gbs(n: int = 8_000_000, reps: int = 5) -> float:
    """Effective memory bandwidth (GB/s) from a STREAM-style triad."""
    b = np.random.rand(n).astype(np.float64)
    c = np.random.rand(n).astype(np.float64)
    s = 3.0
    a = b + s * c  # warm up
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        a = b + s * c
        best = min(best, time.perf_counter() - t0)
    # bytes moved: read b, read c, write a  => 3 arrays * 8 bytes
    bytes_moved = 3.0 * n * 8.0
    _ = a[0]  # keep a alive
    return bytes_moved / best / 1e9


def measure_local(label: str = "x86-dev") -> MachineResult:
    gflops = measure_compute_gflops()
    bw = measure_bandwidth_gbs()
    return MachineResult(
        label=label,
        kind="dev",
        arch="local x86 (float proxy — validates pipeline, not Arm int8)",
        ceilings=Ceilings(
            peak_int8_gops_off=gflops,   # float GFLOP/s used as a stand-in
            peak_int8_gops_on=gflops,    # no i8mm on x86 -> on == off
            mem_bw_gbs=bw,
        ),
        notes="measured locally via numpy BLAS matmul + STREAM triad",
    )
