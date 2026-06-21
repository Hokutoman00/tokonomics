"""Merge the two ablation-variant driver JSONs into one measured machine.

The C driver runs twice (bench_off, bench_on); each emits *its own* peak into
either peak_int8_gops_off or peak_int8_gops_on, plus its own mem_bw_gbs. This
merges them into a single MachineResult(kind="measured") so the same economics
and roofline path used for projection runs on real silicon numbers.

mem_bw is taken from the OFF run (bandwidth is i8mm-independent by construction;
we assert the two agree within tolerance to catch a broken run).
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import MachineResult, Ceilings


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def merge_measured(off_json: Path, on_json: Path) -> MachineResult:
    off = _read(off_json)
    on = _read(on_json)
    if not off.get("correct", False) or not on.get("correct", False):
        raise ValueError("merge_measured: a variant reported correct=false — "
                         "ablation results are not bit-identical, refusing to merge")

    peak_off = float(off["peak_int8_gops_off"])
    peak_on = float(on["peak_int8_gops_on"])
    bw_off = float(off["mem_bw_gbs"])
    bw_on = float(on["mem_bw_gbs"])
    # bandwidth must not depend on the compute ISA; flag a >15% disagreement.
    if abs(bw_off - bw_on) / max(bw_off, bw_on) > 0.15:
        raise ValueError(f"merge_measured: mem_bw disagrees off={bw_off} on={bw_on} "
                         "(>15%) — suspect a noisy run")

    label = off.get("label", "cobalt100-n2")
    return MachineResult(
        label=label,
        kind="measured",
        arch=off.get("arch", "Neoverse N2 (ubuntu-24.04-arm)"),
        ceilings=Ceilings(
            peak_int8_gops_off=peak_off,
            peak_int8_gops_on=peak_on,
            mem_bw_gbs=bw_off,
        ),
        notes=f"measured in CI: off={off.get('kernel_path')} on={on.get('kernel_path')}",
    )
