"""Render the prefill/decode roofline for one machine (i8mm on vs off).

The figure makes the headline visible at a glance: enabling i8mm raises the
horizontal compute ceiling (so the prefill point climbs), while the decode
point stays pinned on the diagonal memory ceiling and barely moves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import MachineResult
from .model import ModelSpec


def plot_roofline(
    machine: MachineResult,
    model: ModelSpec,
    out_path: str | Path,
) -> dict:
    """Write figures/roofline*.png and return the plotted numbers (for tests)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = machine.ceilings
    peak_on = c.peak_int8_gops_on
    peak_off = c.peak_int8_gops_off
    bw = c.mem_bw_gbs

    ai = np.logspace(-1, 3, 400)              # arithmetic intensity sweep (OP/byte)
    mem_line = bw * ai                        # diagonal memory ceiling

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ai, np.minimum(mem_line, peak_off), color="#888",
              lw=2, label=f"i8mm OFF ceiling ({peak_off:.0f} GOP/s)")
    ax.loglog(ai, np.minimum(mem_line, peak_on), color="#c2410c",
              lw=2.4, label=f"i8mm ON ceiling ({peak_on:.0f} GOP/s)")

    # Headroom above the ON ceiling so the prefill marker + its label sit inside
    # the frame instead of colliding with the title.
    ax.set_ylim(top=peak_on * 2.2)

    # decode operating point: low AI -> on the memory diagonal (i8mm-immune).
    # Label sits to the lower-right, clear of the ceiling lines.
    decode_ai = model.decode_ai
    decode_y = bw * decode_ai
    ax.scatter([decode_ai], [decode_y], color="#0f172a", zorder=5, s=70)
    ax.annotate("decode\n(memory-bound,\ni8mm-immune)", (decode_ai, decode_y),
                textcoords="offset points", xytext=(12, -34),
                ha="left", va="top", fontsize=9)

    # prefill operating point: high AI -> on the compute ceiling (i8mm lifts it).
    # Label goes below-left of the ON marker so it clears both the title and the
    # two ceiling lines.
    prefill_ai = 256.0  # representative compute-bound regime for plotting
    ax.scatter([prefill_ai], [peak_off], color="#888", zorder=5, s=55)
    ax.scatter([prefill_ai], [peak_on], color="#c2410c", zorder=5, s=70)
    ax.annotate("prefill\n(compute-bound,\ni8mm lifts it)", (prefill_ai, peak_on),
                textcoords="offset points", xytext=(-46, -52),
                ha="center", va="top", fontsize=9)

    ax.set_xlabel("Arithmetic intensity (OP / byte)")
    ax.set_ylabel("Attainable throughput (GOP/s)")
    tag = {"measured": "MEASURED", "dev": "x86 DEV PROXY",
           "projection": "PROJECTION (specs)"}[machine.kind]
    ax.set_title(f"Tokonomics roofline — {machine.arch}\n[{tag}]", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)

    return {
        "peak_on": peak_on, "peak_off": peak_off, "mem_bw_gbs": bw,
        "decode_ai": decode_ai, "decode_y": decode_y,
        "prefill_on": peak_on, "prefill_off": peak_off,
    }
