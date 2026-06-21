"""The headline finding: the cheapest instance changes with the workload.

Across a grid of (instance generation x prompt:gen ratio) we compute
workload-weighted tokens/$ and report which instance wins each cell. When the
winner flips as the ratio changes, that flip is the crossover — proof that
"newest == cheapest" is false in general.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .schema import MachineResult, PriceEntry, Ceilings
from .model import ModelSpec
from .economics import blended_tokens_per_usd


def crossover_grid(
    machines: dict[str, MachineResult],
    prices: dict[str, PriceEntry],
    model: ModelSpec,
    ratios: list[float],
    i8mm: str = "on",
) -> dict:
    """Return per-instance tokens/$ over the ratio sweep + the winning label per ratio."""
    labels = list(machines.keys())
    table = {lab: [] for lab in labels}
    winners = []
    for r in ratios:
        best_lab, best_val = None, -1.0
        for lab in labels:
            v = blended_tokens_per_usd(machines[lab], prices[lab], model, r, i8mm)
            table[lab].append(v)
            if v > best_val:
                best_lab, best_val = lab, v
        winners.append(best_lab)
    has_crossover = len(set(winners)) > 1
    return {
        "ratios": ratios,
        "labels": labels,
        "tokens_per_usd": table,
        "winners": winners,
        "has_crossover": has_crossover,
    }


def flip_margin(grid: dict) -> dict | None:
    """The tightest decision margin in the sweep: at each ratio, how far ahead
    (relative %) the winner is over the runner-up, reported for the closest cell.

    A *small* margin means the crossover *location* (which ratio it flips at) is
    input-sensitive — a modest change to a price or bandwidth can move it. That
    is a separate, weaker claim from the crossover *ordering* (decode→bandwidth-
    per-$, prefill→compute-per-$), which is structural and does not depend on the
    margin. Surfacing this number is the honest disclosure that the flip *ratio*
    is fragile even though the flip *exists*.
    """
    ratios, labels, tpu = grid["ratios"], grid["labels"], grid["tokens_per_usd"]
    if len(labels) < 2:
        return None
    tightest = None
    for i in range(len(ratios)):
        ranked = sorted(((tpu[lab][i], lab) for lab in labels), reverse=True)
        win_v, win_l = ranked[0]
        run_v, run_l = ranked[1]
        margin = (win_v - run_v) / win_v if win_v else 0.0
        cell = {"ratio": ratios[i], "winner": win_l, "runner_up": run_l,
                "margin_frac": margin}
        if tightest is None or margin < tightest["margin_frac"]:
            tightest = cell
    return tightest


def anchor_on_ceiling(
    machines: dict[str, MachineResult], uplift: float
) -> dict[str, MachineResult]:
    """Return a copy of `machines` with each i8mm-capable on-ceiling pinned to
    off × `uplift`, to test the crossover at a *measured* i8mm magnitude rather
    than the published ~2x in specs.yaml.

    The decode-heavy endpoint winner is set by bandwidth-per-$ (i8mm-independent)
    and the prefill-heavy endpoint winner by off-compute-per-$ scaled by the
    *same* uplift for every i8mm machine — so the *existence* of the flip and its
    two endpoint winners are invariant to `uplift` for any uplift > 1. The flip
    *location* (which ratio it lands at) can still move, because the blended
    tokens/$ at an intermediate ratio mixes the uplift-scaled prefill term with
    the uplift-free decode term — that is the input-sensitivity flip_margin
    discloses, not a defect. Rebuilding the grid at the repo's measured N2 uplift
    thus shows the flip survives at measured magnitudes with no fabricated
    cross-instance measurement. Machines with no i8mm (on == off) are untouched.
    """
    out: dict[str, MachineResult] = {}
    for lab, m in machines.items():
        c = m.ceilings
        if c.peak_int8_gops_on > c.peak_int8_gops_off:  # i8mm-capable
            new_on = c.peak_int8_gops_off * uplift
            out[lab] = replace(m, ceilings=Ceilings(
                c.peak_int8_gops_off, new_on, c.mem_bw_gbs))
        else:
            out[lab] = m
    return out


def plot_crossover(grid: dict, out_path: str | Path, kind: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios = grid["ratios"]
    fig, ax = plt.subplots(figsize=(7.6, 5))
    palette = ["#c2410c", "#0f766e", "#7c3aed", "#888", "#b91c1c"]
    for i, lab in enumerate(grid["labels"]):
        ax.plot(ratios, grid["tokens_per_usd"][lab], marker="o", ms=3,
                color=palette[i % len(palette)], label=lab)

    # mark the crossover region (where the winner changes). Anchor the label in
    # axes-fraction y so it stays inside the frame regardless of the data range.
    winners = grid["winners"]
    for i in range(1, len(winners)):
        if winners[i] != winners[i - 1]:
            xc = (ratios[i] + ratios[i - 1]) / 2
            ax.axvline(xc, color="#111", ls="--", alpha=0.6)
            ax.annotate("crossover", xy=(xc, 0.97),
                        xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(5, -2),
                        ha="left", va="top", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Workload prompt : generated  token ratio")
    ax.set_ylabel("Workload tokens per USD")
    tag = {"measured": "MEASURED", "dev": "x86 DEV PROXY",
           "projection": "PROJECTION (specs)"}[kind]
    note = "newest is NOT always cheapest" if grid["has_crossover"] else "no crossover in range"
    # Two lines + smaller font so the full title never clips at the right edge.
    ax.set_title(f"Tokonomics tokens/$ crossover — {note}\n[{tag}]",
                 fontsize=11)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
