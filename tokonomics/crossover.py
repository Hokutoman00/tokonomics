"""The headline finding: the cheapest instance changes with the workload.

Across a grid of (instance generation x prompt:gen ratio) we compute
workload-weighted tokens/$ and report which instance wins each cell. When the
winner flips as the ratio changes, that flip is the crossover — proof that
"newest == cheapest" is false in general.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import MachineResult, PriceEntry
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
