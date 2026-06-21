"""The crossover claim must be falsifiable: construct two instances engineered
so the winner flips with the workload, and assert the detector finds it — and
construct a case where it does NOT flip, and assert it reports no crossover."""

import pytest

from tokonomics.schema import MachineResult, Ceilings, PriceEntry
from tokonomics.model import ModelSpec
from tokonomics.crossover import crossover_grid

MODEL = ModelSpec("toy", params=1e9, bytes_per_param=0.5)
RATIOS = [1, 2, 4, 8, 16, 32, 64, 128]


def _m(label, off, on, bw):
    return MachineResult(label, "projection", "t", Ceilings(off, on, bw))


def _p(label, usd, watt=40.0):
    return PriceEntry(label, "t", 8, usd, watt, "http://x", "2026-06-20")


def test_winner_flips_with_workload():
    # "decode box": great bandwidth/$, weak compute.
    # "prefill box": great compute/$, weak bandwidth.
    machines = {
        "decodebox": _m("decodebox", off=300, on=400, bw=120),
        "prefillbox": _m("prefillbox", off=900, on=1800, bw=30),
    }
    prices = {"decodebox": _p("decodebox", 0.10),
              "prefillbox": _p("prefillbox", 0.10)}
    grid = crossover_grid(machines, prices, MODEL, RATIOS, i8mm="on")
    assert grid["has_crossover"] is True
    # decode-heavy (r=1) should favour the bandwidth box; prefill-heavy the compute box
    assert grid["winners"][0] == "decodebox"
    assert grid["winners"][-1] == "prefillbox"


def test_no_crossover_when_one_dominates():
    machines = {
        "good": _m("good", off=900, on=1800, bw=120),
        "bad": _m("bad", off=300, on=400, bw=30),
    }
    prices = {"good": _p("good", 0.10), "bad": _p("bad", 0.10)}
    grid = crossover_grid(machines, prices, MODEL, RATIOS, i8mm="on")
    assert grid["has_crossover"] is False
    assert set(grid["winners"]) == {"good"}


def test_grid_shapes_match():
    machines = {"a": _m("a", 300, 400, 120), "b": _m("b", 900, 1800, 30)}
    prices = {"a": _p("a", 0.1), "b": _p("b", 0.1)}
    grid = crossover_grid(machines, prices, MODEL, RATIOS, i8mm="on")
    assert len(grid["winners"]) == len(RATIOS)
    for lab in machines:
        assert len(grid["tokens_per_usd"][lab]) == len(RATIOS)
