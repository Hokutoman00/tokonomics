"""The crossover claim must be falsifiable: construct two instances engineered
so the winner flips with the workload, and assert the detector finds it — and
construct a case where it does NOT flip, and assert it reports no crossover."""

import pytest

from tokonomics.schema import MachineResult, Ceilings, PriceEntry
from tokonomics.model import ModelSpec
from tokonomics.crossover import (
    crossover_grid, flip_margin, anchor_on_ceiling)

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


def test_flip_margin_reports_tightest_cell():
    # decodebox wins at r=1, prefillbox at r=128; somewhere between they are
    # close — flip_margin must return a non-negative fraction at a real ratio.
    machines = {
        "decodebox": _m("decodebox", off=300, on=400, bw=120),
        "prefillbox": _m("prefillbox", off=900, on=1800, bw=30),
    }
    prices = {"decodebox": _p("decodebox", 0.10),
              "prefillbox": _p("prefillbox", 0.10)}
    grid = crossover_grid(machines, prices, MODEL, RATIOS, i8mm="on")
    fm = flip_margin(grid)
    assert fm is not None
    assert fm["ratio"] in RATIOS
    assert 0.0 <= fm["margin_frac"] <= 1.0
    assert fm["winner"] != fm["runner_up"]
    # the tightest margin is no larger than any single cell's margin
    for i in range(len(RATIOS)):
        vals = sorted((grid["tokens_per_usd"][l][i] for l in grid["labels"]),
                      reverse=True)
        cell_margin = (vals[0] - vals[1]) / vals[0]
        assert fm["margin_frac"] <= cell_margin + 1e-12


def test_flip_survives_at_measured_uplift():
    # The load-bearing D1 claim, stated at the strength it actually holds: the
    # *existence* of the flip and its decode/prefill *endpoint winners* are
    # invariant to the i8mm uplift magnitude (decode→bandwidth-per-$, immune to
    # uplift; prefill→compute-per-$, scaled uniformly across i8mm machines). The
    # flip *location* may move — that is the disclosed input-sensitivity — so we
    # assert the structural invariant, NOT full per-ratio equality. (This test
    # was deliberately tightened after an earlier version over-claimed full
    # ordering invariance and this very assertion caught it.)
    machines = {
        "decodebox": _m("decodebox", off=300, on=600, bw=120),   # 2x i8mm
        "prefillbox": _m("prefillbox", off=900, on=1800, bw=30),  # 2x i8mm
    }
    prices = {"decodebox": _p("decodebox", 0.10),
              "prefillbox": _p("prefillbox", 0.10)}
    base = crossover_grid(machines, prices, MODEL, RATIOS, i8mm="on")
    anchored = crossover_grid(
        anchor_on_ceiling(machines, 1.19), prices, MODEL, RATIOS, i8mm="on")
    assert anchored["has_crossover"] == base["has_crossover"] is True
    assert anchored["winners"][0] == base["winners"][0]    # decode-heavy endpoint
    assert anchored["winners"][-1] == base["winners"][-1]  # prefill-heavy endpoint


def test_flip_ordering_invariant_on_shipped_grid():
    # On THIS repo's actual specs/pricing the invariant is even stronger: the
    # full per-ratio ordering is unchanged from the optimistic 2x down to the
    # measured 1.19x. Lock that in so a future spec edit that quietly breaks it
    # is caught. (Uses the real loaders, not synthetic boxes.)
    import pathlib
    from tokonomics.loaders import load_prices, load_spec_machines
    from tokonomics.model import MODELS
    root = pathlib.Path(__file__).resolve().parent.parent
    prices = load_prices(root / "econ" / "pricing.yaml")
    machines = load_spec_machines(root / "projection" / "specs.yaml")
    model = MODELS["llama-3.2-1b-q4_0"]   # the model cmd_project actually sweeps
    base = crossover_grid(machines, prices, model, RATIOS, i8mm="on")
    for uplift in (1.19, 1.29, 2.0):
        anchored = crossover_grid(
            anchor_on_ceiling(machines, uplift), prices, model, RATIOS, i8mm="on")
        assert anchored["winners"] == base["winners"], f"broke at uplift {uplift}"


def test_anchor_leaves_non_i8mm_untouched():
    # A machine with on == off (no i8mm) must be returned unchanged; an
    # i8mm-capable one must have on re-pinned to off * uplift.
    machines = {
        "noi8mm": _m("noi8mm", off=560, on=560, bw=44),
        "hasi8mm": _m("hasi8mm", off=680, on=1360, bw=46),
    }
    out = anchor_on_ceiling(machines, 1.19)
    assert out["noi8mm"].ceilings.peak_int8_gops_on == 560
    assert out["hasi8mm"].ceilings.peak_int8_gops_on == pytest.approx(680 * 1.19)
    assert out["hasi8mm"].ceilings.peak_int8_gops_off == 680
