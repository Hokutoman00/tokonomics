"""The measured-ingest firewall: merge_measured() turns the two CI driver JSONs
into one measured machine, and is the gate that refuses a non-bit-exact ablation
or a bandwidth that disagrees between the two runs. SUBMISSION claims this in
code; these tests pin the claim so it cannot silently rot."""

import json
import re
from pathlib import Path

import pytest

from tokonomics.merge_measured import merge_measured
from tokonomics.loaders import load_prices
from tokonomics.schema import MachineResult

REPO_ROOT = Path(__file__).resolve().parents[1]


def _driver_json(tmp_path, name, *, correct=True, off=600.0, on=900.0, bw=46.0):
    """Write one driver-style JSON (only the keys merge_measured reads)."""
    is_on = "_on" in name   # 'bench_on.json' (not '.json' substring of either)
    doc = {
        "label": "cobalt100-n2",
        "kind": "measured",
        "arch": "Neoverse N2 (ubuntu-24.04-arm)",
        "correct": correct,
        "mem_bw_gbs": bw,
        "kernel_path": "i8mm/smmla" if is_on else "dotprod/sdot",
    }
    doc["peak_int8_gops_on" if is_on else "peak_int8_gops_off"] = on if is_on else off
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_merge_happy_path(tmp_path):
    off = _driver_json(tmp_path, "bench_off.json", off=600.0, bw=46.0)
    on = _driver_json(tmp_path, "bench_on.json", on=900.0, bw=47.0)
    m = merge_measured(off, on)
    assert isinstance(m, MachineResult)
    assert m.kind == "measured" and m.label == "cobalt100-n2"
    assert m.ceilings.peak_int8_gops_off == pytest.approx(600.0)
    assert m.ceilings.peak_int8_gops_on == pytest.approx(900.0)
    # bandwidth is taken from the OFF run (i8mm-independent by construction)
    assert m.ceilings.mem_bw_gbs == pytest.approx(46.0)


def test_merge_rejects_non_bit_exact_run(tmp_path):
    # If either variant disagreed with the scalar oracle, the ablation is
    # meaningless and the merge must refuse rather than report a speedup.
    off = _driver_json(tmp_path, "bench_off.json", correct=False)
    on = _driver_json(tmp_path, "bench_on.json", correct=True)
    with pytest.raises(ValueError, match="bit-identical"):
        merge_measured(off, on)


def test_merge_rejects_bandwidth_disagreement(tmp_path):
    # Memory bandwidth must not depend on the compute ISA; a >15% gap between the
    # two runs signals a noisy machine, not an i8mm effect — reject it.
    off = _driver_json(tmp_path, "bench_off.json", bw=46.0)
    on = _driver_json(tmp_path, "bench_on.json", bw=60.0)  # +30%
    with pytest.raises(ValueError, match="mem_bw disagrees"):
        merge_measured(off, on)


def test_driver_label_has_a_pricing_row():
    """The C driver hardcodes the instance label; cli.py cmd_measured aborts
    (SystemExit) if that label has no row in pricing.yaml. This pins the
    cross-language contract between driver.c and the pricing table so the
    measured CI run cannot fail on its first invocation with an unknown label."""
    driver_c = (REPO_ROOT / "bench" / "microkernel" / "driver.c").read_text(
        encoding="utf-8")
    # driver.c emits the label inside a printf, so the quotes are escaped:
    #   printf("  \"label\": \"cobalt100-n2\",\n");
    m = re.search(r'\\"label\\"\s*:\s*\\"([^\\"]+)\\"', driver_c)
    assert m, "could not find the emitted label literal in driver.c"
    emitted = m.group(1)
    prices = load_prices(REPO_ROOT / "econ" / "pricing.yaml")
    assert emitted in prices, (
        f"driver.c emits label '{emitted}' but pricing.yaml has {list(prices)}; "
        f"cmd_measured would SystemExit on the first measured CI run")


def test_merged_label_prices_through_to_economics(tmp_path):
    """End-to-end of the label contract: a driver-style JSON pair merges to a
    machine whose label resolves in pricing.yaml — the exact lookup cmd_measured
    does before it can emit a measured economics table."""
    off = _driver_json(tmp_path, "bench_off.json", off=600.0)
    on = _driver_json(tmp_path, "bench_on.json", on=900.0)
    m = merge_measured(off, on)
    prices = load_prices(REPO_ROOT / "econ" / "pricing.yaml")
    assert m.label in prices, (
        f"merged label '{m.label}' missing from pricing.yaml — cmd_measured "
        f"would SystemExit")


def test_merge_output_survives_schema_firewall(tmp_path):
    # The merged machine must itself pass schema validation (on >= off etc.),
    # i.e. the ingest path and the load path agree on what a valid machine is.
    from tokonomics.schema import MachineResult as MR
    off = _driver_json(tmp_path, "bench_off.json", off=600.0)
    on = _driver_json(tmp_path, "bench_on.json", on=900.0)
    m = merge_measured(off, on)
    # round-trip through the validating loader
    MR.from_dict(m.to_json(), where="roundtrip")
