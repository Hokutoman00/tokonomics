"""The schema is the honesty firewall: malformed or implausible JSON must be
rejected before it can become an economics number (L3/RC-3)."""

import pytest

from tokonomics.schema import (
    MachineResult, Ceilings, PriceEntry, validate_table_kind, TABLE_KINDS)


def _good():
    return {
        "label": "n2", "kind": "measured", "arch": "neoverse-n2",
        "ceilings": {"peak_int8_gops_off": 500.0,
                     "peak_int8_gops_on": 1000.0, "mem_bw_gbs": 50.0},
        "notes": "ok",
    }


def test_good_roundtrip():
    m = MachineResult.from_dict(_good())
    assert m.ceilings.peak_int8_gops_on == 1000.0
    assert MachineResult.from_dict(m.to_json()).label == "n2"


def test_missing_key_rejected():
    d = _good()
    del d["ceilings"]
    with pytest.raises(ValueError):
        MachineResult.from_dict(d)


def test_missing_nested_key_rejected():
    d = _good()
    del d["ceilings"]["mem_bw_gbs"]
    with pytest.raises(ValueError):
        MachineResult.from_dict(d)


def test_unknown_kind_rejected():
    d = _good()
    d["kind"] = "guess"
    with pytest.raises(ValueError):
        MachineResult.from_dict(d)


def test_wrong_type_rejected():
    d = _good()
    d["ceilings"]["mem_bw_gbs"] = "fast"
    with pytest.raises(ValueError):
        MachineResult.from_dict(d)


def test_nonpositive_and_nan_inf_rejected():
    for bad in (0, -5.0, float("nan"), float("inf")):
        d = _good()
        d["ceilings"]["mem_bw_gbs"] = bad
        with pytest.raises(ValueError):
            MachineResult.from_dict(d)


def test_i8mm_on_below_off_rejected():
    d = _good()
    d["ceilings"]["peak_int8_gops_on"] = 100.0   # < off 500 -> implausible
    with pytest.raises(ValueError):
        MachineResult.from_dict(d)


def test_table_kind_firewall_accepts_four_kinds():
    # All four advertised table kinds (incl. the "roofline" fourth label that
    # VALID_KINDS does not cover) must pass and be returned unchanged.
    assert set(TABLE_KINDS) == {"measured", "roofline", "dev", "projection"}
    for k in TABLE_KINDS:
        assert validate_table_kind(k) == k


def test_table_kind_firewall_rejects_unknown_and_missing():
    # A projection mislabelled, a dropped kind, or a typo must all raise — this
    # is the check report.generate_report runs on every economics table so a
    # projected number can't render under a measured heading.
    for bad in ("guess", "", None, "Measured", "rooflines"):
        with pytest.raises(ValueError):
            validate_table_kind(bad, where="t.json")


def test_price_entry_requires_provenance():
    base = {"label": "n2", "arch": "x", "vcpu": 8, "usd_per_hour": 0.1,
            "tdp_watt": 40.0, "source_url": "http://x", "retrieved": "2026-06-20"}
    assert PriceEntry.from_dict(base).usd_per_hour == 0.1
    for missing in ("source_url", "retrieved", "usd_per_hour"):
        d = dict(base)
        del d[missing]
        with pytest.raises(ValueError):
            PriceEntry.from_dict(d)
