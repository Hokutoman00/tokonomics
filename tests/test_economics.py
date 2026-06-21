"""Verify the economics formulas on known inputs, including the asymmetry that
i8mm helps prefill but not decode, plus boundary/negative asserts."""

import math

import pytest

from tokonomics.schema import MachineResult, Ceilings, PriceEntry
from tokonomics.model import ModelSpec, decode_tok_s, prefill_tok_s
from tokonomics.economics import (
    compute, _tokens_per_usd, _tokens_per_joule, blended_tokens_per_usd,
)


def _machine(label="n2", off=500.0, on=1000.0, bw=50.0):
    return MachineResult(label, "measured", "test", Ceilings(off, on, bw))


def _price(label="n2", usd=0.10, watt=40.0):
    return PriceEntry(label, "test", 8, usd, watt, "http://example", "2026-06-20")


MODEL = ModelSpec("toy", params=1e9, bytes_per_param=0.5)  # 0.5 GB model


def test_tokens_per_usd_exact():
    # 100 tok/s for $0.10/hr -> 100*3600/0.1 = 3,600,000 tokens/$
    assert _tokens_per_usd(100.0, 0.10) == pytest.approx(3_600_000.0)


def test_tokens_per_joule_exact():
    assert _tokens_per_joule(80.0, 40.0) == pytest.approx(2.0)


def test_zero_and_negative_price_rejected():
    with pytest.raises(ValueError):
        _tokens_per_usd(100.0, 0.0)
    with pytest.raises(ValueError):
        _tokens_per_usd(100.0, -1.0)
    with pytest.raises(ValueError):
        _tokens_per_joule(100.0, 0.0)


def test_decode_is_memory_bound_formula():
    # tok/s = bw*1e9 / model_bytes ; model_bytes = 1e9*0.5 = 5e8
    assert decode_tok_s(50.0, MODEL) == pytest.approx(50e9 / 5e8)


def test_decode_roofline_min_matches_closed_form_and_caps_on_compute():
    # Passing peak evaluates the genuine min(compute, mem_bw*AI). On a realistic
    # machine the memory term wins, so it equals the closed form (i8mm-immune):
    assert decode_tok_s(50.0, MODEL, peak_gops=1000.0) == pytest.approx(
        decode_tok_s(50.0, MODEL))
    # but an implausible huge-bandwidth / tiny-compute machine is compute-capped,
    # so decode falls BELOW the memory ceiling — proving min() is real, not assumed.
    capped = decode_tok_s(5000.0, MODEL, peak_gops=10.0)
    assert capped == pytest.approx(10e9 / (2 * MODEL.params))
    assert capped < decode_tok_s(5000.0, MODEL)


def test_prefill_is_compute_bound_formula():
    # At a long prompt the roofline is compute-bound: tok/s = peak*1e9 / (2*params)
    # (mem_bw*AI = 50 * decode_ai*512 >> peak, so attainable == peak)
    assert prefill_tok_s(1000.0, 50.0, MODEL) == pytest.approx(1000e9 / 2e9)


def test_prefill_falls_back_to_memory_ceiling_when_starved():
    # Roofline min() is genuinely evaluated: with a tiny memory ceiling the
    # attainable rate is mem_bw*AI, NOT the compute peak — so prefill drops below
    # the compute-bound value. This exercises the min(compute, mem_bw*AI) branch.
    ai = MODEL.prefill_ai(512)            # decode_ai(=4) * 512
    starved = prefill_tok_s(1000.0, 0.1, MODEL)   # mem_bw*AI = 0.1*2048 = 204.8 < 1000
    assert starved == pytest.approx((0.1 * ai) * 1e9 / (2 * MODEL.params))
    assert starved < prefill_tok_s(1000.0, 50.0, MODEL)


def test_i8mm_lifts_prefill_but_not_decode():
    m, p = _machine(), _price()
    pf_off = compute(m, p, MODEL, "prefill", "off").tok_s
    pf_on = compute(m, p, MODEL, "prefill", "on").tok_s
    dc_off = compute(m, p, MODEL, "decode", "off").tok_s
    dc_on = compute(m, p, MODEL, "decode", "on").tok_s
    # prefill on/off ratio tracks the ceiling ratio (1000/500 = 2x)
    assert pf_on == pytest.approx(2.0 * pf_off)
    # decode is bandwidth-bound: i8mm makes NO difference (negative assert)
    assert dc_on == pytest.approx(dc_off)
    assert dc_on != pytest.approx(2.0 * dc_off)


def test_label_mismatch_rejected():
    with pytest.raises(ValueError):
        compute(_machine(label="n2"), _price(label="v2"), MODEL, "prefill", "on")


def test_bad_phase_and_i8mm_rejected():
    with pytest.raises(ValueError):
        compute(_machine(), _price(), MODEL, "nonsense", "on")
    with pytest.raises(ValueError):
        compute(_machine(), _price(), MODEL, "prefill", "maybe")


def test_blended_is_finite_and_monotone_in_ratio():
    m, p = _machine(), _price()
    v_small = blended_tokens_per_usd(m, p, MODEL, 1.0, "on")
    v_large = blended_tokens_per_usd(m, p, MODEL, 64.0, "on")
    assert math.isfinite(v_small) and math.isfinite(v_large)
    # prefill is faster per token than decode here, so more prompt share raises tokens/$
    assert v_large > v_small
