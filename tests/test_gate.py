"""The tokens/$ regression gate (`action.yml`'s teeth) as unit tests.

The reusable composite action's whole value proposition is "fail my build when a
kernel/quant change makes inference more expensive." That claim is only credible
if the gate logic is *exercised*, not just shipped. We can't run the Arm
microkernel here, but the gate decision is pure given economics rows — so we pin
pass / measure-only / below-floor / missing-row deterministically on x86."""

import pytest

from tokonomics.gate import evaluate_gate, GateError, GateResult


def _rows(prefill_on_tpu):
    """Minimal economics rows; only the prefill/i8mm-on tokens/$ matters."""
    return [
        {"phase": "prefill", "i8mm": "off", "tokens_per_usd": prefill_on_tpu * 0.6},
        {"phase": "prefill", "i8mm": "on", "tokens_per_usd": prefill_on_tpu},
        {"phase": "decode", "i8mm": "on", "tokens_per_usd": prefill_on_tpu * 0.2},
    ]


def test_gate_passes_when_above_floor():
    r = evaluate_gate(_rows(9_000_000), floor=8_000_000)
    assert isinstance(r, GateResult)
    assert r.passed
    assert r.tokens_per_usd == pytest.approx(9_000_000)
    assert "at or above floor" in r.message


def test_gate_measure_only_when_floor_zero():
    # floor 0 = the default: read a baseline, never fail.
    r = evaluate_gate(_rows(123_456), floor=0)
    assert r.passed
    assert "measure-only" in r.message


def test_gate_fails_below_floor():
    # this is the regression the action exists to catch.
    with pytest.raises(GateError, match="below floor"):
        evaluate_gate(_rows(7_000_000), floor=8_000_000)


def test_gate_errors_when_no_prefill_on_row():
    rows = [{"phase": "decode", "i8mm": "on", "tokens_per_usd": 1_000_000}]
    with pytest.raises(GateError, match="no prefill/i8mm-on row"):
        evaluate_gate(rows, floor=0)


def test_gate_boundary_equal_floor_passes():
    # exactly at the floor is not a regression (strict <).
    r = evaluate_gate(_rows(8_000_000), floor=8_000_000)
    assert r.passed
