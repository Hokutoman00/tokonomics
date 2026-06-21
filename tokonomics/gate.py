"""The tokens/$ regression gate as a first-class, testable function.

This is the logic `action.yml` runs to fail a build when a quant/kernel change
makes prefill inference *more expensive* on Arm. It used to live as an inline
heredoc inside the composite action — untestable YAML that could silently rot.
Lifting it here lets the x86 dev suite *exercise* the gate deterministically
(no Arm hardware, no shipped "measured" numbers), so the reusability claim is
proven, not asserted.

The gate reads only the `prefill` / `i8mm==on` row — that is the cell `i8mm`
actually lifts and the one a tokens/$ regression would show up in.
"""

from __future__ import annotations

from dataclasses import dataclass


class GateError(ValueError):
    """A gate violation: either no eligible row, or tokens/$ below the floor."""


@dataclass(frozen=True)
class GateResult:
    tokens_per_usd: float
    floor: float
    passed: bool
    message: str


def evaluate_gate(rows: list[dict], floor: float) -> GateResult:
    """Evaluate the measured economics `rows` against a tokens/$ `floor`.

    Returns a GateResult when the gate is satisfied (including the floor==0
    measure-only case). Raises GateError when no prefill/i8mm-on row exists or
    when measured tokens/$ is strictly below a positive floor.
    """
    on_prefill = [r for r in rows
                  if r.get("phase") == "prefill" and r.get("i8mm") == "on"]
    if not on_prefill:
        raise GateError("no prefill/i8mm-on row produced")

    got = on_prefill[0]["tokens_per_usd"]
    if floor > 0 and got < floor:
        raise GateError(
            f"tokens/$ {got:,.0f} below floor {floor:,.0f}")

    note = ("measure-only (no floor set)" if floor <= 0
            else f"at or above floor {floor:,.0f}")
    return GateResult(
        tokens_per_usd=got, floor=floor, passed=True,
        message=f"prefill tokens/$ (i8mm on) = {got:,.0f} - {note}")
