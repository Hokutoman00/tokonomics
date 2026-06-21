"""Turn measured throughput + public prices into tokens/$ and tokens/Joule.

Formulas (kept deliberately small and testable):

    tokens_per_usd   = tok_s * 3600 / usd_per_hour
    tokens_per_joule = tok_s / avg_power_watt

avg_power_watt is estimated from the instance TDP (we do not have socket-level
power telemetry); it is therefore a *projection*, labelled as such everywhere
it surfaces. tokens/$ uses real published on-demand pricing (provenance in
pricing.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import MachineResult, PriceEntry
from .model import ModelSpec, decode_tok_s, prefill_tok_s


@dataclass(frozen=True)
class Economics:
    label: str
    arch: str
    kind: str                 # measured | dev | projection (inherited from inputs)
    model: str
    phase: str                # "prefill" or "decode"
    i8mm: str                 # "on" or "off"
    tok_s: float
    tokens_per_usd: float
    tokens_per_joule: float
    power_is_estimate: bool   # always True for now (TDP-derived)


def _tokens_per_usd(tok_s: float, usd_per_hour: float) -> float:
    if usd_per_hour <= 0:
        raise ValueError("usd_per_hour must be > 0")
    return tok_s * 3600.0 / usd_per_hour


def _tokens_per_joule(tok_s: float, watt: float) -> float:
    if watt <= 0:
        raise ValueError("avg_power_watt must be > 0")
    return tok_s / watt


def compute(
    machine: MachineResult,
    price: PriceEntry,
    model: ModelSpec,
    phase: str,
    i8mm: str = "on",
) -> Economics:
    """Economics for one (machine, model, phase, i8mm) point."""
    if phase not in ("prefill", "decode"):
        raise ValueError(f"phase must be prefill|decode, got {phase!r}")
    if i8mm not in ("on", "off"):
        raise ValueError(f"i8mm must be on|off, got {i8mm!r}")
    if machine.label != price.label:
        raise ValueError(
            f"machine/price label mismatch: {machine.label} != {price.label}"
        )

    peak = (machine.ceilings.peak_int8_gops_on if i8mm == "on"
            else machine.ceilings.peak_int8_gops_off)

    if phase == "prefill":
        # roofline: compute-bound at realistic prompt length -> i8mm helps
        tok_s = prefill_tok_s(peak, machine.ceilings.mem_bw_gbs, model)
    else:
        # roofline min(compute, mem_bw*AI): decode's low AI makes this the memory
        # term, so i8mm's higher `peak` leaves decode unchanged — derived, not assumed.
        tok_s = decode_tok_s(machine.ceilings.mem_bw_gbs, model, peak)

    return Economics(
        label=machine.label,
        arch=machine.arch,
        kind=machine.kind,
        model=model.name,
        phase=phase,
        i8mm=i8mm,
        tok_s=tok_s,
        tokens_per_usd=_tokens_per_usd(tok_s, price.usd_per_hour),
        tokens_per_joule=_tokens_per_joule(tok_s, price.tdp_watt),
        power_is_estimate=True,
    )


def blended_tokens_per_usd(
    machine: MachineResult,
    price: PriceEntry,
    model: ModelSpec,
    prompt_to_gen_ratio: float,
    i8mm: str = "on",
) -> float:
    """Workload-weighted tokens/$ for a given prompt:gen ratio.

    A request of r prompt tokens : 1 generated token spends time in both
    phases. We weight cost by time share (time = tokens / tok_s) so that a
    decode-heavy workload (small r) is dominated by the memory ceiling and a
    prefill-heavy workload (large r) by the compute ceiling. This is exactly
    where the crossover between instances appears.
    """
    pf = compute(machine, price, model, "prefill", i8mm)
    dc = compute(machine, price, model, "decode", i8mm)
    # time to serve r prompt tokens + 1 generated token
    t_prefill = prompt_to_gen_ratio / pf.tok_s
    t_decode = 1.0 / dc.tok_s
    total_tokens = prompt_to_gen_ratio + 1.0
    total_cost = (t_prefill + t_decode) * (price.usd_per_hour / 3600.0)
    if total_cost <= 0:
        raise ValueError("non-positive cost")
    return total_tokens / total_cost
