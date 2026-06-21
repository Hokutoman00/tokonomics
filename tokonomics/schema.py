"""Typed schema + validation for every JSON the lab produces or consumes.

Why this exists: untrusted JSON (CI artifacts, hand-edited specs) must be
type-guarded before use (project rule L3/RC-3 — never trust parsed JSON shape).
Every loader here raises on a malformed document instead of silently
propagating a bad value into an economics number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# The three confidence kinds a *machine* can carry: its ceilings are either
# measured on silicon, a dev-proxy (x86 float), or a published-spec projection.
# Anything outside this set is rejected so a projection can never be silently
# relabelled as a measurement.
VALID_KINDS = ("measured", "dev", "projection")

# Economics *tables* (per-model tok/s) can additionally be a "roofline": a
# measured/projected ceiling projected onto a model (modeled tok/s, decode ==
# prefill within an i8mm setting), which is neither a raw machine kind nor a
# real-inference measurement. The firewall enforces this fourth label at the
# table level via validate_table_kind so "roofline" can't be silently dropped
# or a real machine kind mislabelled — closing the gap that VALID_KINDS (which
# only governs MachineResult) left for the economics tables.
TABLE_KINDS = ("measured", "roofline", "dev", "projection")


def validate_table_kind(kind: Any, where: str = "economics table") -> str:
    """Reject any economics-table `kind` outside TABLE_KINDS. Returns the kind so
    callers can use it inline. This is the table-level twin of the MachineResult
    `kind not in VALID_KINDS` check, so the advertised four-kind firewall is real
    in code rather than just narrated in the README."""
    if kind not in TABLE_KINDS:
        raise ValueError(f"{where}: kind {kind!r} not in {TABLE_KINDS}")
    return kind


def _require(d: dict, key: str, types: tuple[type, ...], where: str) -> Any:
    if key not in d:
        raise ValueError(f"{where}: missing required key '{key}'")
    val = d[key]
    if not isinstance(val, types):
        raise ValueError(
            f"{where}: key '{key}' must be {types}, got {type(val).__name__}"
        )
    return val


def _pos(x: float, key: str, where: str) -> float:
    if not (x > 0 and x == x and x != float("inf")):  # >0, not NaN, not inf
        raise ValueError(f"{where}: '{key}' must be a finite positive number, got {x!r}")
    return float(x)


def _nonempty(s: str, key: str, where: str) -> str:
    # A present-but-blank provenance string is as orphaned as a missing one, so
    # the firewall rejects it (SUBMISSION claims "a price without a source URL +
    # retrieval date" is refused — this makes that literally true).
    if not s.strip():
        raise ValueError(f"{where}: '{key}' must be a non-empty string")
    return s


@dataclass(frozen=True)
class Ceilings:
    """Roofline ceilings for one machine.

    peak_int8_gops_off : compute ceiling with i8mm disabled (dotprod/SDOT path)
    peak_int8_gops_on  : compute ceiling with i8mm enabled  (SMMLA path)
    mem_bw_gbs         : memory ceiling (STREAM-triad effective bandwidth)

    For the x86 dev proxy these are float GFLOP/s and GB/s (clearly a proxy,
    not Arm int8 — see `kind`).
    """

    peak_int8_gops_off: float
    peak_int8_gops_on: float
    mem_bw_gbs: float


@dataclass(frozen=True)
class MachineResult:
    label: str          # e.g. "n2", "v2", "x86-dev"
    kind: str           # one of VALID_KINDS
    arch: str           # human arch name, e.g. "neoverse-n2 (armv9.0)"
    ceilings: Ceilings
    notes: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict, where: str = "MachineResult") -> "MachineResult":
        label = _require(d, "label", (str,), where)
        kind = _require(d, "kind", (str,), where)
        if kind not in VALID_KINDS:
            raise ValueError(f"{where}: kind '{kind}' not in {VALID_KINDS}")
        arch = _require(d, "arch", (str,), where)
        c = _require(d, "ceilings", (dict,), where)
        cw = f"{where}.ceilings"
        ceilings = Ceilings(
            peak_int8_gops_off=_pos(
                _require(c, "peak_int8_gops_off", (int, float), cw),
                "peak_int8_gops_off", cw),
            peak_int8_gops_on=_pos(
                _require(c, "peak_int8_gops_on", (int, float), cw),
                "peak_int8_gops_on", cw),
            mem_bw_gbs=_pos(
                _require(c, "mem_bw_gbs", (int, float), cw),
                "mem_bw_gbs", cw),
        )
        # i8mm must not *slow down* compute; if on < off the run is bogus.
        if ceilings.peak_int8_gops_on < ceilings.peak_int8_gops_off:
            raise ValueError(
                f"{where}: peak_int8_gops_on ({ceilings.peak_int8_gops_on}) "
                f"< off ({ceilings.peak_int8_gops_off}) — implausible, rejecting"
            )
        return MachineResult(
            label=label, kind=kind, arch=arch, ceilings=ceilings,
            notes=str(d.get("notes", "")),
        )


@dataclass(frozen=True)
class PriceEntry:
    """One row of pricing.yaml — reference data with provenance.

    usd_per_hour / tdp_watt are inputs the user refreshes; each row carries
    its source_url + retrieved date so a number is never orphaned from where
    it came from (project rule: no magic numbers).
    """

    label: str
    arch: str
    vcpu: int
    usd_per_hour: float
    tdp_watt: float
    source_url: str
    retrieved: str

    @staticmethod
    def from_dict(d: dict, where: str = "PriceEntry") -> "PriceEntry":
        return PriceEntry(
            label=_require(d, "label", (str,), where),
            arch=_require(d, "arch", (str,), where),
            vcpu=int(_require(d, "vcpu", (int,), where)),
            usd_per_hour=_pos(
                _require(d, "usd_per_hour", (int, float), where),
                "usd_per_hour", where),
            tdp_watt=_pos(
                _require(d, "tdp_watt", (int, float), where),
                "tdp_watt", where),
            source_url=_nonempty(
                _require(d, "source_url", (str,), where), "source_url", where),
            retrieved=_nonempty(
                _require(d, "retrieved", (str,), where), "retrieved", where),
        )


def load_machine(path: str | Path) -> MachineResult:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return MachineResult.from_dict(data, where=str(p))


def dump_machine(result: MachineResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
