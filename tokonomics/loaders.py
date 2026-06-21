"""Load pricing.yaml and projection/specs.yaml into validated objects."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import MachineResult, Ceilings, PriceEntry


def load_prices(path: str | Path) -> dict[str, PriceEntry]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "machines" not in raw:
        raise ValueError(f"{path}: expected top-level 'machines' list")
    out: dict[str, PriceEntry] = {}
    for i, row in enumerate(raw["machines"]):
        pe = PriceEntry.from_dict(row, where=f"{path}#machines[{i}]")
        if pe.label in out:
            raise ValueError(f"{path}: duplicate label {pe.label}")
        out[pe.label] = pe
    return out


def load_spec_machines(path: str | Path) -> dict[str, MachineResult]:
    """Read specs.yaml -> MachineResult objects tagged kind='projection'."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "machines" not in raw:
        raise ValueError(f"{path}: expected top-level 'machines' list")
    out: dict[str, MachineResult] = {}
    for i, row in enumerate(raw["machines"]):
        where = f"{path}#machines[{i}]"
        label = row["label"]
        mr = MachineResult(
            label=label,
            kind="projection",
            arch=row["arch"],
            ceilings=Ceilings(
                peak_int8_gops_off=float(row["peak_int8_gops_off"]),
                peak_int8_gops_on=float(row["peak_int8_gops_on"]),
                mem_bw_gbs=float(row["mem_bw_gbs"]),
            ),
            notes=f"projection from specs.yaml: {row.get('source', '')}",
        )
        # round-trip through validation to enforce invariants (on >= off etc.)
        out[label] = MachineResult.from_dict(mr.to_json(), where=where)
    return out
