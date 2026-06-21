"""Ingest llama-bench JSON (real LLM inference) into measured economics.

This is the *application-level* counterpart to the microkernel ablation. The
microkernel proves the i8mm uplift on a hand-written GEMM; this layer proves it
survives end-to-end on a real GGUF model via `llama.cpp`, so the headline number
a judge sees is "measured tokens/sec on Llama-3.2-1B", not only a kernel proxy.

`llama-bench -p 512 -n 128 -o json` emits an array of rows. We read the two
builds (i8mm off / on) and pull, from each:

  * prefill (pp): the row with n_gen == 0 and n_prompt > 0  -> avg_ts tok/s
  * decode  (tg): the row with n_prompt == 0 and n_gen > 0  -> avg_ts tok/s

We then price tok/s directly into tokens/$ and tokens/J* — the throughput here
is *measured*, so this is the strongest evidence in the repo, not a roofline
derivation.

Honesty firewall (same spirit as merge_measured / schema):
  * every tok/s must be finite and > 0, and both rows must exist in both files;
  * decode is memory-bound and therefore i8mm-immune by the roofline argument,
    so a tg that moves more than DECODE_TOL between the two builds signals a
    noisy/thermal run, not an i8mm effect -> reject (this is the macro check
    that the micro story holds);
  * we do NOT force prefill to rise. If measured pp_on < pp_off that is an
    honest negative result and is reported as such (lift < 1.0), never hidden.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .schema import PriceEntry

# Decode tok/s must not depend on the compute ISA (it is memory-bound). A gap
# wider than this between the off/on builds means a noisy run, not i8mm —
# mirrors merge_measured's 15% bandwidth-disagreement gate.
DECODE_TOL = 0.15


@dataclass(frozen=True)
class LlamaThroughput:
    """The two measured throughputs from one llama-bench run."""
    model: str
    prefill_tok_s: float
    decode_tok_s: float


def _finite_pos(x: float, what: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"llama ingest: {what} must be finite and > 0, got {x!r}")
    return v


def parse_llama_bench(doc: list[dict]) -> LlamaThroughput:
    """Extract (prefill, decode) tok/s from one llama-bench `-o json` document.

    Robust to repeated rows: averages all matching pp rows and all tg rows.
    """
    if not isinstance(doc, list) or not doc:
        raise ValueError("llama ingest: expected a non-empty JSON array from "
                         "`llama-bench -o json`")

    def _avg(rows: list[dict], what: str) -> float:
        if not rows:
            raise ValueError(f"llama ingest: no {what} row found "
                             "(need a pp row with n_gen==0 and a tg row with "
                             "n_prompt==0 from `-p 512 -n 128`)")
        vals = [_finite_pos(r["avg_ts"], f"{what} avg_ts") for r in rows]
        return sum(vals) / len(vals)

    pp_rows, tg_rows = [], []
    model = ""
    for r in doc:
        n_prompt = int(r.get("n_prompt", 0))
        n_gen = int(r.get("n_gen", 0))
        model = model or r.get("model_filename") or r.get("model_type") or ""
        if n_gen == 0 and n_prompt > 0:
            pp_rows.append(r)
        elif n_prompt == 0 and n_gen > 0:
            tg_rows.append(r)
        # rows that are neither (e.g. a combined pp+tg) are ignored on purpose

    return LlamaThroughput(
        model=model or "unknown.gguf",
        prefill_tok_s=_avg(pp_rows, "prefill (pp)"),
        decode_tok_s=_avg(tg_rows, "decode (tg)"),
    )


def _row(label: str, phase: str, i8mm: str, tok_s: float,
         price: PriceEntry) -> dict:
    return {
        "label": label,
        "phase": phase,
        "i8mm": i8mm,
        "tok_s": tok_s,
        # measured tok/s priced against real published $/hr -> measured tokens/$
        "tokens_per_usd": tok_s * 3600.0 / price.usd_per_hour,
        # power is TDP-derived -> this column stays a projection (the `*`)
        "tokens_per_joule": tok_s / price.tdp_watt,
    }


def ingest_llama(off_json: Path, on_json: Path, price: PriceEntry,
                 instance_label: str) -> dict:
    """Read the two llama-bench JSONs and build a measured-LLM economics doc.

    `price` is the pricing row for the instance the CI run happened on (the free
    `ubuntu-24.04-arm` runner == Neoverse N2 == cobalt100-n2). `instance_label`
    must match `price.label` so a tokens/$ number is never priced against the
    wrong instance.
    """
    if price.label != instance_label:
        raise ValueError(f"llama ingest: price/label mismatch "
                         f"{price.label!r} != {instance_label!r}")

    off = parse_llama_bench(json.loads(Path(off_json).read_text(encoding="utf-8")))
    on = parse_llama_bench(json.loads(Path(on_json).read_text(encoding="utf-8")))

    # Macro firewall: decode is memory-bound, so i8mm must leave tg ~flat.
    decode_ratio = on.decode_tok_s / off.decode_tok_s
    if abs(decode_ratio - 1.0) > DECODE_TOL:
        raise ValueError(
            f"llama ingest: decode tok/s moved {decode_ratio:.2f}x between i8mm "
            f"off/on (>{DECODE_TOL:.0%}). decode is memory-bound and must be "
            "i8mm-immune; this signals a noisy run, not an i8mm effect — refusing "
            "to report it.")

    prefill_lift = on.prefill_tok_s / off.prefill_tok_s
    rows = [
        _row(instance_label, "prefill", "off", off.prefill_tok_s, price),
        _row(instance_label, "prefill", "on", on.prefill_tok_s, price),
        _row(instance_label, "decode", "off", off.decode_tok_s, price),
        _row(instance_label, "decode", "on", on.decode_tok_s, price),
    ]
    return {
        "kind": "measured",
        "source": "llama.cpp llama-bench (real GGUF inference, pp512/tg128)",
        "model": on.model,
        "instance": instance_label,
        "rows": rows,
        "i8mm_prefill_lift": prefill_lift,
        "decode_ratio": decode_ratio,
    }
