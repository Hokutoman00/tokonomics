"""Command-line entry point: build results, figures and REPORT from inputs.

Subcommands:
  project   specs.yaml -> projected economics + roofline/crossover figures
  dev       measure local x86 -> dev economics + roofline (pipeline validation)
  measured  merge microkernel CI driver JSONs -> measured economics
  llama     ingest llama-bench JSONs -> measured-LLM (real GGUF) economics
  gate      fail the build if measured prefill tokens/$ regresses below a floor
  report    regenerate REPORT.md from whatever results exist
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .loaders import load_prices, load_spec_machines
from .schema import dump_machine
from .measure_x86 import measure_local
from .model import MODELS
from .economics import compute
from .roofline import plot_roofline
from .crossover import crossover_grid, plot_crossover

ROOT = Path(__file__).resolve().parent.parent
RATIOS = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_MODEL = "llama-3.2-1b-q4_0"


def _econ_rows(machines, prices, model):
    rows = []
    for lab, m in machines.items():
        if lab not in prices:
            continue
        for phase in ("prefill", "decode"):
            for i8mm in ("off", "on"):
                e = compute(m, prices[lab], model, phase, i8mm)
                rows.append(asdict(e))
    return rows


def cmd_project(args):
    prices = load_prices(ROOT / "econ" / "pricing.yaml")
    machines = load_spec_machines(ROOT / "projection" / "specs.yaml")
    model = MODELS[args.model]

    out_results = ROOT / "results" / "projected"
    out_figs = ROOT / "figures"
    out_results.mkdir(parents=True, exist_ok=True)

    rows = _econ_rows(machines, prices, model)
    (out_results / "economics.json").write_text(
        json.dumps({"kind": "projection", "model": model.name, "rows": rows}, indent=2),
        encoding="utf-8")

    grid = crossover_grid(machines, prices, model, RATIOS, i8mm="on")
    (out_results / "crossover.json").write_text(json.dumps(grid, indent=2),
                                                encoding="utf-8")
    plot_crossover(grid, out_figs / "crossover_projected.png", kind="projection")

    # roofline for the two i8mm-capable anchors
    for lab in ("cobalt100-n2", "graviton4-v2"):
        plot_roofline(machines[lab], model, out_figs / f"roofline_{lab}.png")

    print(f"[project] crossover detected: {grid['has_crossover']} "
          f"(winners: {grid['winners']})")
    print(f"[project] wrote {out_results}/economics.json, crossover.json and figures/")


def cmd_dev(args):
    prices = load_prices(ROOT / "econ" / "pricing.yaml")
    model = MODELS[args.model]
    m = measure_local()
    out_results = ROOT / "results" / "x86-dev"
    out_figs = ROOT / "figures"
    dump_machine(m, out_results / "x86-dev.json")

    # price the dev box against a representative row so tokens/$ is exercised
    price = prices["graviton4-v2"]
    price = type(price)(**{**asdict_price(price), "label": m.label})
    rows = []
    for phase in ("prefill", "decode"):
        rows.append(asdict(compute(m, price, model, phase, "on")))
    (out_results / "economics.json").write_text(
        json.dumps({"kind": "dev", "model": model.name, "rows": rows}, indent=2),
        encoding="utf-8")
    plot_roofline(m, model, out_figs / "roofline_x86dev.png")
    print(f"[dev] measured x86 ceilings: compute={m.ceilings.peak_int8_gops_on:.1f} "
          f"GFLOP/s, bw={m.ceilings.mem_bw_gbs:.1f} GB/s")
    print(f"[dev] wrote {out_results}/ and figures/roofline_x86dev.png")


def asdict_price(p):
    from dataclasses import asdict as _a
    return _a(p)


def cmd_measured(args):
    """Merge CI driver JSONs (results/measured/bench_{off,on}.json) into a
    measured machine, then run the same economics + roofline path on it."""
    from .merge_measured import merge_measured
    prices = load_prices(ROOT / "econ" / "pricing.yaml")
    model = MODELS[args.model]
    out_results = ROOT / "results" / "measured"
    out_figs = ROOT / "figures"

    m = merge_measured(out_results / "bench_off.json", out_results / "bench_on.json")
    dump_machine(m, out_results / f"{m.label}.json")

    if m.label not in prices:
        raise SystemExit(f"[measured] no pricing row for label '{m.label}' "
                         f"in pricing.yaml (have {list(prices)})")
    price = prices[m.label]
    rows = []
    for phase in ("prefill", "decode"):
        for i8mm in ("off", "on"):
            rows.append(asdict(compute(m, price, model, phase, i8mm)))
    (out_results / "economics.json").write_text(
        json.dumps({"kind": "measured", "model": model.name, "rows": rows}, indent=2),
        encoding="utf-8")
    plot_roofline(m, model, out_figs / f"roofline_{m.label}_measured.png")
    print(f"[measured] {m.label}: peak off={m.ceilings.peak_int8_gops_off:.1f} "
          f"on={m.ceilings.peak_int8_gops_on:.1f} GOP/s, "
          f"bw={m.ceilings.mem_bw_gbs:.1f} GB/s (i8mm lift "
          f"{m.ceilings.peak_int8_gops_on / m.ceilings.peak_int8_gops_off:.2f}x)")


def cmd_llama(args):
    """Ingest the two llama-bench JSONs (results/measured/llama_{off,on}.json)
    into a measured-LLM economics table — real GGUF inference, not a proxy."""
    from .llama_ingest import ingest_llama
    prices = load_prices(ROOT / "econ" / "pricing.yaml")
    out_results = ROOT / "results" / "measured"
    label = args.instance
    if label not in prices:
        raise SystemExit(f"[llama] no pricing row for '{label}' "
                         f"(have {list(prices)})")
    doc = ingest_llama(out_results / "llama_off.json",
                       out_results / "llama_on.json",
                       prices[label], label)
    (out_results / "llama_economics.json").write_text(
        json.dumps(doc, indent=2), encoding="utf-8")
    print(f"[llama] {label} model={doc['model']}: "
          f"i8mm prefill lift {doc['i8mm_prefill_lift']:.2f}x, "
          f"decode ratio {doc['decode_ratio']:.2f}x (memory-bound, ~flat) -> "
          f"results/measured/llama_economics.json")


def cmd_gate(args):
    """Evaluate the tokens/$ regression gate against results/measured/economics.json.
    Emits GitHub Actions `::error::`/`::notice::` annotations and exits non-zero
    on a violation — this is exactly what action.yml runs."""
    from .gate import evaluate_gate, GateError
    econ = ROOT / "results" / "measured" / "economics.json"
    if not econ.exists():
        print(f"::error::no measured economics at {econ} - run `tokonomics measured` first")
        raise SystemExit(1)
    rows = json.loads(econ.read_text(encoding="utf-8"))["rows"]
    try:
        result = evaluate_gate(rows, float(args.min_tokens_per_usd))
    except GateError as e:
        print(f"::error::{e}")
        raise SystemExit(1)
    print(result.message)
    print("::notice::tokens/$ gate passed")


def cmd_report(args):
    from .report import generate_report
    path = generate_report(ROOT)
    print(f"[report] wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tokonomics")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("project", help="projected economics + figures from specs.yaml")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    p.set_defaults(func=cmd_project)

    d = sub.add_parser("dev", help="measure local x86 and validate the pipeline")
    d.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    d.set_defaults(func=cmd_dev)

    me = sub.add_parser("measured", help="merge CI driver JSONs into measured economics")
    me.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    me.set_defaults(func=cmd_measured)

    la = sub.add_parser("llama", help="ingest llama-bench JSONs into measured-LLM economics")
    la.add_argument("--instance", default="cobalt100-n2",
                    help="pricing label for the runner (free ubuntu-24.04-arm == N2)")
    la.set_defaults(func=cmd_llama)

    g = sub.add_parser("gate", help="fail the build if measured tokens/$ regresses below a floor")
    g.add_argument("--min-tokens-per-usd", default="0", dest="min_tokens_per_usd",
                   help="floor for prefill tokens/$ (i8mm on); 0 = measure-only")
    g.set_defaults(func=cmd_gate)

    r = sub.add_parser("report", help="regenerate REPORT.md")
    r.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
