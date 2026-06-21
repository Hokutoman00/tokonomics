"""The real-LLM measured layer: parse llama-bench JSON and price it, with the
same firewall discipline as the microkernel merge. We cannot run llama-bench
locally (no Arm runner, no C toolchain), so these tests pin the parser and the
honesty gates against *synthetic* llama-bench documents — the numbers stay
gated on CI, but the ingest logic is proven here so it cannot silently rot."""

import json

import pytest

from tokonomics.llama_ingest import (
    parse_llama_bench, ingest_llama, LlamaThroughput, DECODE_TOL,
)
from tokonomics.schema import PriceEntry


def _bench_doc(pp_ts, tg_ts, model="Llama-3.2-1B-Instruct-Q4_0.gguf"):
    """A minimal llama-bench `-o json` document: one pp row (n_gen==0) and one
    tg row (n_prompt==0), each carrying avg_ts — the only keys we read."""
    return [
        {"model_filename": model, "n_prompt": 512, "n_gen": 0, "avg_ts": pp_ts},
        {"model_filename": model, "n_prompt": 0, "n_gen": 128, "avg_ts": tg_ts},
    ]


def _price():
    return PriceEntry(
        label="cobalt100-n2", arch="Azure Cobalt 100 / Neoverse-N2",
        vcpu=4, usd_per_hour=0.14, tdp_watt=45,
        source_url="https://azure.microsoft.com/pricing", retrieved="2026-06-20",
    )


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_parse_picks_pp_and_tg_rows():
    t = parse_llama_bench(_bench_doc(120.0, 35.0))
    assert isinstance(t, LlamaThroughput)
    assert t.prefill_tok_s == pytest.approx(120.0)
    assert t.decode_tok_s == pytest.approx(35.0)
    assert t.model.endswith(".gguf")


def test_parse_averages_repeated_rows():
    doc = _bench_doc(100.0, 30.0)
    doc.append({"model_filename": "m.gguf", "n_prompt": 512, "n_gen": 0, "avg_ts": 140.0})
    t = parse_llama_bench(doc)
    assert t.prefill_tok_s == pytest.approx(120.0)  # mean of 100 and 140


def test_parse_rejects_missing_phase():
    only_pp = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0}]
    with pytest.raises(ValueError, match="no decode"):
        parse_llama_bench(only_pp)


def test_parse_rejects_nonpositive_ts():
    with pytest.raises(ValueError, match="finite and > 0"):
        parse_llama_bench(_bench_doc(0.0, 30.0))


def test_ingest_happy_path_prices_measured(tmp_path):
    off = _write(tmp_path, "llama_off.json", _bench_doc(80.0, 34.0))
    on = _write(tmp_path, "llama_on.json", _bench_doc(150.0, 35.0))  # pp ~1.9x, tg ~flat
    doc = ingest_llama(off, on, _price(), "cobalt100-n2")
    assert doc["kind"] == "measured"
    assert doc["i8mm_prefill_lift"] == pytest.approx(150.0 / 80.0)
    assert doc["decode_ratio"] == pytest.approx(35.0 / 34.0)
    # 4 rows: prefill/decode x off/on, priced into tokens/$
    assert len(doc["rows"]) == 4
    pp_on = next(r for r in doc["rows"] if r["phase"] == "prefill" and r["i8mm"] == "on")
    assert pp_on["tokens_per_usd"] == pytest.approx(150.0 * 3600.0 / 0.14)
    assert pp_on["tokens_per_joule"] == pytest.approx(150.0 / 45.0)


def test_ingest_rejects_noisy_decode(tmp_path):
    # decode is memory-bound -> i8mm-immune. A tg that moves well past DECODE_TOL
    # between builds is a noisy run, not an i8mm effect: refuse it.
    off = _write(tmp_path, "llama_off.json", _bench_doc(80.0, 34.0))
    on = _write(tmp_path, "llama_on.json", _bench_doc(150.0, 34.0 * (1 + DECODE_TOL + 0.1)))
    with pytest.raises(ValueError, match="decode is memory-bound"):
        ingest_llama(off, on, _price(), "cobalt100-n2")


def test_ingest_allows_honest_negative_prefill(tmp_path):
    # If i8mm did NOT help prefill, that is an honest result (lift < 1), not a
    # firewall violation — we must report it, never hide it.
    off = _write(tmp_path, "llama_off.json", _bench_doc(120.0, 34.0))
    on = _write(tmp_path, "llama_on.json", _bench_doc(110.0, 34.0))
    doc = ingest_llama(off, on, _price(), "cobalt100-n2")
    assert doc["i8mm_prefill_lift"] < 1.0


def test_ingest_rejects_price_label_mismatch(tmp_path):
    off = _write(tmp_path, "llama_off.json", _bench_doc(80.0, 34.0))
    on = _write(tmp_path, "llama_on.json", _bench_doc(150.0, 35.0))
    with pytest.raises(ValueError, match="price/label mismatch"):
        ingest_llama(off, on, _price(), "graviton4-v2")
