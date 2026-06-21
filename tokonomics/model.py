"""First-order roofline model of LLM inference on a CPU.

This is the standard decode-vs-prefill decomposition (see the roofline
literature, e.g. arXiv:2402.16363 "LLM Inference Unveiled"): given the two
hardware ceilings (compute GOP/s, memory GB/s) and a model's size, we derive
the attainable prefill and decode throughput. No fabricated tok/s — the
numbers fall out of measured ceilings and published model dimensions.

Why prefill and decode behave differently:

  decode (one token at a time): every generated token must stream the full
    weight matrix from memory once. bytes/token ~= model_bytes, FLOPs/token
    ~= 2*params. arithmetic intensity AI = 2*params / model_bytes is *low*,
    so decode sits on the memory ceiling:  tok/s_decode = mem_bw / model_bytes.
    => raising the compute ceiling (i8mm) does almost nothing for decode.

  prefill (many prompt tokens at once): weights are reused across the prompt,
    so AI is *high* and prefill sits on the compute ceiling:
    tok/s_prefill = peak_ops / (2*params).
    => i8mm (which lifts the compute ceiling) speeds prefill up.

That asymmetry is the headline finding the lab makes reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A quantised decoder-only LLM, the unit of work we price.

    name        : human label
    params      : total parameter count
    bytes_per_param : effective bytes/weight after quantisation
                      (Q4_0 4-bit ~= 0.5 + a little for scales ~= 0.56)
    """

    name: str
    params: float
    bytes_per_param: float

    @property
    def model_bytes(self) -> float:
        return self.params * self.bytes_per_param

    @property
    def decode_ai(self) -> float:
        """OP/byte for decode — low, hence memory-bound."""
        return (2.0 * self.params) / self.model_bytes

    def prefill_ai(self, prompt_len: float) -> float:
        """OP/byte for prefill — high because each weight load is amortised
        across `prompt_len` prompt tokens. Finite (not inf), so the roofline
        min(compute, mem_bw*AI) is genuinely evaluated rather than assumed
        compute-bound: at a long prompt this is compute-bound, but a very short
        prompt correctly falls back toward the memory ceiling.
        """
        return self.decode_ai * prompt_len


# A couple of small CPU-friendly models, expressed only by public dimensions.
MODELS = {
    "llama-3.2-1b-q4_0": ModelSpec("Llama-3.2-1B Q4_0", 1.24e9, 0.56),
    "qwen2.5-0.5b-q4_0": ModelSpec("Qwen2.5-0.5B Q4_0", 0.49e9, 0.56),
    "llama-3.1-8b-q4_0": ModelSpec("Llama-3.1-8B Q4_0", 8.03e9, 0.56),
}


def decode_tok_s(mem_bw_gbs: float, model: ModelSpec,
                 peak_gops: float | None = None) -> float:
    """Decode throughput (tokens/s) on the roofline.

    Decode's arithmetic intensity is low (`decode_ai = 2/bytes_per_param`), so the
    attainable rate is `min(compute ceiling, mem_bw * AI)` — which the roofline
    *picks* as the memory term on every realistic instance. Pass `peak_gops` to
    evaluate that min() genuinely (so the "decode is memory-bound" claim is
    derived, not assumed: an implausible tiny-bandwidth / huge-compute machine
    would correctly stay compute-capped). With `peak_gops=None` we return the
    memory ceiling directly — the closed form of the same quantity in the
    memory-bound regime (`mem_bw / model_bytes`)."""
    mem_ceiling = (mem_bw_gbs * 1e9) / model.model_bytes
    if peak_gops is None:
        return mem_ceiling
    attainable = attainable_gops(peak_gops, mem_bw_gbs, model.decode_ai)
    return (attainable * 1e9) / (2.0 * model.params)


# Prompt length over which prefill amortises each weight load. Long enough that
# realistic instances are compute-bound (so i8mm helps), but the memory ceiling
# is still genuinely checked via the roofline below.
PREFILL_PROMPT_LEN = 512


def prefill_tok_s(peak_gops: float, mem_bw_gbs: float, model: ModelSpec,
                  prompt_len: float = PREFILL_PROMPT_LEN) -> float:
    """Roofline prefill throughput (tokens/s).

    Uses the *roofline* attainable rate min(compute ceiling, mem_bw * AI) with a
    finite prefill arithmetic intensity, not an unconditional compute-bound
    assumption. At PREFILL_PROMPT_LEN on real instances this is compute-bound, so
    i8mm lifts it; a short prompt would correctly become memory-bound.
    """
    ai = model.prefill_ai(prompt_len)
    attainable = attainable_gops(peak_gops, mem_bw_gbs, ai)
    return (attainable * 1e9) / (2.0 * model.params)


def attainable_gops(peak_gops: float, mem_bw_gbs: float, ai: float) -> float:
    """Roofline: min(compute ceiling, memory ceiling * arithmetic intensity)."""
    return min(peak_gops, mem_bw_gbs * ai)
