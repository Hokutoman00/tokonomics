"""Tokonomics — the Arm64 LLM-inference economics lab.

Measures tokens-per-dollar and tokens-per-joule across Arm Neoverse
generations, isolates the i8mm contribution *within the same silicon*,
and decomposes prefill/decode on a roofline so you can see *why* the
newest instance is not always the cheapest.

Confidence labels (see README):
  - measured   : produced by the on-silicon harness (results/measured/**)
  - dev        : pipeline validated locally on x86 float proxies (results/x86-dev/**)
  - projection : derived from published specs (results/projected/**), CI replaces it
"""

__version__ = "0.1.0"
