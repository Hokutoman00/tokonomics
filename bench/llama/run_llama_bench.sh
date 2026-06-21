#!/usr/bin/env bash
# run_llama_bench.sh — build llama.cpp twice (i8mm ON/OFF) and run llama-bench,
# emitting one JSON per variant so the economics layer can price real pp/tg
# (prefill/decode) throughput from a real model on real silicon.
#
# This is the "macro" confirmation of the microkernel "micro" finding: the same
# i8mm ablation that lifts the SMMLA microkernel should lift llama.cpp prefill
# (pp) while leaving decode (tg) ~flat, because tg is memory-bound.
#
# Designed for the free `ubuntu-24.04-arm` runner (Neoverse N2 = i8mm). Runs
# only after push (user/Codex approval); never invoked locally on x86.
set -euo pipefail

MODEL_URL="${MODEL_URL:-https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf}"
OUT_DIR="${OUT_DIR:-results/measured}"
WORK="${WORK:-$(pwd)/.llama-work}"
THREADS="${THREADS:-$(nproc)}"

mkdir -p "$OUT_DIR" "$WORK"
cd "$WORK"

if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi

MODEL_FILE="$WORK/model.gguf"
[ -f "$MODEL_FILE" ] || curl -fL "$MODEL_URL" -o "$MODEL_FILE"

build_and_bench() {
  local label="$1" flags="$2"
  local bdir="build-$label"
  cmake -S llama.cpp -B "$bdir" -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF -DCMAKE_C_FLAGS="$flags" -DCMAKE_CXX_FLAGS="$flags" \
        >/dev/null
  cmake --build "$bdir" --target llama-bench -j "$THREADS" >/dev/null
  # -o json gives machine-readable pp/tg rows
  "$bdir/bin/llama-bench" -m "$MODEL_FILE" -t "$THREADS" \
      -p 512 -n 128 -o json > "$OLDPWD/$OUT_DIR/llama_${label}.json"
  echo "[llama] $label done -> $OUT_DIR/llama_${label}.json"
}

# OFF: dotprod only. ON: enable Armv8.6 MatMulInt8 (i8mm) so KleidiAI/SMMLA path
# is taken for Q4_0 prefill GEMM.
build_and_bench off "-march=armv8.2-a+dotprod"
build_and_bench on  "-march=armv8.6-a+i8mm"

echo "[llama] both variants benched. Prefill(pp) should rise with i8mm; decode(tg) ~flat."
