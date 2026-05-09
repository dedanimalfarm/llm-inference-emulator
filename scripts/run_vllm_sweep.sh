#!/usr/bin/env bash
# vLLM calibration sweep for the inference emulator.
#
# Run this on a 2× RTX 5090 (or any Blackwell/Hopper) vast.ai instance after
# `pip install "vllm>=0.9.0"`. The output JSON+log pairs are what
# scripts/import_vllm_bench.py and emulator/calibrate.py expect.
#
# What this captures (and why):
#
#   1. MULTI-BATCH SWEEP per (model, TP) — needed to jointly fit β and
#      batch_saturation = (batch_max, batch_50pct). One point per config (the
#      first session) leaves these two parameters mathematically degenerate.
#
#   2. EAGER vs CUDA-graphs delta — the first session ran with --enforce-eager
#      everywhere, which hides the optimization we care about. Running both
#      modes at one matched batch size gives the speedup factor for graphs.
#
#   3. PREFIX CACHE on/off — the design doc claims h≈0.83 reduces TTFT 5×;
#      we verify with a fixed system prompt and 100 different user queries.
#
#   4. FP8 KV-CACHE — Blackwell-native; expect 30-50% decode speedup at long
#      context. Hopper-only optimizations (FlashAttention-3, FP8 weights via
#      Machete) are NOT in this script — they need a Hopper instance.
#
#   5. SPECULATIVE — Llama-3.1-8B with Llama-3.2-1B draft. Single-batch
#      regime where speculative actually pays off. Captures real accept_rate
#      on ShareGPT-style prompts (vs literature 0.7).
#
# Estimated runtime: 4-5 hours on 2× RTX 5090. Cost ~$8-10 on vast.ai.

set -euo pipefail

OUT="${OUT:-$HOME/vllm_results}"
mkdir -p "$OUT" "$OUT/logs"

run() {
    # run <tag> <vllm-bench-args...>
    local tag="$1"; shift
    echo "=== [$tag] $* ===" | tee -a "$OUT/sweep.log"
    vllm bench throughput "$@" \
        --output-json "$OUT/${tag}.json" \
        2>&1 | tee "$OUT/logs/${tag}.log"
}

# Common args for AWQ throughput runs
COMMON_AWQ=(--input-len 1024 --output-len 128)

QWEN7B="Qwen/Qwen2.5-7B-Instruct-AWQ"
QWEN32B="Qwen/Qwen2.5-32B-Instruct-AWQ"
LLAMA70B="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
LLAMA8B="meta-llama/Llama-3.1-8B-Instruct"
LLAMA1B="meta-llama/Llama-3.2-1B-Instruct"

# ============================================================
# 1. Multi-batch sweep — fit β and batch_saturation jointly
#    (5 batch points × 4 configs = 20 runs)
# ============================================================

# Qwen-7B AWQ, TP=1 — fits comfortably on one card
for N in 50 100 200 500 1000; do
    run "qwen7b_tp1_b${N}" --model "$QWEN7B" \
        --tensor-parallel-size 1 --num-prompts "$N" "${COMMON_AWQ[@]}"
done

# Qwen-7B AWQ, TP=2 — for cross-check with TP=1 (TP_efficiency)
for N in 50 100 200 500 1000; do
    run "qwen7b_tp2_b${N}" --model "$QWEN7B" \
        --tensor-parallel-size 2 --num-prompts "$N" "${COMMON_AWQ[@]}"
done

# Qwen-32B AWQ, TP=2 — needs more than 32 GB, must split
for N in 50 100 200; do
    run "qwen32b_tp2_b${N}" --model "$QWEN32B" \
        --tensor-parallel-size 2 --num-prompts "$N" "${COMMON_AWQ[@]}"
done

# Llama-70B AWQ, TP=2 — only fits across both cards
for N in 50 100; do
    run "llama70b_tp2_b${N}" --model "$LLAMA70B" \
        --tensor-parallel-size 2 --num-prompts "$N" "${COMMON_AWQ[@]}"
done

# ============================================================
# 2. CUDA graphs ON vs OFF — delta at fixed batch
# ============================================================

# Default (graphs on) was already covered by the sweep above.
# Run one --enforce-eager pair at a representative batch.
run "qwen7b_tp1_b200_eager" --model "$QWEN7B" \
    --tensor-parallel-size 1 --num-prompts 200 \
    --enforce-eager "${COMMON_AWQ[@]}"

# ============================================================
# 3. Prefix cache on/off — TTFT delta
# ============================================================
#
# The throughput-bench tool uses random prompts, which give ~0% APC hit rate.
# To measure prefix caching we need a fixed system prompt.
# Use vllm bench serving with --dataset-name=sharegpt is one option; here we
# use --dataset-name=random with a fixed `--random-prefix-len` so the first
# `prefix_len` tokens are identical across requests.
#
# Run pair: with prefix caching enabled (default) vs --no-enable-prefix-caching.

# 'shared system prompt' = first 512 tokens identical, rest random
run "qwen7b_tp1_b200_apc_on" --model "$QWEN7B" \
    --tensor-parallel-size 1 --num-prompts 200 \
    --random-prefix-len 512 --random-input-len 256 --output-len 64

run "qwen7b_tp1_b200_apc_off" --model "$QWEN7B" \
    --tensor-parallel-size 1 --num-prompts 200 \
    --random-prefix-len 512 --random-input-len 256 --output-len 64 \
    --no-enable-prefix-caching

# ============================================================
# 4. FP8 KV cache (Blackwell native, Hopper too)
# ============================================================

run "qwen7b_tp1_b200_kvfp8" --model "$QWEN7B" \
    --tensor-parallel-size 1 --num-prompts 200 \
    --kv-cache-dtype fp8_e5m2 "${COMMON_AWQ[@]}"

# Long-context variant — KV pressure where fp8 matters most
run "qwen7b_tp1_b50_kvfp8_8k" --model "$QWEN7B" \
    --tensor-parallel-size 1 --num-prompts 50 \
    --input-len 8192 --output-len 256 \
    --kv-cache-dtype fp8_e5m2

# Same long-context with default fp16 KV for direct comparison
run "qwen7b_tp1_b50_kvfp16_8k" --model "$QWEN7B" \
    --tensor-parallel-size 1 --num-prompts 50 \
    --input-len 8192 --output-len 256

# ============================================================
# 5. Speculative decoding (8B main + 1B draft, batch=1)
# ============================================================

run "spec_8b_1b_b1" --model "$LLAMA8B" \
    --tensor-parallel-size 1 --num-prompts 50 \
    --input-len 256 --output-len 128 \
    --speculative-model "$LLAMA1B" --num-speculative-tokens 4

# Reference run without speculative
run "spec_8b_b1_nospec" --model "$LLAMA8B" \
    --tensor-parallel-size 1 --num-prompts 50 \
    --input-len 256 --output-len 128

# ============================================================
# Done — print summary
# ============================================================

echo ""
echo "=== SWEEP COMPLETE ==="
echo "JSON outputs:"
ls -la "$OUT"/*.json | wc -l
echo "Logs:"
ls -la "$OUT/logs"/*.log | wc -l
echo ""
echo "Next steps (off the GPU instance, on host):"
echo "  scp -P <PORT> root@<IP>:$OUT/'*.json' data/raw_vllm/"
echo "  scp -P <PORT> root@<IP>:$OUT/logs/'*.log' data/raw_vllm/logs/"
echo "  python scripts/import_vllm_bench.py data/raw_vllm/qwen*.json data/raw_vllm/llama*.json \\"
echo "      --hw 2xRTX-5090 --output data/leaderboard-2xRTX-5090-vllm.csv"
echo "  python scripts/run_calibration.py"
