#!/usr/bin/env bash
export HF_HOME=/workspace/hf_cache
export VLLM_USE_V1=0
export OPENBLAS_NUM_THREADS=1

echo "=== 2. Serving Benchmark (Qwen-32B) ==="
# Start server in background
taskset -c 0-63 vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ -tp 2 --disable-log-stats --max-model-len 32768 > ~/vllm_results/logs/serve_32b.log 2>&1 &
SERVER_PID=$!

echo "Waiting for server to start..."
for i in {1..150}; do
    curl -s http://localhost:8000/v1/models >/dev/null && break
    sleep 2
done

for Q in 2 5 10; do
    echo "Benchmarking QPS=$Q..."
    python3 -m vllm.benchmarks.benchmark_serving --model Qwen/Qwen2.5-32B-Instruct-AWQ \
        --dataset-name random --random-input-len 256 --random-output-len 64 \
        --num-prompts 200 --request-rate $Q \
        --save-result > ~/vllm_results/serving_32b_qps${Q}.json 2> ~/vllm_results/logs/serving_32b_qps${Q}.log
done

kill -9 $SERVER_PID
echo "=== BONUS COMPLETE ==="
