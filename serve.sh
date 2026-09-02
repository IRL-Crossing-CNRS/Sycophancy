#!/usr/bin/env bash
# vLLM launch commands for the sycophancy surrender runs. Run this ON harold.
#
#   ./serve.sh check              # GPU + vllm version, run this first
#   ./serve.sh ministral          # then one of the five models below
#
# Binds 127.0.0.1:18080 to match the harold-llm SSH forward in ~/.ssh/config
# (LocalForward 8080 localhost:18080), so the endpoint is reachable only through
# the tunnel and not from the rest of the lab network.
#
# From your laptop, once the tunnel is up (ssh -N harold-llm):
#   python run_surrender_inference.py --model ministral-8b

set -euo pipefail

HOST=127.0.0.1
PORT=18080

# All five models are already in the HF cache; offline mode stops vLLM from
# re-authenticating against the gated Ministral and Llama repos.
export HF_HUB_OFFLINE=1

# --max-logprobs 25: server-side ceiling on requested logprobs. Default is 20 and the
#   client asks for 20, which is exactly on the boundary; some vLLM versions count the
#   sampled token against it and reject every request. Costs nothing to raise.
# --max-model-len 4096: longest surrender prompt is ~460 tokens. The stock 32k-128k
#   context would reserve enormous KV cache for nothing.
COMMON=(--host "$HOST" --port "$PORT" --max-logprobs 25 --max-model-len 4096)

case "${1:-}" in

check)
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
  vllm --version
  echo "--- flag names drift between releases; confirm these exist: ---"
  vllm serve --help | grep -E 'max-logprobs|max-model-len|served-model-name' || true
  ;;

ministral)          # ~16GB bf16
  vllm serve mistralai/Ministral-8B-Instruct-2410 \
    --served-model-name ministral-8b \
    "${COMMON[@]}" --gpu-memory-utilization 0.90
  ;;

qwen25)             # ~15GB bf16. Qwen2.5 has no thinking mode:
                    # do NOT pass --disable-thinking on the client for this one.
  vllm serve Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen25-7b \
    "${COMMON[@]}" --gpu-memory-utilization 0.90
  ;;

llama)              # ~16GB bf16
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --served-model-name llama31-8b \
    "${COMMON[@]}" --gpu-memory-utilization 0.90
  ;;

mistral-small)      # ~47GB bf16 -- needs an 80GB card, or add --tensor-parallel-size 2.
                    # Vision model: the mistral-format flags avoid a tokenizer mismatch
                    # in the HF path, and limit-mm-per-prompt skips the vision tower.
  vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
    --served-model-name mistral-small-24b \
    --tokenizer-mode mistral --config-format mistral --load-format mistral \
    --limit-mm-per-prompt '{"image":0}' \
    "${COMMON[@]}" --gpu-memory-utilization 0.92
  ;;

uncensored)         # ~27GB on Hopper/Ada. On Ampere (A100) vLLM dequantizes FP8 to
                    # bf16, so budget ~54GB instead. Verify the repo id first.
                    # If replies open with <think>, add --disable-thinking on the client.
  vllm serve orcarouter/Qwen3.8-27B-Uncensored-FP8 \
    --served-model-name qwen-uncensored \
    "${COMMON[@]}" --gpu-memory-utilization 0.92
  ;;

*)
  echo "usage: $0 {check|ministral|qwen25|llama|mistral-small|uncensored}" >&2
  exit 1
  ;;
esac
