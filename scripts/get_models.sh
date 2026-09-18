#!/bin/bash
# 모델 내려받기. 사용: get_models.sh [bonsai|qwen4b|qwen27b|all]
set -e
W=${1:-bonsai}; mkdir -p models
dl() { huggingface-cli download "$1" "$2" --local-dir models/; }
case "$W" in
  bonsai|all) dl prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf ;;&
  qwen4b|all) dl unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q8_0.gguf ;;&
  qwen27b|all) dl unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf ;;&
esac
ls -la models/
