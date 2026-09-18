#!/bin/bash
# 일반 GGUF(Qwen 등) 판단 서버. 사용: serve_llm.sh <gpu> <ngl> <port> <model.gguf>
G=${1:-0}; NGL=${2:-99}; PORT=${3:-8082}; M=${4:?모델 gguf 경로}
BIN=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}
CUDA_VISIBLE_DEVICES=$G setsid nohup "$BIN" -m "$M" --jinja -c 16384 -b 2048 -ub 512 -np 8 -ngl "$NGL" -t 16 \
  --host 127.0.0.1 --port "$PORT" > "llm_server_$PORT.log" 2>&1 &
echo "pid=$! port=$PORT model=$M"
