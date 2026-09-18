#!/bin/bash
# Ternary Bonsai 2 27B 판단 서버 — PrismML llama.cpp 포크 빌드가 필요하다(스톡은 경고 없이 헛소리를 낸다).
# 사용: serve_bonsai.sh <gpu> <port> <model.gguf> [llama-server 경로]
G=${1:-0}; PORT=${2:-8083}; M=${3:-models/Ternary-Bonsai-2-27B-PQ2_0.gguf}
BIN=${4:-${LLAMA_SERVER:-llama_prism/build/bin/llama-server}}
CUDA_VISIBLE_DEVICES=$G setsid nohup "$BIN" -m "$M" --jinja -c 16384 -b 2048 -ub 512 -np 8 -ngl 99 -t 16 \
  --host 127.0.0.1 --port "$PORT" > "bonsai_server_$PORT.log" 2>&1 &
echo "pid=$! port=$PORT model=$M"; echo "확인: curl -s http://127.0.0.1:$PORT/health"
