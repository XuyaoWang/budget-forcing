#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6,7
export VLLM_USE_V1=0 # Set environment variable to disable V1 engine, so that disable_frontend_multiprocessing parameter can take effect
export CUDA_VISIBLE_DEVICES=4,5,6,7
# Set the project root directory
PROJECT_ROOT="."

MODEL_PATH=/aifs4su/yaodong/models/Qwen2.5-VL-7B-Instruct
MODEL_NAME=vl-rethinker-7b
PORT=8010
HOST=localhost
TP_SIZE=4
GPU_UTILIZATION=0.9
LIMIT_MM_PER_PROMPT="image=10,video=10"
CHAT_TEMPLATE=/aifs4su/yaodong/xuyao/machine-scientist/workshop-s1_v/simple_budget_forcing/scripts/qwen2_5_vl_test-scaling.jinja
ENABLE_PREFIX_CACHING=true
DTYPE=bfloat16
DISABLE_LOG_STATS=true
DISABLE_LOG_REQUESTS=true
DISABLE_FASTAPI_DOCS=true


# benchmark
BENCHMARK=mathvision
DATA_PATH=MathLLMs/MathVision
SPLIT=test

# sampling
TEMPERATURE=0.0
TOP_P=0.1
REPETITION_PENALTY=1.0


# Set other default parameters
RESULTS_DIR="./results"
CACHE_DIR="./cache"
NUM_WORKERS=10
NUM_IGNORE=8
MAX_TOKENS_THINKING=32000

# Execute the main script with vLLM backend
# python main.py \
#   --model-path ${MODEL_PATH} \
#   --model-name ${MODEL_NAME} \
#   --port ${PORT} \
#   --host ${HOST} \
#   --tensor-parallel-size ${TP_SIZE} \
#   --gpu-memory-utilization ${GPU_UTILIZATION} \
#   --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
#   --chat-template ${CHAT_TEMPLATE} \
#   --dtype ${DTYPE} \
#   --benchmark ${BENCHMARK} \
#   --data-path ${DATA_PATH} \
#   --split ${SPLIT} \
#   --temperature ${TEMPERATURE} \
#   --top-p ${TOP_P} \
#   --repetition-penalty ${REPETITION_PENALTY} \
#   --results-dir ${RESULTS_DIR} \
#   --cache-dir ${CACHE_DIR} \
#   --num-workers ${NUM_WORKERS} \
#   --num-ignore ${NUM_IGNORE} \
#   --max-tokens-thinking ${MAX_TOKENS_THINKING} \
#   ${ENABLE_PREFIX_CACHING:+--enable-prefix-caching} \
#   ${DISABLE_LOG_STATS:+--disable-log-stats} \
#   ${DISABLE_LOG_REQUESTS:+--disable-log-requests} \
#   ${DISABLE_FASTAPI_DOCS:+--disable-fastapi-docs} \
#   ${REASONING:+--reasoning}

python main.py \
  --model-path ${MODEL_PATH} \
  --model-name ${MODEL_NAME} \
  --port ${PORT} \
  --host ${HOST} \
  --tensor-parallel-size ${TP_SIZE} \
  --gpu-memory-utilization ${GPU_UTILIZATION} \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
  --dtype ${DTYPE} \
  --benchmark ${BENCHMARK} \
  --data-path ${DATA_PATH} \
  --split ${SPLIT} \
  --temperature ${TEMPERATURE} \
  --top-p ${TOP_P} \
  --repetition-penalty ${REPETITION_PENALTY} \
  --results-dir ${RESULTS_DIR} \
  --cache-dir ${CACHE_DIR} \
  --num-workers ${NUM_WORKERS} \
  ${ENABLE_PREFIX_CACHING:+--enable-prefix-caching} \
  ${DISABLE_LOG_STATS:+--disable-log-stats} \
  ${DISABLE_LOG_REQUESTS:+--disable-log-requests} \
  ${DISABLE_FASTAPI_DOCS:+--disable-fastapi-docs}
