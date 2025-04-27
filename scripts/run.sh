#!/bin/bash

export VLLM_USE_V1=0
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# Set the project root directory
PROJECT_ROOT="."

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
MODEL_NAME=qwen-vl-3b
PORT=8010
HOST=localhost
PP_SIZE=1
TP_SIZE=4
GPU_UTILIZATION=0.95
LIMIT_MM_PER_PROMPT="image=10,video=10"
MAX_SEQ_LEN=32768
CHAT_TEMPLATE=./scripts/qwen2_5_vl_test-scaling.jinja
ENABLE_PREFIX_CACHING=true
DTYPE=bfloat16
DISABLE_LOG_STATS=true
DISABLE_LOG_REQUESTS=true
DISABLE_FASTAPI_DOCS=true


# benchmark
BENCHMARK=mathvista
DATA_PATH=AI4Math/MathVista
SPLIT=testmini

# sampling
TEMPERATURE=0.0
TOP_P=0.1
REPETITION_PENALTY=1.0

REASONING=false

# Set other default parameters
RESULTS_DIR="./results"
CACHE_DIR="./cache/${MODEL_NAME}"
NUM_WORKERS=100
NUM_IGNORE=8
MAX_TOKENS_THINKING=32000

python main.py \
  --model-path ${MODEL_PATH} \
  --model-name ${MODEL_NAME} \
  --port ${PORT} \
  --host ${HOST} \
  --pipeline-parallel-size ${PP_SIZE} \
  --tensor-parallel-size ${TP_SIZE} \
  --gpu-memory-utilization ${GPU_UTILIZATION} \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
  --max-seq-len ${MAX_SEQ_LEN} \
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
  --num-ignore ${NUM_IGNORE} \
  --max-tokens-thinking ${MAX_TOKENS_THINKING} \
  $([ "$ENABLE_PREFIX_CACHING" = "true" ] && echo "--enable-prefix-caching" || echo "") \
  $([ "$DISABLE_LOG_STATS" = "true" ] && echo "--disable-log-stats" || echo "") \
  $([ "$DISABLE_LOG_REQUESTS" = "true" ] && echo "--disable-log-requests" || echo "") \
  $([ "$DISABLE_FASTAPI_DOCS" = "true" ] && echo "--disable-fastapi-docs" || echo "") \
  $([ "$REASONING" = "true" ] && echo "--reasoning" || echo "")









