#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# Set the project root directory
PROJECT_ROOT="."

MODEL_PATH=/aifs4su/hansirui_1st/models/Qwen2.5-7B-Instruct
MODEL_NAME=Qwen2.5-7B-Instruct
PORT=8877
HOST=localhost
PP_SIZE=1
TP_SIZE=1
GPU_UTILIZATION=0.95
LIMIT_MM_PER_PROMPT="image=10,video=10"
MAX_SEQ_LEN=32768
# CHAT_TEMPLATE=./scripts/qwen2_5_vl_test-scaling.jinja
ENABLE_PREFIX_CACHING=true
DTYPE=bfloat16
DISABLE_LOG_STATS=true
DISABLE_LOG_REQUESTS=true
DISABLE_FASTAPI_DOCS=true


# benchmark
BENCHMARK=latent_jailbreak
DATA_PATH=benchmarks/latent_jailbreak/
SPLIT=train

# sampling
TEMPERATURE=0.0
TOP_P=0.1
REPETITION_PENALTY=1.0

REASONING=false

# Set other default parameters
RESULTS_DIR="./results"
CACHE_DIR="./cache/${MODEL_NAME}"
NUM_WORKERS=40

export API_KEY="sk-xxx"
export API_BASE=""

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
  --split ${SPLIT} \
  --temperature ${TEMPERATURE} \
  --top-p ${TOP_P} \
  --repetition-penalty ${REPETITION_PENALTY} \
  --results-dir ${RESULTS_DIR} \
  --cache-dir ${CACHE_DIR} \
  --num-workers ${NUM_WORKERS} \
  --data-path ${DATA_PATH} \
  $([ "$ENABLE_PREFIX_CACHING" = "true" ] && echo "--enable-prefix-caching" || echo "") \
  $([ "$DISABLE_LOG_STATS" = "true" ] && echo "--disable-log-stats" || echo "") \
  $([ "$DISABLE_LOG_REQUESTS" = "true" ] && echo "--disable-log-requests" || echo "") \
  $([ "$DISABLE_FASTAPI_DOCS" = "true" ] && echo "--disable-fastapi-docs" || echo "") \
  $([ "$REASONING" = "true" ] && echo "--reasoning" || echo "")









