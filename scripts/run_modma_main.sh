#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT/run"

MODEL_NAME="${MODEL_NAME:-Transfromer}"
EMB_SIZE="${EMB_SIZE:-50}"
DEPTH="${DEPTH:-8}"
NUM_EPOCH="${NUM_EPOCH:-300}"
SEED="${SEED:-1024}"

python3 Cross_MODMA.py \
    --model_name "$MODEL_NAME" \
    --emb_size "$EMB_SIZE" \
    --depth "$DEPTH" \
    --num_epoch "$NUM_EPOCH" \
    --seed "$SEED" \
    --original_data True \
    --sampling_data True
