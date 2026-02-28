#!/usr/bin/env bash
# Train CCAN (CMLM + context-aware cross-attention) on WMT14 En-De.
# Env: DATA_DIR (path to binarized data), SAVE_DIR (checkpoint root), [GPU].
set -e
DATA_DIR="${DATA_DIR:?Set DATA_DIR=path/to/databin}"
SAVE_DIR="${SAVE_DIR:?Set SAVE_DIR=path/to/save}"
RUN_NAME="${1:?Usage: $0 <run_name>}"
GPU="${GPU:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${SAVE_DIR}/${RUN_NAME}"
log_dir="${SAVE_DIR}/${RUN_NAME}.log"
mkdir -p "$(dirname "$model_dir")"

if [ -n "$GPU" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

python "$ROOT/train.py" "$DATA_DIR" \
  --arch bert_transformer_seq2seq \
  --share-all-embeddings \
  --criterion label_smoothed_length_cross_entropy \
  --label-smoothing 0.1 \
  --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 \
  --task translation_self \
  --max-tokens 8192 --update-freq 8 --weight-decay 0.01 --dropout 0.2 \
  --encoder-layers 6 --encoder-embed-dim 512 \
  --decoder-layers 6 --decoder-embed-dim 512 \
  --fp16 --ddp-backend=no_c10d \
  --max-source-positions 10000 --max-target-positions 10000 \
  --max-update 300000 --seed 1 \
  --save-dir "$model_dir" \
  --keep-last-epochs 1 \
  --no-progress-bar --log-format simple --log-interval 100 \
  --save-interval-updates 4000 \
  --decoder-layers-to-apply-local '1,2,3,4,5,6' \
  --win-size 9 \
  2>&1 | tee "$log_dir"
