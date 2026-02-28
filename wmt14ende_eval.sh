#!/usr/bin/env bash
# Decode and compute BLEU. Env: DATA, CHECKPOINT, [RUN_NAME], [GPU], [SUBSET], [OUT].
set -e
RUN_NAME="${1:-default}"
GPU="${2:-0}"
DATA="${DATA:-databin/ende/distill_de}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=path/to/checkpoint_dir}"
CKPT="${CKPT:-checkpoint_best.pt}"
SUBSET="${SUBSET:-test}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-wmt14ende_rst/${RUN_NAME}}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="$GPU"
python "$ROOT/generate_cmlm.py" "$DATA" \
  --path "$CHECKPOINT/$CKPT" \
  --task translation_self \
  --gen-subset "$SUBSET" \
  --remove-bpe \
  --decoding-iterations 10 \
  --decoding-strategy mask_predict \
  --max-sentences 90 \
  > "$OUT/out.out" 2>&1

if [ -n "$REF" ] && [ -f "$REF" ]; then
  grep ^H "$OUT/out.out" | sed 's/^H\-//' | sort -n -k 1 | awk -F '\t' '{print $NF}' > "$OUT/out.hyp"
  sacrebleu "$REF" -i "$OUT/out.hyp" -b
else
  bash "$ROOT/scripts/BLEU.sh" wmt14/full en de "$OUT/out.out"
fi
