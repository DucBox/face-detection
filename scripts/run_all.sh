#!/bin/bash
# Full pipeline: setup -> baseline eval of pretrained -> train -> eval best & last checkpoints.
# Usage: NAME=exp1 EPOCHS=300 BATCH=8 IMG=640 GPUS=0 bash scripts/run_all.sh [extra tools/train.py args]
# Run detached:  nohup bash scripts/run_all.sh > run_all.log 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"
NAME="${NAME:-exp}"
export NAME GPUS   # eval_widerface.sh runs on the first id of GPUS

bash scripts/setup.sh

if [ ! -f runs/inference/widerface_baseline_pretrained/widerface_result.txt ]; then
  log "baseline: pretrained/yolov6s_face.pt (expected ~96.2 / 94.7 / 85.1)"
  bash scripts/eval_widerface.sh pretrained/yolov6s_face.pt widerface_baseline_pretrained
fi

bash scripts/train.sh "$@"
RUN_DIR=$(ls -td runs/train/"$NAME"*/ | head -1)

for ck in best_ckpt last_ckpt; do
  bash scripts/eval_widerface.sh "${RUN_DIR}weights/$ck.pt" "widerface_$(basename "$RUN_DIR")_$ck"
done

log "SUMMARY (WIDER val AP)"
for d in runs/inference/widerface_baseline_pretrained runs/inference/widerface_"$(basename "$RUN_DIR")"_*; do
  echo "== $(basename "$d")"; grep "Val AP" "$d/widerface_result.txt"
done
