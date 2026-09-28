#!/bin/bash
# WIDER FACE val benchmark (Easy / Medium / Hard AP), same protocol as the README numbers.
# Usage: GPUS=0 bash scripts/eval_widerface.sh <weights.pt> [name]     (env: IMG=640 CONF_THRES=0.02)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"
# single-GPU inference on the first id of GPUS; infer.py indexes cuda:<n> inside the visible set -> 0
if [ "$GPUS" = "cpu" ]; then
  DEVICE=cpu
else
  export CUDA_VISIBLE_DEVICES="${GPUS%%,*}"; DEVICE=0
fi

WEIGHTS="${1:?usage: eval_widerface.sh <weights.pt> [name]}"
NAME="${2:-widerface_$(basename "$(dirname "$(dirname "$WEIGHTS")")")_$(basename "$WEIGHTS" .pt)}"
IMG="${IMG:-640}"
CONF_THRES="${CONF_THRES:-0.02}"
OUT="runs/inference/$NAME"

[ -f "$DATA_ROOT/.unpacked_ok" ] || { echo "dataset not ready at $DATA_ROOT -> run scripts/setup.sh first"; exit 1; }
write_data_yaml
rm -rf "$OUT"

log "1/2 inference on val (3226 images, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES device=$DEVICE): $WEIGHTS -> $OUT"
PYTHONWARNINGS=ignore $PYTHON tools/infer.py --weights "$WEIGHTS" --source "$DATA_ROOT/images/val" --yaml "$DATA_YAML" \
  --img-size "$IMG" "$IMG" --conf-thres "$CONF_THRES" --device "$DEVICE" \
  --not-save-img --save-txt-widerface --project runs/inference --name "$NAME"
n=$(find "$OUT/labels/val" -name '*.txt' | wc -l)
[ "$n" -eq 3226 ] || { echo "expected 3226 prediction files, got $n"; exit 1; }

log "2/2 WIDER evaluation"
( cd widerface_evaluate && PYTHONWARNINGS=ignore $PYTHON evaluation.py --pred "../$OUT/labels/val/" 2>&1 \
    | tr '\r' '\n' | grep -A5 "Results" ) | tee "$OUT/widerface_result.txt"
log "saved: $OUT/widerface_result.txt"
