#!/bin/bash
# Continue training YOLOv6s-face from pretrained/yolov6s_face.pt on WIDER FACE.
# Usage: NAME=exp1 EPOCHS=300 BATCH=8 IMG=640 GPUS=0 bash scripts/train.sh [extra tools/train.py args]
#   GPUS=0,1,2,3 -> DDP via torchrun, BATCH is the total batch across GPUs.
#   tools/train.py sets CUDA_VISIBLE_DEVICES from --device itself, so --device gets the same physical ids.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

NAME="${NAME:-exp}"
CONF="${CONF:-configs/yolov6s_face_continue.py}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-8}"
IMG="${IMG:-640}"
WORKERS="${WORKERS:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"

[ -f "$DATA_ROOT/.unpacked_ok" ] || { echo "dataset not ready at $DATA_ROOT -> run scripts/setup.sh first"; exit 1; }
write_data_yaml
mkdir -p runs/train
LOG="runs/train/$NAME.log"

ARGS=(tools/train.py --conf-file "$CONF" --data-path "$DATA_YAML" --fuse_ab
      --epochs "$EPOCHS" --batch-size "$BATCH" --img-size "$IMG" --workers "$WORKERS"
      --device "$GPUS" --eval-interval "$EVAL_INTERVAL" --name "$NAME" "$@")
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
if [ "$NGPU" -gt 1 ]; then
  CMD=(torchrun --nproc_per_node "$NGPU" "${ARGS[@]}")
else
  CMD=("$PYTHON" "${ARGS[@]}")
fi

log "train on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES: ${CMD[*]}"
log "log file: $LOG"
COLUMNS=200 PYTHONWARNINGS=ignore "${CMD[@]}" 2>&1 | tee "$LOG"
# train.py appends a number when the name exists (exp, exp1, ...): report the newest run dir
RUN_DIR=$(ls -td runs/train/"$NAME"*/ | head -1)
log "done. checkpoints: ${RUN_DIR}weights/"
