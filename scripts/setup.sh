#!/bin/bash
# Offline setup: check python deps, build the WIDER eval extension, unpack the dataset from datasets_pack/.
# Usage: bash scripts/setup.sh            (env: DATA_ROOT, PYTHON)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

log "1/3 python deps ($($PYTHON --version))"
$PYTHON - <<'EOF'
import importlib.util, sys
mods = "torch torchvision numpy cv2 yaml scipy tqdm addict tensorboard pycocotools thop matplotlib requests PIL".split()
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"MISSING python packages: {missing}  (pip names: opencv-python-headless for cv2, PyYAML for yaml, Pillow for PIL)")
import torch
print(f"  torch {torch.__version__}  cuda={torch.version.cuda}  gpus={torch.cuda.device_count()}")
EOF

log "2/3 build widerface_evaluate/bbox (optional: evaluation.py falls back to numpy)"
( cd widerface_evaluate && $PYTHON setup.py build_ext --inplace > /dev/null 2>&1 && rm -rf build && echo "  built bbox" ) \
  || echo "  could not build bbox (no Cython/compiler) -> numpy fallback will be used"

log "3/3 dataset -> $DATA_ROOT"
if [ -f "$DATA_ROOT/.unpacked_ok" ]; then
  echo "  already unpacked, skip"
else
  ( cd datasets_pack && shasum -a 256 -c SHA256SUMS --quiet 2>/dev/null || sha256sum -c SHA256SUMS --quiet ) \
    || { echo "checksum mismatch in datasets_pack/ (git lfs pull not done?)"; exit 1; }
  mkdir -p "$(dirname "$DATA_ROOT")"
  tmp="$(dirname "$DATA_ROOT")/.unpack_tmp"; rm -rf "$tmp"; mkdir -p "$tmp"
  for s in train val; do cat datasets_pack/widerface_$s.tar.part_* | tar -xf - -C "$tmp"; done
  rm -rf "$DATA_ROOT"; mv "$tmp/widerface" "$DATA_ROOT"; rm -rf "$tmp"
  for s in train val; do
    echo "  $s: $(find "$DATA_ROOT/images/$s" -name '*.jpg' | wc -l) images, $(find "$DATA_ROOT/labels/$s" -name '*.txt' | wc -l) labels"
  done
  touch "$DATA_ROOT/.unpacked_ok"
fi
write_data_yaml
log "setup done. data yaml: $DATA_YAML"
