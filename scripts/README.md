# YOLOv6s-face: continue training on WIDER FACE (offline server)

Code is tag `0.3.1` of `meituan/YOLOv6` branch `yolov6-face` (the version that produced the released
`yolov6s_face.pt`), plus small fixes for torch>=2.6 / numpy 2 / CPU. Everything needed is in the repo,
no network access required:

| Path | Content |
|---|---|
| `pretrained/yolov6s_face.pt` | released YOLOv6s-face weights (WIDER val 96.2 / 94.7 / 85.1) |
| `configs/yolov6s_face_continue.py` | `yolov6s_finetune.py` with `pretrained` pointing to the file above |
| `datasets_pack/` | WIDER FACE train (12880) + val (3226) images and YOLO-face labels (15 cols, 5 landmarks), tar split in 95MB parts + `SHA256SUMS` |
| `widerface_evaluate/ground_truth/` | official Easy / Medium / Hard val ground truth |

## Run

```bash
bash scripts/setup.sh                       # check deps, build eval ext, unpack data to ./datasets/widerface
NAME=exp1 EPOCHS=300 BATCH=8 IMG=640 GPUS=0 nohup bash scripts/run_all.sh > run_all.log 2>&1 &
```

`run_all.sh` = setup -> baseline eval of the pretrained model -> train -> Easy/Medium/Hard eval of
`best_ckpt.pt` and `last_ckpt.pt`, then prints a summary. The steps can also be run separately:

```bash
NAME=exp1 EPOCHS=300 BATCH=8 IMG=640 GPUS=0 bash scripts/train.sh     # GPUS=0,1,2,3 -> torchrun DDP
bash scripts/eval_widerface.sh runs/train/exp1/weights/best_ckpt.pt       # -> runs/inference/<name>/widerface_result.txt
```

Env vars: `GPUS` (physical GPU ids, sets `CUDA_VISIBLE_DEVICES`; train uses all, eval uses the first; `GPUS=cpu` for CPU), `DATA_ROOT` (default `./datasets/widerface`), `PYTHON`, `WORKERS`, `EVAL_INTERVAL`, `CONF`,
`CONF_THRES` (eval, default 0.02). Extra args after the script name go straight to `tools/train.py`.

Python packages needed (install from a local wheel mirror on the server): torch, torchvision, numpy,
opencv-python(-headless), PyYAML, scipy, tqdm, addict, tensorboard, pycocotools, thop, matplotlib,
requests, Pillow. Optional: Cython (fast eval; otherwise a numpy fallback with identical results), seaborn.

Notes
- `best_ckpt.pt` is selected on the same val set that is reported, so `last_ckpt.pt` is the unbiased number.
- Solver in the config (lr0=0.0032, 300 epochs) is the COCO->WIDER finetune recipe, not tuned for
  continuing from an already converged face model.
- Measured on an RTX 4060 8GB: batch 8 / img 800 OOMs in label assignment (falls back to slow CPU
  batches); batch 1 / img 640 uses 1.3GB at ~3 it/s. Larger batches at img 640 were not measured yet.
  If the log shows `CPU Mode for This Batch`, lower BATCH or IMG.
