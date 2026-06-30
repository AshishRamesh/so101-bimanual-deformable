#!/usr/bin/env bash
# Fine-tune the sim tops model (tops_fold_80k) on YOUR OWN recorded data
# (Pandeyjiya024/so101_cups_merged — "fold the shirt", your robot/cameras).
# Camera keys + shapes + fps already match the model, so NO rename_map needed.
# This is the best path to real-world folding on your rig.
#
# Runs in the venv. Run AFTER ./setup.sh.   ->   ./finetune_mydata.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venv-so101}"
PRETRAINED="${PRETRAINED:-AshishRamesh/tops_fold_80k}"   # sim model; or a local path to its pretrained_model
DATA_REPO="Pandeyjiya024/so101_cups_merged"
DATA_DIR="Datasets/example/mydata"
CFG="configs/train_mydata_finetune.yaml"

# ---- knobs: ~15 epochs over 58 eps / ~67k frames (small data -> fine-tune, don't over-train) ----
BATCH=16
STEPS=60000           # ~14 epochs @ batch 16  (5090: ~4-5 h)
SAVE_FREQ=6000        # ~every 1.5 epochs -> ~10 checkpoints (best is likely EARLY)
LR=3.0e-5             # low LR: adapt to your domain without erasing sim-learned folding
# -------------------------------------------------------------------------------------------------

# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$DATA_DIR" configs logs outputs

echo "==> [1/3] download your dataset + verify it loads"
hf download "$DATA_REPO" --repo-type dataset --local-dir "$DATA_DIR"
python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("x", root="$DATA_DIR")
print(f"    loaded: {ds.num_episodes} eps, {ds.num_frames} frames")
_ = ds[0]; _ = ds[ds.num_frames - 1]
print("    frames readable OK")
PY

echo "==> [2/3] write fine-tune config -> $CFG"
cat > "$CFG" <<YAML
dataset:
  repo_id: $DATA_REPO
  root: $DATA_DIR
  image_transforms:                 # light aug for lighting robustness
    enable: true
    max_num_transforms: 3
    random_order: true
    tfs:
      brightness: { type: ColorJitter, weight: 1.0, kwargs: { brightness: [0.7, 1.3] } }
      contrast:   { type: ColorJitter, weight: 1.0, kwargs: { contrast:   [0.7, 1.3] } }
      saturation: { type: ColorJitter, weight: 1.0, kwargs: { saturation: [0.7, 1.3] } }

# NO rename_map — your dataset already uses top_rgb/left_rgb/right_rgb (matches the model)

policy:
  type: smolvla
  pretrained_path: $PRETRAINED
  device: cuda
  push_to_hub: false
  optimizer_lr: $LR
  scheduler_warmup_steps: 300
  scheduler_decay_steps: $STEPS
  optimizer_weight_decay: 1.0e-4

  input_features:
    observation.state:            { type: STATE,  shape: [12] }
    observation.images.top_rgb:   { type: VISUAL, shape: [3, 480, 640] }
    observation.images.left_rgb:  { type: VISUAL, shape: [3, 480, 640] }
    observation.images.right_rgb: { type: VISUAL, shape: [3, 480, 640] }
  output_features:
    action: { type: ACTION, shape: [12] }

output_dir: outputs/train/finetune_mydata
batch_size: $BATCH
steps: $STEPS
save_freq: $SAVE_FREQ
log_freq: 100
num_workers: 8

wandb:
  enable: false
YAML

echo "==> [3/3] SMOKE TEST (20 steps) then full fine-tune"
rm -rf outputs/train/finetune_mydata_smoke
if lerobot-train --config_path="$CFG" --steps=20 --save_freq=20 \
     --output_dir=outputs/train/finetune_mydata_smoke >logs/smoke_mydata.log 2>&1; then
  echo "    smoke test PASSED"; rm -rf outputs/train/finetune_mydata_smoke
else
  echo "    !! SMOKE TEST FAILED — see logs/smoke_mydata.log"; tail -25 logs/smoke_mydata.log; exit 1
fi

echo "==> launching full fine-tune on YOUR data ($STEPS steps, ~14 epochs)"
echo "    58 eps is small -> best checkpoint is likely EARLY; save_freq=$SAVE_FREQ, robot-test a few"
lerobot-train --config_path="$CFG" 2>&1 | tee "logs/finetune_mydata_$(date +%Y%m%d_%H%M%S).log"
