#!/usr/bin/env bash
# Fine-tune the 4-GARMENT sim model (smolvla-4type-fold-test) on YOUR OWN data
# (Pandeyjiya024/so101_cups_merged — "fold the shirt", your robot/cameras).
# The 4-type model already transfers relatively well to real (more garment diversity
# in sim -> better generalization), so adapting IT to your domain is the best bet.
# Camera keys/shapes/fps match -> NO rename_map needed.
#
# Runs in the venv. Run AFTER ./setup.sh.   ->   ./finetune_4type_mydata.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venv-so101}"
PRETRAINED="${PRETRAINED:-AshishRamesh/smolvla-4type-fold-test}"   # the 4-type sim model (best real transfer)
DATA_REPO="Pandeyjiya024/so101_cups_merged"
DATA_DIR="Datasets/example/mydata"
CFG="configs/train_mydata_4type.yaml"

# ---- knobs: fine-tune a skilled model on 58 eps -> low LR, modest epochs, pick early ----
BATCH=16
STEPS=60000           # ~14 epochs @ batch 16 (5090: ~4-5 h)
SAVE_FREQ=6000        # ~every 1.5 epochs -> ~10 checkpoints (best likely EARLY)
LR=3.0e-5             # low LR: adapt to your domain without erasing learned folding
# ----------------------------------------------------------------------------------------

# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$DATA_DIR" configs logs outputs

echo "==> [1/3] ensure your dataset is present + loads"
if [ -f "$DATA_DIR/meta/info.json" ]; then echo "    dataset present"; else
  hf download "$DATA_REPO" --repo-type dataset --local-dir "$DATA_DIR"; fi
python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("x", root="$DATA_DIR")
print(f"    loaded: {ds.num_episodes} eps, {ds.num_frames} frames")
PY

echo "==> [2/3] write config -> $CFG"
cat > "$CFG" <<YAML
dataset:
  repo_id: $DATA_REPO
  root: $DATA_DIR
  image_transforms:
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

output_dir: outputs/train/finetune_4type_mydata
batch_size: $BATCH
steps: $STEPS
save_freq: $SAVE_FREQ
log_freq: 100
num_workers: 8

wandb:
  enable: false
YAML

echo "==> [3/3] SMOKE TEST (20 steps) then full fine-tune"
rm -rf outputs/train/finetune_4type_mydata_smoke
if lerobot-train --config_path="$CFG" --steps=20 --save_freq=20 \
     --output_dir=outputs/train/finetune_4type_mydata_smoke >logs/smoke_4type_mydata.log 2>&1; then
  echo "    smoke test PASSED"; rm -rf outputs/train/finetune_4type_mydata_smoke
else
  echo "    !! SMOKE TEST FAILED — see logs/smoke_4type_mydata.log"; tail -25 logs/smoke_4type_mydata.log; exit 1
fi

echo "==> launching fine-tune of 4-type model on YOUR data ($STEPS steps, ~14 epochs)"
echo "    58 eps -> best checkpoint likely EARLY; save_freq=$SAVE_FREQ, robot-test a few"
lerobot-train --config_path="$CFG" 2>&1 | tee "logs/finetune_4type_mydata_$(date +%Y%m%d_%H%M%S).log"
