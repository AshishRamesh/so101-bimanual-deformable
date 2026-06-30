#!/usr/bin/env bash
# CONTROL: train on YOUR data (Pandeyjiya024/so101_cups_merged) from smolvla_base,
# i.e. NO sim pretrain. Compare on the robot vs finetune_mydata.sh to confirm how much
# the sim 80k pretrain helps. Expect this to be WORSE — 58 eps is very little to learn
# folding from scratch; this is a control, not a deployable model.
#
# Runs in the venv. Run AFTER ./setup.sh (and ideally after finetune_mydata.sh which
# already downloaded the dataset).   ->   ./frombase_mydata.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venv-so101}"
DATA_REPO="Pandeyjiya024/so101_cups_merged"
DATA_DIR="Datasets/example/mydata"
CFG="configs/train_mydata_frombase.yaml"

# ---- knobs: from-base needs higher LR + more epochs than the fine-tune ----
BATCH=16
STEPS=100000          # ~24 epochs @ batch 16 over 58 eps / ~67k frames (5090: ~6-7 h)
SAVE_FREQ=10000       # ~every 2.4 epochs -> ~10 checkpoints
LR=1.0e-4             # higher LR: training the action expert from base, not adapting
# ---------------------------------------------------------------------------

# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$DATA_DIR" configs logs outputs

echo "==> [1/3] ensure your dataset is present (reuses finetune_mydata.sh's download)"
if [ -f "$DATA_DIR/meta/info.json" ]; then
  echo "    dataset present"
else
  hf download "$DATA_REPO" --repo-type dataset --local-dir "$DATA_DIR"
fi
python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("x", root="$DATA_DIR")
print(f"    loaded: {ds.num_episodes} eps, {ds.num_frames} frames")
PY

echo "==> [2/3] write from-base config -> $CFG"
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

# NO rename_map — your dataset already uses top_rgb/left_rgb/right_rgb

policy:
  type: smolvla
  pretrained_path: lerobot/smolvla_base    # <-- from BASE, no sim pretrain
  device: cuda
  push_to_hub: false
  optimizer_lr: $LR
  scheduler_warmup_steps: 2000
  scheduler_decay_steps: $STEPS
  optimizer_weight_decay: 1.0e-4

  input_features:
    observation.state:            { type: STATE,  shape: [12] }
    observation.images.top_rgb:   { type: VISUAL, shape: [3, 480, 640] }
    observation.images.left_rgb:  { type: VISUAL, shape: [3, 480, 640] }
    observation.images.right_rgb: { type: VISUAL, shape: [3, 480, 640] }
  output_features:
    action: { type: ACTION, shape: [12] }

output_dir: outputs/train/frombase_mydata
batch_size: $BATCH
steps: $STEPS
save_freq: $SAVE_FREQ
log_freq: 100
num_workers: 8

wandb:
  enable: false
YAML

echo "==> [3/3] SMOKE TEST (20 steps) then full from-base training"
rm -rf outputs/train/frombase_mydata_smoke
if lerobot-train --config_path="$CFG" --steps=20 --save_freq=20 \
     --output_dir=outputs/train/frombase_mydata_smoke >logs/smoke_mydata_frombase.log 2>&1; then
  echo "    smoke test PASSED"; rm -rf outputs/train/frombase_mydata_smoke
else
  echo "    !! SMOKE TEST FAILED — see logs/smoke_mydata_frombase.log"; tail -25 logs/smoke_mydata_frombase.log; exit 1
fi

echo "==> launching from-base training on YOUR data ($STEPS steps, ~24 epochs) — CONTROL run"
lerobot-train --config_path="$CFG" 2>&1 | tee "logs/frombase_mydata_$(date +%Y%m%d_%H%M%S).log"
