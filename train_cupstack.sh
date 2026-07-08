#!/usr/bin/env bash
# Train a bimanual cup-stacking SmolVLA from base on Pandeyjiya024/bimanual-cup-stacking-v2.
# Same-env inference (no domain gap, no camera rename). Pure lerobot (no Isaac). Run after ./setup.sh.
#   ./train_cupstack.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venv-so101}"
DATA_DIR="Datasets/cupstack_v2"
CFG="configs/train_cupstack.yaml"
DATASET_REPO="Pandeyjiya024/bimanual-cup-stacking-v2"   # 82 eps (35 orig + 47 new), already merged

# ---- knobs (82 eps / 95k frames; ~25 epochs @ batch 16; ~14 h on 5090 @ 0.34 s/step) ----
BATCH=16
STEPS=150000
SAVE_FREQ=15000       # ~10 checkpoints -> in-env test a few, pick best (likely not the last)
LR=1.0e-4
# -----------------------------------------------------------------------------------------

# shellcheck disable=SC1091
# activate the venv only if it exists; otherwise assume an env is already active (e.g. conda)
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; else echo "    using current env (no venv at $VENV — e.g. conda)"; fi
mkdir -p configs logs outputs

echo "==> [1/4] download dataset"
if [ -f "$DATA_DIR/meta/info.json" ]; then
  echo "    present"
else
  hf download "$DATASET_REPO" --repo-type dataset --local-dir "$DATA_DIR"
fi

echo "==> [2/4] verify it loads (catches any file_index metadata quirk)"
python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("x", root="$DATA_DIR")
_ = ds[0]; _ = ds[ds.num_frames-1]
print(f"    loads OK: {ds.num_episodes} eps, {ds.num_frames} frames")
PY

echo "==> [3/4] write config -> $CFG"
cat > "$CFG" <<YAML
dataset:
  repo_id: cupstack
  root: $DATA_DIR
  image_transforms:
    enable: true
    max_num_transforms: 3
    random_order: true
    tfs:
      brightness: { type: ColorJitter, weight: 1.0, kwargs: { brightness: [0.7, 1.3] } }
      contrast:   { type: ColorJitter, weight: 1.0, kwargs: { contrast:   [0.7, 1.3] } }

policy:
  type: smolvla
  pretrained_path: lerobot/smolvla_base
  device: cuda
  push_to_hub: false
  optimizer_lr: $LR
  scheduler_warmup_steps: 2000
  scheduler_decay_steps: $STEPS
  optimizer_weight_decay: 1.0e-4

  input_features:
    observation.state:                       { type: STATE,  shape: [12] }
    observation.images.left_wrist:           { type: VISUAL, shape: [3, 480, 640] }
    observation.images.right_wrist:          { type: VISUAL, shape: [3, 480, 640] }
    observation.images.right_wrist_depth:    { type: VISUAL, shape: [3, 480, 640] }
    observation.images.right_front:          { type: VISUAL, shape: [3, 480, 640] }
  output_features:
    action: { type: ACTION, shape: [12] }

output_dir: outputs/train/cupstack
batch_size: $BATCH
steps: $STEPS
save_freq: $SAVE_FREQ
log_freq: 100
num_workers: 8

wandb:
  enable: false
YAML

echo "==> [4/4] SMOKE TEST (20 steps)"
rm -rf outputs/train/cupstack_smoke
if lerobot-train --config_path="$CFG" --steps=20 --save_freq=20 \
     --output_dir=outputs/train/cupstack_smoke >logs/smoke_cupstack.log 2>&1; then
  echo "    smoke test PASSED"
  rm -rf outputs/train/cupstack_smoke
else
  echo "    !! SMOKE TEST FAILED — see logs/smoke_cupstack.log"; tail -25 logs/smoke_cupstack.log; exit 1
fi

echo "==> launching cup-stacking training ($STEPS steps, ~25 epochs)"
lerobot-train --config_path="$CFG" 2>&1 | tee "logs/cupstack_$(date +%Y%m%d_%H%M%S).log"
