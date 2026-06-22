#!/usr/bin/env bash
# Train on ONLY the real lehome tops data, from smolvla_base (NO sim pretrain).
# This is the control/baseline vs finetune_real.sh (which fine-tunes the sim 80k):
# comparing the two on the robot tells you how much the sim pretrain actually helped,
# given the sim->real camera change (top vs front).
#
# Reuses the real_tops_merged dataset built by finetune_real.sh (builds it if absent).
# Runs in the venv. Run AFTER ./setup.sh.   ->   ./train_real_frombase.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venv-so101}"
DATA_DIR="Datasets/example"
CFG="configs/train_real_frombase.yaml"

# ---- knobs: from-base needs a higher LR + a few more epochs than the fine-tune ----
BATCH=16
STEPS=175000          # ~25 epochs @ batch 16 over the 250 real eps (~111k frames)
SAVE_FREQ=17500       # ~every 2.5 epochs -> ~10 checkpoints
LR=1.0e-4             # higher LR: training the action expert from base, not gently adapting
# -----------------------------------------------------------------------------------

# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$DATA_DIR" configs logs outputs

echo "==> [1/4] ensure real tops data is downloaded"
for sub in top_long_merged top_short_merged; do
  [ -d "$DATA_DIR/real_raw/$sub" ] && { echo "    real/$sub present"; continue; }
  echo "    syncing real/$sub …"
  hf download lehome/dataset_challenge_real --include "$sub/*" \
    --repo-type dataset --local-dir "$DATA_DIR/real_raw"
done

echo "==> [2/4] patch + merge -> real_tops_merged (reuses it if finetune_real.sh already built it)"
if [ -f "$DATA_DIR/real_tops_merged/meta/info.json" ]; then
  echo "    real_tops_merged present"
else
  rm -rf "$DATA_DIR/real_tops_merged"
  python - <<'PY'
from pathlib import Path
import glob, pandas as pd
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
base = "Datasets/example/real_raw"
for name in ["top_long_merged", "top_short_merged"]:
    p = sorted(glob.glob(f"{base}/{name}/meta/episodes/chunk-000/*.parquet"))[0]
    ep = pd.read_parquet(p); ep["data/chunk_index"] = 0; ep["data/file_index"] = 0
    ep.to_parquet(p, index=False); print(f"    patched {name}")
aggregate_datasets(
    repo_ids=["local/rtl", "local/rts"],
    aggr_repo_id="local/real_tops_merged",
    roots=[Path(f"{base}/top_long_merged"), Path(f"{base}/top_short_merged")],
    aggr_root=Path("Datasets/example/real_tops_merged"),
)
ds = LeRobotDataset("x", root="Datasets/example/real_tops_merged")
assert ds.num_episodes == 250, f"expected 250 real eps, got {ds.num_episodes}"
print(f"    merged + verified: {ds.num_episodes} eps, {ds.num_frames} frames")
PY
fi

echo "==> [3/4] write from-base config -> $CFG"
cat > "$CFG" <<YAML
dataset:
  repo_id: repo_real_tops
  root: Datasets/example/real_tops_merged
  image_transforms:
    enable: true
    max_num_transforms: 3
    random_order: true
    tfs:
      brightness: { type: ColorJitter, weight: 1.0, kwargs: { brightness: [0.7, 1.3] } }
      contrast:   { type: ColorJitter, weight: 1.0, kwargs: { contrast:   [0.7, 1.3] } }
      saturation: { type: ColorJitter, weight: 1.0, kwargs: { saturation: [0.7, 1.3] } }

# remap REAL camera keys -> the slots SmolVLA uses
rename_map:
  observation.images.left_wrist:  observation.images.left_rgb
  observation.images.right_wrist: observation.images.right_rgb
  observation.images.right_front: observation.images.top_rgb

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

output_dir: outputs/train/real_frombase
batch_size: $BATCH
steps: $STEPS
save_freq: $SAVE_FREQ
log_freq: 100
num_workers: 8

wandb:
  enable: false
YAML

echo "==> [4/4] SMOKE TEST (20 steps)"
rm -rf outputs/train/real_frombase_smoke
if lerobot-train --config_path="$CFG" --steps=20 --save_freq=20 \
     --output_dir=outputs/train/real_frombase_smoke >logs/smoke_frombase.log 2>&1; then
  echo "    smoke test PASSED"
  rm -rf outputs/train/real_frombase_smoke
else
  echo "    !! SMOKE TEST FAILED — see logs/smoke_frombase.log"
  echo "    If it's a shape error, set top_rgb shape to [3,720,1280] in $CFG and re-run."
  tail -25 logs/smoke_frombase.log
  exit 1
fi

echo "==> launching from-base real training ($STEPS steps, ~25 epochs)"
lerobot-train --config_path="$CFG" 2>&1 | tee "logs/real_frombase_$(date +%Y%m%d_%H%M%S).log"
