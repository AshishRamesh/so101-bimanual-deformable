#!/usr/bin/env bash
# Fine-tune the sim tops model (tops_fold_80k) on the REAL lehome tops data, on the 5090.
# Real rig = front + 2 wrist cams (matches the deployment robot); we remap the real camera
# keys into the slots the model expects (front -> top slot) via rename_map.
#
# Runs in the venv (no Docker needed for training). Run AFTER ./setup.sh has set up the venv+torch.
#   ./finetune_real.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venv-so101}"
PRETRAINED="${PRETRAINED:-AshishRamesh/tops_fold_80k}"   # HF id, or a local path to the 80k pretrained_model
DATA_DIR="Datasets/example"
CFG="configs/train_real_finetune.yaml"

# ---- knobs (sized for ~20 epochs over the 250 real eps / ~111k frames at batch 16) ----
BATCH=16
STEPS=140000          # ~20 epochs @ batch 16  (5090: ~4-6 h frozen-vision)
SAVE_FREQ=14000       # ~every 2 epochs -> ~10 checkpoints to robot-test
LR=3.0e-5             # low LR: adapt to real without erasing sim-learned folding
# ---------------------------------------------------------------------------------------

# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$DATA_DIR" configs logs outputs

echo "==> [1/4] download real tops (long+short)"
for sub in top_long_merged top_short_merged; do
  echo "    syncing real/$sub …"
  hf download lehome/dataset_challenge_real --include "$sub/*" \
    --repo-type dataset --local-dir "$DATA_DIR/real_raw"
done

echo "==> [2/4] patch data file_index + merge -> real_tops_merged (+verify)"
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
# same metadata quirk as the sim subsets: all data is in file-000 but episodes claim file_index 0..N
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

echo "==> [3/4] write fine-tune config -> $CFG"
cat > "$CFG" <<YAML
dataset:
  repo_id: repo_real_tops
  root: Datasets/example/real_tops_merged
  image_transforms:                 # light aug for real lighting robustness (not heavy DR)
    enable: true
    max_num_transforms: 3
    random_order: true
    tfs:
      brightness: { type: ColorJitter, weight: 1.0, kwargs: { brightness: [0.7, 1.3] } }
      contrast:   { type: ColorJitter, weight: 1.0, kwargs: { contrast:   [0.7, 1.3] } }
      saturation: { type: ColorJitter, weight: 1.0, kwargs: { saturation: [0.7, 1.3] } }

# remap REAL camera keys -> the slots the 80k model was trained on
rename_map:
  observation.images.left_wrist:  observation.images.left_rgb
  observation.images.right_wrist: observation.images.right_rgb
  observation.images.right_front: observation.images.top_rgb

policy:
  type: smolvla
  pretrained_path: $PRETRAINED
  device: cuda
  push_to_hub: false
  optimizer_lr: $LR
  scheduler_warmup_steps: 500
  scheduler_decay_steps: $STEPS
  optimizer_weight_decay: 1.0e-4

  input_features:
    observation.state:            { type: STATE,  shape: [12] }
    observation.images.top_rgb:   { type: VISUAL, shape: [3, 480, 640] }
    observation.images.left_rgb:  { type: VISUAL, shape: [3, 480, 640] }
    observation.images.right_rgb: { type: VISUAL, shape: [3, 480, 640] }
  output_features:
    action: { type: ACTION, shape: [12] }

output_dir: outputs/train/real_finetune_80k
batch_size: $BATCH
steps: $STEPS
save_freq: $SAVE_FREQ
log_freq: 100
num_workers: 8

wandb:
  enable: false
YAML

echo "==> [4/4] SMOKE TEST (20 steps) — catches rename/shape errors fast"
rm -rf outputs/train/real_finetune_80k_smoke
if lerobot-train --config_path="$CFG" --steps=20 --save_freq=20 \
     --output_dir=outputs/train/real_finetune_80k_smoke >logs/smoke.log 2>&1; then
  echo "    smoke test PASSED — rename + shapes OK"
  rm -rf outputs/train/real_finetune_80k_smoke
else
  echo "    !! SMOKE TEST FAILED — see logs/smoke.log (likely the 720x1280 top-cam shape)."
  echo "    If it's a shape error, set top_rgb shape to [3,720,1280] in $CFG and re-run."
  tail -25 logs/smoke.log
  exit 1
fi

echo "==> launching full fine-tune ($STEPS steps, ~20 epochs)"
echo "    save checkpoints every $SAVE_FREQ -> robot-test a few, pick the best (likely NOT the last)"
lerobot-train --config_path="$CFG" 2>&1 | tee "logs/real_finetune_$(date +%Y%m%d_%H%M%S).log"
