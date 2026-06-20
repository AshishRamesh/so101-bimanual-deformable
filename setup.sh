#!/usr/bin/env bash
# One-time setup for TRAINING the tops SmolVLA model on a CUDA box (e.g. RTX 5090).
# Idempotent: safe to re-run. Does NOT need Docker / Isaac Sim (training only).
#
#   ./setup.sh
#
# Creates a venv, installs LeRobot (+ Blackwell-capable torch), downloads the
# top_long + top_short datasets, merges them into tops_merged, and pre-caches the
# SmolVLA base. After it finishes, run ./train_tops.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venv-so101}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.4.3}"
DATA_DIR="Datasets/example"

echo "==> [1/6] venv ($VENV)"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  echo "    created"
else
  echo "    exists"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q -U pip wheel

echo "==> [2/6] LeRobot $LEROBOT_VERSION + hf CLI"
if python -c "import lerobot, sys; sys.exit(0 if lerobot.__version__=='$LEROBOT_VERSION' else 1)" 2>/dev/null; then
  echo "    lerobot $LEROBOT_VERSION present"
else
  pip install -q "lerobot==$LEROBOT_VERSION"
fi
pip install -q -U "huggingface_hub[cli]"

echo "==> [3/6] ensure Blackwell-capable torch (RTX 50xx needs CUDA 12.8+)"
# If torch can't see the GPU, install the cu128 build. Comment out if not on Blackwell.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "    installing torch (cu128)…"
  pip install -q -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
python - <<'PY'
import torch
print(f"    torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA GPU not visible — fix torch install before training"
print(f"    gpu={torch.cuda.get_device_name(0)}")
PY

echo "==> [4/6] download tops datasets (long + short)"
mkdir -p "$DATA_DIR"
for sub in top_long_merged top_short_merged; do
  if [ -d "$DATA_DIR/$sub" ]; then
    echo "    $sub present"
  else
    echo "    downloading $sub …"
    hf download lehome/dataset_challenge_merged --include "$sub/*" \
      --repo-type dataset --local-dir "$DATA_DIR"
  fi
done

echo "==> [5/6] merge -> tops_merged"
if [ -d "$DATA_DIR/tops_merged" ]; then
  echo "    tops_merged present"
else
  python - <<'PY'
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
aggregate_datasets(
    repo_ids=["local/top_long", "local/top_short"],
    aggr_repo_id="local/tops_merged",
    roots=[Path("Datasets/example/top_long_merged"), Path("Datasets/example/top_short_merged")],
    aggr_root=Path("Datasets/example/tops_merged"),
)
print("    merged -> Datasets/example/tops_merged")
PY
fi

echo "==> [6/6] pre-cache SmolVLA base"
hf download lerobot/smolvla_base >/dev/null 2>&1 || echo "    (will auto-download at train time)"

echo
echo "==> SETUP DONE.  Next:  source $VENV/bin/activate && ./train_tops.sh"
