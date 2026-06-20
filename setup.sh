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

echo "==> [2/6] LeRobot $LEROBOT_VERSION (pulls a compatible huggingface_hub[cli])"
if python -c "import lerobot, sys; sys.exit(0 if lerobot.__version__=='$LEROBOT_VERSION' else 1)" 2>/dev/null; then
  echo "    lerobot $LEROBOT_VERSION present"
else
  pip install -q "lerobot==$LEROBOT_VERSION"
fi
# Do NOT upgrade huggingface_hub past lerobot's pin (<0.36); 1.x also drops the [cli] extra.
# Pin it in-range and ensure the `hf` CLI is present.
pip install -q "huggingface_hub[cli,hf-transfer]>=0.34.2,<0.36.0"

echo "==> [3/6] ensure torch actually runs on this GPU (Blackwell sm_120 needs cu128)"
# NOTE: torch.cuda.is_available() can be True while kernels are missing for the
# GPU's arch (e.g. cu126 torch on an RTX 5090). So we launch a REAL kernel.
gpu_ok() {
  python - <<'PY' 2>/dev/null
import torch, sys
try:
    assert torch.cuda.is_available()
    (torch.randn(8, device="cuda") * 2).sum().item()   # forces a kernel launch
except Exception:
    sys.exit(1)
PY
}
if gpu_ok; then
  echo "    CUDA kernels OK"
else
  echo "    GPU kernels incompatible (likely cu126 on Blackwell) -> installing cu128 torch…"
  pip install -q -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
  gpu_ok || { echo "    STILL failing — see https://pytorch.org/get-started/locally/ (need a build for your GPU arch)"; exit 1; }
  echo "    fixed"
fi
python -c "import torch; print('    torch', torch.__version__, '| gpu', torch.cuda.get_device_name(0))"

echo "==> [4/6] download tops datasets (long + short)"
mkdir -p "$DATA_DIR"
# Always call hf download — it's idempotent and COMPLETES partial dirs
# (a previous run may have left a folder that exists but is missing files).
for sub in top_long_merged top_short_merged; do
  echo "    syncing $sub …"
  hf download lehome/dataset_challenge_merged --include "$sub/*" \
    --repo-type dataset --local-dir "$DATA_DIR"
done

echo "==> [5/6] merge -> tops_merged"
# Only treat as done if the merge actually finished (meta/info.json present);
# otherwise wipe any half-built dir and redo it.
if [ -f "$DATA_DIR/tops_merged/meta/info.json" ]; then
  echo "    tops_merged present"
else
  rm -rf "$DATA_DIR/tops_merged"
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
