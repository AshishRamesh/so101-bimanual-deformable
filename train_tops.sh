#!/usr/bin/env bash
# Train the tops SmolVLA model. Run ./setup.sh first.
# Native step/loss progress prints straight to the terminal (also tee'd to a log).
#
#   ./train_tops.sh                 # fresh run
#   ./train_tops.sh --resume=true   # continue from the latest checkpoint
set -euo pipefail

VENV="${VENV:-$HOME/.venv-so101}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

mkdir -p outputs logs
LOG="logs/train_tops_$(date +%Y%m%d_%H%M%S).log"

echo "==> training -> outputs/train/smolvla_tops   (log: $LOG)"
echo "==> watch lr: it should ramp to ~1e-4 by step 2000 then decay slowly across 100k"
lerobot-train --config_path=configs/train_smolvla_tops.yaml "$@" 2>&1 | tee "$LOG"
