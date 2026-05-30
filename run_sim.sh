#!/usr/bin/env bash
# LeHome Challenge — run the Isaac Sim eval with visualization.
#
# Usage:
#   ./run_sim.sh gui     # live Isaac Sim window (default)
#   ./run_sim.sh video   # headless, records mp4 to outputs/eval_videos/
#
# Notes:
# - Uses the built-in "custom" random policy (no trained checkpoint needed).
#   Swap to a trained model by editing the POLICY_* / extra args below.
# - Physics runs on CPU (--device cpu) as required by the challenge; the GPU
#   is used for rendering/cameras.

set -euo pipefail

MODE="${1:-gui}"
IMAGE="lehome-challenge"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Eval knobs — tweak freely
GARMENT="top_long"      # top_long | top_short | pant_long | pant_short | custom
NUM_EPISODES=1
MAX_STEPS=300

# Allow the container to talk to your X server (needed for the GUI window AND
# for the repo's pynput import, which connects to the display on startup).
xhost +local:docker >/dev/null

COMMON_DOCKER_ARGS=(
  --rm --gpus all
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y
  -e DISPLAY="${DISPLAY}"
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -v /tmp/.X11-unix:/tmp/.X11-unix
  -v "${REPO_DIR}/Assets:/opt/lehome-challenge/Assets"
  -v "${REPO_DIR}/Datasets:/opt/lehome-challenge/Datasets"
  -v "${REPO_DIR}/outputs:/opt/lehome-challenge/outputs"
  --entrypoint bash
)

case "${MODE}" in
  gui)
    EVAL_FLAGS="--enable_cameras --device cpu"            # no --headless => GUI window
    ;;
  video)
    EVAL_FLAGS="--enable_cameras --device cpu --headless --save_video --video_dir outputs/eval_videos"
    ;;
  *)
    echo "Unknown mode '${MODE}'. Use: gui | video" >&2
    exit 1
    ;;
esac

docker run "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" -c "
  source /opt/lehome-challenge/.venv/bin/activate
  cd /opt/lehome-challenge
  python -m scripts.eval \
    --policy_type custom \
    --garment_type ${GARMENT} \
    --num_episodes ${NUM_EPISODES} \
    --max_steps ${MAX_STEPS} \
    ${EVAL_FLAGS}
"
