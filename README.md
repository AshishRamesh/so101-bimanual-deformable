# so101-bimanual-deformable

Tooling and configs for training a **SmolVLA** vision-language-action policy for
**bimanual SO-101 deformable-object (garment) manipulation** — folding cloth in
an Isaac Lab + LeRobot simulator.

> This repo contains **only my own files** (training configs + a sim/eval
> launcher) — no third-party simulator code, assets, or datasets. The commands
> below run against an external Isaac Lab garment-manipulation environment
> (provided as a Docker image), which supplies the sim, `scripts.eval`, and
> `lerobot-train`; install that separately.

## Contents

| File | Purpose |
|------|---------|
| `configs/train_smolvla_demo.yaml` | Short (~800-step) SmolVLA fine-tune from `lerobot/smolvla_base`. Memory-safe for an 8 GB GPU. Verifies the train→sim pipeline; under-trained by design. |
| `configs/train_smolvla_fold.yaml` | Full 30k-step SmolVLA fine-tune on the `top_long` dataset (same memory-safe setup). The real training run. |
| `run_sim.sh` | Launch the Isaac Sim eval with visualization — `./run_sim.sh gui` (live window) or `./run_sim.sh video` (headless, records mp4). Handles X-forwarding into the container. |

## Setup (Docker route)

1. Get the LeHome challenge Docker image (`lehome-challenge:latest`) per the
   official [docker install guide](https://github.com/lehome-official/lehome-challenge/blob/main/docs/docker_installation.md).
2. Download the challenge assets + a dataset subset:
   ```bash
   hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
   hf download lehome/dataset_challenge_merged --include "top_long_merged/*" \
     --repo-type dataset --local-dir Datasets/example
   ```
3. Copy the files from this repo into the challenge repo root (so `configs/` and
   `run_sim.sh` sit alongside `scripts/`, `source/`, etc.).

## Train

```bash
docker run --rm --gpus all --shm-size=16g --ipc=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e LEHOME_DISABLE_KEYBOARD=1 \
  -e HF_HOME=/opt/lehome-challenge/outputs/hf_cache \
  -v "$PWD/configs:/opt/lehome-challenge/configs" \
  -v "$PWD/Datasets:/opt/lehome-challenge/Datasets" \
  -v "$PWD/outputs:/opt/lehome-challenge/outputs" \
  --entrypoint bash lehome-challenge -c '
    source /opt/lehome-challenge/.venv/bin/activate && cd /opt/lehome-challenge
    lerobot-train --config_path=configs/train_smolvla_fold.yaml
  '
```

## Evaluate in sim (with video)

```bash
xhost +local:docker
docker run --rm --gpus all --shm-size=8g --ipc=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HF_HOME=/opt/lehome-challenge/outputs/hf_cache \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD/Assets:/opt/lehome-challenge/Assets" \
  -v "$PWD/Datasets:/opt/lehome-challenge/Datasets" \
  -v "$PWD/outputs:/opt/lehome-challenge/outputs" \
  --entrypoint bash lehome-challenge -c '
    source /opt/lehome-challenge/.venv/bin/activate && cd /opt/lehome-challenge
    python -m scripts.eval --policy_type lerobot \
      --policy_path outputs/train/smolvla_top_long_fold/checkpoints/last/pretrained_model \
      --garment_type top_long --dataset_root Datasets/example/top_long_merged \
      --num_episodes 2 --max_steps 300 --enable_cameras --device cpu \
      --save_video --video_dir outputs/eval_videos \
      --task_description "fold the garment on the table" --headless
  '
```

## Results so far

- Full 30k-step fine-tune (batch 2, from `smolvla_base`) on `top_long`:
  **20.83 % success (5/24)** across the 12 evaluation garments; up to 100 %
  (2/2) on individual seen garments. **0 %** on the two *unseen* garments — a
  generalization gap consistent with under-training.

## Gotchas (learned the hard way)

- **8 GB GPU + Isaac Sim RTX rendering + SmolVLA inference is tight.** If a
  previous training/eval Docker container is still holding VRAM, eval dies with
  `NVTT Error: cudaMalloc ... out of memory` and *every* garment fails to load —
  looking like the policy "failed everything" when it didn't. **Always clear
  leftover containers and check `nvidia-smi` before an eval; never run two GPU
  jobs at once:**
  ```bash
  docker ps --filter ancestor=lehome-challenge -q | xargs -r docker stop
  ```
- **DataLoader workers crash in Docker** (`worker exited unexpectedly`) unless
  you pass `--shm-size=16g --ipc=host` (default `/dev/shm` is 64 MB).
- **The repo imports `pynput` on startup**, which needs an X display even in
  headless mode — forward `$DISPLAY` + mount `/tmp/.X11-unix` and run
  `xhost +local:docker`.
- **`batch 2 × 30k steps ≈ 0.7 epoch`** over the 83k-frame dataset, vs the
  baseline's `batch 32 × 30k ≈ 11 epochs`. To improve: more samples-seen
  (`steps: 150000`+ or larger batch) and the multi-garment `four_types_merged`
  dataset.

## Notes

- Physics runs CPU-only (`--device cpu`); the GPU handles rendering + policy inference.
- Training/eval artifacts (`outputs/`), downloaded `Assets/`, and `Datasets/` are
  gitignored — they're large and reproducible from the steps above.
