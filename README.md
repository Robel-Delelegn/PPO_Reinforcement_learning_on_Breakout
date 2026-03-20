# PPO Reinforcement Learning on Atari Breakout

End-to-end project for training Breakout agents with:

1. **Imitation Learning + RL**: Human demonstrations -> Behavior Cloning (BC) -> PPO fine-tuning  
2. **OpenAI-style PPO from scratch**: Atari wrapper stack and hyperparameters aligned with the classic PPO Atari recipe

This repository is designed for practical experimentation: collect data, train, continue runs, and watch policies with minimal friction.

## Contents

1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Requirements](#requirements)
4. [Quick Start](#quick-start)
5. [Path A: Human Demos -> BC -> PPO](#path-a-human-demos---bc---ppo)
6. [Path B: OpenAI-style PPO from Scratch](#path-b-openai-style-ppo-from-scratch)
7. [Continue OpenAI-style Training](#continue-openai-style-training)
8. [Watch Trained Models](#watch-trained-models)
9. [Data and Model Artifacts](#data-and-model-artifacts)
10. [Key Concepts](#key-concepts)
11. [Troubleshooting](#troubleshooting)
12. [GitHub Push Notes (Large Files)](#github-push-notes-large-files)

## Project Overview

This repo intentionally provides **two PPO environments**:

- A **custom BC/PPO environment** (`ALE/Breakout-v5` + custom stack wrapper) used to warm-start PPO from imitation.
- An **OpenAI-style Atari environment** (`BreakoutNoFrameskip-v4` + canonical wrappers) used for direct PPO baselines and continued training.

The two paths are not identical by design. This makes it easier to compare:

- when demonstrations help,
- when canonical Atari shaping is enough,
- and where environment/wrapper choices dominate outcome.

## Repository Layout

- `collect_breakout_data.py`: Record human gameplay to compressed `.npz` datasets.
- `train_breakout_bc.py`: Train supervised behavior cloning actor from datasets.
- `train_breakout_ppo_from_bc.py`: Initialize PPO from BC weights and fine-tune.
- `watch_breakout_ppo.py`: Watch PPO models from the BC path.
- `train_breakout_ppo_openai_style.py`: Train PPO from scratch with OpenAI-style wrappers.
- `continue_breakout_ppo_openai_style.py`: Resume/extend an OpenAI-style PPO run.
- `watch_breakout_ppo_openai_style.py`: Watch OpenAI-style models, including legacy checkpoint layouts.
- `data/`: Demonstration datasets.
- `models/`: Trained BC/PPO checkpoints.
- `scripts/`: Auxiliary scripts.

## Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install gymnasium ale-py stable-baselines3 torch numpy pillow pygame
```

## Quick Start

### A) BC + PPO path

```bash
.venv/bin/python collect_breakout_data.py --output data/session3.npz
.venv/bin/python train_breakout_bc.py --datasets data/session2.npz data/session3.npz --output models/breakout_bc_actor.pt
.venv/bin/python train_breakout_ppo_from_bc.py --bc-checkpoint models/breakout_bc_actor.pt --output-dir models/ppo_breakout_bc --human-render-freq 0
.venv/bin/python watch_breakout_ppo.py --model-path models/ppo_breakout_bc/best_model/best_model.zip
```

### B) OpenAI-style PPO path

```bash
.venv/bin/python train_breakout_ppo_openai_style.py
.venv/bin/python continue_breakout_ppo_openai_style.py --additional-timesteps 5000000 --n-envs 8
.venv/bin/python watch_breakout_ppo_openai_style.py --output-dir models/openai_ppo_breakout_style
```

## Path A: Human Demos -> BC -> PPO

### Step 1: Collect demonstrations

```bash
.venv/bin/python collect_breakout_data.py --output data/session3.npz
```

Controls:

- `Left` / `A`: move left
- `Right` / `D`: move right
- `Space`: fire
- `Esc`: save and exit

Important defaults:

- `env_frameskip=1` during play for precise control
- saved dataset `frameskip=4`
- saved metadata includes both `env_frameskip` and saved `frameskip`

### Step 2: Train behavior cloning

```bash
.venv/bin/python train_breakout_bc.py \
  --datasets data/session2.npz data/session3.npz \
  --output models/breakout_bc_actor.pt
```

BC details:

- class weighting via inverse-frequency weights (`--class-weight-power`)
- left-right symmetry augmentation on the training split
- validation reporting includes balanced accuracy, macro-F1, and per-class recall
- best checkpoint selected by balanced metrics, not raw accuracy

### Step 3: PPO fine-tuning from BC

```bash
.venv/bin/python train_breakout_ppo_from_bc.py \
  --bc-checkpoint models/breakout_bc_actor.pt \
  --output-dir models/ppo_breakout_bc \
  --human-render-freq 0
```

Key defaults:

- `total_timesteps=5_000_000`
- `n_envs=4`
- `n_steps=128`
- `batch_size=256`
- `n_epochs=4`
- `frameskip=4`

### Step 4: Watch BC/PPO models

```bash
.venv/bin/python watch_breakout_ppo.py --output-dir models/ppo_breakout_bc
```

Explicit best checkpoint:

```bash
.venv/bin/python watch_breakout_ppo.py \
  --model-path models/ppo_breakout_bc/best_model/best_model.zip
```

## Path B: OpenAI-style PPO from Scratch

Train from scratch:

```bash
.venv/bin/python train_breakout_ppo_openai_style.py
```

Default environment recipe:

- `BreakoutNoFrameskip-v4`
- `NoopResetEnv`
- `MaxAndSkipEnv(skip=4)`
- `EpisodicLifeEnv`
- `FireResetEnv`
- grayscale warp to `84x84`
- `ClipRewardEnv` (`-1/0/+1`)
- `VecFrameStack(4)`

Default PPO recipe:

- `total_timesteps=10_000_000`
- `n_envs=8`
- `n_steps=128`
- `n_minibatches=4` (effective batch size `256`)
- `n_epochs=4`
- linear schedules for learning rate (`2.5e-4 -> 0`) and clip range (`0.1 -> 0`)

## Continue OpenAI-style Training

Resume an existing OpenAI-style model and train longer:

```bash
.venv/bin/python continue_breakout_ppo_openai_style.py \
  --additional-timesteps 5000000 \
  --n-envs 8
```

Resume behavior:

- auto-resolves model from `--model-path` or latest in `--output-dir`
- preserves timestep counter (`reset_num_timesteps=False`)
- writes rolling checkpoints in the same folder
- updates final alias `breakout_openai_style_final.zip`
- writes numbered snapshot `breakout_openai_style_<timesteps>_steps.zip`

Compatibility:

- continuation script auto-detects and supports both:
  - legacy observation layout checkpoints `(1, 336, 84)`
  - corrected layout checkpoints `(4, 84, 84)`

## Watch Trained Models

### Watch OpenAI-style model

```bash
.venv/bin/python watch_breakout_ppo_openai_style.py \
  --model-path models/openai_ppo_breakout_style/breakout_openai_style_final.zip
```

Auto-resolve from output dir:

```bash
.venv/bin/python watch_breakout_ppo_openai_style.py \
  --output-dir models/openai_ppo_breakout_style
```

### Watch with stochastic actions

```bash
.venv/bin/python watch_breakout_ppo_openai_style.py \
  --output-dir models/openai_ppo_breakout_style \
  --stochastic
```

## Data and Model Artifacts

### Common model outputs

- `models/ppo_breakout_bc/`
- `models/openai_ppo_breakout_style/`

### BC path outputs

- `ppo_bc_initialized.zip`
- `ppo_bc_final.zip`
- `checkpoints/breakout_ppo_bc_*_steps.zip`
- `best_model/best_model.zip`

### OpenAI-style outputs

- `breakout_openai_style_final.zip`
- `breakout_openai_style_<timesteps>_steps.zip`
- `checkpoints/openai_style_breakout_ppo_*_steps.zip`

## Key Concepts

### What is one PPO timestep?

One PPO timestep is one transition per environment:

- policy receives the current stacked observation
- picks one action
- action is repeated for 4 emulator frames by `MaxAndSkip(4)`
- env returns one reward and next observation

With `n_envs=8`, each synchronized vectorized step increases global timesteps by `8`.

### Reward clipping vs normalization

OpenAI-style path uses reward clipping (`ClipRewardEnv`), not z-score normalization:

- positive reward -> `+1`
- zero reward -> `0`
- negative reward -> `-1`

This improves optimization stability while sacrificing reward magnitude information.

## Troubleshooting

### `ResetNeeded: Cannot call env.render() before env.reset()`

Use the current watcher scripts in this repo. This order is already handled.

### Observation shape mismatch when watching OpenAI-style models

Use:

```bash
.venv/bin/python watch_breakout_ppo_openai_style.py --model-path <exact_model.zip>
```

The script auto-detects legacy vs corrected checkpoint layouts.

### `Model file not found`

Double-check the exact model path and extension (`.zip`).

### PPO watcher chooses final model instead of best model

Pass `--model-path` explicitly to enforce the exact checkpoint.

## GitHub Push Notes (Large Files)

This project can produce many large model checkpoints. GitHub normal git pushes reject files larger than 100 MB.

If you want to version large checkpoints/datasets in GitHub:

- use Git LFS, or
- keep large artifacts out of git and publish them as release assets or external storage links.

For normal code-only updates, commit scripts and README only.
