# PPO Reinforcement Learning on Breakout

This repository has two complete training paths for Atari Breakout:

1. `Human demos -> Behavior Cloning (BC) -> PPO fine-tuning`
2. `OpenAI-style PPO from scratch` (Atari wrapper stack)

It also includes scripts to watch models and continue OpenAI-style training from the latest checkpoint.

## Repo structure

- `collect_breakout_data.py`: collect keyboard demos and save `.npz`
- `train_breakout_bc.py`: supervised BC training
- `train_breakout_ppo_from_bc.py`: warm-start PPO from BC checkpoint
- `watch_breakout_ppo.py`: watch BC/PPO-from-BC models
- `train_breakout_ppo_openai_style.py`: OpenAI-style Atari PPO from scratch
- `continue_breakout_ppo_openai_style.py`: continue OpenAI-style PPO training
- `watch_breakout_ppo_openai_style.py`: watch OpenAI-style PPO models

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install gymnasium ale-py stable-baselines3 torch numpy pillow pygame
```

## Path A: Demo -> BC -> PPO

### 1) Collect demos

```bash
python3 collect_breakout_data.py --output data/session3.npz
```

Controls:

- `Left` / `A`: move left
- `Right` / `D`: move right
- `Space`: fire
- `Esc`: save and exit

Important behavior:

- play/render defaults to `env_frameskip=1` for smooth control
- saved dataset defaults to `frameskip=4`
- file metadata stores both `env_frameskip` and saved `frameskip`

### 2) Train behavior cloning

```bash
python3 train_breakout_bc.py \
  --datasets data/session2.npz data/session3.npz \
  --output models/breakout_bc_actor.pt
```

Notes:

- all datasets in one run must share the same saved `frameskip`
- class weighting is enabled (with `--class-weight-power`)
- left-right augmentation is applied on training split
- best checkpoint selection uses balanced metrics, not raw accuracy

### 3) Train PPO from BC

```bash
python3 train_breakout_ppo_from_bc.py \
  --bc-checkpoint models/breakout_bc_actor.pt \
  --output-dir models/ppo_breakout_bc \
  --human-render-freq 0
```

Key defaults:

- `total_timesteps=5_000_000`
- `n_envs=4`, `n_steps=128`, `n_epochs=4`
- `frameskip=4`

Artifacts:

- `models/ppo_breakout_bc/ppo_bc_initialized.zip`
- `models/ppo_breakout_bc/checkpoints/breakout_ppo_bc_*_steps.zip`
- `models/ppo_breakout_bc/best_model/best_model.zip`
- `models/ppo_breakout_bc/ppo_bc_final.zip`

### 4) Watch PPO-from-BC

```bash
python3 watch_breakout_ppo.py \
  --model-path models/ppo_breakout_bc/best_model/best_model.zip
```

If you omit `--model-path`, the watcher auto-resolves from `--output-dir` and may choose latest/final before `best_model`.

## Path B: OpenAI-style PPO from scratch

### 1) Train from scratch

```bash
python3 train_breakout_ppo_openai_style.py
```

Default setup:

- env: `BreakoutNoFrameskip-v4`
- wrappers: `NoopReset -> MaxAndSkip(4) -> EpisodicLife -> FireReset -> Warp84Gray -> ClipReward -> FrameStack(4)`
- PPO defaults: `10_000_000` timesteps, `8` envs, `n_steps=128`, `4` minibatches, `4` epochs
- linear schedules for learning rate and clip range

Artifacts:

- `models/openai_ppo_breakout_style/checkpoints/openai_style_breakout_ppo_*_steps.zip`
- `models/openai_ppo_breakout_style/breakout_openai_style_final.zip`

### 2) Continue training existing OpenAI-style model

This script loads the latest OpenAI-style model and trains it further.

```bash
python3 continue_breakout_ppo_openai_style.py \
  --additional-timesteps 5000000 \
  --n-envs 8
```

What it does:

- auto-resolves latest model from `models/openai_ppo_breakout_style`
- keeps checkpointing in the same `checkpoints/` folder
- writes updated final alias:
  - `breakout_openai_style_final.zip`
- writes a numbered snapshot:
  - `breakout_openai_style_<timesteps>_steps.zip`

### 3) Watch OpenAI-style models

```bash
python3 watch_breakout_ppo_openai_style.py \
  --model-path models/openai_ppo_breakout_style/breakout_openai_style_final.zip
```

Or auto-resolve from output dir:

```bash
python3 watch_breakout_ppo_openai_style.py \
  --output-dir models/openai_ppo_breakout_style
```

## Quick start commands

### BC + PPO path

```bash
python3 collect_breakout_data.py --output data/session3.npz
python3 train_breakout_bc.py --datasets data/session2.npz data/session3.npz --output models/breakout_bc_actor.pt
python3 train_breakout_ppo_from_bc.py --bc-checkpoint models/breakout_bc_actor.pt --output-dir models/ppo_breakout_bc --human-render-freq 0
python3 watch_breakout_ppo.py --model-path models/ppo_breakout_bc/best_model/best_model.zip
```

### OpenAI-style path

```bash
python3 train_breakout_ppo_openai_style.py
python3 continue_breakout_ppo_openai_style.py --additional-timesteps 5000000 --n-envs 8
python3 watch_breakout_ppo_openai_style.py --output-dir models/openai_ppo_breakout_style
```

## Important concepts

### What is one PPO timestep here?

One PPO timestep is one transition per environment:

- policy sees a stacked observation
- picks one action
- `MaxAndSkip(4)` repeats that action over 4 emulator frames
- wrapper returns one next observation + one reward signal

With `n_envs=8`, each synchronized environment step advances global timesteps by 8.

### Reward clipping vs normalization

OpenAI-style training uses reward clipping (`ClipRewardEnv`), mapping per-step reward sign to `-1/0/+1`.

- this is not z-score normalization
- it improves optimization stability
- it removes reward magnitude information but keeps event frequency signal

## Troubleshooting

### `ResetNeeded: Cannot call env.render() before env.reset()`

Use current scripts from this repo. This is already handled in watchers.

### Observation shape mismatch while watching OpenAI-style models

`watch_breakout_ppo_openai_style.py` supports both:

- legacy layout checkpoints (trained before frame-stack layout fix)
- corrected layout checkpoints

If mismatch still appears, ensure you are loading the model with:

```bash
python3 watch_breakout_ppo_openai_style.py --model-path <exact_model_zip>
```

### OpenCV import error

OpenAI-style scripts in this repo use Pillow-based grayscale/resize wrapper, so `opencv-python` is not required for that path.

## Notes on current local data

- `data/session1_frameskip1_backup.npz`: backup from older collection format
- `data/session2.npz`, `data/session3.npz`: current datasets used by BC runs
