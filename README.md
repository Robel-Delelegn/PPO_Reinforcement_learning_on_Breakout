# PPO Reinforcement Learning on Atari Breakout

This repository contains two complete training paths and one uploaded pretrained OpenAI-style PPO checkpoint that you can run immediately.

## Uploaded and Ready to Use

- Code for data collection, behavior cloning, custom PPO fine-tuning, OpenAI-style PPO training, continuation, and viewers.
- Demo datasets:
  - `data/session1_frameskip1_backup.npz`
  - `data/session2.npz`
  - `data/session3.npz`
- Pretrained OpenAI-style PPO model (GitHub-safe):
  - `models/pretrained/openai_style/breakout_openai_style_best_fp16_inference.zip`
  - SHA256: `ed3747832f10c8815bf9370489b44083fd0428970339a56eaa003b84b99c2e69`

The pretrained model above is the one intended for users to download and run from GitHub.

## Repository Layout

- `collect_breakout_data.py`: Play Breakout and save imitation-learning data.
- `train_breakout_bc.py`: Train behavior cloning (BC) policy on saved demos.
- `train_breakout_ppo_from_bc.py`: Initialize PPO from BC and continue RL training.
- `watch_breakout_ppo.py`: Watch models from the custom BC -> PPO pipeline.
- `train_breakout_ppo_openai_style.py`: Train PPO from scratch using OpenAI-style Atari wrappers.
- `continue_breakout_ppo_openai_style.py`: Continue OpenAI-style PPO from a saved checkpoint.
- `watch_breakout_ppo_openai_style.py`: Watch OpenAI-style PPO checkpoints.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install gymnasium ale-py stable-baselines3 torch numpy pillow pygame
```

If Breakout environments fail to load, install Atari extras:

```bash
pip install "gymnasium[atari,accept-rom-license]"
```

## Quick Start: Run the Uploaded Pretrained OpenAI-Style Model

```bash
.venv/bin/python watch_breakout_ppo_openai_style.py \
  --model-path models/pretrained/openai_style/breakout_openai_style_best_fp16_inference.zip
```

Optional stochastic policy actions:

```bash
.venv/bin/python watch_breakout_ppo_openai_style.py \
  --model-path models/pretrained/openai_style/breakout_openai_style_best_fp16_inference.zip \
  --stochastic
```

## Path A: Train OpenAI-Style PPO From Scratch

Train:

```bash
.venv/bin/python train_breakout_ppo_openai_style.py \
  --output-dir models/openai_ppo_breakout_style
```

Default recipe:

- Env: `BreakoutNoFrameskip-v4`
- Wrappers: `NoopResetEnv -> MaxAndSkip(4) -> EpisodicLifeEnv -> FireResetEnv -> Warp84Gray -> ClipRewardEnv -> FrameStack(4)`
- PPO: `n_envs=8`, `n_steps=128`, `n_epochs=4`, minibatches=4, total timesteps=`10_000_000`

Continue for +5M steps:

```bash
.venv/bin/python continue_breakout_ppo_openai_style.py \
  --output-dir models/openai_ppo_breakout_style \
  --additional-timesteps 5000000 \
  --n-envs 8
```

Watch the latest checkpoint in that directory:

```bash
.venv/bin/python watch_breakout_ppo_openai_style.py \
  --output-dir models/openai_ppo_breakout_style
```

## Path B: Train Custom BC -> PPO

### 1) Collect demonstrations

```bash
.venv/bin/python collect_breakout_data.py --output data/session4.npz
```

Controls:

- `Left/A`: move left
- `Right/D`: move right
- `Space/Up/W`: fire
- `Esc`: save and exit

Important defaults:

- Play/rendering env uses `--env-frameskip 1` (precise control while collecting).
- Saved dataset is converted to `--frameskip 4` by default.

### 2) Train behavior cloning

```bash
.venv/bin/python train_breakout_bc.py \
  --datasets data/session2.npz data/session3.npz data/session4.npz \
  --output models/breakout_bc_actor.pt
```

### 3) Fine-tune PPO from BC

```bash
.venv/bin/python train_breakout_ppo_from_bc.py \
  --bc-checkpoint models/breakout_bc_actor.pt \
  --output-dir models/ppo_breakout_bc \
  --human-render-freq 0
```

### 4) Watch custom PPO

```bash
.venv/bin/python watch_breakout_ppo.py \
  --output-dir models/ppo_breakout_bc
```

Or force a specific model:

```bash
.venv/bin/python watch_breakout_ppo.py \
  --model-path models/ppo_breakout_bc/best_model/best_model.zip
```

## Which Script Should I Use to Watch Which Model?

- OpenAI-style model (`BreakoutNoFrameskip-v4` wrappers): use `watch_breakout_ppo_openai_style.py`
- Custom BC/PPO model (`ALE/Breakout-v5` custom wrapper): use `watch_breakout_ppo.py`

Using the wrong watcher can cause observation-shape errors.

## Notes on Model Files and GitHub Size Limits

- GitHub rejects regular git files over 100 MB.
- This repo tracks only one pretrained OpenAI-style checkpoint under `models/pretrained/openai_style/`.
- Full training checkpoints under `models/openai_ppo_breakout_style/` and `models/ppo_breakout_bc/` are intentionally ignored by default.
- If you want to publish large checkpoints, use Git LFS.
