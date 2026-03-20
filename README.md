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

## Demo Video

<video controls width="860" src="https://github.com/Robel-Delelegn/PPO_Reinforcement_learning_on_Breakout/raw/main/assets/breakout_ppo_demo.mp4"></video>

If video playback does not start in your browser, use the direct file link:
[assets/breakout_ppo_demo.mp4](assets/breakout_ppo_demo.mp4)

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

## Training Transparency (Default Config)

All values below are script defaults. You can override them with CLI flags.

### OpenAI-style PPO (`train_breakout_ppo_openai_style.py`)

- Env: `BreakoutNoFrameskip-v4`
- Wrappers: `NoopReset -> MaxAndSkip(4) -> EpisodicLife -> FireReset -> Warp84Gray -> ClipReward -> FrameStack(4)`
- Total timesteps: `10,000,000`
- Parallel envs: `8`
- Rollout length (`n_steps`): `128`
- Rollout size per PPO iteration: `8 * 128 = 1,024` transitions
- Minibatches: `4` -> minibatch size `256`
- PPO epochs per iteration: `4`
- Approx PPO iterations for 10M steps: `10,000,000 / 1,024 ~= 9,766`
- Learning rate: linear schedule `2.5e-4 -> 0`
- Clip range: linear schedule `0.1 -> 0`
- Gamma / GAE lambda: `0.99 / 0.95`
- Entropy / value coefficients: `0.01 / 0.5`
- Max grad norm: `0.5`
- Checkpoint frequency: every `1,000,000` env steps

### Continue OpenAI-style PPO (`continue_breakout_ppo_openai_style.py`)

- Loads latest model from output directory unless `--model-path` is passed
- Additional timesteps per run: `5,000,000` (default)
- Parallel envs during continuation: `8` (default)
- Keeps global step counter (`reset_num_timesteps=False`)

### Behavior Cloning (`train_breakout_bc.py`)

- Input: stacked grayscale observations of shape `(4, 84, 84)`
- Default epochs: `20`
- Batch size: `256`
- Learning rate: `1e-4`
- Weight decay: `1e-5`
- Validation split: `10%`
- Class weighting: inverse-frequency, power `1.0`
- Data augmentation: left-right flip of observations, with label swap `LEFT <-> RIGHT`

### Custom PPO from BC (`train_breakout_ppo_from_bc.py`)

- Env: `ALE/Breakout-v5`, `obs_type=grayscale`, `full_action_space=False`
- Env frameskip: `4`
- Repeat action probability: `0.0`
- Observation wrapper: custom crop/resize/max-over-last-frame + frame stack
- Total timesteps: `5,000,000`
- Parallel envs: `4`
- Rollout length (`n_steps`): `128`
- Rollout size per PPO iteration: `4 * 128 = 512` transitions
- Batch size: `256`
- PPO epochs per iteration: `4`
- Approx PPO iterations for 5M steps: `5,000,000 / 512 ~= 9,766`
- Learning rate: constant `2.5e-4`
- Clip range: constant `0.1`
- Gamma / GAE lambda: `0.99 / 0.95`
- Entropy / value coefficients: `0.01 / 0.5`
- Max grad norm: `0.5`
- Eval frequency: every `50,000` env steps, `5` episodes
- Checkpoint frequency: every `100,000` env steps

## Which Script Should I Use to Watch Which Model?

- OpenAI-style model (`BreakoutNoFrameskip-v4` wrappers): use `watch_breakout_ppo_openai_style.py`
- Custom BC/PPO model (`ALE/Breakout-v5` custom wrapper): use `watch_breakout_ppo.py`

Using the wrong watcher can cause observation-shape errors.

## Model Files and GitHub Size Limits

- This repository includes one downloadable pretrained OpenAI-style checkpoint:
  `models/pretrained/openai_style/breakout_openai_style_best_fp16_inference.zip`
- Large rolling training checkpoints are not included in the repository because they exceed normal GitHub file limits.
- To generate full checkpoint histories locally, run the training commands in this README.

