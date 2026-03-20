#!/usr/bin/env python3
"""Train Breakout from scratch with the OpenAI Atari PPO setup.

This mirrors the published OpenAI Baselines Atari PPO2 configuration as closely
as practical in the current local stack:

- env id: BreakoutNoFrameskip-v4
- Atari wrappers: random no-ops, skip=4, max-pool over last 2 frames,
  episodic life, FIRE reset, reward clipping, grayscale 84x84, frame stack 4
- parallel envs: 8
- total timesteps: 10,000,000
- n_steps: 128
- minibatches: 4  -> batch_size 256 for 8 * 128 rollout size
- n_epochs: 4
- gamma: 0.99
- gae_lambda: 0.95
- ent_coef: 0.01
- vf_coef: 0.5
- max_grad_norm: 0.5
- learning_rate: linearly annealed from 2.5e-4 to 0
- clip_range: linearly annealed from 0.1 to 0
- policy: Nature CNN with 512-d feature layer and linear actor/value heads

Notes:
- This is a direct local Python script using gymnasium + ale-py +
  stable-baselines3, not the original TensorFlow 1 Baselines code.
- FIRE reset is intentionally enabled because it is part of the original
  Atari wrapper stack OpenAI used for PPO.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import ale_py
    import gymnasium as gym
    import numpy as np
    import torch.nn as nn
    from PIL import Image
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.atari_wrappers import (
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency.\n"
        "Use the project virtualenv or install:\n"
        "  pip install gymnasium ale-py stable-baselines3 torch pillow\n"
    ) from exc

if hasattr(gym, "register_envs"):
    gym.register_envs(ale_py)


def linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * initial_value

    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="BreakoutNoFrameskip-v4")
    parser.add_argument("--output-dir", type=Path, default=Path("models/openai_ppo_breakout_style"))
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-minibatches", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    return parser.parse_args()


class WarpFramePIL(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 1),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame):
        image = Image.fromarray(frame).convert("L")
        image = image.resize((self.width, self.height), Image.Resampling.BOX)
        array = np.array(image, dtype=np.uint8)
        return array[:, :, None]


def wrap_openai_atari(env: gym.Env) -> gym.Env:
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFramePIL(env, width=84, height=84)
    env = ClipRewardEnv(env)
    return env


def build_env(args: argparse.Namespace):
    batch_size = (args.n_envs * args.n_steps) // args.n_minibatches
    if batch_size < 2:
        raise SystemExit("Batch size must be at least 2.")

    vec_env_cls = DummyVecEnv if args.n_envs == 1 else SubprocVecEnv
    env = make_vec_env(
        args.env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        wrapper_class=wrap_openai_atari,
        vec_env_cls=vec_env_cls,
    )
    env = VecFrameStack(env, n_stack=4)
    return env, batch_size


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env, batch_size = build_env(args)

    policy_kwargs = dict(
        net_arch=[],
        activation_fn=nn.ReLU,
        ortho_init=False,
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=linear_schedule(args.learning_rate),
        n_steps=args.n_steps,
        batch_size=batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=linear_schedule(args.clip_range),
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(args.tensorboard_log) if args.tensorboard_log else None,
        seed=args.seed,
        verbose=1,
        device=args.device,
    )

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
                save_path=str(args.output_dir / "checkpoints"),
                name_prefix="openai_style_breakout_ppo",
            )
        )

    print(f"Environment: {args.env_id}")
    print("Wrapper stack: NoopReset -> MaxAndSkip(4) -> EpisodicLife -> FireReset -> Warp84Gray -> ClipReward -> FrameStack(4)")
    print("Policy: NatureCNN(32x8/4, 64x4/2, 64x3/1, fc512) + linear actor/value heads")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Rollout size: {args.n_envs * args.n_steps}")
    print(f"Minibatch size: {batch_size}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Output dir: {args.output_dir}")
    print()

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks or None)
    model.save(str(args.output_dir / "breakout_openai_style_final"))
    env.close()

    print()
    print(f"Saved final model to {args.output_dir / 'breakout_openai_style_final.zip'}")


if __name__ == "__main__":
    main()
