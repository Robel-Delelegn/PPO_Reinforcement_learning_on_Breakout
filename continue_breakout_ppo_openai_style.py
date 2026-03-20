#!/usr/bin/env python3
"""Continue training an OpenAI-style Breakout PPO model."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecTransposeImage

from train_breakout_ppo_openai_style import (
    PPO,
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    make_vec_env,
    wrap_openai_atari,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models/openai_ppo_breakout_style"))
    parser.add_argument("--env-id", default="BreakoutNoFrameskip-v4")
    parser.add_argument("--additional-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    return parser.parse_args()


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None

    checkpoint_pattern = re.compile(r"_(\d+)_steps\.zip$")
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("*.zip"):
        match = checkpoint_pattern.search(path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), path))

    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path is not None:
        if not args.model_path.exists():
            raise SystemExit(f"Model file not found: {args.model_path}")
        return args.model_path

    final_model = args.output_dir / "breakout_openai_style_final.zip"
    checkpoint = latest_checkpoint(args.output_dir / "checkpoints")
    latest_training_artifacts = [path for path in [final_model, checkpoint] if path is not None and path.exists()]
    if latest_training_artifacts:
        return max(latest_training_artifacts, key=lambda path: path.stat().st_mtime)

    raise SystemExit(
        f"No OpenAI-style PPO model found in {args.output_dir}.\n"
        "Expected one of:\n"
        "  breakout_openai_style_final.zip\n"
        "  checkpoints/openai_style_breakout_ppo_*_steps.zip"
    )


def build_env(env_id: str, n_envs: int, seed: int, channels_order: str | None):
    vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=wrap_openai_atari,
        vec_env_cls=vec_env_cls,
    )
    env = VecFrameStack(env, n_stack=4, channels_order=channels_order)
    if channels_order == "first":
        env = VecTransposeImage(env)
    return env


def checkpoint_layout(model_path: Path, device: str) -> tuple[str | None, str, int]:
    preview_model = PPO.load(model_path, device=device)
    expected_shape = tuple(int(v) for v in preview_model.observation_space.shape)
    existing_timesteps = int(preview_model.num_timesteps)
    del preview_model

    if expected_shape == (1, 336, 84):
        return "first", "legacy stacked-first layout", existing_timesteps
    if expected_shape == (4, 84, 84):
        return None, "correct channel-last stack with automatic CHW transpose", existing_timesteps

    raise SystemExit(
        f"Unsupported saved observation shape {expected_shape} in {model_path}.\n"
        "Expected either:\n"
        "  (1, 336, 84) for the legacy layout\n"
        "  (4, 84, 84) for the corrected layout"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args)
    channels_order, layout_name, existing_timesteps = checkpoint_layout(model_path, args.device)
    env = build_env(args.env_id, args.n_envs, args.seed, channels_order)
    model = PPO.load(model_path, env=env, device=args.device)

    if args.tensorboard_log is not None:
        model.tensorboard_log = str(args.tensorboard_log)

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
                save_path=str(args.output_dir / "checkpoints"),
                name_prefix="openai_style_breakout_ppo",
            )
        )

    target_timesteps = existing_timesteps + args.additional_timesteps

    print(f"Loaded model: {model_path}")
    print(f"Observation layout: {layout_name}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Current total timesteps: {existing_timesteps}")
    print(f"Additional timesteps: {args.additional_timesteps}")
    print(f"Target total timesteps: {target_timesteps}")
    print(f"Output dir: {args.output_dir}")
    print()

    model.learn(
        total_timesteps=args.additional_timesteps,
        callback=callbacks or None,
        reset_num_timesteps=False,
        tb_log_name="openai_style_breakout",
    )

    final_alias = args.output_dir / "breakout_openai_style_final"
    numbered_snapshot = args.output_dir / f"breakout_openai_style_{model.num_timesteps}_steps"
    model.save(str(final_alias))
    model.save(str(numbered_snapshot))
    env.close()

    print()
    print(f"Saved final model to {final_alias.with_suffix('.zip')}")
    print(f"Saved numbered snapshot to {numbered_snapshot.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
