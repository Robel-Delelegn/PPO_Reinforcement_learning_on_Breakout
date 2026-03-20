#!/usr/bin/env python3
"""Watch a Breakout PPO model trained with the OpenAI-style Atari setup."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from train_breakout_ppo_openai_style import PPO, DummyVecEnv, VecFrameStack, make_vec_env, wrap_openai_atari
from stable_baselines3.common.vec_env import VecTransposeImage

try:
    import pygame
except ModuleNotFoundError:
    pygame = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models/openai_ppo_breakout_style"))
    parser.add_argument("--env-id", default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render-scale", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="auto")
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
        f"No PPO model found in {args.output_dir}.\n"
        "Expected one of:\n"
        "  breakout_openai_style_final.zip\n"
        "  checkpoints/openai_style_breakout_ppo_*_steps.zip"
    )


def ensure_pygame() -> None:
    if pygame is None:
        raise SystemExit(
            "Missing dependency.\n"
            "Install pygame in your virtual environment:\n"
            "  pip install pygame"
        )
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()


def process_events() -> bool:
    assert pygame is not None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
    return True


def draw_frame(screen, frame: np.ndarray, render_scale: int) -> None:
    assert pygame is not None
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    if render_scale != 1:
        surface = pygame.transform.scale(
            surface,
            (frame.shape[1] * render_scale, frame.shape[0] * render_scale),
        )
    screen.blit(surface, (0, 0))
    pygame.display.flip()


def build_env(env_id: str, seed: int, channels_order: str | None = None):
    env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        wrapper_class=wrap_openai_atari,
        vec_env_cls=DummyVecEnv,
    )
    env = VecFrameStack(env, n_stack=4, channels_order=channels_order)
    env = VecTransposeImage(env)
    return env


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args)
    ensure_pygame()
    assert pygame is not None

    model = PPO.load(model_path, device=args.device)
    expected_shape = tuple(int(v) for v in model.observation_space.shape)
    if expected_shape == (1, 336, 84):
        channels_order = "first"
        layout_name = "legacy stacked-first layout"
    elif expected_shape == (4, 84, 84):
        channels_order = None
        layout_name = "correct channel-last stack transposed to CHW"
    else:
        raise SystemExit(
            f"Unsupported saved observation shape {expected_shape} in {model_path}. "
            "Expected either (1, 336, 84) for the legacy layout or (4, 84, 84) for the corrected layout."
        )

    env = build_env(args.env_id, args.seed, channels_order=channels_order)

    obs = env.reset()
    first_frame = env.envs[0].render()
    if first_frame is None:
        raise SystemExit("The environment did not return a frame for render_mode='rgb_array'.")

    window_size = (first_frame.shape[1] * args.render_scale, first_frame.shape[0] * args.render_scale)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Breakout PPO viewer (OpenAI style)")
    clock = pygame.time.Clock()
    draw_frame(screen, first_frame, args.render_scale)
    episode_return = 0.0
    episode_steps = 0
    episode_index = 1
    deterministic = not args.stochastic

    print(f"Loaded model: {model_path}")
    print("Environment stack: OpenAI-style Atari wrappers + frame stack 4")
    print(f"Observation layout: {layout_name}")
    print(f"Action mode: {'stochastic sampling' if args.stochastic else 'deterministic argmax'}")
    print("Controls:")
    print("  Esc / Q / close window -> quit")
    print()

    try:
        running = True
        while running:
            running = process_events()
            if not running:
                break

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, _ = env.step(action)
            episode_return += float(rewards[0])
            episode_steps += 1

            frame = env.envs[0].render()
            if frame is not None:
                draw_frame(screen, frame, args.render_scale)
            clock.tick(args.fps)

            if bool(dones[0]):
                print(
                    f"Episode {episode_index} finished: "
                    f"return={episode_return:.1f}, steps={episode_steps}"
                )
                episode_index += 1
                episode_return = 0.0
                episode_steps = 0
                obs = env.reset()
                frame = env.envs[0].render()
                if frame is not None:
                    draw_frame(screen, frame, args.render_scale)
    finally:
        env.close()
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
