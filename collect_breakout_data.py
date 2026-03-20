#!/usr/bin/env python3
"""Play Atari Breakout with the keyboard and save imitation-learning data."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import ale_py
    import gymnasium as gym
    import pygame
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency.\n"
        "Create a virtual environment and install:\n"
        "  pip install gymnasium ale-py pygame"
    ) from exc

if hasattr(gym, "register_envs"):
    gym.register_envs(ale_py)


class BreakoutStackWrapper(gym.Wrapper):
    """Crop, resize, max-pool, and stack grayscale Breakout frames."""

    def __init__(
        self,
        env: gym.Env,
        screen_size: int = 84,
        stack_size: int = 4,
        crop_top: int = 34,
        crop_bottom: int = 194,
        max_over_last_frame: bool = True,
    ) -> None:
        super().__init__(env)
        self.screen_size = screen_size
        self.stack_size = stack_size
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.max_over_last_frame = max_over_last_frame

        cropped_height = crop_bottom - crop_top
        self._row_idx = np.linspace(0, cropped_height - 1, screen_size).astype(np.int32)
        self._col_idx = np.linspace(0, env.observation_space.shape[1] - 1, screen_size).astype(np.int32)
        self._frames: deque[np.ndarray] = deque(maxlen=stack_size)
        self._previous_frame: np.ndarray | None = None

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(stack_size, screen_size, screen_size),
            dtype=np.uint8,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        cropped = frame[self.crop_top:self.crop_bottom, :]
        resized = cropped[np.ix_(self._row_idx, self._col_idx)].astype(np.uint8, copy=False)
        if self.max_over_last_frame and self._previous_frame is not None:
            resized = np.maximum(resized, self._previous_frame)
        self._previous_frame = resized
        return resized

    def _stack(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._previous_frame = None
        frame = self._preprocess(obs)
        self._frames.clear()
        for _ in range(self.stack_size):
            self._frames.append(frame)
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._preprocess(obs)
        self._frames.append(frame)
        return self._stack(), reward, terminated, truncated, info


class TrajectoryRecorder:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.terminateds: list[bool] = []
        self.truncateds: list[bool] = []
        self.episode_ids: list[int] = []

    def record(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        episode_id: int,
    ) -> None:
        self.observations.append(observation.copy())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.terminateds.append(bool(terminated))
        self.truncateds.append(bool(truncated))
        self.episode_ids.append(int(episode_id))

    def arrays(self, observation_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        if self.observations:
            observations = np.asarray(self.observations, dtype=np.uint8)
        else:
            observations = np.empty((0, *observation_shape), dtype=np.uint8)
        return {
            "observations": observations,
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "terminateds": np.asarray(self.terminateds, dtype=np.bool_),
            "truncateds": np.asarray(self.truncateds, dtype=np.bool_),
            "episode_ids": np.asarray(self.episode_ids, dtype=np.int32),
        }


def convert_dataset_frameskip(
    dataset: dict[str, np.ndarray],
    output_frameskip: int,
    source_frameskip: int,
) -> dict[str, np.ndarray]:
    if output_frameskip < source_frameskip:
        raise ValueError(
            f"Output frameskip {output_frameskip} cannot be smaller than source env frameskip {source_frameskip}"
        )
    if output_frameskip % source_frameskip != 0:
        raise ValueError(
            f"Output frameskip {output_frameskip} must be divisible by source env frameskip {source_frameskip}"
        )

    stride = output_frameskip // source_frameskip
    if stride == 1:
        return dataset

    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminateds = dataset["terminateds"]
    truncateds = dataset["truncateds"]
    episode_ids = dataset["episode_ids"]

    if observations.shape[0] == 0:
        return dataset

    stack_size = int(observations.shape[1])
    converted_observations: list[np.ndarray] = []
    converted_actions: list[int] = []
    converted_rewards: list[float] = []
    converted_terminateds: list[bool] = []
    converted_truncateds: list[bool] = []
    converted_episode_ids: list[int] = []

    start = 0
    total_steps = int(actions.shape[0])
    while start < total_steps:
        end = start + 1
        episode_id = int(episode_ids[start])
        while end < total_steps and int(episode_ids[end]) == episode_id:
            end += 1

        decision_indices = np.arange(start, end, stride, dtype=np.int64)
        boundary_frames = observations[decision_indices, -1, :, :]

        for offset, raw_idx in enumerate(decision_indices):
            frame_history = boundary_frames[max(0, offset - stack_size + 1): offset + 1]
            if frame_history.shape[0] < stack_size:
                pad = np.repeat(frame_history[:1], stack_size - frame_history.shape[0], axis=0)
                stacked_obs = np.concatenate([pad, frame_history], axis=0)
            else:
                stacked_obs = frame_history

            chunk_end = min(int(raw_idx) + stride, end)
            converted_observations.append(stacked_obs.astype(np.uint8, copy=False))
            converted_actions.append(int(actions[raw_idx]))
            converted_rewards.append(float(rewards[raw_idx:chunk_end].sum()))
            converted_terminateds.append(bool(terminateds[raw_idx:chunk_end].any()))
            converted_truncateds.append(bool(truncateds[raw_idx:chunk_end].any()))
            converted_episode_ids.append(episode_id)

        start = end

    return {
        "observations": np.asarray(converted_observations, dtype=np.uint8),
        "actions": np.asarray(converted_actions, dtype=np.int64),
        "rewards": np.asarray(converted_rewards, dtype=np.float32),
        "terminateds": np.asarray(converted_terminateds, dtype=np.bool_),
        "truncateds": np.asarray(converted_truncateds, dtype=np.bool_),
        "episode_ids": np.asarray(converted_episode_ids, dtype=np.int32),
    }


def save_dataset(output_path: Path, dataset: dict[str, np.ndarray], metadata: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, metadata=np.asarray(json.dumps(metadata)), **dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/breakout_demos.npz"))
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stack-size", type=int, default=4)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument(
        "--frameskip",
        type=int,
        default=4,
        help="Frameskip stored in the saved dataset.",
    )
    parser.add_argument(
        "--env-frameskip",
        type=int,
        default=1,
        help="Frameskip used while playing and rendering the environment.",
    )
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--render-scale", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> gym.Env:
    env = gym.make(
        args.env_id,
        render_mode="rgb_array",
        obs_type="grayscale",
        full_action_space=False,
        frameskip=args.env_frameskip,
        repeat_action_probability=args.repeat_action_probability,
    )
    return BreakoutStackWrapper(
        env,
        screen_size=args.screen_size,
        stack_size=args.stack_size,
    )


def render_frame(screen: pygame.Surface, frame: np.ndarray, scale: int) -> None:
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    if scale != 1:
        surface = pygame.transform.scale(
            surface,
            (frame.shape[1] * scale, frame.shape[0] * scale),
        )
    screen.blit(surface, (0, 0))
    pygame.display.flip()


def current_action(
    keys: pygame.key.ScancodeWrapper,
    noop_action: int,
    fire_action: int,
    right_action: int,
    left_action: int,
) -> int:
    left_pressed = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right_pressed = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire_pressed = keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]

    if left_pressed and not right_pressed:
        return left_action
    if right_pressed and not left_pressed:
        return right_action
    if fire_pressed:
        return fire_action
    return noop_action


def prompt_continue(
    screen: pygame.Surface,
    last_frame: np.ndarray | None,
    scale: int,
    episode_id: int,
    episode_return: float,
    episode_steps: int,
) -> bool:
    prompt_font = pygame.font.SysFont(None, 28)
    info_font = pygame.font.SysFont(None, 22)
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))

    if last_frame is not None:
        render_frame(screen, last_frame, scale)
    screen.blit(overlay, (0, 0))

    lines = [
        f"Episode {episode_id} finished",
        f"Return: {episode_return:.1f}    Steps: {episode_steps}",
        "Press Y or Enter to play another game",
        "Press N, Esc, or close the window to save and quit",
    ]
    fonts = [prompt_font, info_font, info_font, info_font]

    center_x = screen.get_width() // 2
    top_y = screen.get_height() // 2 - 60
    for idx, (line, font) in enumerate(zip(lines, fonts, strict=True)):
        text = font.render(line, True, (255, 255, 255))
        rect = text.get_rect(center=(center_x, top_y + idx * 32))
        screen.blit(text, rect)
    pygame.display.flip()

    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            return False
        if event.type != pygame.KEYDOWN:
            continue
        if event.key in (pygame.K_y, pygame.K_RETURN, pygame.K_KP_ENTER):
            return True
        if event.key in (pygame.K_n, pygame.K_ESCAPE, pygame.K_q):
            return False


def main() -> None:
    args = parse_args()
    env = make_env(args)
    recorder = TrajectoryRecorder()

    action_meanings = (
        list(env.unwrapped.get_action_meanings())
        if hasattr(env.unwrapped, "get_action_meanings")
        else ["NOOP", "FIRE", "RIGHT", "LEFT"]
    )
    action_lookup = {name: idx for idx, name in enumerate(action_meanings)}
    noop_action = action_lookup.get("NOOP", 0)
    fire_action = action_lookup.get("FIRE", 1)
    right_action = action_lookup.get("RIGHT", 2)
    left_action = action_lookup.get("LEFT", 3)

    pygame.init()
    obs, _ = env.reset(seed=args.seed)
    first_frame = env.render()
    if first_frame is None:
        raise RuntimeError("The environment did not return a frame after reset().")

    window_size = (first_frame.shape[1] * args.render_scale, first_frame.shape[0] * args.render_scale)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Breakout demo collection")
    clock = pygame.time.Clock()

    print("Controls:")
    print("  Left / A  -> move left")
    print("  Right / D -> move right")
    print("  Space     -> FIRE")
    print("  Esc       -> save and quit")
    print("  After each episode: Y/Enter to continue, N/Esc to stop")
    print(f"Display env frameskip: {args.env_frameskip}")
    print(f"Saved dataset frameskip: {args.frameskip}")
    print()

    render_frame(screen, first_frame, args.render_scale)

    running = True
    episode_id = 0
    episode_return = 0.0
    episode_steps = 0
    completed_episodes = 0
    total_reward = 0.0

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if not running:
                break

            action = current_action(
                pygame.key.get_pressed(),
                noop_action=noop_action,
                fire_action=fire_action,
                right_action=right_action,
                left_action=left_action,
            )
            current_obs = obs
            obs, reward, terminated, truncated, _ = env.step(action)
            recorder.record(current_obs, action, reward, terminated, truncated, episode_id)

            episode_return += reward
            episode_steps += 1
            total_reward += reward

            frame = env.render()
            if frame is not None:
                render_frame(screen, frame, args.render_scale)
            clock.tick(args.fps)

            if terminated or truncated:
                completed_episodes += 1
                print(
                    f"Episode {episode_id} finished: return={episode_return:.1f}, "
                    f"steps={episode_steps}"
                )
                continue_playing = prompt_continue(
                    screen=screen,
                    last_frame=frame,
                    scale=args.render_scale,
                    episode_id=episode_id,
                    episode_return=episode_return,
                    episode_steps=episode_steps,
                )
                if not continue_playing:
                    break
                episode_id += 1
                episode_return = 0.0
                episode_steps = 0
                obs, _ = env.reset()
                frame = env.render()
                if frame is not None:
                    render_frame(screen, frame, args.render_scale)
    finally:
        observation_shape = tuple(int(v) for v in env.observation_space.shape)
        raw_dataset = recorder.arrays(observation_shape)
        converted_dataset = convert_dataset_frameskip(
            raw_dataset,
            output_frameskip=args.frameskip,
            source_frameskip=args.env_frameskip,
        )
        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "env_id": args.env_id,
            "observation_shape": list(converted_dataset["observations"].shape[1:]),
            "action_meanings": action_meanings,
            "stack_size": args.stack_size,
            "screen_size": args.screen_size,
            "frameskip": args.frameskip,
            "env_frameskip": args.env_frameskip,
            "repeat_action_probability": args.repeat_action_probability,
            "seed": args.seed,
            "completed_episodes": completed_episodes,
            "total_steps": int(converted_dataset["actions"].shape[0]),
            "total_reward": float(converted_dataset["rewards"].sum()),
            "controls": {
                "left": ["left", "a"],
                "right": ["right", "d"],
                "fire": ["space"],
                "quit": ["escape", "window close"],
            },
        }
        if args.frameskip != args.env_frameskip:
            metadata["conversion_method"] = "decision_boundary_restack"
            metadata["conversion_stride"] = args.frameskip // args.env_frameskip
            metadata["raw_total_steps"] = int(raw_dataset["actions"].shape[0])
            metadata["raw_total_reward"] = float(raw_dataset["rewards"].sum())
        save_dataset(args.output, converted_dataset, metadata)
        env.close()
        pygame.quit()

    print()
    print(f"Saved {metadata['total_steps']} transitions to {args.output}")


if __name__ == "__main__":
    main()
