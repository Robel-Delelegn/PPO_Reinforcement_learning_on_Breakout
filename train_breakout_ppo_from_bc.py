#!/usr/bin/env python3
"""Warm-start PPO on Breakout from a behavior-cloned actor."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    import ale_py
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency.\n"
        "Create a virtual environment and install:\n"
        "  pip install gymnasium ale-py stable-baselines3"
    ) from exc

if hasattr(gym, "register_envs"):
    gym.register_envs(ale_py)

try:
    import pygame
except ModuleNotFoundError:
    pygame = None


class BreakoutStackWrapper(gym.Wrapper):
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

        self.observation_space = spaces.Box(
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


class BreakoutEncoder(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], features_dim: int = 512) -> None:
        super().__init__()
        channels = observation_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_shape, dtype=torch.float32)).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs))


class BreakoutCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        self.encoder = BreakoutEncoder(tuple(int(v) for v in observation_space.shape), features_dim=features_dim)
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


class HumanRenderCallback(BaseCallback):
    def __init__(
        self,
        env_fn,
        render_freq: int,
        deterministic: bool = False,
        render_scale: int = 4,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.env_fn = env_fn
        self.render_freq = render_freq
        self.deterministic = deterministic
        self.render_scale = render_scale
        self.render_env = None
        self._enabled = render_freq > 0
        self._obs: np.ndarray | None = None
        self._episode_return = 0.0
        self._episode_steps = 0
        self._episode_count = 0
        self._screen = None

    def _disable(self, message: str) -> None:
        print(message)
        self._enabled = False
        if self.render_env is not None:
            self.render_env.close()
            self.render_env = None
        self._obs = None
        if pygame is not None and self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None

    def _ensure_env(self) -> bool:
        if not self._enabled:
            return False
        if pygame is None:
            self._disable("Human render disabled: pygame is required for the live viewer")
            return False
        if self.render_env is not None:
            return True
        try:
            self.render_env = self.env_fn()
            return True
        except Exception as exc:  # pragma: no cover - depends on local display setup
            self._disable(f"Human render disabled: could not create render env ({exc})")
            return False

    def _process_window_events(self) -> bool:
        assert pygame is not None
        if not pygame.get_init() or not pygame.display.get_init():
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._disable("Human render window closed; PPO training continues in hidden environments")
                return False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                self._disable("Human render disabled from keyboard; PPO training continues in hidden environments")
                return False
        return True

    def _draw_frame(self) -> bool:
        if not self._ensure_env():
            return False
        assert self.render_env is not None
        assert pygame is not None

        frame = self.render_env.render()
        if frame is None:
            return True

        if self._screen is None:
            if not pygame.get_init():
                pygame.init()
            if not pygame.display.get_init():
                pygame.display.init()
            window_size = (frame.shape[1] * self.render_scale, frame.shape[0] * self.render_scale)
            self._screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("Breakout PPO live viewer")

        if not self._process_window_events():
            return False

        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        if self.render_scale != 1:
            surface = pygame.transform.scale(
                surface,
                (frame.shape[1] * self.render_scale, frame.shape[0] * self.render_scale),
            )
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()
        return True

    def _reset_render_episode(self) -> None:
        assert self.render_env is not None
        self._obs, _ = self.render_env.reset()
        self._episode_return = 0.0
        self._episode_steps = 0
        self._draw_frame()

    def _step_render_env(self) -> None:
        if not self._ensure_env():
            return

        try:
            if self._obs is None:
                self._reset_render_episode()
            if not self._enabled:
                return

            if not self._process_window_events():
                return

            action, _ = self.model.predict(self._obs, deterministic=self.deterministic)
            self._obs, reward, terminated, truncated, _ = self.render_env.step(int(np.asarray(action).item()))
            self._episode_return += float(reward)
            self._episode_steps += 1
            self._draw_frame()

            if terminated or truncated:
                self._episode_count += 1
                print(
                    f"Live render episode {self._episode_count}: "
                    f"return={self._episode_return:.1f}, steps={self._episode_steps}, "
                    f"timesteps={self.num_timesteps}"
                )
                self._reset_render_episode()
        except Exception as exc:  # pragma: no cover - depends on local display setup
            self._disable(f"Human render disabled after runtime error: {exc}")

    def _on_training_start(self) -> None:
        if self._enabled:
            mode = "deterministic argmax" if self.deterministic else "stochastic policy sampling"
            print(f"Human render enabled: showing one live policy-controlled environment ({mode})")
            if self._ensure_env():
                self._reset_render_episode()

    def _on_step(self) -> bool:
        if not self._enabled:
            return True
        if self.n_calls % self.render_freq != 0:
            return True
        self._step_render_env()
        return True

    def _on_training_end(self) -> None:
        if self._enabled:
            self._disable("Human render closed at end of PPO training")
        else:
            if self.render_env is not None:
                self.render_env.close()
                self.render_env = None
            self._obs = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/ppo_breakout_bc"))
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--human-render-freq", type=int, default=1)
    parser.add_argument("--human-render-deterministic", action="store_true")
    parser.add_argument("--human-render-scale", type=int, default=4)
    return parser.parse_args()


def choose_device(raw_device: str) -> str:
    if raw_device != "auto":
        return raw_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_env_fn(
    env_id: str,
    seed: int,
    frameskip: int,
    repeat_action_probability: float,
    render_mode: str | None = None,
):
    def _init():
        env = gym.make(
            env_id,
            render_mode=render_mode,
            obs_type="grayscale",
            full_action_space=False,
            frameskip=frameskip,
            repeat_action_probability=repeat_action_probability,
        )
        env = BreakoutStackWrapper(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def load_bc_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {"encoder_state", "policy_head_state", "observation_shape", "num_actions", "features_dim"}
    missing = required_keys.difference(checkpoint)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Checkpoint {path} is missing required keys: {missing_keys}")
    return checkpoint


def checkpoint_policy_hidden_dim(checkpoint: dict) -> int:
    return int(checkpoint.get("policy_hidden_dim", 0))


def warm_start_policy(model: PPO, checkpoint: dict) -> None:
    expected_obs_shape = tuple(int(v) for v in checkpoint["observation_shape"])
    if expected_obs_shape != tuple(int(v) for v in model.observation_space.shape):
        raise ValueError(
            f"BC observation shape {expected_obs_shape} does not match PPO env shape "
            f"{tuple(int(v) for v in model.observation_space.shape)}"
        )
    expected_actions = int(checkpoint["num_actions"])
    if expected_actions != int(model.action_space.n):
        raise ValueError(
            f"BC action count {expected_actions} does not match PPO action count {model.action_space.n}"
        )

    model.policy.features_extractor.encoder.load_state_dict(checkpoint["encoder_state"])
    policy_hidden_dim = checkpoint_policy_hidden_dim(checkpoint)
    if policy_hidden_dim > 0:
        policy_mlp_state = checkpoint.get("policy_mlp_state")
        if policy_mlp_state is None:
            raise ValueError("BC checkpoint is missing policy_mlp_state for a nonzero policy_hidden_dim")
        model.policy.mlp_extractor.policy_net.load_state_dict(policy_mlp_state)
    model.policy.action_net.load_state_dict(checkpoint["policy_head_state"])


def build_callbacks(args: argparse.Namespace, eval_env):
    callbacks = []

    human_render_callback = HumanRenderCallback(
        env_fn=make_env_fn(
            env_id=args.env_id,
            seed=args.seed + 20_000,
            frameskip=args.frameskip,
            repeat_action_probability=args.repeat_action_probability,
            render_mode="rgb_array",
        ),
        render_freq=args.human_render_freq,
        deterministic=args.human_render_deterministic,
        render_scale=args.human_render_scale,
    )
    if human_render_callback.render_freq > 0:
        callbacks.append(human_render_callback)

    if args.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(args.output_dir / "best_model"),
                log_path=str(args.output_dir / "eval_logs"),
                eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            )
        )

    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
                save_path=str(args.output_dir / "checkpoints"),
                name_prefix="breakout_ppo_bc",
            )
        )

    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]

    return CallbackList(callbacks)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size > args.n_steps * args.n_envs:
        raise SystemExit("--batch-size must be <= n_steps * n_envs for PPO.")

    checkpoint = load_bc_checkpoint(args.bc_checkpoint)
    device = choose_device(args.device)
    policy_hidden_dim = checkpoint_policy_hidden_dim(checkpoint)
    if policy_hidden_dim > 0:
        net_arch = dict(pi=[policy_hidden_dim], vf=[policy_hidden_dim])
    else:
        net_arch = dict(pi=[], vf=[])

    policy_kwargs = dict(
        features_extractor_class=BreakoutCNNExtractor,
        features_extractor_kwargs=dict(features_dim=int(checkpoint["features_dim"])),
        net_arch=net_arch,
        activation_fn=nn.ReLU,
        ortho_init=False,
        share_features_extractor=True,
    )

    vec_cls = DummyVecEnv if args.n_envs == 1 else SubprocVecEnv
    train_env = VecMonitor(
        vec_cls(
            [
                make_env_fn(
                    env_id=args.env_id,
                    seed=args.seed + idx,
                    frameskip=args.frameskip,
                    repeat_action_probability=args.repeat_action_probability,
                    render_mode=None,
                )
                for idx in range(args.n_envs)
            ]
        )
    )
    eval_env = VecMonitor(
        DummyVecEnv(
            [
                make_env_fn(
                    env_id=args.env_id,
                    seed=args.seed + 10_000,
                    frameskip=args.frameskip,
                    repeat_action_probability=args.repeat_action_probability,
                    render_mode=None,
                )
            ]
        )
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(args.tensorboard_log) if args.tensorboard_log else None,
        seed=args.seed,
        verbose=1,
        device=device,
    )

    warm_start_policy(model, checkpoint)
    model.save(str(args.output_dir / "ppo_bc_initialized"))

    print(f"Loaded BC checkpoint from {args.bc_checkpoint}")
    print(f"Policy hidden dim: {policy_hidden_dim}")
    print(f"Training PPO for {args.total_timesteps} timesteps on {args.n_envs} env(s)")
    print(f"Device: {device}")
    print()

    callbacks = build_callbacks(args, eval_env)
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    model.save(str(args.output_dir / "ppo_bc_final"))

    train_env.close()
    eval_env.close()

    print()
    print(f"Saved initialized model to {args.output_dir / 'ppo_bc_initialized.zip'}")
    print(f"Saved final PPO model to {args.output_dir / 'ppo_bc_final.zip'}")


if __name__ == "__main__":
    main()
