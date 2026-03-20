#!/usr/bin/env python3
"""Train a behavior-cloning actor on Breakout demonstration data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class IndexedBreakoutDataset(Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray) -> None:
        self.observations = observations
        self.actions = actions

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, item: int):
        obs = torch.from_numpy(self.observations[item]).float()
        action = torch.tensor(int(self.actions[item]), dtype=torch.long)
        return obs, action


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


class BreakoutActor(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        num_actions: int,
        features_dim: int = 512,
        policy_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if policy_hidden_dim < 1:
            raise ValueError("policy_hidden_dim must be >= 1")
        self.encoder = BreakoutEncoder(observation_shape, features_dim=features_dim)
        self.policy_mlp = nn.Sequential(
            nn.Linear(features_dim, policy_hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(policy_hidden_dim, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs / 255.0
        features = self.encoder(obs)
        policy_latent = self.policy_mlp(features)
        return self.policy_head(policy_latent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        required=True,
        help="One or more .npz files produced by collect_breakout_data.py",
    )
    parser.add_argument("--output", type=Path, default=Path("models/breakout_bc_actor.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--features-dim", type=int, default=512)
    parser.add_argument("--policy-hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to inverse-frequency class weights. "
            "Use values > 1.0 to further emphasize rare actions."
        ),
    )
    return parser.parse_args()


def choose_device(raw_device: str) -> torch.device:
    if raw_device != "auto":
        return torch.device(raw_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_demo_data(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    observations = []
    actions = []
    metadata = []
    expected_frameskip: int | None = None

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            dataset_metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
            dataset_frameskip = dataset_metadata.get("frameskip")
            if dataset_frameskip is not None:
                dataset_frameskip = int(dataset_frameskip)
                if expected_frameskip is None:
                    expected_frameskip = dataset_frameskip
                elif dataset_frameskip != expected_frameskip:
                    raise ValueError(
                        f"Mismatched dataset frameskip: expected {expected_frameskip}, "
                        f"got {dataset_frameskip} for {path}"
                    )

            observations.append(data["observations"])
            actions.append(data["actions"].astype(np.int64))

            applied_metadata = dict(dataset_metadata)
            applied_metadata["source_path"] = str(path)
            metadata.append(applied_metadata)

    merged_observations = np.concatenate(observations, axis=0)
    merged_actions = np.concatenate(actions, axis=0)
    return merged_observations, merged_actions, metadata


def resolve_action_names(source_metadata: list[dict], num_actions: int) -> list[str]:
    action_names = [f"action_{idx}" for idx in range(num_actions)]
    for metadata in source_metadata:
        candidate_names = metadata.get("action_meanings")
        if candidate_names and len(candidate_names) == num_actions:
            return [str(name) for name in candidate_names]
    return action_names


def augment_left_right_symmetry(
    observations: np.ndarray,
    actions: np.ndarray,
    action_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, int] | None]:
    action_lookup = {name: idx for idx, name in enumerate(action_names)}
    left_action = action_lookup.get("LEFT")
    right_action = action_lookup.get("RIGHT")

    if left_action is None or right_action is None or left_action == right_action:
        return observations, actions, None

    flipped_observations = np.flip(observations, axis=-1).copy()
    flipped_actions = actions.copy()
    left_mask = actions == left_action
    right_mask = actions == right_action
    flipped_actions[left_mask] = right_action
    flipped_actions[right_mask] = left_action

    augmented_observations = np.concatenate([observations, flipped_observations], axis=0)
    augmented_actions = np.concatenate([actions, flipped_actions], axis=0)
    augmentation_info = {
        "left_action": int(left_action),
        "right_action": int(right_action),
        "added_examples": int(flipped_actions.shape[0]),
    }
    return augmented_observations, augmented_actions, augmentation_info


def summarize_confusion(confusion: torch.Tensor) -> dict[str, float | list[float]]:
    confusion = confusion.to(torch.float64)
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    true_positives = confusion.diag()

    recall = torch.zeros_like(true_positives)
    precision = torch.zeros_like(true_positives)

    recall_mask = support > 0
    precision_mask = predicted > 0
    recall[recall_mask] = true_positives[recall_mask] / support[recall_mask]
    precision[precision_mask] = true_positives[precision_mask] / predicted[precision_mask]

    f1 = torch.zeros_like(true_positives)
    f1_mask = (precision + recall) > 0
    f1[f1_mask] = 2 * precision[f1_mask] * recall[f1_mask] / (precision[f1_mask] + recall[f1_mask])

    total = float(confusion.sum().item())
    valid_classes = recall_mask

    accuracy = float(true_positives.sum().item() / total) if total > 0 else 0.0
    balanced_accuracy = float(recall[valid_classes].mean().item()) if valid_classes.any() else 0.0
    macro_f1 = float(f1[valid_classes].mean().item()) if valid_classes.any() else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "per_class_recall": [float(value) for value in recall.tolist()],
    }


def run_epoch(
    model: BreakoutActor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float | list[float]]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    num_actions = int(model.policy_head.out_features)
    confusion = torch.zeros((num_actions, num_actions), dtype=torch.int64)

    for batch_obs, batch_actions in loader:
        batch_obs = batch_obs.to(device, non_blocking=True)
        batch_actions = batch_actions.to(device, non_blocking=True)

        logits = model(batch_obs)
        loss = criterion(logits, batch_actions)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        predictions = logits.detach().argmax(dim=1)
        batch_size = batch_obs.shape[0]
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        pairs = (
            batch_actions.detach().to(torch.int64).cpu() * num_actions
            + predictions.to(torch.int64).cpu()
        )
        confusion += torch.bincount(pairs, minlength=num_actions * num_actions).reshape(num_actions, num_actions)

    mean_loss = total_loss / max(total_examples, 1)
    metrics = summarize_confusion(confusion)
    metrics["loss"] = mean_loss
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    observations, actions, source_metadata = load_demo_data(args.datasets)
    if observations.shape[0] == 0:
        raise SystemExit("The datasets are empty, so there is nothing to train on.")

    observation_shape = tuple(int(v) for v in observations.shape[1:])
    num_actions = int(actions.max()) + 1
    dataset_class_counts = np.bincount(actions, minlength=num_actions)
    action_names = resolve_action_names(source_metadata, num_actions)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(observations.shape[0])
    val_size = int(observations.shape[0] * args.val_split)
    if args.val_split > 0 and observations.shape[0] > 1:
        val_size = min(max(val_size, 1), observations.shape[0] - 1)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_observations = observations[train_indices]
    train_actions = actions[train_indices]
    val_observations = observations[val_indices]
    val_actions = actions[val_indices]

    train_observations, train_actions, augmentation_info = augment_left_right_symmetry(
        train_observations,
        train_actions,
        action_names,
    )
    train_class_counts = np.bincount(train_actions, minlength=num_actions)

    train_dataset = IndexedBreakoutDataset(train_observations, train_actions)
    val_dataset = IndexedBreakoutDataset(val_observations, val_actions)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = choose_device(args.device)
    model = BreakoutActor(
        observation_shape=observation_shape,
        num_actions=num_actions,
        features_dim=args.features_dim,
        policy_hidden_dim=args.policy_hidden_dim,
    ).to(device)

    safe_class_counts = np.maximum(train_class_counts, 1).astype(np.float64)
    weights = safe_class_counts.sum() / safe_class_counts
    weights = np.power(weights, args.class_weight_power)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_selection_key: tuple[float, float, float] | None = None
    best_val_loss = float("inf")
    best_balanced_accuracy = 0.0
    best_macro_f1 = 0.0

    print(f"Loaded {observations.shape[0]} transitions from {len(args.datasets)} file(s)")
    print(f"Observation shape: {observation_shape}")
    print(f"Dataset action counts: {dataset_class_counts.tolist()}")
    print(f"Training action counts: {train_class_counts.tolist()}")
    print(f"Policy hidden dim: {args.policy_hidden_dim}")
    print(f"Class weights: {[round(float(weight), 4) for weight in weights.tolist()]}")
    if augmentation_info is not None:
        print(
            "Left-right augmentation: enabled "
            f"(LEFT={augmentation_info['left_action']}, RIGHT={augmentation_info['right_action']}, "
            f"added_examples={augmentation_info['added_examples']})"
        )
    else:
        print("Left-right augmentation: skipped (LEFT/RIGHT actions were not found)")
    for metadata in source_metadata:
        source_path = metadata.get("source_path", "<unknown>")
        frameskip = metadata.get("frameskip", "?")
        total_steps = metadata.get("total_steps", "?")
        print(f"  {source_path}: frameskip={frameskip}, steps={total_steps}")
    print(f"Device: {device}")
    print()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        if len(val_dataset) > 0:
            val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
            selection_key = (
                float(val_metrics["balanced_accuracy"]),
                float(val_metrics["macro_f1"]),
                -float(val_metrics["loss"]),
            )
            if best_selection_key is None or selection_key > best_selection_key:
                best_selection_key = selection_key
                best_balanced_accuracy = float(val_metrics["balanced_accuracy"])
                best_macro_f1 = float(val_metrics["macro_f1"])
                best_val_loss = float(val_metrics["loss"])
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={float(train_metrics['loss']):.4f} "
                f"train_acc={float(train_metrics['accuracy']):.4f} "
                f"train_bal_acc={float(train_metrics['balanced_accuracy']):.4f} | "
                f"val_loss={float(val_metrics['loss']):.4f} "
                f"val_acc={float(val_metrics['accuracy']):.4f} "
                f"val_bal_acc={float(val_metrics['balanced_accuracy']):.4f} "
                f"val_macro_f1={float(val_metrics['macro_f1']):.4f}"
            )
            val_recalls = ", ".join(
                f"{name}={recall:.3f}"
                for name, recall in zip(action_names, val_metrics["per_class_recall"], strict=True)
            )
            print(f"           val_recall: {val_recalls}")
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_selection_key = (
                float(train_metrics["balanced_accuracy"]),
                float(train_metrics["macro_f1"]),
                -float(train_metrics["loss"]),
            )
            best_balanced_accuracy = float(train_metrics["balanced_accuracy"])
            best_macro_f1 = float(train_metrics["macro_f1"])
            best_val_loss = float(train_metrics["loss"])
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={float(train_metrics['loss']):.4f} "
                f"train_acc={float(train_metrics['accuracy']):.4f} "
                f"train_bal_acc={float(train_metrics['balanced_accuracy']):.4f} "
                f"train_macro_f1={float(train_metrics['macro_f1']):.4f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)

    checkpoint = {
        "model_state": model.state_dict(),
        "encoder_state": model.encoder.state_dict(),
        "policy_mlp_state": model.policy_mlp.state_dict(),
        "policy_head_state": model.policy_head.state_dict(),
        "observation_shape": observation_shape,
        "num_actions": num_actions,
        "features_dim": args.features_dim,
        "policy_hidden_dim": args.policy_hidden_dim,
        "class_counts": dataset_class_counts.tolist(),
        "train_class_counts": train_class_counts.tolist(),
        "class_weights": weights.tolist(),
        "class_weight_power": args.class_weight_power,
        "action_names": action_names,
        "symmetry_augmentation": augmentation_info,
        "best_metric": best_balanced_accuracy,
        "best_metric_name": "val_balanced_accuracy",
        "best_balanced_accuracy": best_balanced_accuracy,
        "best_macro_f1": best_macro_f1,
        "best_val_loss": best_val_loss,
        "source_datasets": [str(path) for path in args.datasets],
        "source_metadata": source_metadata,
        "seed": args.seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print()
    print(f"Saved behavior-cloning checkpoint to {args.output}")


if __name__ == "__main__":
    main()
