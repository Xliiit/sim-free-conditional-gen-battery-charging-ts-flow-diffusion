from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .distributed_mode import is_main_process
from .utils import plot_trajectories


def resolve_data_paths(config_root: Path, paths: Sequence[str]) -> List[Path]:
    resolved = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (config_root / candidate).resolve()
        resolved.append(candidate)
    return resolved


def parse_str_to_100d_array(policy: str) -> np.ndarray:
    segments = policy.split("-")
    if not segments:
        raise ValueError("Invalid charge policy string.")

    pattern = r"^(\d+(?:\.\d+)?)\((\d+)%\)$"
    parsed = []
    for segment in segments:
        match = re.match(pattern, segment.strip())
        if not match:
            raise ValueError(f"Invalid segment format: {segment}")
        value = float(match.group(1))
        percentage = int(match.group(2))
        parsed.append((value, percentage))

    percentages = [item[1] for item in parsed]
    if percentages[-1] != 100:
        raise ValueError("The last percentage must be 100.")
    if percentages != sorted(percentages):
        raise ValueError("Percentages must be monotonically increasing.")

    arr = np.zeros(100, dtype=np.float32)
    prev_pct = 0
    for value, current_pct in parsed:
        arr[prev_pct:current_pct] = value
        prev_pct = current_pct
    return arr


def inverse_padding(
    data_i: np.ndarray,
    data_u: np.ndarray,
    data_qc: np.ndarray,
) -> Tuple[Iterable[np.ndarray], Iterable[np.ndarray], Iterable[np.ndarray], np.ndarray]:
    data_i = np.asarray(data_i)
    data_u = np.asarray(data_u)
    data_qc = np.asarray(data_qc)
    if data_i.ndim == 1:
        data_i = data_i.reshape(1, -1)
        data_u = data_u.reshape(1, -1)
        data_qc = data_qc.reshape(1, -1)
    if data_i.ndim != 2:
        raise ValueError(f"Expected 1D or 2D arrays, got {data_i.ndim}D")
    if data_i.shape != data_u.shape or data_i.shape != data_qc.shape:
        raise ValueError("Input arrays must have the same shape.")

    non_zero_mask = ~np.isclose(data_i, 0, atol=1e-6)
    lengths = np.zeros(data_i.shape[0], dtype=int)
    for idx in range(data_i.shape[0]):
        sample_non_zero = non_zero_mask[idx]
        if np.any(sample_non_zero):
            lengths[idx] = np.max(np.where(sample_non_zero)[0]) + 1

    processed_i = [data_i[idx, : lengths[idx]] for idx in range(data_i.shape[0])]
    processed_u = [data_u[idx, : lengths[idx]] for idx in range(data_i.shape[0])]
    processed_qc = [data_qc[idx, : lengths[idx]] for idx in range(data_i.shape[0])]
    if len(processed_i) == 1:
        return processed_i[0], processed_u[0], processed_qc[0], lengths[0]
    return processed_i, processed_u, processed_qc, lengths


def _flatten_samples(data_paths: Sequence[Path]):
    data_i = []
    data_u = []
    data_qc = []
    data_soh = []
    charge_policies = []
    raw_charge_policies = []
    data_labels = []

    for path in data_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with open(path, "rb") as handle:
            cells = pickle.load(handle)

        for cell in cells:
            label_str = cell.get("label", "Healthy")
            label = 1 if "ISC" in label_str else 0
            policy = parse_str_to_100d_array(cell["charge_policy"])
            for cycle in cell["cycles"]:
                data_i.append(np.asarray(cycle["I"], dtype=np.float32))
                data_u.append(np.asarray(cycle["V"], dtype=np.float32))
                data_qc.append(np.asarray(cycle["Qc"], dtype=np.float32))
                data_soh.append(np.float32(cycle["SOH"]))
                charge_policies.append(policy.copy())
                raw_charge_policies.append(cell["charge_policy"])
                data_labels.append(label)

        if is_main_process():
            print(
                f"Loaded {path} with {len(cells)} cells and {len(data_i)} total samples."
            )

    return (
        data_i,
        data_u,
        data_qc,
        data_soh,
        charge_policies,
        raw_charge_policies,
        data_labels,
    )


def _compute_stats(values: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    concatenated = np.concatenate(values)
    return (
        np.array([np.mean(concatenated)], dtype=np.float32),
        np.array([np.std(concatenated) + 1e-7], dtype=np.float32),
    )


def _compute_protocol_stats(values: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(values, axis=0)
    return (
        stacked.mean(axis=0).astype(np.float32),
        (stacked.std(axis=0) + 1e-7).astype(np.float32),
    )


def _default_stats_path(args) -> Path:
    if args.resume:
        candidate = Path(args.resume).resolve().parent / "normalization.pkl"
        if candidate.exists():
            return candidate
        fallback = Path(args.resume).resolve().parent.parent / "normalization.pkl"
        if fallback.exists():
            return fallback
    return Path(args.run_dir) / "normalization.pkl"


def build_dataset(args):
    data_cfg = args.dataset_params
    save_path = Path(args.run_dir)
    if args.eval_only and data_cfg.get("valid_files"):
        rel_paths = data_cfg.get("valid_files", [])
    else:
        rel_paths = data_cfg.get("battery_files", [])

    paths = resolve_data_paths(Path(args.config_root), rel_paths)
    (
        train_data_i,
        train_data_u,
        train_qc,
        train_soh,
        charge_policies,
        raw_charge_policies,
        data_labels,
    ) = _flatten_samples(paths)

    if is_main_process() and train_data_i:
        plot_trajectories(
            data_i=train_data_i[:36],
            data_u=train_data_u[:36],
            data_qc=train_qc[:36],
            data_soh=train_soh[:36],
            labels=data_labels[:36],
            save_path=save_path / "dataset_preview.png",
        )

    stats_path = Path(args.normalization_file) if args.normalization_file else _default_stats_path(args)
    if args.eval_only:
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalization file not found for evaluation: {stats_path}"
            )
        with open(stats_path, "rb") as handle:
            stats = pickle.load(handle)
        global_mean_u, global_std_u = np.array(stats["U"][0]), np.array(stats["U"][1])
        global_mean_i, global_std_i = np.array(stats["I"][0]), np.array(stats["I"][1])
        global_mean_qc, global_std_qc = np.array(stats["Qc"][0]), np.array(stats["Qc"][1])
        protocol_mean, protocol_std = np.array(stats["protocol"][0]), np.array(stats["protocol"][1])
    else:
        masked_train_data_i, masked_train_data_u, masked_train_qc, _ = inverse_padding(
            train_data_i, train_data_u, train_qc
        )
        global_mean_i, global_std_i = _compute_stats(masked_train_data_i)
        global_mean_u, global_std_u = _compute_stats(masked_train_data_u)
        global_mean_qc, global_std_qc = _compute_stats(masked_train_qc)
        protocol_mean, protocol_std = _compute_protocol_stats(charge_policies)

    train_normed_u = [
        (item - global_mean_u[0]) / global_std_u[0] for item in train_data_u
    ]
    train_normed_i = [
        (item - global_mean_i[0]) / global_std_i[0] for item in train_data_i
    ]
    train_normed_qc = [
        (item - global_mean_qc[0]) / global_std_qc[0] for item in train_qc
    ]
    train_normed_protocols = [
        (item - protocol_mean) / protocol_std for item in charge_policies
    ]

    mean_std = {
        "I": [global_mean_i.tolist(), global_std_i.tolist()],
        "U": [global_mean_u.tolist(), global_std_u.tolist()],
        "Qc": [global_mean_qc.tolist(), global_std_qc.tolist()],
        "protocol": [protocol_mean.tolist(), protocol_std.tolist()],
    }
    if is_main_process() and not args.eval_only:
        with open(stats_path, "wb") as handle:
            pickle.dump(mean_std, handle)

    dataset = BatteryDataset(
        train_normed_i,
        train_normed_u,
        train_normed_qc,
        train_soh,
        train_normed_protocols,
        raw_charge_policies,
        data_labels,
    )
    return dataset, mean_std


class BatteryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_i,
        data_u,
        data_qc,
        data_soh,
        charge_policies,
        raw_charge_policies,
        data_labels,
    ):
        self.data_i = data_i
        self.data_u = data_u
        self.data_qc = data_qc
        self.data_soh = data_soh
        self.charge_policies = charge_policies
        self.raw_charge_policies = raw_charge_policies
        self.data_labels = data_labels

    def __len__(self):
        return len(self.data_i)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data_i[idx], dtype=torch.float32),
            torch.tensor(self.data_u[idx], dtype=torch.float32),
            torch.tensor(self.data_qc[idx], dtype=torch.float32),
            torch.tensor(self.data_soh[idx], dtype=torch.float32),
            torch.tensor(self.charge_policies[idx], dtype=torch.float32),
            self.raw_charge_policies[idx],
            torch.tensor(self.data_labels[idx], dtype=torch.long),
        )
