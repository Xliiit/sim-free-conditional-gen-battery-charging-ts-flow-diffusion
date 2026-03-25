from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import trange

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None

import flash_battery.distributed_mode as dist_utils
from flash_battery.distributed_mode import barrier, get_rank, init_distributed_mode, is_main_process


def setup_config(args):
    if not args.config_file:
        raise ValueError("Must provide --config_file")
    config_path = Path(args.config_file).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    args.config_file = str(config_path)
    args.config_root = str(config_path.parent)
    args.expr = config.get("expr", "default_experiment")
    args.dataset_params = config.get("dataset_params", {})
    args.dit_model_params = config.get("dit_model_params", {})

    flow_params = config.get("flow_params", {})
    for key, value in flow_params.items():
        if hasattr(args, key) and getattr(args, key) is not None:
            if is_main_process():
                print(f"[Override] {key}: CLI value overrides config value {value}")
        else:
            setattr(args, key, value)

    train_params = config.get("train_params", {})
    for key, value in train_params.items():
        if hasattr(args, key) and getattr(args, key) is not None:
            if is_main_process():
                print(f"[Override] {key}: CLI value overrides config value {value}")
        else:
            setattr(args, key, value)
    return args


def get_args_parser():
    parser = argparse.ArgumentParser("FLASH training script", add_help=False)
    parser.add_argument(
        "--config_file",
        type=str,
        default="./configs/flash_attention_dit.json",
        help="Path to a JSON config file.",
    )
    parser.add_argument("--device", default="cuda", help="Device to use.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory.")
    parser.add_argument("--ema", dest="use_ema", action="store_true", help="Use EMA.")
    parser.add_argument("--skewed_timesteps", action="store_true")
    parser.add_argument("--eval_frequency", type=int)
    parser.add_argument("--test_samples", type=int)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument("--ode_method", type=str, help="ODE solver method.")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--cfg_training", action="store_true")
    parser.add_argument("--normalization_file", default="", type=str)
    parser.add_argument(
        "--keep_padded_length",
        action="store_true",
        help="Evaluate on full padded sequence length instead of trimmed length.",
    )
    return parser


def _make_run_dir(args) -> Path:
    current_time = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{current_time}_{args.expr}_eval_{args.eval_only}"
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run directory created: {run_dir}")
    return run_dir


def main(args):
    from flash_battery.dataloader import build_dataset
    from flash_battery.eval_loop import eval_model
    from flash_battery.model import build_model, load_model
    from flash_battery.train_loop import train_one_epoch

    init_distributed_mode(args)
    args = setup_config(args)
    args.run_dir = str(_make_run_dir(args))

    writer = None
    if is_main_process() and not args.eval_only and SummaryWriter is not None:
        writer = SummaryWriter(logdir=args.run_dir)
    args.tbx = writer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset, mean_std = build_dataset(args)
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_utils.get_world_size(),
            rank=dist_utils.get_rank(),
            shuffle=not args.eval_only,
        )
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        shuffle=sampler is None and not args.eval_only,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = build_model(args)
    model.to(device)
    if args.use_ema:
        from flash_battery.ema import EMA

        model = EMA(model, decay=0.9999)
        model.to(device)

    model_without_ddp = model
    load_model(args=args, model_without_ddp=model_without_ddp)

    if args.distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)
        else:
            model = DDP(model, find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model_without_ddp.parameters()),
        lr=args.lr,
    )
    total_steps = max(args.epochs * max(len(data_loader), 1), 1)
    scheduler = OneCycleLR(
        optimizer,
        args.lr,
        total_steps=total_steps,
        pct_start=0.25,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=1e5,
    )

    if is_main_process():
        print(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable params: {num_params}")

    start_time = time.time()
    epoch_iter = trange(args.epochs, desc="Epoch") if is_main_process() else range(args.epochs)
    for epoch in epoch_iter:
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                args=args,
            )
            if is_main_process():
                if writer is not None:
                    writer.add_scalar("train/loss", train_stats["loss"], epoch)
                epoch_iter.set_description(f"train_loss: {train_stats['loss']:.6f}", refresh=True)

        should_eval = (
            (args.eval_frequency and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        )
        if should_eval:
            if is_main_process():
                (Path(args.run_dir) / f"epoch{epoch + 1}").mkdir(parents=True, exist_ok=True)
            if args.distributed:
                barrier()
            if not args.eval_only:
                model.train(False)
                if is_main_process():
                    checkpoint_path = Path(args.run_dir) / f"epoch{epoch + 1}" / "checkpoint.pth"
                    torch.save({"model": model_without_ddp.state_dict()}, checkpoint_path)
                    stats_path = Path(args.run_dir) / f"epoch{epoch + 1}" / "normalization.pkl"
                    with open(stats_path, "wb") as handle:
                        import pickle

                        pickle.dump(mean_std, handle)

            eval_model(
                model,
                data_loader,
                Path(args.run_dir),
                device,
                epoch=epoch,
                fid_samples=36 if not args.eval_only else 288,
                mean_std=mean_std,
                args=args,
            )
            if args.eval_only or args.test_run:
                break

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time}")
    if writer is not None:
        writer.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FLASH training script", parents=[get_args_parser()])
    main(parser.parse_args())
