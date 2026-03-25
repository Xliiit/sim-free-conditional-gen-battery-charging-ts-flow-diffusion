"""Helpers for single-node distributed training."""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def init_distributed_mode(args) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True

        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.set_device(args.gpu)
            backend = "nccl"
        else:
            backend = "gloo"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
            timeout=datetime.timedelta(minutes=30),
        )
        dist.barrier()
        setup_for_distributed(args.rank == 0)
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0


def setup_for_distributed(is_master: bool) -> None:
    import builtins as builtin

    builtin_print = builtin.print

    def wrapped_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtin.print = wrapped_print


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def reduce_mean(tensor: torch.Tensor, nprocs: int) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= nprocs
    return reduced


def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
