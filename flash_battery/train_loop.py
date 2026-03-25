from __future__ import annotations

import torch
from flow_matching.path import CondOTProbPath
from torch.nn.parallel import DistributedDataParallel

from .distributed_mode import get_world_size, reduce_mean
from .ema import EMA


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    p_mean = -1.2
    p_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * p_std + p_mean).exp()
    time = 1 / (1 + sigma)
    return torch.clip(time, min=0.0001, max=1.0)


def train_one_epoch(model, data_loader, optimizer, scheduler, device, args):
    model.train(True)
    path = CondOTProbPath()
    running_loss = 0.0
    total_steps = 0
    drop_prob = 0.2

    for batch in data_loader:
        data_i, data_u, data_qc, data_soh, charge_policies, _, labels = batch
        x_1 = torch.stack([data_i, data_u, data_qc], dim=1).to(device, non_blocking=True)
        condition = charge_policies.to(device, non_blocking=True)
        data_soh = data_soh.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = x_1.shape[0]

        extra_args = None
        if args.cfg_training:
            extra_args = {
                "label": labels,
                "drop_mask": torch.rand(batch_size, device=device) < drop_prob,
            }

        x_0 = torch.randn_like(x_1, device=device)
        if args.skewed_timesteps:
            t = skewed_timestep_sample(batch_size, device)
        else:
            t = torch.rand(batch_size, device=device)

        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        pred_v = model(path_sample.x_t, t, data_soh, condition, extra=extra_args)
        loss = torch.pow(pred_v - path_sample.dx_t, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if isinstance(model, EMA):
            model.update_ema()
        elif isinstance(model, DistributedDataParallel) and isinstance(model.module, EMA):
            model.module.update_ema()

        running_loss += loss.detach()
        total_steps += 1

    avg_loss = running_loss / max(total_steps, 1)
    if args.distributed:
        avg_loss = reduce_mean(avg_loss, get_world_size())
    return {"loss": avg_loss.item()}
