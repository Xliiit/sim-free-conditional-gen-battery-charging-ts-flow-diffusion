from __future__ import annotations

import os
import pickle
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable

import torch
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from torch.nn.parallel import DistributedDataParallel

from .dataloader import inverse_padding
from .distributed_mode import barrier, get_rank, get_world_size, is_main_process
from .utils import analyze_and_plot_results


class CFGWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def forward(self, x, t, **kwargs):
        label = kwargs.get("label")
        cfg_scale = kwargs.get("cfg_scale", 0.0)
        soh = kwargs.get("soh")
        protocols = kwargs.get("protocols")
        with torch.no_grad():
            if cfg_scale > 0 and label is not None:
                cond_out = self.model(x, t, soh, protocols, extra={"label": label})
                drop_mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
                uncond_out = self.model(
                    x,
                    t,
                    soh,
                    protocols,
                    extra={"label": label, "drop_mask": drop_mask},
                )
                return uncond_out + cfg_scale * (cond_out - uncond_out)
            extra = {"label": label} if label is not None else {}
            return self.model(x, t, soh, protocols, extra=extra)


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    run_dir: Path,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    mean_std: Dict[str, torch.Tensor],
    args: Namespace,
):
    cfg_scaled_model = CFGWrapper(model=model)
    cfg_scaled_model.train(False)
    solver = ODESolver(velocity_model=cfg_scaled_model)
    ode_opts = getattr(args, "ode_options", {})
    num_synthetic = 0
    local_results = []
    seq_len = args.dit_model_params.get("seq_len", 2048)

    u_mean = torch.tensor(mean_std["U"][0][0], dtype=torch.float32, device=device)
    u_std = torch.tensor(mean_std["U"][1][0], dtype=torch.float32, device=device)
    i_mean = torch.tensor(mean_std["I"][0][0], dtype=torch.float32, device=device)
    i_std = torch.tensor(mean_std["I"][1][0], dtype=torch.float32, device=device)
    qc_mean = torch.tensor(mean_std["Qc"][0][0], dtype=torch.float32, device=device)
    qc_std = torch.tensor(mean_std["Qc"][1][0], dtype=torch.float32, device=device)

    test_scales = [1.0, 4.0, 8.0] if args.cfg_training else [0.0]

    with torch.no_grad():
        for batch in data_loader:
            data_i, data_u, data_qc, data_soh, charge_policies, raw_charge_policies, labels = batch
            x_1 = torch.stack([data_i, data_u, data_qc], dim=1).to(device, non_blocking=True)
            protocols = charge_policies.to(device, non_blocking=True)
            data_soh = data_soh.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = x_1.shape[0]
            x_0 = torch.randn(x_1.shape, dtype=torch.float32, device=device)
            time_grid = torch.tensor([0.0, 1.0], device=device)
            batch_results = {}

            for scale in test_scales:
                synthetic_samples = solver.sample(
                    time_grid=time_grid,
                    x_init=x_0,
                    method=args.ode_method,
                    return_intermediates=False,
                    atol=ode_opts.get("atol", 1e-5),
                    rtol=ode_opts.get("rtol", 1e-5),
                    step_size=ode_opts.get("step_size"),
                    soh=data_soh,
                    protocols=protocols,
                    label=labels if args.cfg_training else None,
                    cfg_scale=scale,
                )
                batch_results[scale] = {
                    "gen_i": (synthetic_samples[:, 0, :] * i_std + i_mean).cpu().numpy(),
                    "gen_u": (synthetic_samples[:, 1, :] * u_std + u_mean).cpu().numpy(),
                    "gen_qc": (synthetic_samples[:, 2, :] * qc_std + qc_mean).cpu().numpy(),
                }

            real_i = (x_1[:, 0, :] * i_std + i_mean).cpu().numpy()
            real_u = (x_1[:, 1, :] * u_std + u_mean).cpu().numpy()
            real_qc = (x_1[:, 2, :] * qc_std + qc_mean).cpu().numpy()
            soh_np = data_soh.cpu().numpy()
            _, _, _, mask_lengths = inverse_padding(real_i, real_u, real_qc)

            for idx in range(batch_size):
                length = int(mask_lengths[idx]) if not args.keep_padded_length else seq_len
                base_scale = test_scales[0]
                gen_u_dict = {s: batch_results[s]["gen_u"][idx, :length] for s in test_scales}
                gen_i_dict = {s: batch_results[s]["gen_i"][idx, :length] for s in test_scales}
                sample_res = {
                    "real_i": real_i[idx, :length],
                    "real_u": real_u[idx, :length],
                    "real_qc": real_qc[idx, :length],
                    "gen_i": batch_results[base_scale]["gen_i"][idx, :length],
                    "gen_u": batch_results[base_scale]["gen_u"][idx, :length],
                    "gen_qc": batch_results[base_scale]["gen_qc"][idx, :length],
                    "gen_u_dict": gen_u_dict,
                    "gen_i_dict": gen_i_dict,
                    "real_i_padded": real_i[idx, :].copy(),
                    "real_u_padded": real_u[idx, :].copy(),
                    "real_qc_padded": real_qc[idx, :].copy(),
                    "gen_i_padded": batch_results[base_scale]["gen_i"][idx, :].copy(),
                    "gen_u_padded": batch_results[base_scale]["gen_u"][idx, :].copy(),
                    "gen_qc_padded": batch_results[base_scale]["gen_qc"][idx, :].copy(),
                    "soh": float(soh_np[idx]),
                    "label": int(labels[idx]),
                    "raw_protocol": raw_charge_policies[idx],
                }
                if len(test_scales) > 1:
                    sample_res["gen_i_padded_scale2"] = batch_results[test_scales[1]]["gen_i"][idx, :].copy()
                    sample_res["gen_u_padded_scale2"] = batch_results[test_scales[1]]["gen_u"][idx, :].copy()
                    sample_res["gen_qc_padded_scale2"] = batch_results[test_scales[1]]["gen_qc"][idx, :].copy()
                    sample_res["gen_i_padded_scale3"] = batch_results[test_scales[2]]["gen_i"][idx, :].copy()
                    sample_res["gen_u_padded_scale3"] = batch_results[test_scales[2]]["gen_u"][idx, :].copy()
                    sample_res["gen_qc_padded_scale3"] = batch_results[test_scales[2]]["gen_qc"][idx, :].copy()
                local_results.append(sample_res)

            num_synthetic += batch_size
            if args.test_run and num_synthetic > 0:
                break
            if fid_samples != -1 and num_synthetic >= fid_samples:
                break

    save_dir = Path(run_dir) / f"epoch{epoch + 1}"
    os.makedirs(save_dir, exist_ok=True)
    rank_pkl = save_dir / f"_temp_rank{get_rank()}.pkl"
    with open(rank_pkl, "wb") as handle:
        pickle.dump(local_results, handle)

    barrier()
    if is_main_process():
        flattened_results = []
        for rank in range(get_world_size()):
            tmp_path = save_dir / f"_temp_rank{rank}.pkl"
            with open(tmp_path, "rb") as handle:
                flattened_results.extend(pickle.load(handle))
            tmp_path.unlink()

        pkl_path = save_dir / "generated_data.pkl"
        with open(pkl_path, "wb") as handle:
            pickle.dump(flattened_results, handle)
        if args.eval_only:
            analyze_and_plot_results(run_dir, epoch, flattened_results)
    barrier()
