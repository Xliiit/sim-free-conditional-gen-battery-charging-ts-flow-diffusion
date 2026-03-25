# FLASH

FLASH is a simulation-free conditional generative framework for lithium-ion battery charging time-series under mixed charging protocols. This repository packages the core `flow matching + diffusion transformer` implementation into a public, reproducible research codebase for training and evaluating synthetic battery trajectories conditioned on state of health (SOH) and charging protocol.

## Overview

The code follows the main ideas described in the paper:

- `Flow Matching` formulates generation as an ODE transport problem instead of a standard diffusion denoising chain.
- `Full-resolution temporal projection` keeps the aligned sequence at length `2048` to preserve transient electrochemical responses.
- `Joint condition encoding` fuses pseudo-time, SOH, and a dense `100`-dimensional protocol vector.
- `Factorized dual-stream attention` decouples short-term fluctuations and long-range degradation trends.
- `Simulation-free generation` produces `I / V / Qc` trajectories directly from data without running an electrochemical simulator.

According to the manuscript, the method is evaluated on held-out mixed charging protocols and reports average validation RMSEs of approximately `0.184 C` for current, `0.010 V` for voltage, and `0.002 Ah` for charge capacity.

## Repository Layout

```text
.
├── configs/
│   └── flash_attention_dit.json
├── docs/
│   ├── checkpoints.md
│   ├── dataset.md
│   └── reproduction.md
├── flash_battery/
│   ├── dataloader.py
│   ├── distributed_mode.py
│   ├── ema.py
│   ├── eval_loop.py
│   ├── model.py
│   ├── train.py
│   ├── train_loop.py
│   └── utils.py
├── scripts/
│   ├── eval.sh
│   └── train.sh
├── LICENSE
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a compatible PyTorch build for your CUDA version first if needed.

## Data Preparation

This repository does not ship datasets or checkpoints. You should prepare preprocessed pickle files that already follow the paper's fixed-length alignment strategy.

See:

- [`docs/dataset.md`](docs/dataset.md)
- [`docs/checkpoints.md`](docs/checkpoints.md)

The default config expects:

```json
{
  "dataset_params": {
    "battery_files": ["./data/train.pkl"],
    "valid_files": ["./data/valid.pkl"]
  }
}
```

## Training

Single GPU:

```bash
python -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --output_dir ./outputs
```

Quick sanity check:

```bash
python -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --output_dir ./outputs \
  --test_run
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --output_dir ./outputs
```

## Evaluation

```bash
python -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --resume ./outputs/<run>/epoch10/checkpoint.pth \
  --normalization_file ./outputs/<run>/epoch10/normalization.pkl \
  --eval_only
```

Evaluation writes generated trajectories and summary plots to the corresponding `epoch*` directory.

## Notes On Implementation

- The public code keeps `AttentionDiT` as the primary model path because it matches the paper's full-resolution temporal projection and factorized dual-stream attention design more closely.
- Charging protocols are converted from strings such as `6(40%)-3(80%)-1(100%)` into dense `100`-dimensional vectors, where each entry represents a `1%` SOC interval.
- Normalization statistics for `I`, `V`, `Qc`, and protocol vectors are saved alongside checkpoints for reproducible evaluation.

## Reproduction Checklist

- Update dataset paths in [`configs/flash_attention_dit.json`](configs/flash_attention_dit.json)
- Run a `--test_run` sanity check
- Launch full training
- Evaluate with a saved checkpoint and matching normalization file

Detailed steps are documented in [`docs/reproduction.md`](docs/reproduction.md).

## Citation

If you use this repository in your research, please cite the associated paper:

```bibtex
@article{li2026flash,
  title   = {Simulation-free conditional generative modeling of lithium-ion battery charging time-series under mixed protocols using flow matching and diffusion transformers},
  author  = {Li, Xiaotian and Huang, Xinghao and Tao, Shengyu and Liang, Chen and Wang, Runhua and Wang, Junpeng and Zhang, Jiale and Xia, Bizhong},
  journal = {To be updated},
  year    = {2026}
}
```

## Limitations

- The repository assumes preprocessing to fixed length has already been completed.
- No public dataset download link or pretrained checkpoint is bundled yet.
- Some evaluation plots are intended for research analysis rather than production benchmarking.
