# Reproduction Guide

## 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a PyTorch build matching your CUDA runtime before installing the remaining dependencies if needed.

## 2. Prepare Data

Create a `data/` directory or point the config to your own dataset location:

```bash
mkdir -p data
```

Then update [`configs/flash_attention_dit.json`](../configs/flash_attention_dit.json).

## 3. Quick Sanity Run

```bash
python -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --output_dir ./outputs \
  --test_run
```

## 4. Full Training

Single GPU:

```bash
python -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --output_dir ./outputs
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --output_dir ./outputs
```

## 5. Evaluation

```bash
python -m flash_battery.train \
  --config_file ./configs/flash_attention_dit.json \
  --resume ./outputs/<run>/epoch10/checkpoint.pth \
  --normalization_file ./outputs/<run>/epoch10/normalization.pkl \
  --eval_only
```
