# Checkpoints

This repository does not currently bundle pretrained checkpoints.

## Training Outputs

By default, training writes artifacts to:

```text
outputs/<timestamp>_<experiment_name>_eval_False/
```

Each evaluation epoch creates:

- `epoch*/checkpoint.pth`: model weights
- `epoch*/normalization.pkl`: normalization statistics for `I`, `V`, `Qc`, and protocol vectors
- `epoch*/generated_data.pkl`: generated samples collected during evaluation

## Evaluation Requirements

For evaluation you need:

- a config file
- a checkpoint file
- a matching normalization file

Example:

```bash
bash scripts/eval.sh ./configs/flash_attention_dit.json ./outputs/.../epoch10/checkpoint.pth ./outputs/.../epoch10/normalization.pkl
```

If `--normalization_file` is omitted, the code tries to infer it from the checkpoint directory.
