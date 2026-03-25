# Dataset Format

This repository does not ship training data. Please prepare your own pickle files and point the config to them.

## Expected File Structure

Each dataset file should be a Python `list` serialized by `pickle`. Each element represents one battery:

```python
[
    {
        "charge_policy": "6(40%)-3(80%)-1(100%)",
        "label": "Healthy",
        "cycles": [
            {
                "I": np.ndarray,
                "V": np.ndarray,
                "Qc": np.ndarray,
                "SOH": float,
            },
            ...
        ],
    },
    ...
]
```

## Supported Fields

- `charge_policy`: mixed charging protocol string. It is converted into a 100-dimensional dense protocol vector, where each index corresponds to 1% SOC.
- `label`: optional discrete label. `Healthy` is treated as class `0`, and strings containing `ISC` are treated as class `1`. This is only used when classifier-free guidance training is enabled.
- `I`: current trajectory.
- `V`: voltage trajectory.
- `Qc`: charge capacity trajectory.
- `SOH`: scalar state-of-health value for the cycle.

## Sequence Length

The model expects fixed-length sequences, typically `2048`, following the paper's preprocessing:

- current is padded with zeros
- voltage is padded with its last observed value
- charge capacity is padded with its last observed value

The repository assumes your data has already been preprocessed to this aligned format.

## Config Example

Update [`configs/flash_attention_dit.json`](../configs/flash_attention_dit.json):

```json
{
  "dataset_params": {
    "battery_files": ["./data/train.pkl"],
    "valid_files": ["./data/valid.pkl"]
  }
}
```

Paths are resolved relative to the config file location.
