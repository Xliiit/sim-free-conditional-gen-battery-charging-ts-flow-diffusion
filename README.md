# FLASH: Flow-matching Lithium-ion Attention-decoupled Simulation-free High-fidelity Generative System for Mixed Charging Protocols

## Overview

**FLASH** is a generative modeling framework for lithium-ion battery time-series data under mixed charging protocols. It leverages flow matching and a dual-stream attention architecture to produce high-fidelity synthetic battery data without requiring any simulator, guided jointly by state-of-health (SOH) and charging-protocol conditions.

## Highlights

1. **Simulation-free generative model** yields high-fidelity battery time-series.
2. **Fused state of health and charging protocols** dynamically guide data generation.
3. **Dual-stream attention** separates charging fluctuations and long-term degradation trends.
4. **Validated on 29 unseen mixed charging protocols** with high generation accuracy.

## Key Features

| Feature | Description |
|---|---|
| Flow Matching | Straight-trajectory ODE transport replaces diffusion, enabling fast and stable training without simulation |
| Conditional Generation | SOH value and charging-protocol label are jointly fused as generation conditions |
| Dual-Stream Attention | One stream captures high-frequency charging fluctuations; the other tracks slow capacity degradation |
| Generalization | Evaluated on 29 held-out mixed charging protocols not seen during training |

## Citation

If you use FLASH in your research, please cite the corresponding paper (to be updated upon publication).
