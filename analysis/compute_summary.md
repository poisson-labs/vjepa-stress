# Compute summary — sweep-b077627

- **Total wallclock**: 6,729.7 s = 1.87 h
- **GPU rate**: $1.89/hr (RTX PRO 6000, community cloud)
- **Estimated cost**: $3.53

## Per-row wallclock distribution (seconds)

| stat | value |
|------|-------|
| mean | 0.1516 |
| median | 0.1529 |
| p95 | 0.2525 |
| max | 0.4970 |

## Per-model wallclock

| model | total_sec | total_hr | est_cost_usd |
|-------|-----------|----------|--------------|
| vjepa-2.1-vit-base | 1,127.6 | 0.313 | $0.59 |
| vjepa-2.1-vit-large | 1,095.5 | 0.304 | $0.58 |
| vjepa-2.1-vit-giant | 1,827.9 | 0.508 | $0.96 |
| vjepa-2.1-vit-gigantic | 2,678.6 | 0.744 | $1.41 |

## Per-perturbation wallclock

| perturbation | total_sec | total_hr |
|--------------|-----------|----------|
| frame_drops | 1,282.8 | 0.356 |
| occlusion | 1,277.0 | 0.355 |
| motion_blur | 1,222.2 | 0.339 |
| low_light | 1,200.3 | 0.333 |
| gaussian_noise | 1,186.0 | 0.329 |
| jpeg | 177.4 | 0.049 |
| identity | 129.4 | 0.036 |
| reverse | 128.7 | 0.036 |
| horizontal_flip | 125.9 | 0.035 |
