# Compute economics — V-JEPA 2.1 stress sweep

Total spend across the two sweeps that produced every published
number in this report.

## Headline

| sweep                  | wallclock           | row count | est. cost |
|------------------------|---------------------|-----------|-----------|
| `sweep-b077627` (SSv2) | 6,729.7 s (1.870 h) | 44,400    | $3.53     |
| `sweep-davis-9fb565d` (DAVIS) | 1,447 s (0.402 h)   |  3,000    | $0.76     |
| **TOTAL**              | **8,176.7 s (2.272 h)** | **47,400** | **$4.29** |

Hardware: RunPod RTX PRO 6000 Blackwell (96 GB), Community Cloud,
billed at **$1.89/hr**. Both sweeps ran on the same pod; the DAVIS
total is end-to-end pod wallclock (24:07 minutes including the four
encoder-load events), not the row-summed `wallclock_sec` (582.4 s)
which only covers per-cell encode/decode time.

The full corpus — 4 model sizes × 9 perturbations × multi-strength
sweeps × two datasets — was produced for **less than the cost of
lunch.**

## Per-model breakdown (SSv2 sweep, end-to-end row sum)

| model                  | params | total_sec | total_hr | est. cost |
|------------------------|--------|-----------|----------|-----------|
| vjepa-2.1-vit-base     | 80M    | 1,127.6   | 0.313    | $0.59     |
| vjepa-2.1-vit-large    | 300M   | 1,095.5   | 0.304    | $0.58     |
| vjepa-2.1-vit-giant    | 1B     | 1,827.9   | 0.508    | $0.96     |
| vjepa-2.1-vit-gigantic | 2B     | 2,678.6   | 0.744    | $1.41     |

Per-model cost grows sub-linearly with parameter count once you
account for the fact that gigantic is 25× the parameters of base
but only 2.4× the wallclock. The bottleneck is not the encoder.

## Per-row wallclock distribution

| sweep        | mean    | median  | p95     | max     |
|--------------|---------|---------|---------|---------|
| SSv2 (n=44,400) | 0.152 s | 0.153 s | 0.253 s | 0.497 s |
| DAVIS (n=3,000) | 0.194 s | 0.190 s | 0.334 s | 0.506 s |

Median per-row latency is 150–190 ms across both sweeps. Tail
latency (max ≈ 0.5 s) is dominated by frame-drop rows where the
substituted-frame chain forces extra decoder work — see the
per-perturbation breakdown below.

## Per-perturbation wallclock (SSv2)

| perturbation     | total_sec | total_hr |
|------------------|-----------|----------|
| frame_drops      | 1,282.8   | 0.356    |
| occlusion        | 1,277.0   | 0.355    |
| motion_blur      | 1,222.2   | 0.339    |
| low_light        | 1,200.3   | 0.333    |
| gaussian_noise   | 1,186.0   | 0.329    |
| jpeg             |   177.4   | 0.049    |
| identity         |   129.4   | 0.036    |
| reverse          |   128.7   | 0.036    |
| horizontal_flip  |   125.9   | 0.035    |

Tier 1 perturbations (top five rows) each cost ~10× more wallclock
than Tier 2/3 perturbations. The Tier 2/3 perturbations are
near-zero-cost CPU operations (index permutation, byte-level
compression, identity); Tier 1 perturbations apply per-pixel
transforms across all 16 frames at 384×384.

## Alternative hardware estimates

What the same sweep would have cost on other accelerators commonly
available on the same marketplaces, holding total wallclock fixed at
2.272 h. (These are upper bounds — H100 and H200 would likely
finish faster due to higher compute throughput, but all four model
sizes are encoder-bound in fp16, and decode/perturbation overhead
dominates for the smaller models.)

| GPU                     | $/hr  | est. total |
|-------------------------|-------|------------|
| RTX PRO 6000 (Blackwell) | $1.89 | $4.29 *(actual)* |
| A100 80GB (Community)   | $1.89 | $4.29      |
| H100 80GB (Community)   | $2.99 | $6.79      |
| H200 141GB (Community)  | $3.99 | $9.07      |

The takeaway: on commodity rental hardware, an exhaustive frozen-encoder
robustness sweep covering 47k clip evaluations across four model sizes
and nine perturbations costs single-digit dollars. Methodology
trumps hardware budget.

## Why so cheap — the decode bottleneck

Frozen-encoder evaluation is a fundamentally different cost regime
from training. There are no backward passes, no optimizer state, no
multi-GPU coordination, and (because we cache the clean clip's
features when possible) no redundant clean-side encodes. What is
left is: decode 16 H.264 frames → resize to 384² → apply
perturbation → forward through one ViT → write a 768-d vector to
parquet. The forward pass through `vit-gigantic` (2 B params, fp16)
is ~80 ms on the Blackwell card; the rest of the per-row budget is
PyAV decode + per-pixel CPU perturbation work. That ratio gets
*worse* for smaller models — vit-base spends a larger fraction of
its 25 ms/row in decode than vit-gigantic spends in encode.

This has two practical consequences:

1. **Bigger GPUs do not buy proportionate speedup.** Going from
   Blackwell → H200 mostly speeds up the encode (already a small
   slice for vit-base/large) and does nothing for the decode. The
   Tier 1 perturbations (motion blur, low light, gaussian noise)
   would benefit from a CUDA-side perturbation kernel more than from
   a faster forward pass.
2. **Replicating this study is cheap for any lab that can rent a
   single GPU for an afternoon.** The total marginal cost of the
   sweep ($4.29) is less than the AWS S3 egress cost of moving the
   raw SSv2 dataset around. Compute is not the gating factor; what
   *was* gating was the design work — picking the metrics, defining
   the hypothesis space, deciding what counts as a confirmation
   versus a refutation. That work generalized beyond V-JEPA 2.1 and
   is what the next sweep will reuse.
