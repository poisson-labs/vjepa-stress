# Pre-Registered Predictions

> **Pre-registration is binding.** This file mirrors EXPERIMENT.md §7 and is
> committed *before* any sweep is run. Predictions written here cannot be
> revised in light of results; if a prediction is later found to be wrong,
> that is a finding, not a reason to edit the file.

## Status

- Recorded on: _TBD — fill in when EXPERIMENT.md §7 is finalized._
- Sweep launched on: _TBD_
- Manifest hash at pre-registration: _TBD (paste sha256 of `manifests/manifest.json`)._

## Predictions

> **Placeholder.** Copy the full set of pre-registered hypotheses from
> EXPERIMENT.md §7 into this file. Every prediction should be:
>
> 1. **Specific** — names the metric (M1 / M2 / M3), the model variant(s),
>    the perturbation, and the strength range.
> 2. **Falsifiable** — states a quantitative threshold or directional claim
>    that the sweep can disconfirm.
> 3. **Tied to a mechanism** — one sentence on *why* this is expected, so a
>    null result is informative.
>
> Suggested template per prediction:
>
> ```
> ### H<n>: <one-line claim>
>
> - Metric: M<k>
> - Models: <variant set>
> - Perturbation: <P-id> at strength <range>
> - Direction / threshold: <e.g. "M1 increases monotonically with strength;
>   ViT-G < ViT-B at every strength tested">
> - Mechanism: <one sentence>
> - Disconfirmation: <what observation would falsify this>
> ```
