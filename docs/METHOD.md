# BRPA: implementation notes (English)

This document describes **only what is implemented in the supplied BRPA Python file**. The separately supplied manuscript is about TCPELIA, so it is not a verified BRPA method reference.

## 1. Inputs and branch shapes

- Input/output: `X, Y` are `(batch, channels, height, width)`.
- Seed predictor: depthwise `3×3` convolution → batch normalization → SiLU → `1×1` convolution → sigmoid, returning `S0` of shape `(batch, 1, height, width)`.
- Boundary predictor: an independent branch with the same layer types and a sigmoid output, returning `B` of shape `(batch, 1, height, width)`.
- Channel predictor (optional): global average pooling → `1×1` convolution → ReLU → `1×1` convolution → sigmoid, returning `C` of shape `(batch, channels, 1, 1)`; its hidden width is `max(channels // 8, 16)`.

## 2. Boundary-guided propagation

At iteration `t`, define a candidate neighborhood response with operator `P` and a nonnegative increment:

$$
D_t=\operatorname{ReLU}\left(P(S_t)-S_t\right).
$$

The seed response is updated as:

$$
S_{t+1}=\operatorname{clip}\left(S_t+p\,G(B)\odot D_t,0,1\right).
$$

`p` is either `sigmoid(eta)` (`eta` initialized to `0.5`) or the fixed scalar `FIXED_P`. In the implementation, `G(B)` can be:

| Barrier mode | Implemented gate | Interpretation |
|---|---|---|
| `none` | `1` | No spatial barrier |
| `soft` | `1 - B` | Suppress updates near larger predicted boundary responses |
| `hard` | `1[B < threshold]` | Non-differentiable threshold gate |
| `shifted` | `1 - roll(B)` | Shift the barrier spatially (ablation control) |
| `power` | `(1 - B)^gamma` | Adjust suppression strength |

`P` can be max-pooling (`3×3` or `5×5`), average-pooling (`3×3`), or a learnable `3×3` single-channel convolution with candidate clamping. For `conv3`, the candidate is `max(S_t, conv3(S_t))`. All modes apply `ReLU` to the increment, so propagation does not decrease the current map through a negative increment.

**Boundary semantics:** The boundary map is learned end-to-end. The code does **not** implement a separate pixel-level boundary-label supervision loss. Calling `B` a predicted propagation barrier is supported; describing it as supervised ground-truth edge detection is not supported by the uploaded code.

## 3. Spatial attention and feature output

When refinement is enabled, concatenate `S_K`, initial `S0`, and predicted `B`, and run a `3×3` convolution plus sigmoid:

$$
A=\sigma\big(\operatorname{Conv}_{3\times3}([S_K,S_0,B])\big).
$$

When refinement is disabled, `A = S_K`. The output is selected by `RECALIBRATION_MODE`:

| Mode | Output |
|---|---|
| `spatial` | `Y = A * X` |
| `spatial_channel` | `Y = A * X * C` (requires channel branch) |
| `spatial_residual` | `Y = X + A * X` |
| `full` | `Y = X + A * X * C` if channel exists; otherwise `Y = X + A * X` |

Here `*` denotes broadcast elementwise multiplication. A `full` **component variant** creates all branches and enables channel modulation; the `full` **recalibration mode** only specifies the output formula and can still operate in reduced variants without channel modulation. The two `full` strings mean different things.

## 4. Practical details and caveats

1. **Source default:** `COMPONENT_VARIANT="seed_prop_boundary_refine"` and `PROP_STEPS=4` in the supplied source. Therefore the uploaded default does not enable the channel branch.
2. **Constructor override:** `BRPA(channels, steps=N)` uses `N` instead of the global propagation depth.
3. **Initialization of learnable rate:** `sigmoid(0.5)` is approximately `0.622459`; this is an initial value, not the value after training.
4. **Hard barrier:** The threshold comparison is not differentiable with respect to the boundary values; it is intended as an ablation control.
5. **Tag warning:** The original `get_brpa_ablation_tag()` uses the global `PROP_STEPS` and an `ENABLE_CHANNEL` switch, not necessarily the instantiated module's effective settings. A checkpoint must record actual model parameters/settings separately.
6. **Numerical equivalence:** `tests/test_source_compatibility.py` checks several representative configurations against the archived original implementation. No source-provided training logs or BRPA publication metrics are present.

## 5. Searchable identifiers

- Public entry point: [`BRPA.py`](../BRPA.py), class `BRPA`.
- Relevant terms: `Boundary-guided Region Propagation Attention`, `boundary-guided attention`, `region propagation`, `seed attention`, `boundary barrier`, `PyTorch attention module`.
- Original archival entry: [`reference/BRPA_ablation_original.py`](../reference/BRPA_ablation_original.py).
