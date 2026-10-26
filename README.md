<div align="center">

# BRPA
### Boundary-guided Region Propagation Attention

**Seed discovery · Boundary-controlled spreading · Spatial refinement · Channel recalibration**

[![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)](BRPA.py)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](requirements.txt)
[![Module](https://img.shields.io/badge/Type-Attention_Module-5261A3?style=flat-square)](BRPA.py)
[![Docs](https://img.shields.io/badge/Docs-English%20%7C%20%E4%B8%AD%E6%96%87-13A89E?style=flat-square)](README_zh-CN.md)

**[English](README.md) | [简体中文](README_zh-CN.md)**

</div>

---

> [!IMPORTANT]
> **Source fidelity.** The provided Python file implements BRPA and ablation settings. The supplied LaTeX manuscript describes **TCPELIA**, a different module. Therefore, this repository **does not** attribute that manuscript's paper title, numerical results, or publication information to BRPA. The exact uploaded Python code is preserved at [`reference/BRPA_ablation_original.py`](reference/BRPA_ablation_original.py).

## Overview

**BRPA (Boundary-guided Region Propagation Attention)** is a PyTorch attention module that starts from an estimated seed map and iteratively extends spatial responses under an optional boundary barrier. The resulting attention map can be refined and used with an optional channel modulation branch to recalibrate the input feature map.

The implementation provides switches for **component ablations**, **propagation depth**, **spreading operators**, **propagation rate**, **boundary barriers**, and **output recalibration**. It is a **feature-processing module**, not a complete detector or an end-to-end training pipeline.

### Module at a glance

```mermaid
flowchart LR
    X["Input feature X"] --> S["Seed prediction"]
    X --> B["Boundary prediction"]
    S --> P["Iterative region propagation"]
    B -. "barrier G(B)" .-> P
    S -. "original seed S₀" .-> R["Attention refinement"]
    B -. "boundary map B" .-> R
    P --> R
    X --> C["Channel modulation"]
    R --> F["Output recalibration"]
    C -. "optional C" .-> F
    X --> F
    F --> Y["Output feature Y"]
    style S fill:#e5f2ff,stroke:#4784c7,color:#153b60
    style B fill:#fff0e9,stroke:#e58b62,color:#633719
    style P fill:#e8f5ed,stroke:#52a978,color:#194d31
    style R fill:#f0ebff,stroke:#9473d1,color:#432c69
    style F fill:#e9f0ff,stroke:#6a88cf,color:#273d72
```

*The diagram shows the complete module. Individual ablation variants omit the corresponding optional branches.*

## Quick start

### 1. Install the requirement

Use an environment with a working PyTorch installation:

```bash
python -m pip install -r requirements.txt
```

The only runtime Python dependency in the uploaded BRPA implementation is **PyTorch**. No specific minimum version has been established by the supplied files.

### 2. Run a standalone forward pass

```python
import torch
from BRPA import BRPA

x = torch.randn(2, 64, 40, 40)  # [batch, channels, height, width]
module = BRPA(channels=64, steps=2)
y = module(x)

print(x.shape, y.shape)  # torch.Size([2, 64, 40, 40]) x 2
```

Try the runnable example:

```bash
python examples/basic_usage.py
```

### 3. Select the complete component configuration

**The supplied Python file defaults to an ablation**, not to the complete configuration:

```python
# BRPA.py — settings in the uploaded file
COMPONENT_VARIANT = "seed_prop_boundary_refine"
PROP_STEPS = 4
ENABLE_CHANNEL = False
```

To activate **all BRPA branches**, edit `BRPA.py` *before instantiating the model*:

```python
COMPONENT_VARIANT = "full"
PROP_OPERATOR = "maxpool3"
BARRIER_MODE = "soft"
RECALIBRATION_MODE = "full"
# Set PROP_STEPS to the depth required by your experiment.
```

`BRPA(channels=64, steps=2)` explicitly overrides `PROP_STEPS` for that instance. **No particular propagation depth or measured score is presented here as the paper-validated BRPA configuration**, because the supplied manuscript is about TCPELIA.

## The mechanism

Let the input feature map be $X\in\mathbb{R}^{B\times C\times H\times W}$. The seed and boundary branches produce $S_0,B\in[0,1]^{B\times1\times H\times W}$. At propagation step $t$:

$$
\Delta_t=\operatorname{ReLU}\big(\mathcal{P}(S_t)-S_t\big),\qquad
S_{t+1}=\operatorname{clip}\big(S_t+p\,G(B)\odot\Delta_t,0,1\big).
$$

Here $\mathcal{P}$ is the selected spreading operator and $p$ is the effective propagation rate. In the original soft-barrier configuration, $G(B)=1-B$. With refinement enabled:

$$
A=\sigma\Big(\operatorname{Conv}_{3\times3}\big([S_K,S_0,B]\big)\Big).
$$

When the channel branch and full output recalibration are enabled:

$$
C_w=\sigma\big(\operatorname{MLP}(\operatorname{GAP}(X))\big),\qquad
Y=X+A\odot X\odot C_w.
$$

The spatial and channel weights are broadcast over compatible dimensions. These equations describe the **supplied implementation**, not an independently verified claim from the provided manuscript. See [method notes](docs/METHOD.md) and [中文原理说明](docs/METHOD_zh-CN.md).

## Ablation options

| Setting in `BRPA.py` | Choices / effect |
|---|---|
| `COMPONENT_VARIANT` | `identity`, `seed`, `seed_prop`, `seed_prop_boundary`, `seed_prop_boundary_refine`, `full` |
| `PROP_STEPS` | `0` through `5`; may be overridden by `steps=` |
| `PROP_OPERATOR` | `maxpool3`, `avgpool3`, `maxpool5`, `conv3` |
| `PROP_RATE_MODE` | `learnable` (`sigmoid(eta)`) or `fixed` (`FIXED_P`) |
| `BARRIER_MODE` | `none`, `soft`, `hard`, `shifted`, `power` |
| `RECALIBRATION_MODE` | `spatial`, `spatial_channel`, `spatial_residual`, `full` |
| `ENABLE_CHANNEL` | Enable channel branch in non-`full` variants |

> [!NOTE]
> In the original source, the helper `get_brpa_ablation_tag()` is based on *global settings*. It does **not** capture an instance-level `steps` override or post-training changes to the learnable propagation rate. Additionally, the original `Chan/NoChan` suffix reflects `ENABLE_CHANNEL` rather than the `full` variant's effective channel state. Record the instantiated module settings and its `get_effective_p()` value when reporting experiments.

## Project layout

```text
BRPA-GitHub-Release/
├── BRPA.py                            # Searchable, documented BRPA implementation
├── README.md                          # English documentation (this page)
├── README_zh-CN.md                    # 中文说明
├── requirements.txt                   # Runtime dependency
├── .gitignore
├── examples/
│   ├── basic_usage.py                 # Forward pass and backward check
│   └── integration_example.py         # Minimal feature-block integration
├── tests/
│   ├── test_brpa.py                   # Behavior and shape checks
│   └── test_source_compatibility.py   # Compare against the uploaded source
├── docs/
│   ├── METHOD.md                      # Implementation-based method description
│   ├── METHOD_zh-CN.md                # 中文数学原理与参数说明
│   ├── GITHUB_PUBLISH_zh-CN.md        # Publishing / search discoverability
│   └── SOURCE_PROVENANCE.md          # Supplied-source audit and hash
└── reference/
    └── BRPA_ablation_original.py      # Byte-for-byte copy of uploaded BRPA code
```

## Tests and source provenance

Run:

```bash
python -m unittest discover -s tests -v
python BRPA.py
```

See [source provenance and SHA-256](docs/SOURCE_PROVENANCE.md). The compatibility tests compare numerical outputs from `BRPA.py` to the **uploaded original code** under matched initialization, for a selection of component and propagation options. They cannot establish the accuracy, training reproducibility, or completeness of an external detection system.

## Scope and availability

| Item | Included? |
|---|---|
| BRPA attention module | ✅ |
| Original uploaded BRPA code | ✅ |
| Bilingual method/use documentation | ✅ |
| Standalone examples and tests | ✅ |
| Dataset and annotation files | ❌ Not supplied |
| Detector integration patch / training commands | ❌ Not supplied |
| Trained weights and benchmark logs | ❌ Not supplied |
| BRPA publication DOI / verified citation | ❌ Not supplied |
| Open-source license | ❌ Not supplied; choose a license before granting reuse permissions |

For detector integration, instantiate `BRPA(channels=<feature_channels>)` at the desired feature-processing point. Integration registration for specific frameworks must be handled in the corresponding framework project; **this archive is not a ready-to-train Ultralytics fork**.

## Citation and license

A **BRPA-specific published citation cannot be verified from the supplied manuscript**; do not cite the TCPELIA paper as the BRPA paper. Add a verified bibliographic entry after the corresponding BRPA manuscript is identified. No license is included because none was supplied or authorized; public GitHub visibility does not by itself grant reuse rights.

---

<div align="center">

**BRPA** · Boundary-guided Region Propagation Attention

[Source code](BRPA.py) · [Method](docs/METHOD.md) · [中文版](README_zh-CN.md) · [Publishing guide](docs/GITHUB_PUBLISH_zh-CN.md)

</div>
