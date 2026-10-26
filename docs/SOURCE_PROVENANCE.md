# Source provenance / 源码来源说明

## Materials actually supplied / 本次实际上传材料

| Supplied filename | Inspected content | Appropriate use |
|---|---|---|
| `BRPA(20261026-014953).py` | Standalone `BRPA(nn.Module)` with settings for component, propagation, barrier, and recalibration ablations | Primary authority for the code in this repository |
| `manuscript(6).tex` | Paper entitled *Prediction-Error-Driven Iterative Evidence Selection With Lateral Competition for Insulator Shackle Corrosion Detection* and describing **TCPELIA / TCPELIA-Net** | Useful research-domain context **only**; **not a verified BRPA paper** |

## Original source conservation / 原代码完整性

The archive [`reference/BRPA_ablation_original.py`](../reference/BRPA_ablation_original.py) is an unchanged, byte-for-byte copy of the uploaded Python file.

`SHA-256`: `7c4444072cc3c08797777c6d168add8d8c44457e44853c195542d993d7e8d014`

The public [`BRPA.py`](../BRPA.py) is a readability-focused edition: it adds English/Chinese comments, more complete class/method docstrings, clearer sectioning and removes an unused `math` import. It **retains the original default configuration, public entry points, control flow, calculations, layers, and parameter names**. No new research mechanism is inserted.

`tests/test_source_compatibility.py` compares module weights and outputs against the original under identical random seeds for six representative configurations. This provides a narrow implementation-level equivalence check; it does not demonstrate paper-level experimental reproducibility.

## What's intentionally not included / 有意未编造的内容

- BRPA-specific publication metadata, DOI, claimed performance, and official BibTeX.
- YOLO/Ultralytics integration patches, original complete network YAML, and training or validation scripts.
- Dataset distribution and download links, pretrained weights, benchmark recordings.
- License text or author declarations not provided by the upload.

Add these resources only after obtaining their original, authorized and verified versions.
