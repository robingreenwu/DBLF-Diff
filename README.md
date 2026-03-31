# DBLF-LAGF: Dual-Branch Layer-wise Adaptive Gated Fusion for EEG Depression Detection

> ⚠️ **Research Code Notice (Unpublished Work)**  
> This repository contains source code for an **unpublished** paper under active development.  
> APIs, configs, and reported results may change before submission.

## 1. Overview

This project studies EEG-based depression detection with a diffusion-enhanced training pipeline and classifier evaluation.

Core model idea:
- **Dual-branch UNet backbone**
- **Layer-wise cross-branch fusion**
- **Adaptive gated fusion (LAGF)** as the main method
- Fusion ablation naming convention:
  - `DBLF-Concat`
  - `DBLF-Gate`
  - `DBLF-SE`
  - `DBLF-LAGF (Ours)`

## 2. Repository Structure

```text
.
├── run/
│   ├── Cross_MODMA.py                 # MODMA classifier training/evaluation
│   ├── Cross_PRED_CT.py               # PRED+CT classifier training/evaluation
│   ├── Pretraining_stage_MODMA.py     # Diffusion pretraining (MODMA)
│   ├── Pretraining_stage_PRED_CT.py   # Diffusion pretraining (PRED+CT)
│   ├── Sampling_stage_MODMA.py        # Diffusion sampling (MODMA)
│   ├── Sampling_stage_PRED_CT.py      # Diffusion sampling (PRED+CT)
│   ├── model/                         # Classifier & diffusion wrappers
│   ├── diffusion/                     # UNet/diffusion core
│   ├── datas/                         # Data loading/preprocessing
│   ├── augment_method_roc/            # ROC export & merge utilities
│   └── scripts/
│       ├── run_modma_main.sh
│       ├── export_one_fusion_roc.sh
│       └── merge_fusion_ablation_roc.sh
├── datas/                             # Sample/generated data files (outside run)
├── results/                           # Outputs (local)
└── README.md
```

## 3. Environment

Recommended: Linux/WSL + Python 3.9/3.10 + CUDA-enabled PyTorch.

Typical dependencies used by this project:
- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `h5py`
- `mamba-ssm`

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `mamba-ssm` has platform-specific build issues, install core deps first and then install it separately:

```bash
pip install numpy scipy scikit-learn matplotlib h5py tqdm einops
pip install torch torchvision
pip install mamba-ssm
```

## 4. Quick Start

### 4.1 Main training (MODMA)

```bash
bash run/scripts/run_modma_main.sh
```

Optional override:

```bash
MODEL_NAME=Transfromer EMB_SIZE=50 DEPTH=8 NUM_EPOCH=300 SEED=1024 \
  bash run/scripts/run_modma_main.sh
```

### 4.2 Export ROC for one fusion method

```bash
FUSION_METHOD=dblf_lagf SERVER_TAG=serverA \
  bash run/scripts/export_one_fusion_roc.sh
```

Supported method tags:
- `dblf_concat` → `DBLF-Concat`
- `dblf_gate` → `DBLF-Gate`
- `dblf_se` → `DBLF-SE`
- `dblf_lagf` → `DBLF-LAGF (Ours)`

### 4.3 Merge fusion ablation ROC curves

```bash
bash run/scripts/merge_fusion_ablation_roc.sh
```

Outputs:
- Figure: `results/MODMA_augment_roc/fusion_ablation_roc_4methods_no_attn.png`
- Summary: `results/MODMA_augment_roc/merged_server_roc_auc_summary.csv`

## 5. Data

This repository expects `.mat` EEG files loaded via `run/datas/data_pre.py`.

If your dataset is restricted by license/ethics policy:
- do **not** publish raw data directly,
- provide acquisition/application instructions instead,
- keep preprocessing scripts and schema descriptions for reproducibility.

## 6. Reproducibility Notes

- Fix random seeds (already supported in training scripts).
- Keep the same fold settings and sample counts when comparing methods.
- Use consistent fusion labels across servers before merging ROC outputs.

