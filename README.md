# MFVLR: Multi-domain Fine-grained Vision-Language Reconstruction for Generalizable Diffusion Face Forgery Detection and Localization

[![Tests](https://img.shields.io/badge/tests-93%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](requirements.txt)
[![Reproducibility](https://img.shields.io/badge/reproducibility-paper--faithful-blueviolet.svg)](docs/reproduction_notes.md)

This repository contains a **paper-faithful reproduction implementation of MFVLR** with documented assumptions for details not specified by the academic paper:
> **MFVLR: Multi-domain Fine-grained Vision-Language Reconstruction for Generalizable Diffusion Face Forgery Detection and Localization**  
> *arXiv:2605.10071v1*

---

## 1. Project Status & Reproducibility Disclaimer

- **Status:** **PHASE 12 COMPLETE (Ready for Experimental Reproduction)**.
- **Traceability:** 100% of the paper's mathematical equations (Eq. 1 through Eq. 27), architectural parameter sharing rules, and loss functions have been implemented, verified, and mapped in the [Paper-Code Traceability Matrix](docs/paper_code_traceability.md).
- **Test Suite:** **93 unit and integration tests passing** across 23 test modules in under 90 seconds.
- **Disclaimer:** This codebase is an independent academic reproduction and does NOT claim to be the official codebase. No benchmark tables or metric numbers (AUC/ACC/mIoU) have been fabricated without running full GenFace training.

---

## 2. Architecture Overview

MFVLR is designed for generalizable face forgery detection and pixel-level localization through multi-domain vision-language reconstruction:

```
[TRAINING FORWARD GRAPH]
======================================================================================================
Image I [B, 3, 224, 224]                          Hierarchical Prompt T [B, 308]
       |                                                         |
       v                                                         v
Image Encoder (Conv + Transformer)                     Language Embeddings (s=49408, d=512)
       |                                                         |
  +----+-------------------------+                               v
  |                              |                      Language Encoder (E=12 blocks)
  v                              v                               ^
I_loc [B, 1024, 14, 14]       I_g [B, 512]                       | (VIM Injection using I_v)
  |                              |                               v
Vision Decoder (Shared UNet)     |                      T_hig^e [B, 308, 512] --> T_l = Last Token [B, 512]
  |              |               |                               |
  v              v               |                               v
I_pre          M_pre             |                      Language Decoder (D=7 blocks, causal MMHA)
[B,3,224,224] [B,2,224,224]     |                               ^
  |                              |                               | (Cross-MHA on T_hig^e + VIM)
  v                              v                               v
Residual I_r = |I_pre - I|       |                      T_rec^d [B, 308, 512]
  |                              |                               |
  v                              |                               v (Tied W_voc^T)
Residual Encoder (Shared IE)     |                      T_pre [B, 308, 49408]
  |                              |
  v                              |
I_rg [B, 512]                    |
  |                              |
  +-------------(+)--------------+
                 |
                 v
     I_v = I_g + I_rg [B, 512]
     /          |            \
    v           v             v
 Adapter   Detection Head   VIM Keys/Values
    |           |           (LE & LD blocks)
    v           v
  T_lpre      y_pre
 [B, 512]    [B, 2]
======================================================================================================
```

### Critical Architectural Constraints Verified in Code:
1. **Weight Sharing:** Residual Encoder (RE) is the exact same network object and parameter memory as Image Encoder (IE).
2. **Decoder Sharing:** Appearance Decoder (AD) and Mask Decoder (MD) share the exact same U-Net decoder trunk.
3. **Vocabulary Tying:** Language Decoder vocabulary projection is tied to the token embedding matrix $W_{\text{voc}}^T$.
4. **Singleton VIM:** Vision Injection Module preserves the paper's single visual key/value token ($I_v \in \mathbb{R}^{B \times 1 \times 512}$), causing dot-product softmax attention over keys to be identically 1.0.
5. **Last-Token Language Feature:** Global language representation $T_l$ is extracted strictly as the last token of $T_{\text{hig}}^e$ (index -1), without any final LayerNorm.

---

## 3. Training vs. Inference Separation

- **Training Mode:** Executes full multi-modal forward pass (MVE, VD, FLT, VIM, Adapter, Detection Head) and optimizes all 6 multi-task objectives:
  $$\mathcal{L} = \lambda_{\text{fd}} \mathcal{L}_{\text{fd}} + \lambda_{\text{lr}} \mathcal{L}_{\text{lr}} + \lambda_{\text{cmc}} \mathcal{L}_{\text{cmc}} + \lambda_{\text{fl}} \mathcal{L}_{\text{fl}} + \lambda_{\text{ar}} \mathcal{L}_{\text{ar}} + \lambda_{\text{kl}} \mathcal{L}_{\text{kl}}$$
  where all $\lambda_k = 1.0$.
- **Standard Inference Mode:** **Image-only forward pass** (`forward_image_only`). Bypasses text tokenization, prompt generation, Language Encoder, Language Decoder, FLT, and language VIM paths entirely.

---

## 4. Installation & Setup

### Requirements:
- Python 3.10+
- PyTorch 2.x
- NumPy, Pillow, PyYAML, scikit-learn, pytest

```bash
# Clone the repository
git clone <repository_url>
cd Paper

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Repository Structure

```
├── configs/
│   ├── mfvlr.yaml              # Main architecture, loss, and optimization config
│   └── dataset.yaml            # Dataset and mask generation config
├── datasets/
│   ├── manifest_dataset.py     # Generic manifest dataset loader (.jsonl, .json, .csv)
│   ├── mask_generator.py       # Ground-truth mask generation (Eq. 18 procedure)
│   ├── prompt_generator.py     # L1-L4 hierarchical prompt metadata generator
│   ├── tokenizer.py            # Offline deterministic tokenizer boundary (s=49408, n=308)
│   ├── dummy_dataset.py        # Synthetic test dataset generator
│   └── transforms.py           # Standard RGB float preprocessing
├── models/
│   ├── vision/
│   │   ├── unet_encoder.py     # Local conv U-Net encoder (I_loc)
│   │   ├── transformer.py      # Image Transformer (B=4 blocks, I_g)
│   │   ├── mve.py              # Multi-domain Vision Encoder (IE + RE + Fusion)
│   │   └── vision_decoder.py   # Shared U-Net trunk, AD head, MD head
│   ├── language/
│   │   ├── embeddings.py       # Token & positional embeddings (W_voc, P_e, P_d)
│   │   ├── vim.py              # Vision Injection Module (Eq. 4-11)
│   │   ├── language_encoder.py # E=12 LE blocks with VIM
│   │   ├── language_decoder.py # D=7 LD blocks with causal MMHA, cross-attention, tied vocab
│   │   └── flt.py              # Fine-Grained Language Transformer wrapper
│   ├── heads/
│   │   ├── adapter.py          # Visual-to-language adapter (T_lpre)
│   │   └── detection_head.py   # 2-class detection MLP (y_pre)
│   ├── losses/
│   │   ├── detection_loss.py   # L_fd (Eq. 26)
│   │   ├── language_reconstruction_loss.py # L_lr (Eq. 24-25)
│   │   ├── appearance_reconstruction_loss.py # L_ar MSE (Eq. 17)
│   │   ├── localization_loss.py# L_fl (Eq. 18)
│   │   ├── kl_loss.py          # L_kl (Eq. 19, tau=0.5)
│   │   ├── cmc_loss.py         # L_cmc (Eq. 20-23, unnormalized dot product, trainable tau)
│   │   └── total_loss.py       # Multi-task loss combiner (Eq. 27)
│   └── mfvlr.py                # Full model orchestration & image-only inference
├── utils/
│   ├── trainer.py              # Adam optimizer, StepLR, train_step, train_one_epoch
│   ├── evaluator.py            # Image-only evaluation pipeline
│   ├── checkpoint.py           # State persistence (weights, optimizer, scheduler, log_tau)
│   ├── metrics.py              # ACC, continuous AUC, mIoU computation
│   ├── seed.py                 # Deterministic seed utilities
│   └── logger.py               # Structured logging
├── train.py                    # Training CLI with --dry-run and --resume
├── evaluate.py                 # Evaluation CLI (image-only)
├── infer.py                    # Single-image inference CLI
├── docs/                       # Complete Phase 1-12 reports, notes, and traceability matrix
└── tests/                      # 23 comprehensive unit and integration test suites
```

---

## 6. Manifest Schema

The dataset interface is manifest-driven and does not require an arbitrary directory tree:

```jsonl
{"image_path": "images/real_001.jpg", "label": 0, "manipulation_type": "Real"}
{"image_path": "images/fake_001.jpg", "label": 1, "manipulation_type": "EFS", "generator": "DDPM", "family": "diffusion"}
{"image_path": "images/fake_002.jpg", "source_image_path": "images/real_002.jpg", "label": 1, "manipulation_type": "AM", "generator": "DiffAE", "family": "diffusion"}
```

Paths are resolved relative to `dataset.root` or the manifest file directory.

---

## 7. Command Line Interface (CLI) Usage

### 1. Smoke Testing / Dry-Run:
Execute a single training step (all 6 losses, backward pass, optimizer update) plus an image-only evaluation batch:
```bash
python train.py --config configs/mfvlr.yaml --dry-run --device cpu
```

### 2. Full Training:
```bash
python train.py --config configs/mfvlr.yaml --epochs 30 --batch-size 8 --device cuda
```

### 3. Checkpoint Resume:
```bash
python train.py --config configs/mfvlr.yaml --resume checkpoints/checkpoint_epoch_15.pt --epochs 30
```

### 4. Image-Only Evaluation:
```bash
python evaluate.py --config configs/mfvlr.yaml --checkpoint checkpoints/best_model.pt --manifest datasets/test_manifest.jsonl
```

### 5. Single-Image Inference:
```bash
python infer.py --config configs/mfvlr.yaml --checkpoint checkpoints/best_model.pt --image sample_face.jpg --output-mask predicted_mask.png
```

---

## 8. Paper-Specified Hyperparameters vs. Documented Assumptions

| Parameter | Value | Classification | Notes |
| :--- | :--- | :--- | :--- |
| **Image Resolution** | $224 \times 224 \times 3$ | `PAPER_SPECIFIED` | Standard input dimension |
| **Local Feature Shape** | $[B, 1024, 14, 14]$ | `PAPER_SPECIFIED` | Output of U-Net Encoder $I_{\text{loc}}$ |
| **Global Feature Dim** | $d = 512$ | `PAPER_SPECIFIED` | Transformer embed dim |
| **Image Transformer Blocks** | $B = 4$ | `PAPER_SPECIFIED` | Visual Transformer depth |
| **Language Encoder Blocks** | $E = 12$ | `PAPER_SPECIFIED` | Text encoder depth |
| **Language Decoder Blocks** | $D = 7$ | `PAPER_SPECIFIED` | Text decoder depth |
| **Sequence Length** | $n = 308$ | `PAPER_SPECIFIED` | Text token sequence length |
| **Vocabulary Size** | $s = 49,408$ | `PAPER_SPECIFIED` | Output vocabulary space |
| **Batch Size** | $b = 8$ | `PAPER_SPECIFIED` | Training batch size |
| **Optimizer** | Adam ($\text{lr}=10^{-4}, \text{wd}=10^{-3}$) | `PAPER_SPECIFIED` | Standard Adam optimizer |
| **LR Schedule** | StepLR ($\text{step}=15, \gamma=0.1$) | `PAPER_SPECIFIED` | Reduce LR by $10\times$ every 15 epochs |
| **Loss Weights** | All $\lambda_k = 1.0$ | `PAPER_SPECIFIED` | Equal multi-task weighting |
| **CMC Temperature** | Trainable, init $\tau=0.07$ | `PAPER_SPECIFIED` | Unnormalized dot-product similarity |
| **KL Temperature** | Fixed $\tau = 0.5$ | `PAPER_SPECIFIED` | Direction: $D_{\text{KL}}(P(T_l) \parallel Q(T_{\text{lpre}}))$ |
| **Mask Threshold** | $0.1$ | `PAPER_SPECIFIED` | AM/FS absolute difference threshold |
| **Input Float Scaling** | Float RGB in $[0, 1]$ | `ASSUMPTION_FROM_PAPER_GAP` | Reconstruction-compatible engineering choice |
| **Prompt Packing Strategy** | $4 \times 77 = 308$ | `ASSUMPTION_FROM_PAPER_GAP` | Padding/concatenation of L1-L4 prompts |
| **Special Token IDs** | $\text{BOS}=49406, \text{EOS}=49407, \text{PAD}=0$ | `ASSUMPTION_FROM_PAPER_GAP` | Standard CLIP-compatible IDs |
| **Grayscale Luminance Weights** | $(0.299, 0.587, 0.114)$ | `ASSUMPTION_FROM_PAPER_GAP` | ITU-R BT.601 weights for mask diff conversion |
| **Total Training Epochs** | Configurable (default null) | `ASSUMPTION_FROM_PAPER_GAP` | Total epochs unspecified in paper |

---

## 9. Running Tests

To run the complete test suite across all 23 test modules:
```bash
python -m pytest -v
```
All 93 tests execute and pass in $\approx 86$ seconds.

---

## 10. License

This reproduction implementation is licensed under the Apache 2.0 License.
