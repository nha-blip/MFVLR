# CollDiff Official Validation Report

**Status:** `CollDiff = OFFICIAL_READY`  
**Date:** 2026-09-19  
**Target Generator:** CollDiff (Entire Face Synthesis - EFS)  
**Architecture Family:** Diffusion  
**Reproduced Environment:** `latdiff_env` (Conda, Python 3.8.20, PyTorch 1.13.1+cu117)

---

## 1. Hardware & Runtime Audit

| Parameter | Value |
| :--- | :--- |
| **GPU Name** | NVIDIA GeForce RTX 4050 Laptop GPU (actual detected) |
| **Actual VRAM Total** | 6.00 GB (6,438,780,928 bytes / 6,141 MiB) |
| **System RAM Total** | 15.71 GB (Available: 7.36 GB at start) |
| **CUDA Driver** | 581.86 (CUDA 13.0) |
| **CUDA Runtime** | 11.7 (`torch.version.cuda = 11.7`) |
| **PyTorch Version** | 1.13.1+cu117 |
| **Python Version** | 3.8.20 |

---

## 2. Official Implementation Specifications

| Property | Value |
| :--- | :--- |
| **Official Repository** | `https://github.com/ziqihuangg/Collaborative-Diffusion.git` |
| **Git Commit** | `a4a0d1f13bc83f2315cb885ef1095bb498c5984c` |
| **Paper** | *Collaborative Diffusion for Multi-Modal Face Generation and Editing* (Huang et al., CVPR 2023) |
| **Local Clone Path** | `external/colldiff` |
| **Primary Model Class** | `ldm.models.diffusion.ddpm_compose.LatentDiffusionCompose` |
| **Sub-Model Classes** | `ldm.modules.diffusionmodules.compose_openaimodel.ComposeUNet` (833.52M params)<br/>`ldm.models.diffusion.ddpm.LatentDiffusion` (seg_mask: 403.62M params)<br/>`ldm.models.diffusion.ddpm.LatentDiffusion` (text: 403.62M params)<br/>`ldm.models.autoencoder.AutoencoderKL` (vae: 34.1M params) |
| **Total Parameter Count** | **1,674,860,000 parameters** (~1.67B parameters across 4 branches) |
| **Native Output Resolution** | $256 \times 256$ (3 channels RGB) |
| **Final Dataset Resolution** | $224 \times 224$ (bilinear resize) |
| **Sampling Algorithm** | Multi-modal DDIM (50 steps, eta=1.0) |

---

## 3. Checkpoint Verification

All checkpoints were downloaded directly from the official Google Drive provided in the paper's official repository into `external/colldiff/pretrained/` and `checkpoints/EFS/CollDiff/`:

| Checkpoint | File Size | SHA256 Checksum |
| :--- | :--- | :--- |
| **`256_codiff_mask_text.ckpt`** | 4.21 GB (`4,524,429,737` bytes) | `a1a2801566530f35caed8e32e072de6bbc6a512ca48aef6f11ac6e520793f1df` |
| **`256_mask.ckpt`** | 4.73 GB (`5,081,213,901` bytes) | `6a236982d6af7d511301670360e9e8b1a96e89f14d97a6984be60549232ac70a` |
| **`256_text.ckpt`** | 6.65 GB (`7,144,705,537` bytes) | `83c80c5bcf1828c4b9c3db43d4a31f048d08704c07a3ff1ce90a148ed3f5c265` |
| **`256_vae.ckpt`** | 0.70 GB (`756,128,105` bytes) | `e1e07b18297851e4087e3104a4db7bf23dc2ec8fbd483becef7c5c19af0c5a42` |

---

## 4. Local Execution & Resource Benchmark

Official inference was executed in pure local mode using batch size 1, `torch.no_grad()`, and native DDIM 50-step sampling on the RTX 4050 GPU.

| Metric | Measured Value |
| :--- | :--- |
| **Number of Real Samples** | 10 (`colldiff_000000.png` to `colldiff_000009.png`) |
| **Peak VRAM** | **4,471.1 MB** (4.37 GB / 6.00 GB limit — 72.8% VRAM utilization) |
| **Peak System RAM** | **2,220.9 MB** (2.17 GB / 15.71 GB limit — 13.8% RAM utilization) |
| **Average Runtime / Sample** | **15.62 seconds / image** (3.4 it/s DDIM) |
| **Sample Resolution** | $224 \times 224 \times 3$ (PNG) |
| **Mask Value** | Full face synthetic (all pixels `255`, $224 \times 224$ single channel PNG) |
| **Visual Verification Grid** | `MFVLR_Dataset/logs/verification_samples/colldiff_visual_grid.png` |

### Sample Performance Breakdown

| Sample | Seed | Time (s) | Peak VRAM (MB) | RAM (MB) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `colldiff_000000.png` | 0 | 19.06 | 4,465.6 | 2,204.7 | PASS |
| `colldiff_000001.png` | 1 | 15.68 | 4,471.1 | 2,211.2 | PASS |
| `colldiff_000002.png` | 2 | 15.50 | 4,471.1 | 2,215.2 | PASS |
| `colldiff_000003.png` | 3 | 15.52 | 4,471.1 | 2,216.0 | PASS |
| `colldiff_000004.png` | 4 | 14.99 | 4,471.1 | 2,216.0 | PASS |
| `colldiff_000005.png` | 5 | 14.70 | 4,471.1 | 2,217.0 | PASS |
| `colldiff_000006.png` | 6 | 15.08 | 4,471.1 | 2,217.0 | PASS |
| `colldiff_000007.png` | 7 | 15.23 | 4,471.1 | 2,217.7 | PASS |
| `colldiff_000008.png` | 8 | 15.75 | 4,471.1 | 2,219.4 | PASS |
| `colldiff_000009.png` | 9 | 14.64 | 4,471.1 | 2,220.9 | PASS |

---

## 5. Quality Gate Checklist (A–J)

- [x] **A. Official repository?** `https://github.com/ziqihuangg/Collaborative-Diffusion` (commit `a4a0d1f13bc83f2315cb885ef1095bb498c5984c`).
- [x] **B. Official architecture?** Multi-modal Collaborative Diffusion (`ComposeDiffusion` composed of mask LDM, text LDM, composition UNet, and VAE).
- [x] **C. Official checkpoint?** All 4 official checkpoints verified and loaded (`256_codiff_mask_text.ckpt`, `256_mask.ckpt`, `256_text.ckpt`, `256_vae.ckpt`).
- [x] **D. SHA256 recorded?** All 4 checkpoints recorded and verified.
- [x] **E. Real inference?** Yes, 10 official DDIM sampling runs executed on local RTX 4050 GPU without surrogate, mocks, or quantization.
- [x] **F. 5–10 samples?** Exactly 10 samples generated (`colldiff_000000.png` to `colldiff_000009.png`).
- [x] **G. Correct source/target pairing?** N/A (EFS does not require source/target pairing; text condition "This woman is in her forties." + segmentation mask conditioning).
- [x] **H. Correct mask?** Yes, full face mask (all pixels `255`, 8-bit grayscale, $224 \times 224$).
- [x] **I. Correct architecture label?** `Diffusion`.
- [x] **J. Correct L1–L4 prompts?**
  - L1: `A photo of a fake face`
  - L2: `A photo of an entire synthesized face`
  - L3: `A photo generated by the diffusion-based model`
  - L4: `The source generative model of this photo is CollDiff`
