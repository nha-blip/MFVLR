# LatTrans Official Validation Report

**Status:** `LatTrans = OFFICIAL_READY`  
**Date:** 2026-09-19  
**Target Generator:** LatTrans (Attribute Manipulation - AM)  
**Architecture Family:** GAN  
**Reproduced Environment:** `lattrans_env` (Conda, Python 3.8.20, PyTorch 1.13.1+cu117, CUDA Toolkit 11.7, MSVC 2022 v14.44)

---

## 1. Hardware & Toolchain Audit

| Parameter | Value |
| :--- | :--- |
| **GPU Name** | NVIDIA GeForce RTX 4050 Laptop GPU (actual detected) |
| **Actual VRAM Total** | 6.00 GB (6,438,780,928 bytes / 6,141 MiB) |
| **System RAM Total** | 15.71 GB (Available: 5.61 GB at start) |
| **CUDA Driver** | 581.86 (CUDA 13.0) |
| **CUDA Runtime** | 11.7 (`torch.version.cuda = 11.7`) |
| **CUDA Toolkit / Compiler** | NVIDIA CUDA Toolkit 11.7 (`nvcc.exe` at `lattrans_env\Library\bin\nvcc.exe`) |
| **Host C++ Compiler** | Microsoft Visual Studio 2022 Community MSVC `cl.exe` (v14.44.35207) |
| **Build Tool** | Ninja 1.13.2 |
| **PyTorch Version** | 1.13.1+cu117 |
| **Python Version** | 3.8.20 |

---

## 2. Official Implementation Specifications

| Property | Value |
| :--- | :--- |
| **Official Repository** | `https://github.com/InterDigitalInc/latent-transformer.git` |
| **Git Commit** | `00155b9a8cb404c0ae38d3809214732aa57a2fdf` |
| **Paper** | *A Latent Transformer for Disentangled Face Editing in Images and Videos* (Yao et al., ICCV 2021) |
| **Local Clone Path** | `external/latent-transformer` |
| **Auxiliary Dependencies** | `pixel2style2pixel` (`https://github.com/eladrich/pixel2style2pixel.git`) |
| **Model Classes** | `nets.F_mapping` (Latent Transformer T-Net: 18 layers, 512 fmaps, 4.73M params)<br/>`pixel2style2pixel.models.psp.pSp` (pSp GradualStyleEncoder: 26.68M params)<br/>`pixel2style2pixel.models.stylegan2.model.Generator` (StyleGAN2 1024x1024: 30.03M params) |
| **Total Parameter Count** | **61,440,000 parameters** (~61.44M parameters) |
| **Native Output Resolution** | $1024 \times 1024$ (3 channels RGB) |
| **Final Dataset Resolution** | $224 \times 224$ (bilinear resize) |
| **Manipulation Protocol** | Source face $\rightarrow$ pSp inversion to $W^+$ latent code ($1 \times 18 \times 512$) $\rightarrow$ LatTrans edit (Smiling $+1.5$) $\rightarrow$ StyleGAN2 synthesis |
| **CUDA Extensions** | Official `fused_bias_act` and `upfirdn2d` compiled and loaded via `torch.utils.cpp_extension` with Windows MSVC 2022 + CUDA 11.7 toolchain. |

---

## 3. Checkpoint Verification

All checkpoints were downloaded directly from the official Google Drive provided in the paper's official repository into `external/latent-transformer/`:

| Checkpoint | Path / Official Source | File Size | SHA256 Checksum |
| :--- | :--- | :--- | :--- |
| **pSp FFHQ Encoder (`psp_ffhq_encode.pt`)** | `external/latent-transformer/pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt` | 1,145.85 MB (`1,201,489,451` bytes) | `f786ac5cf18fcdbc5c57c9f3f12cbedca69381fbdf5694dd132294e3133749d7` |
| **Smiling Transformer (`tnet_31.pth.tar`)** | `external/latent-transformer/logs/001/tnet_31.pth.tar` | 18.05 MB (`18,926,901` bytes) | `66fe1e5f0de8fe20d500033ba62ba002c512b7fe7ad7c72a9fe21404b2354a40` |
| **Latent Classifier (`latent_classifier_epoch_20.pth`)** | `external/latent-transformer/models/latent_classifier_epoch_20.pth` | 76.09 MB (`79,788,381` bytes) | `5af28f11222d4a1c076e7552a823b44ffe646b47c9af12820ceae8a4be44e05b` |

---

## 4. Local Execution & Resource Benchmark

Official inference was executed in pure local mode using batch size 1, `torch.no_grad()`, native pSp encoder inversion, LatTrans $W^+$ manipulation, and official StyleGAN2 CUDA decoder on the RTX 4050 GPU.

| Metric | Measured Value |
| :--- | :--- |
| **Number of Real Samples** | 10 (`lattrans_000000.png` to `lattrans_000009.png`) |
| **Peak VRAM** | **1,606.4 MB** (1.57 GB / 6.00 GB limit — 26.2% VRAM utilization) |
| **Peak System RAM** | **3,534.9 MB** (3.45 GB / 15.71 GB limit — 22.0% RAM utilization) |
| **Average Runtime / Sample** | **0.24 seconds / image** (warm: 0.09s / image) |
| **Sample Resolution** | $224 \times 224 \times 3$ (PNG) |
| **Mask Value** | MFVLR-compliant attribute difference mask: $|fake - corresponding\_source| \rightarrow \text{ITU-R BT.601 grayscale} \rightarrow \text{normalize } /255.0 \rightarrow \text{threshold} > 0.1 \rightarrow \text{binary } \{0, 255\} \rightarrow \text{no morphology} \rightarrow 224 \times 224$ |
| **Source Image Pairing** | Paired 1-to-1 with real faces in `MFVLR_Dataset/source/AM/LatTrans/` |
| **Visual Verification Grid** | `MFVLR_Dataset/logs/verification_samples/lattrans_visual_grid.png` |

### Sample Performance Breakdown

| Sample | Source Image | Time (s) | Peak VRAM (MB) | RAM (MB) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `lattrans_000000.png` | `000000.png` | 1.53 | 1,569.4 | 3,506.8 | PASS |
| `lattrans_000001.png` | `000001.png` | 0.11 | 1,605.4 | 3,531.3 | PASS |
| `lattrans_000002.png` | `000002.png` | 0.08 | 1,605.4 | 3,532.8 | PASS |
| `lattrans_000003.png` | `000003.png` | 0.09 | 1,605.4 | 3,532.8 | PASS |
| `lattrans_000004.png` | `000004.png` | 0.09 | 1,605.4 | 3,532.8 | PASS |
| `lattrans_000005.png` | `000005.png` | 0.09 | 1,605.4 | 3,533.5 | PASS |
| `lattrans_000006.png` | `000006.png` | 0.09 | 1,605.4 | 3,533.5 | PASS |
| `lattrans_000007.png` | `000007.png` | 0.09 | 1,605.4 | 3,534.2 | PASS |
| `lattrans_000008.png` | `000008.png` | 0.09 | 1,605.4 | 3,534.3 | PASS |
| `lattrans_000009.png` | `000009.png` | 0.10 | 1,606.4 | 3,534.9 | PASS |

---

## 5. Quality Gate Checklist (A–J)

- [x] **A. Official repository?** `https://github.com/InterDigitalInc/latent-transformer` (commit `00155b9a8cb404c0ae38d3809214732aa57a2fdf`).
- [x] **B. Official architecture?** Latent Transformer (`F_mapping`) with `pixel2style2pixel` encoder and StyleGAN2 decoder.
- [x] **C. Official checkpoint?** All checkpoints verified and loaded (`psp_ffhq_encode.pt`, `tnet_31.pth.tar`, `latent_classifier_epoch_20.pth`).
- [x] **D. SHA256 recorded?** All 3 checkpoints recorded and verified.
- [x] **E. Real inference?** Yes, 10 official inversion + attribute manipulation runs executed on local RTX 4050 GPU without surrogate or mocks.
- [x] **F. 5–10 samples?** Exactly 10 samples generated (`lattrans_000000.png` to `lattrans_000009.png`).
- [x] **G. Correct source/target pairing?** Yes, 1-to-1 paired with real source images in `MFVLR_Dataset/source/AM/LatTrans/`.
- [x] **H. Correct mask?** Audited and harmonized per MFVLR reproduction protocol: computed strictly as $|fake - source| \rightarrow \text{RGB-to-grayscale (ITU-R BT.601)} \rightarrow /255.0 \rightarrow \text{threshold} > 0.1 \rightarrow \text{binary } \{0, 255\}$ at $224 \times 224$. Morphology/dilation completely removed, difference pair bound to corresponding source face (see [MASK_PROTOCOL_AUDIT.md](file:///c:/Ổ%20đĩa%20D/MFVLR/MASK_PROTOCOL_AUDIT.md)).
- [x] **I. Correct architecture label?** `GAN`.
- [x] **J. Correct L1–L4 prompts?**
  - L1: `A photo of a fake face`
  - L2: `A photo of an attribute-manipulated face`
  - L3: `A photo generated by the gan-based model`
  - L4: `The source generative model of this photo is LatTrans`
