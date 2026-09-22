# Master Report: All Generators Official Validation

**Project:** MFVLR (Multi-Granularity Face Video & Image Forgery Localization and Recognition) Dataset Reproduction  
**Date:** 2026-09-19  
**Scope:** Complete Official Generator Audit and Local GPU Validation (10 Generators + Real)  
**Strict Reproduction Policy:** No surrogates, no mock substitutions, no SDEdit replacement, official architectures and weights strictly audited.

---

## 1. Hardware & Environment Audit

Detected and audited at runtime on the local validation machine:

| Property | Value |
| :--- | :--- |
| **GPU Name** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **VRAM Total** | 6,438,780,928 bytes (6.00 GB / 6,141 MiB) |
| **System RAM Total** | 16,088 MB (15.71 GB) |
| **CUDA Driver** | 581.86 (CUDA 13.0) |
| **CUDA Runtime** | 11.7 (PyTorch `1.13.1+cu117`) |
| **PyTorch Version** | `1.13.1+cu117` |
| **Python Version** | `3.8.20` |
| **Host OS & Toolchain** | Windows 11, MSVC 2022 v14.44, Ninja 1.11.1 |

---

## 2. Executive Summary Table

| # | Generator | Category | Architecture Family | Final Status | Local Execution (RTX GPU) | Checkpoints & Specifications |
|---|---|---|---|:---:|:---:|---|
| 0 | **Real** | REAL | REAL | **`OFFICIAL_READY`** | PASS (15 samples) | CelebA-HQ real source faces ($224 \times 224$). |
| 1 | **DDPM** | EFS | Diffusion | **`OFFICIAL_READY`** | PASS (15 samples) | `google/ddpm-celebahq-256`, 113.7M params. |
| 2 | **DiffAE** | AM | Diffusion | **`OFFICIAL_READY`** | PASS (10 samples) | `ffhq256_autoenc` (168.5M params) + classifier, smile edit. |
| 3 | **StyleGAN3** | EFS | GAN | **`OFFICIAL_READY`** | PASS (10 samples) | `stylegan3-r-ffhq-1024x1024.pkl` (61.4M params, 305 MB). |
| 4 | **LatDiff** | EFS | Diffusion | **`OFFICIAL_READY`** | PASS (10 samples) | `CompVis/latent-diffusion` CelebA 256x256 (329.4M params, 2.25 GB). |
| 5 | **CollDiff** | EFS | Diffusion | **`OFFICIAL_READY`** | PASS (10 samples) | `256_codiff_mask_text.ckpt` + 3 LDMs (1.67B params). Peak VRAM 4.37 GB, 15.62s/img. |
| 6 | **LatTrans** | AM | GAN | **`OFFICIAL_READY`** | PASS (10 samples) | pSp FFHQ (1.20 GB) + T-Net (705 MB) + StyleGAN2 decoder. Peak VRAM 1.57 GB, 0.24s/img. |
| 7 | **IAFaces** | AM | GAN | **`OFFICIAL_READY`** | PASS (10 samples) | `iafaces-celebahq-256.pth` (582 MB, 46.8M params). Peak VRAM 0.37 GB, 0.141s/img. |
| 8 | **DiffFace** | FS | Diffusion | **`MANUAL_DOWNLOAD_REQUIRED`** | Checkpoint 404 | Official repository cloned. Author's GIST SharePoint link dead (HTTP 404). Slurm script ready. |
| 9 | **FSLSD** | FS | GAN | **`MANUAL_DOWNLOAD_REQUIRED`** | Permission Denied | Official repository cloned. Google Drive link permission restricted. Linux NCCL backend + CUDA JIT required. |
| 10 | **FaceSwapper** | FS | GAN | **`OFFICIAL_READY`** | PASS (10 samples) | `faceswapper.ckpt` (306 MB) + `model_ir_se50.pth` (175 MB) + `wing.ckpt` (194 MB). 0.34s/img, peak VRAM 688.7 MB. |

---

## 3. Detailed Generator Audit & Evidence

### 3.1 Entire Face Synthesis (EFS)

#### DDPM (`OFFICIAL_READY`)
- **Repository:** `google/ddpm-celebahq-256` (Hugging Face diffusers).
- **Architecture:** `UNet2DModel` (113,742,083 parameters).
- **Checkpoints:** `diffusion_pytorch_model.bin` (455 MB, SHA256: `6b81d77cb077d018788417c88b4ddc89a7444c1dcae9e0350a4ec9ec51787c8d`).
- **Runtime:** 15 samples generated, 50-step DDIM, $256 \times 256 \rightarrow 224 \times 224$. Mask: all 255.

#### StyleGAN3 (`OFFICIAL_READY`)
- **Repository:** `https://github.com/NVlabs/stylegan3` (commit `07d031548e24483733230a103c80081bf582531d`).
- **Architecture:** `SynthesisNetwork` / `Generator` (61,353,235 parameters).
- **Checkpoints:** `stylegan3-r-ffhq-1024x1024.pkl` (305 MB, SHA256: `a93b3dcfaee08d3e913a89047196fa0bb843c08253fbbeec76b3dd7e1e63a8e9`).
- **Runtime:** 10 samples generated, peak VRAM 1.10 GB, $1024 \times 1024 \rightarrow 224 \times 224$. Report: [`STYLEGAN3_OFFICIAL_VALIDATION.md`](file:///c:/Ổ đĩa D/MFVLR/STYLEGAN3_OFFICIAL_VALIDATION.md).

#### LatDiff (`OFFICIAL_READY`)
- **Repository:** `https://github.com/CompVis/latent-diffusion` (commit `a506df5756472e2ebaf9078affdde2c4f1502cd4`).
- **Architecture:** `ldm.models.diffusion.ddpm.LatentDiffusion` with VQ-f4 autoencoder (329,378,945 parameters).
- **Checkpoints:** `model.ckpt` (2.25 GB, SHA256: `aa782e2c4f9b49e70be153add726450f4165ed157690fda8258aaaef782b8002`) + `vq-f4/model.ckpt` (721 MB).
- **Runtime:** 10 samples generated, 50-step DDIM, 3.9s/img, peak VRAM 2.72 GB. Report: [`LATDIFF_OFFICIAL_VALIDATION.md`](file:///c:/Ổ đĩa D/MFVLR/LATDIFF_OFFICIAL_VALIDATION.md).

#### CollDiff (`OFFICIAL_READY`)
- **Repository:** `https://github.com/ziqihuangg/Collaborative-Diffusion` (commit `a4a0d1f13bc83f2315cb885ef1095bb498c5984c`).
- **Architecture:** Multi-modal Collaborative Diffusion (`ComposeDiffusion` composed of mask LDM, text LDM, composition UNet, VAE; 1.67B parameters).
- **Checkpoints:**
  - `256_codiff_mask_text.ckpt` (4.52 GB, SHA256: `a1a280155bbf89ba7e2b1ea739d486259e81b6aa2828b6d193d56b005183dbd0`)
  - `256_mask.ckpt` (5.07 GB, SHA256: `6a2369829377461ab1d06371cb769a23c34ff8c3cbfd72ee20c57c744f479d2b`)
  - `256_text.ckpt` (7.14 GB, SHA256: `83c80c5bd4eaeb8092a41a45ae785507be169542a15998a44b9319eeb3ee3794`)
  - `256_vae.ckpt` (756 MB, SHA256: `e1e07b189ff4c23dbd4c4bfaec739e4a3aa52674e2a632007823f6e16fdfdb22`)
- **Optimization:** Sequential model loading with garbage collection eliminated peak RAM spiking from 17.5 GB to 2.17 GB.
- **Runtime:** 10 samples generated, 50-step DDIM, 15.62s/img, peak VRAM 4.37 GB (well within 6.00 GB limit). Report: [`COLLDIFF_OFFICIAL_VALIDATION.md`](file:///c:/Ổ đĩa D/MFVLR/COLLDIFF_OFFICIAL_VALIDATION.md).

---

### 3.2 Attribute Manipulation (AM)

#### DiffAE (`OFFICIAL_READY`)
- **Repository:** `https://github.com/phizaz/diffae` / `external/diffae`.
- **Architecture:** Diffusion Autoencoders (`LitModel` 168.49M params + `ClsModel`).
- **Checkpoints:** `ffhq256_autoenc/last.ckpt` (SHA256: `9bd2ba9e...`) + `ffhq256_autoenc_cls/last.ckpt` (SHA256: `a8338109...`).
- **Runtime:** 10 samples generated with real smile attribute manipulation. Report: [`DIFFAE_OFFICIAL_VALIDATION.md`](file:///c:/Ổ đĩa D/MFVLR/DIFFAE_OFFICIAL_VALIDATION.md).

#### LatTrans (`OFFICIAL_READY`)
- **Repository:** `https://github.com/InterDigitalInc/latent-transformer` (commit `00155b9a8cb404c0ae38d3809214732aa57a2fdf`).
- **Architecture:** Latent Transformer `F_mapping` + `LCNet` + `pixel2style2pixel` StyleGAN2 generator.
- **Checkpoints:**
  - `psp_ffhq_encode.pt` (1.20 GB, SHA256: `f786ac5c8e3aa7738ee055375dbfb0eecb08f86f687da6a0572e01dfd2943714`)
  - `tnet_31.pth.tar` (705 MB, SHA256: `66fe1e5fb5ca53e1cb07357dd8d0f19cfa12f38c3aa74b29c9b19dfb2d699eec`)
  - `latent_classifier_epoch_20.pth` (79.8 MB, SHA256: `5af28f11ec32203e3a966779a9da8a502c9a96e216e254dfbfa8e64c24e6500c`)
- **Toolchain Resolution:** Successfully compiled official StyleGAN2 CUDA C++ extensions (`fused_bias_act`, `upfirdn2d`) on Windows using MSVC 2022 + CUDA Toolkit 11.7 with Win32 8.3 short paths.
- **Runtime:** 10 samples generated, 0.24s/img (warm: 0.09s/img), peak VRAM 1.57 GB. Report: [`LATTRANS_OFFICIAL_VALIDATION.md`](file:///c:/Ổ đĩa D/MFVLR/LATTRANS_OFFICIAL_VALIDATION.md).

#### IAFaces (`OFFICIAL_READY`)
- **Repository:** `https://github.com/CMACH508/IA-FaceS` (commit `50e18fc091f03f5ce6bbd377b7f58cb2e93b4dd6`).
- **Architecture:** `model.iafaces_256.Encoder` + `model.iafaces_256.Generator` with Component Adaptive Modulation (CAM, 46.8M params).
- **Checkpoints:**
  - `iafaces-celebahq-256.pth` (582 MB, SHA256: `b08552b6c1adcc372b665e005cf1a4fa3a50112ba747391060e268db41b6cab2`)
- **Runtime:** 10 samples generated, 0.141s/img (warm: 0.024s/img), peak VRAM 0.37 GB. Report: [`IAFACES_OFFICIAL_VALIDATION.md`](file:///c:/Ổ đĩa D/MFVLR/IAFACES_OFFICIAL_VALIDATION.md).

---

### 3.3 Face Swapping (FS)

#### DiffFace (`MANUAL_DOWNLOAD_REQUIRED`)
- **Repository:** `https://github.com/hxngiee/DiffFace` (commit `53ea455a538561be0d5bb64c7ee608a0d0144f80`).
- **Architecture:** Conditional DDPM with identity/segmentation/gaze facial guidance.
- **Checkpoints Required:** `Arcface.tar`, `FaceParser.pth`, `GazeEstimator.pt`, `Model.pt`.
- **Status Reason:** The author's official GIST SharePoint link returned HTTP 404 (storage expired). Slurm script ready. Report: [`DIFFFACE_OFFICIAL_VALIDATION.md`](file:///c:/Ổ đĩa D/MFVLR/DIFFFACE_OFFICIAL_VALIDATION.md).

#### FSLSD (`MANUAL_DOWNLOAD_REQUIRED`)
- **Repository:** `https://github.com/cnnlstm/FSLSD_HiRes` (commit `4cf086965c477444960833bd9fd054d274b1aec5`).
- **Architecture:** Latent Semantics Disentanglement Face Swapper (`GradualLandmarkEncoder`, `GPENEncoder`, `Generator`, `Decoder`, `bald_model`).
- **Checkpoints Required:** `CELEBA-HQ-1024.pt`.
- **Status Reason:** Author's Google Drive link (`1LH4RlxaPnrHAiWEDm3LDp5Sz9H02bzXU`) is access-restricted / quota-exceeded. Slurm script ready. Report: [`FSLSD_OFFICIAL_VALIDATION.md`](file:///c:/Ổ đĩa D/MFVLR/FSLSD_OFFICIAL_VALIDATION.md).

#### FaceSwapper (`OFFICIAL_READY`)
- **Repository:** `https://github.com/liqi-casia/FaceSwapper` (commit `69f53e5e493214736f33cf224c084f74d081e7d2`).
- **Architecture:** One-shot Progressive Face Swapping (`IdentityEncoder`, `AttrEncoder`, `Decoder` with `AdaIN`, `FAN`, `Backbone`).
- **Checkpoints:** `faceswapper.ckpt` (306 MB), `model_ir_se50.pth` (175 MB), `wing.ckpt` (194 MB).
- **Runtime:** 10 samples generated on GPU, 0.34s/img, peak VRAM 688.7 MB. Report: [`FACESWAPPER_OFFICIAL_VALIDATION.md`](file:///c:/Ổ đĩa D/MFVLR/FACESWAPPER_OFFICIAL_VALIDATION.md).

---

## 4. Dataset Integrity & Quality Gate Summary

- **Total Samples in MFVLR_Dataset:** 115 verified samples (`images/`, `source/`, `masks/`, `metadata/all.csv`).
- **Mask Protocol Audit:** Fully audited and harmonized across all AM and FS generators per official MFVLR specification (see [MASK_PROTOCOL_AUDIT.md](file:///c:/Ổ%20đĩa%20D/MFVLR/MASK_PROTOCOL_AUDIT.md)). All masks computed strictly as $|fake - corresponding\_source| \rightarrow \text{RGB-to-grayscale (ITU-R BT.601)} \rightarrow /255.0 \rightarrow \text{threshold} > 0.1 \rightarrow \text{binary } \{0, 255\}$ at $224 \times 224$. All unauthorized morphology (dilation/blur) and intermediate reconstruction subtractions removed.
- **Quality Gate H (Mask Integrity):** **`PASS`** across all active generators.
- **Dataset Verification (`verify_dataset.py`):** **`PASS`**, 0 errors, 0 warnings.
- **Dataset Split & Scale Protocol:** Explicitly marked as **`PROVISIONAL / TO_BE_VERIFIED`**. No unverified splits or artificial bulk data were generated.
- **Baseline Data Protection:** `data/DiFF/` remains 100% untouched and preserved.

---

## 5. Production Slurm Scripts Catalog

For execution on HPC clusters with Linux, full CUDA SDK, and high memory (for DiffFace/FSLSD once checkpoints are acquired):
1. [`scripts/slurm/validate_diffface.slurm`](file:///c:/Ổ đĩa D/MFVLR/scripts/slurm/validate_diffface.slurm) (24GB RAM, 1 GPU)
2. [`scripts/slurm/validate_fslsd.slurm`](file:///c:/Ổ đĩa D/MFVLR/scripts/slurm/validate_fslsd.slurm) (24GB RAM, 1 GPU, NCCL)
