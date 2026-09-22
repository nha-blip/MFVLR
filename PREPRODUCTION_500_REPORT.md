# MFVLR Pre-Production Dataset Master Report (500 Samples / Generator)

> **Execution Date:** September 20, 2026  
> **Status:** **PASS / OFFICIAL_READY**  
> **Total Valid Pre-Production Samples:** **3,880** (8 Generators across EFS, AM, FS)  
> **Pilot Dataset Status:** **115 Pilot Samples in `MFVLR_Dataset/` 100% Intact & Untouched**  
> **Next Step Recommendation:** **STOP & Review** (No automated scaling to 5k; DiffFace and FSLSD remain blocked).

---

## 1. Executive Summary & Benchmark Table

A pre-production dataset generation campaign was conducted to validate reproducibility, metadata compliance, strict MFVLR mask protocols, and computational feasibility across all 8 verified `OFFICIAL_READY` generators prior to large-scale generation.

* **Target:** Up to 500 valid samples per generator.
* **Result:** **3,880 / 3,880 samples generated and verified (100.0% Success Rate).**
* **Verification Outcome:** 0 errors, 0 warnings, 0 duplicate SHA256 hashes, 0 blank/invalid masks.

### Benchmark & Resource Utilization

| STT | Category | Generator | Architecture | Requested | Generated | Valid | Invalid | Dup | Avg Time / Sample | Peak VRAM | Total Disk | Mask Area (Mean±Std [Min, Max]) | Quality Gate |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **EFS** | **StyleGAN3** | GAN | 500 | 500 | 500 | 0 | 0 | 0.2856s | 819.00 MB | 39.56 MB | 1.000±0.000 [1.000, 1.000] | **PASS** |
| 2 | **AM** | **IAFaces** | GAN | 500 | 500 | 500 | 0 | 0 | 0.1395s | 375.05 MB | 34.71 MB | 0.110±0.055 [0.021, 0.226] | **PASS** |
| 3 | **AM** | **LatTrans** | GAN | 500 | 500 | 500 | 0 | 0 | 0.1978s | 2,273.19 MB | 34.01 MB | 0.307±0.105 [0.091, 0.603] | **PASS** |
| 4 | **FS** | **FaceSwapper** | GAN | 380 | 380 | 380 | 0 | 0 | 0.3429s | 514.41 MB | 23.50 MB | 0.076±0.034 [0.025, 0.192] | **PASS** |
| 5 | **EFS** | **DDPM** | Diffusion | 500 | 500 | 500 | 0 | 0 | 6.0964s | 803.04 MB | 35.34 MB | 1.000±0.000 [1.000, 1.000] | **PASS** |
| 6 | **EFS** | **LatDiff** | Diffusion | 500 | 500 | 500 | 0 | 0 | 4.1332s | 2,721.41 MB | 39.56 MB | 1.000±0.000 [1.000, 1.000] | **PASS** |
| 7 | **EFS** | **CollDiff** | Diffusion | 500 | 500 | 500 | 0 | 0 | 13.9603s | 4,471.27 MB | 35.30 MB | 1.000±0.000 [1.000, 1.000] | **PASS** |
| 8 | **AM** | **DiffAE** | Diffusion | 500 | 500 | 500 | 0 | 0 | 8.7573s | 1,645.91 MB | 62.79 MB | 0.894±0.050 [0.680, 0.962] | **PASS** |
| **TỔNG** | | | | **3,880** | **3,880** | **3,880** | **0** | **0** | **4.27s (TB)** | **4,471 MB (Max)** | **462.60 MB** | | **ALL PASS** |

---

## 2. 14-Point Comprehensive Audit Breakdown

### Audit Point 1: Environment & Toolchain Isolation
All generation tasks were executed in isolated, purpose-built Conda environments to prevent dependency conflicts (e.g., PyTorch 1.7 vs 1.12 vs 2.x, CUDA extensions):
* `sg3_env`: Python 3.9, PyTorch 1.11.0+cu113, Ninja, MSVC CL 19.44 (C++ custom ops compiled).
* `iafaces_env`: Python 3.8, PyTorch 1.7.1+cu110.
* `lattrans_env`: Python 3.8, PyTorch 1.7.1+cu110, Ninja, MSVC CL 19.44 (fused ops compiled).
* `faceswapper_env`: Python 3.8, PyTorch 1.8.1+cu111, InsightFace.
* `ddpm_env`: Python 3.8, PyTorch 1.7.1+cu110.
* `latdiff_env`: Python 3.8, PyTorch 1.12.1+cu113, PyTorch Lightning.
* `colldiff_env`: Python 3.8, PyTorch 1.12.1+cu113, PyTorch Lightning, VQ-f4 autoencoder.
* `diffae_env`: Python 3.10, PyTorch 2.6.0+cu124, PyTorch Lightning.

### Audit Point 2: Git Commits & Repository Provenance
Each generator code base is cloned directly from its official public research repository in `external/`:
* `external/stylegan3`: NVlabs/stylegan3 (`origin/main`)
* `external/iafaces`: IAFaces official repo (`origin/main`)
* `external/lattrans`: Latent-Translation official repo (`origin/main`)
* `external/faceswapper`: FaceSwapper official repo (`origin/main`)
* `external/ddpm`: openai/improved-diffusion (`origin/main`)
* `external/latdiff`: CompVis/latent-diffusion (`origin/main`)
* `external/colldiff`: CollaborativeDiffusion official repo (`origin/main`)
* `external/diffae`: phog/diffae official repo (`origin/main`)

### Audit Point 3: Official Checkpoint Integrity & Verification
Checkpoints were validated against official releases and their SHA256 digests recorded:
* **StyleGAN3:** `stylegan3-r-ffhq-1024x1024.pkl` (SHA256: `a5879a7384ff82173ea51f930e386928e44f80077c449c394fcc7ae9cf71bc8f`)
* **IAFaces:** `CelebA-HQ_Smiling.pt` (SHA256: `3b25f82bbcb902a7aa0bbdf72cbceb7405e3f28cf69dbd1cebaae8eb82f14399`)
* **LatTrans:** `stylegan2_ffhq1024.pth` (SHA256: `0c74b8823fdfab7d0ec36f2f35f29910d6e6a17b0cb8f7d98305f8841a54b397`)
* **FaceSwapper:** `faceswapper_model.pth` (SHA256: `951aeaf8242a3cf134c4fcba13c9e378385bbd195972824ff433eb1f9429188d`)
* **DDPM:** `ffhq_1000steps.pt` (SHA256: `ec5715ef5eb4f4cb48e8e7a0ad3a9c7b94ad3161c6b16259074cb821cf3e68bc`)
* **LatDiff:** `ffhq_ldm_kl_f4.ckpt` (SHA256: `4224fbe29fa9be7e6f83ec55ee988cf2332616f73587e974e370a256df2e6dc3`)
* **CollDiff:** `colldiff_ffhq.ckpt` (SHA256: `3a4f89d38c64119d67b2d56e7e59c00ab2f5a6b0c2049e49a8fe10ca1c15f9b4`)
* **DiffAE:** `ffhq1024_diffae.ckpt` + `celeba_cls.ckpt` (Attribute 31: `Smiling`)

### Audit Point 4: Source Datasets & Pristine Real Alignments
* **EFS (DDPM, LatDiff, CollDiff, StyleGAN3):** Entirely synthesized faces unconditionally generated from Gaussian noise $\mathcal{N}(0, I)$ with explicit seeds.
* **AM (LatTrans, DiffAE, IAFaces):** Real source images from FFHQ / CelebA-HQ ($224 \times 224$ synchronized alignments), modified along semantic attribute directions (Smiling, Age, Pose).
* **FS (FaceSwapper):** 20 high-resolution pristine FaceForensics++ source video frames swapped pairwise ($20 \times 19 = 380$ pairs), preserving exact source identity transfer and target background/pose.

### Audit Point 5: Generation Configurations & Reproducibility
* Deterministic seeds were specified and logged for every single sample in `generation_config` JSON fields.
* Sampling parameters adhered to official publications:
  * DDPM: 1,000 diffusion timesteps, linear beta schedule.
  * LatDiff: DDIM 50 steps, $\eta=0.0$, latent downsampling factor $f=4$.
  * CollDiff: Dynamic classifier-free guidance, DDIM 50 steps.
  * DiffAE: DDIM stochastic inversion $T=250$, DDIM generative render $T=100$, linear semantic shift $\alpha \in [-2.5, +2.5]$.
  * StyleGAN3: Truncation $\psi=0.7$, synthesis network at native $1024 \times 1024$.

### Audit Point 6: Strict MFVLR Mask Protocol Audit
* All masks strictly comply with the official MFVLR formulation:
  $$\text{diff} = |I_{fake} - I_{reference}|$$
  $$\text{gray} = 0.299 \cdot \text{diff}_R + 0.587 \cdot \text{diff}_G + 0.114 \cdot \text{diff}_B$$
  $$\text{mask} = (\text{gray} / 255.0 > 0.1) \times 255$$
* **Zero morphology operations:** No dilation, no erosion, no Gaussian blur, no threshold 15 heuristic.
* Format: $224 \times 224$ uint8 lossless PNG, values strictly $\in \{0, 255\}$.
* **EFS:** Entirely synthesized faces have all-255 masks (ratio = 1.000).
* **AM:** Attribute-manipulated faces show focused, realistic altered regions:
  * DiffAE: $0.894 \pm 0.050$
  * LatTrans: $0.307 \pm 0.105$
  * IAFaces: $0.110 \pm 0.055$
* **FS:** FaceSwapper shows precise facial area replacement ($0.076 \pm 0.034$), matching inner-face swapping boundaries.

### Audit Point 7: Metadata Conformance (26 Fields)
The combined pre-production CSV ([`all_preproduction.csv`](file:///c:/Ổ%20đĩa%20D/MFVLR/GenFace_Reproduced/preproduction_500/metadata/all_preproduction.csv)) strictly includes all 26 mandatory fields:
`sample_id`, `image_path`, `source_path`, `target_path`, `mask_path`, `label`, `forgery_type`, `architecture`, `generator`, `split`, `L1`, `L2`, `L3`, `L4`, `original_dataset`, `generator_repo`, `generator_commit`, `checkpoint`, `checkpoint_sha256`, `seed`, `source_id`, `target_id`, `generation_config`, `native_resolution`, `sha256`, `notes`.
* Zero null or missing values across all required keys.

### Audit Point 8: Duplicate Image Audit
* All 3,880 generated images were hashed via SHA256.
* **Result:** **0 duplicate hashes found.** Every sample represents a distinct facial image.

### Audit Point 9: Leakage & Split Isolation
* In accordance with pre-production specifications, `split` is explicitly set to **`UNASSIGNED`** across all 3,880 samples to prevent premature train/val/test data leakage.
* Source faces used for AM/FS are fully recorded by ID to ensure identity-disjoint splitting at production scale.

### Audit Point 10: Pilot Dataset Audit (115 Samples Intact)
* An automated audit of [`MFVLR_Dataset/metadata/all.csv`](file:///c:/Ổ%20đĩa%20D/MFVLR/MFVLR_Dataset/metadata/all.csv) and associated images/masks confirmed that:
  * All 15 real samples and 100 pilot fake samples remain **100% intact**.
  * File timestamps and checksums show **zero modifications or overwrites**.

### Audit Point 11: Visual Inspection Grids
Visual inspection mosaics (showing fake face, reference face, and binary difference mask) were generated and verified for each generator:
* [StyleGAN3 Grid](file:///c:/Ổ%20đĩa%20D/MFVLR/GenFace_Reproduced/preproduction_500/visual_grids/stylegan3_preprod_grid.png)
* [IAFaces Grid](file:///c:/Ổ%20đĩa%20D/MFVLR/GenFace_Reproduced/preproduction_500/visual_grids/iafaces_preprod_grid.png)
* [LatTrans Grid](file:///c:/Ổ%20đĩa%20D/MFVLR/GenFace_Reproduced/preproduction_500/visual_grids/lattrans_preprod_grid.png)
* [FaceSwapper Grid](file:///c:/Ổ%20đĩa%20D/MFVLR/GenFace_Reproduced/preproduction_500/visual_grids/faceswapper_preprod_grid.png)
* [DDPM Grid](file:///c:/Ổ%20đĩa%20D/MFVLR/GenFace_Reproduced/preproduction_500/visual_grids/ddpm_preprod_grid.png)
* [LatDiff Grid](file:///c:/Ổ%20đĩa%20D/MFVLR/GenFace_Reproduced/preproduction_500/visual_grids/latdiff_preprod_grid.png)
* [CollDiff Grid](file:///c:/Ổ%20đĩa%20D/MFVLR/GenFace_Reproduced/preproduction_500/visual_grids/colldiff_preprod_grid.png)
* [DiffAE Grid](file:///c:/Ổ%20đĩa%20D/MFVLR/GenFace_Reproduced/preproduction_500/visual_grids/diffae_preprod_grid.png)

### Audit Point 12: Errors, Warnings & Mitigations
* *PyTorch 2.6 Checkpoint Loading:* Resolved PyTorch Lightning weights-only restriction via `torch.load(..., weights_only=False)`.
* *DiffAE Latent Pre-Inversion:* Pre-caching latent representations `(z_sem, x_T)` for real faces reduced sample time from 53.7s to 8.76s (~6x speedup), eliminating redundant inversion.
* *MSVC C++ Compilation:* Verified and locked MSVC CL 19.44 toolchain for native CUDA extensions (StyleGAN3, LatTrans).

### Audit Point 13: Total Execution Time & Storage Footprint
* **Total Cumulative Generation Time:** ~4.6 hours across all 8 generators.
* **Total Pre-Production Storage:** **462.60 MB** (3,880 images + 3,880 masks + source/target + metadata + visual grids).

### Audit Point 14: Scale-to-5k Feasibility & Resource Projection
Based on measured pre-production benchmarks, scaling to 5,000 samples per generator ($8 \times 5,000 = 40,000$ samples):

| Generator | Measured Time / Sample | Projected Time (5k Samples) | Projected Disk (5k Samples) | Peak VRAM | Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **StyleGAN3** | 0.2856s | ~23.8 minutes | ~395 MB | 819 MB | **High** |
| **IAFaces** | 0.1395s | ~11.6 minutes | ~347 MB | 375 MB | **High** |
| **LatTrans** | 0.1978s | ~16.5 minutes | ~340 MB | 2,273 MB | **High** |
| **FaceSwapper** | 0.3429s | ~28.6 minutes | ~310 MB | 514 MB | **High** (Requires ~265 FF++ source frames) |
| **DDPM** | 6.0964s | ~8.47 hours | ~353 MB | 803 MB | **Medium** |
| **LatDiff** | 4.1332s | ~5.74 hours | ~395 MB | 2,721 MB | **Medium** |
| **CollDiff** | 13.9603s | ~19.39 hours | ~353 MB | 4,471 MB | **Heavy** (Recommend batching / overnight) |
| **DiffAE** | 8.7573s | ~12.16 hours | ~628 MB | 1,646 MB | **Heavy** (With pre-inversion caching) |
| **TỔNG** | | **~46.9 hours** | **~3.12 GB** | | |

---

## 3. Blocked Generators Status
* **DiffFace (FS - Diffusion):** Remains **BLOCKED / MANUAL_DOWNLOAD_REQUIRED**. Checkpoints require manual acquisition from author repository.
* **FSLSD (FS - Diffusion):** Remains **BLOCKED / MANUAL_DOWNLOAD_REQUIRED**. Official checkpoint links require manual retrieval.
* *Protocol:* Zero surrogate, mock, or synthetic replacements were used.

---

## 4. Final Verdict & Stop Instruction

> [!IMPORTANT]
> **Pre-Production Quality Gate H: 100% PASS.**  
> All 3,880 samples have been verified with 0 errors, 0 warnings, and complete compliance with MFVLR provenance, mask protocols, and metadata standards.  
> As required by the project execution protocol, execution is now **STOPPED**. No scale-to-5k generation, no training of MFVLR, and no unblocking of DiffFace/FSLSD will occur without explicit user instructions.
