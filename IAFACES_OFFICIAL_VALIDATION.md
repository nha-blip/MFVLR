# IAFaces Official Validation Report

**Status:** `IAFaces = OFFICIAL_READY`  
**Date:** 2026-09-19  
**Target Generator:** IAFaces (Attribute Manipulation - AM)  
**Architecture Family:** GAN  
**Reproduced Environment:** `lattrans_env` (Conda, Python 3.8.20, PyTorch 1.13.1+cu117)

---

## 1. Hardware & Environment Audit

| Property | Value |
| :--- | :--- |
| **GPU Name** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **VRAM Total** | 6,438,780,928 bytes (6.00 GB / 6,141 MiB) |
| **System RAM Total** | 16,088 MB (15.71 GB) |
| **CUDA Driver** | 581.86 (CUDA 13.0) |
| **CUDA Runtime** | 11.7 (PyTorch `1.13.1+cu117`) |
| **PyTorch Version** | `1.13.1+cu117` |
| **Python Version** | `3.8.20` |
| **Host OS** | Windows 11 (MSVC 2022 v14.44, Ninja 1.11.1) |

---

## 2. Official Implementation Specifications

| Property | Value |
| :--- | :--- |
| **Official Repository** | `https://github.com/CMACH508/IA-FaceS.git` |
| **Git Commit** | `50e18fc091f03f5ce6bbd377b7f58cb2e93b4dd6` |
| **Paper** | *IA-FaceS: A Bidirectional Method for Semantic Face Editing* (Huang, Tu, and Xu, *Neural Networks* 2023) |
| **Local Clone Path** | `external/IA-FaceS` |
| **Model Classes** | `model.iafaces_256.Encoder`, `model.iafaces_256.Generator` |
| **Native Output Resolution** | $256 \times 256$ (3 channels RGB) |
| **Final Dataset Resolution** | $224 \times 224$ (strict MFVLR protocol) |
| **Manipulation Protocol** | Source face $\rightarrow$ Encoder $\rightarrow$ Component transfer (eyes, mouth) $\rightarrow$ CAM Generator |

---

## 3. Checkpoint Verification

Downloaded directly from the official Google Drive provided in the paper's repository:

| Checkpoint | Path / Official Source | File Size | SHA256 Checksum |
| :--- | :--- | :--- | :--- |
| **CelebA-HQ 256 Model (`iafaces-celebahq-256.pth`)** | `checkpoints/AM/IAFaces/iafaces-celebahq-256.pth`<br/>`https://drive.google.com/file/d/1tHXOpMn7AGUYVmgDU-8oVRhcYims9jjZ/` | `610,642,849` bytes (582 MB) | `b08552b6c1adcc372b665e005cf1a4fa3a50112ba747391060e268db41b6cab2` |

---

## 4. Compilation & Execution Resolution

The official IA-FaceS repository integrates rosinality's StyleGAN2 operations (`modules/op/fused_act.py` and `modules/op/upfirdn2d.py`), which compile C++/CUDA kernels on the fly via `torch.utils.cpp_extension.load`.

### Resolved Windows Build Issues:
1. **Windows 8.3 Short Path for MSVC/NVCC:**
   Workspace directory `C:\Ổ đĩa D\MFVLR` contains non-ASCII characters that corrupted MSVC preprocessor arguments. Resolved using Win32 `GetShortPathNameW` (`C:\IAD~1\MFVLR`).
2. **MSVC 14.44 & CUDA 11.7 Compatibility:**
   Passed `-allow-unsupported-compiler` and `-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` to NVCC and MSVC, successfully building `fused.pyd` and `upfirdn2d.pyd`.
3. **Windows `PosixPath` Unpickling:**
   Checkpoints saved on Linux serialized `pathlib.PosixPath`. Safely mapped to `pathlib.WindowsPath` during unpickling.

---

## 5. Local Official Runtime & Pilot Generation Metrics

Pilot generation of 10 samples (`iafaces_000000.png` – `iafaces_000009.png`) with batch size = 1:

| Metric | Measured Value |
| :--- | :--- |
| **Peak VRAM** | **374.83 MB** (0.37 GB) — well within 6.00 GB VRAM limit |
| **Peak System RAM** | **3,441.91 MB** (3.36 GB) |
| **Inference Time / Sample** | **0.141 s** (warm: ~0.024 s/image) |
| **Total Parameter Count** | **46,777,373** (Encoder: 15,363,840, Generator: 31,413,533) |
| **Generated Samples** | 10 real inference samples ($224 \times 224$ PNG) |
| **Paired Source Images** | 10 real source faces in `MFVLR_Dataset/source/AM/IAFaces/` |
| **Difference Masks** | 10 MFVLR-compliant binary masks: $|fake - corresponding\_source| \rightarrow \text{ITU-R BT.601} \rightarrow \text{threshold} > 0.1 \rightarrow \text{binary } \{0, 255\}$, $224 \times 224$ in `MFVLR_Dataset/masks/AM/IAFaces/` |
| **Visual Grid** | [`MFVLR_Dataset/logs/verification_samples/iafaces_visual_grid.png`](file:///c:/Ổ đĩa D/MFVLR/MFVLR_Dataset/logs/verification_samples/iafaces_visual_grid.png) |
| **Dataset Verification** | 115 samples in `MFVLR_Dataset/metadata/all.csv` verified: **0 errors, 0 warnings (PASS)** |

---

## 6. Quality Gate Checklist (A–J)

- [x] **A. Official repository?** `https://github.com/CMACH508/IA-FaceS` (commit `50e18fc091f03f5ce6bbd377b7f58cb2e93b4dd6`).
- [x] **B. Official architecture?** `model.iafaces_256.Encoder` + `model.iafaces_256.Generator` with CAM module.
- [x] **C. Official checkpoint?** `iafaces-celebahq-256.pth` (SHA256: `b08552b6...`).
- [x] **D. SHA256 recorded?** `b08552b6c1adcc372b665e005cf1a4fa3a50112ba747391060e268db41b6cab2`.
- [x] **E. Real inference?** Yes, executed on NVIDIA RTX 4050 Laptop GPU (PyTorch 1.13.1+cu117).
- [x] **F. 5–10 samples?** 10 pilot samples generated (`iafaces_000000.png` – `iafaces_000009.png`).
- [x] **G. Correct source/target pairing?** Yes, 10 paired source faces in `MFVLR_Dataset/source/AM/IAFaces/`.
- [x] **H. Correct mask?** Audited and harmonized per MFVLR reproduction protocol: computed strictly as $|fake - source| \rightarrow \text{RGB-to-grayscale (ITU-R BT.601)} \rightarrow /255.0 \rightarrow \text{threshold} > 0.1 \rightarrow \text{binary } \{0, 255\}$ at $224 \times 224$. Difference pair bound to corresponding source face, no morphology (see [MASK_PROTOCOL_AUDIT.md](file:///c:/Ổ%20đĩa%20D/MFVLR/MASK_PROTOCOL_AUDIT.md)).
- [x] **I. Correct architecture label?** `GAN`.
- [x] **J. Correct L1–L4 prompts?**
  - L1: `A photo of a fake face`
  - L2: `A photo of an attribute-manipulated face`
  - L3: `A photo generated by the gan-based model`
  - L4: `The source generative model of this photo is IAFaces`
