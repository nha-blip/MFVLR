# FaceSwapper Official Validation Report

**Status:** `FaceSwapper = OFFICIAL_READY`  
**Date:** 2026-09-19  
**Target Generator:** FaceSwapper (Face Swapping - FS)  
**Architecture Family:** GAN  
**Reproduced Environment:** `latdiff_env` (Conda, Python 3.8.20, PyTorch 1.13.1+cu117)

---

## 1. Official Implementation Specifications

| Property | Value |
| :--- | :--- |
| **Official Repository** | `https://github.com/liqi-casia/FaceSwapper.git` |
| **Git Commit** | `69f53e5e493214736f33cf224c084f74d081e7d2` |
| **Paper** | *FaceSwapper: Learning Disentangled Representation for One-shot Progressive Face Swapping* (Li et al., *IEEE Transactions on Pattern Analysis and Machine Intelligence* 2024) |
| **Local Clone Path** | `external/FaceSwapper` |
| **Model Classes** | `core.model.Generator` (composed of `IdentityEncoder`, `AttrEncoder`, `Decoder` with `AdaIN`), `core.face_model.Backbone` (ArcFace IR-SE50), `core.wing.FAN` (Facial Alignment Network) |
| **Native Output Resolution** | $256 \times 256$ (3 channels RGB) |
| **Final Dataset Resolution** | $224 \times 224$ (bilinear / area resize) |
| **Face Swapping Protocol** | Source face (identity) + Target face (attribute/context) $\rightarrow$ Disentangled representation & AdaIN decoding $\rightarrow$ Post-processed swapped face |

---

## 2. Checkpoint Verification

All three required checkpoints were downloaded directly from the official Google Drive provided in the paper's repository:

| Checkpoint | Path / Official Source | File Size | SHA256 Checksum |
| :--- | :--- | :--- | :--- |
| **Face Swapper (`faceswapper.ckpt`)** | `external/FaceSwapper/pretrained_checkpoints/faceswapper.ckpt`<br/>`https://drive.google.com/file/d/1Tb3V09wbaGe6SaiN3BZkOcCy7VJ0KYC8/` | `321,299,697` bytes (306 MB) | `74d781083057091ab8e74b9e86ce2c84a5b66c72dd1f218290e0aeaa0d87b4ee` |
| **Face Recognition (`model_ir_se50.pth`)** | `external/FaceSwapper/pretrained_checkpoints/model_ir_se50.pth`<br/>`https://drive.google.com/file/d/1-lxc-jZGIFNdwFUXQ9tDS9OSuhadj6AC/` | `175,004,821` bytes (175 MB) | `a035c768259b98ab1ce0e646312f48b9e1e218197a0f80ac6765e88f8b6ddf28` |
| **Face Alignment (`wing.ckpt`)** | `external/FaceSwapper/pretrained_checkpoints/wing.ckpt`<br/>`https://drive.google.com/file/d/1lBt4x4P5qaClB2ZN_POBV-ue41hdlaoJ/` | `203,791,241` bytes (194 MB) | `bbfd137307a4c7debd5c283b9b0ce539466cee417ac0a155e184d857f9f2899c` |

---

## 3. Real Pilot Generation Evidence (10 Samples, Pairs 0–9)

Executed via `run_faceswapper_pilot.py` using official FaceSwapper generator + post-processing:

### Runtime Metrics per Sample

| Sample ID | Source ID | Target Context | Native Res | Dataset Res | Inference Time (s) | Peak VRAM (MB) | RAM (MB) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `faceswapper_000000` | raw_003_0 | raw_000_0 | 256x256 | 224x224 | 1.49s | 684.4 MB | 2794.2 MB | PASS |
| `faceswapper_000001` | raw_003_1 | raw_000_1 | 256x256 | 224x224 | 0.24s | 688.7 MB | 2799.4 MB | PASS |
| `faceswapper_000002` | raw_003_2 | raw_000_2 | 256x256 | 224x224 | 0.21s | 688.7 MB | 2800.7 MB | PASS |
| `faceswapper_000003` | raw_003_3 | raw_000_3 | 256x256 | 224x224 | 0.21s | 688.7 MB | 2802.2 MB | PASS |
| `faceswapper_000004` | raw_003_4 | raw_000_4 | 256x256 | 224x224 | 0.22s | 688.7 MB | 2802.4 MB | PASS |
| `faceswapper_000005` | raw_003_5 | raw_000_5 | 256x256 | 224x224 | 0.22s | 688.7 MB | 2803.5 MB | PASS |
| `faceswapper_000006` | raw_003_6 | raw_000_6 | 256x256 | 224x224 | 0.21s | 688.7 MB | 2803.5 MB | PASS |
| `faceswapper_000007` | raw_003_7 | raw_000_7 | 256x256 | 224x224 | 0.22s | 688.7 MB | 2803.6 MB | PASS |
| `faceswapper_000008` | raw_003_8 | raw_000_8 | 256x256 | 224x224 | 0.20s | 688.7 MB | 2804.9 MB | PASS |
| `faceswapper_000009` | raw_003_9 | raw_000_9 | 256x256 | 224x224 | 0.20s | 688.7 MB | 2805.6 MB | PASS |

---

## 4. Quality Gate Checklist (A–J)

- [x] **A. Official repository?** `https://github.com/liqi-casia/FaceSwapper` (commit `69f53e5e493214736f33cf224c084f74d081e7d2`).
- [x] **B. Official architecture?** Disentangled representation face swapper (`IdentityEncoder`, `AttrEncoder`, `Decoder` with `AdaIN`, `FAN`, `ArcFace`).
- [x] **C. Official checkpoint?** `faceswapper.ckpt` (306 MB), `model_ir_se50.pth` (175 MB), `wing.ckpt` (194 MB).
- [x] **D. SHA256 recorded?** Primary checkpoint SHA256 recorded (`74d781083057091ab8e74b9e86ce2c84a5b66c72dd1f218290e0aeaa0d87b4ee`).
- [x] **E. Real inference?** Yes, executed on NVIDIA GPU via `run_faceswapper_pilot.py`.
- [x] **F. 5–10 samples?** Exactly 10 samples generated.
- [x] **G. Correct source/target pairing?** Yes, source identity paired with target attribute context from official FF++ test set.
- [x] **H. Correct mask?** Audited and harmonized per MFVLR reproduction protocol: computed strictly as $|fake - target| \rightarrow \text{RGB-to-grayscale (ITU-R BT.601)} \rightarrow /255.0 \rightarrow \text{threshold} > 0.1 \rightarrow \text{binary } \{0, 255\}$ at $224 \times 224$ without morphology (see [MASK_PROTOCOL_AUDIT.md](file:///c:/Ổ%20đĩa%20D/MFVLR/MASK_PROTOCOL_AUDIT.md)).
- [x] **I. Correct architecture label?** `GAN`.
- [x] **J. Correct L1–L4 prompts?**
  - L1: `A photo of a fake face`
  - L2: `A photo of a face-swapped face`
  - L3: `A photo generated by the gan-based model`
  - L4: `The source generative model of this photo is FaceSwapper`

---

## 5. Visual Verification
- Visual grid generated at: `MFVLR_Dataset/logs/verification_samples/faceswapper_visual_grid.png` (`SOURCE_ID | TARGET_ATTR | FAKE_SWAP | MASK`).
- Dataset verification (`verify_dataset.py`): **`PASS`**, 85 total samples, 0 errors, 0 warnings.
