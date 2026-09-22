# Mask Protocol Audit Report: MFVLR / GenFace Reproduction

**Date:** 2026-09-19  
**Scope:** Strict Audit and Harmonization of Localization Masks for AM (Attribute Manipulation) and FS (Face Swapping) Generators.  
**Audited Generators:** DiffAE, LatTrans, IAFaces, FaceSwapper.

---

## 1. MFVLR Mask Reproduction Protocol Definition

Per the official MFVLR / GenFace reproduction specification (`generate_masks.py`):

$$\text{diff} = |\text{fake} - \text{corresponding source}|$$

1. **Pixel-wise Subtraction:** Absolute difference between the manipulated face image and the corresponding source face image.
2. **RGB-to-Grayscale Conversion:** ITU-R BT.601 luminance weights:
   $$\text{gray} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
3. **Normalization:**
   $$\text{norm\_gray} = \frac{\text{gray}}{255.0} \in [0.0, 1.0]$$
4. **Binarization:**
   $$\text{mask} = \begin{cases} 255 & \text{if } \text{norm\_gray} > 0.1 \\ 0 & \text{otherwise} \end{cases}$$
   *(Note: Threshold $0.1$ corresponds to $25.5$ on a $[0, 255]$ scale).*
5. **Morphological Operations:** **STRICTLY NONE** (no dilation, no erosion, no Gaussian blur, no manual boundary expansion).
6. **Resolution & Alignment:** Strictly synchronized at $224 \times 224$ pixels, lossless PNG format (`uint8` $\{0, 255\}$).

---

## 2. Generator Mask Audit & Harmonization Table

| Generator | Category | Difference Pair | Threshold | Morphology | Resolution | Previous Status | Corrected Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DiffAE** | AM | `fake - source` | $> 0.1$ (normalized) | None | $224 \times 224$ | **PASS** | **PASS** (100% compliant) |
| **LatTrans** | AM | `fake - source` *(was fake - recon)* | $> 0.1$ *(was unweighted > 15)* | None *(was dilation iter=3)* | $224 \times 224$ | **NON-COMPLIANT** | **PASS** (Regenerated) |
| **IAFaces** | AM | `fake - source` *(was fake - src_recon)* | $> 0.1$ *(was unweighted > 15)* | None | $224 \times 224$ | **NON-COMPLIANT** | **PASS** (Regenerated) |
| **FaceSwapper** | FS | `fake - target_context` | $> 0.1$ *(was unweighted > 0.1)* | None | $224 \times 224$ | **MINOR DIVERGENCE** | **PASS** (Regenerated) |

---

## 3. Analysis: `fake - source` vs. `fake - reconstruction`

A critical distinction must be maintained between the **original source image** and an intermediate **generator reconstruction**:

### Why Did Earlier Scripts Use `reconstruction`?
In GAN-based attribute manipulation (such as LatTrans using pSp / StyleGAN2, or IAFaces using its component encoder):
- The model first inverts/encodes an input image $x$ into latent space $w = \text{Enc}(x)$ or node features.
- The generator can decode this unedited latent representation back to an image: $x_{\text{recon}} = \text{Dec}(w)$.
- Because GAN inversion is lossy, $\|x - x_{\text{recon}}\|_2 > 0$ (residual inversion errors in hair, background, or micro-textures).
- Subtracting `fake - recon` captures purely the synthetic delta introduced by the latent transformation ($\Delta w$), isolating the edited component from the generator's inversion error.

### Why MFVLR Protocol Demands `fake - source`:
- In real-world face forgery detection, the baseline truth is the **actual pristine source image** ($x$), not the internal autoencoder reconstruction ($x_{\text{recon}}$).
- The MFVLR localization task assesses whether a detector can identify all regions altered with respect to the authentic person's original photo.
- Under the MFVLR reproduction protocol, the ground truth mask must always be derived from the **corresponding source image** stored in `MFVLR_Dataset/source/`.
- Therefore, treating $x_{\text{recon}}$ as equivalent to $x$ violates protocol.

---

## 4. Specific Corrections Applied

### A. LatTrans (AM)
- **Previous Implementation:**
  - Subtraction: $|\text{fake} - \text{recon}|$
  - Threshold: Unweighted RGB mean $> 15.0$
  - Morphology: `scipy.ndimage.binary_dilation(iterations=3)`
- **Issues:** Violates MFVLR protocol on three counts (wrong difference pair, non-standard threshold, unauthorized morphological dilation). Quality Gate H was marked PENDING/FAIL.
- **Correction:**
  - Preserved the official generated fake images (`lattrans_000000.png` – `lattrans_000009.png`).
  - Recomputed all 10 masks using `compute_difference_mask(fake, source, threshold=0.1)`.
  - Resulting masks are localized binary $\{0, 255\}$ masks with 11.7% – 43.0% active pixels focused on facial attribute modifications.
  - Quality Gate H: **PASS**.

### B. IAFaces (AM)
- **Previous Implementation:**
  - Subtraction: $|\text{fake} - \text{src\_recon}|$
  - Threshold: Unweighted RGB mean $> 15.0$
  - Morphology: None
- **Issues:** Used intermediate reconstruction instead of the corresponding source face image; used unweighted RGB mean threshold.
- **Correction:**
  - Preserved the official generated fake images (`iafaces_000000.png` – `iafaces_000009.png`).
  - Recomputed all 10 masks using `compute_difference_mask(fake, source, threshold=0.1)`.
  - Resulting masks are localized binary $\{0, 255\}$ masks with 2.2% – 21.4% active pixels.
  - Quality Gate H: **PASS**.

### C. FaceSwapper (FS)
- **Previous Implementation:**
  - Subtraction: $|\text{fake} - \text{target\_context}|$ (where `target_context` is the target attribute image stored as source).
  - Threshold: Unweighted RGB mean $> 0.1$
  - Morphology: None
- **Issues:** Used unweighted mean $(R+G+B)/3$ rather than ITU-R BT.601 luminance weights $(0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B)$, causing minor discrepancies (~200–400 pixels).
- **Correction:**
  - Recomputed all 10 masks using exact `compute_difference_mask(fake, source, threshold=0.1)`.
  - Quality Gate H: **PASS**.

### D. DiffAE (AM)
- **Verification:** Already used `generate_masks.py` with exact formula `abs(fake - source)` $\rightarrow$ luminance grayscale $\rightarrow$ threshold $> 0.1$ $\rightarrow$ binary $\{0, 255\}$.
- **Compliance:** 10/10 samples matched protocol with 0 mismatched pixels.
- **Status:** **PASS** (no regeneration required).

---

## 5. Verification Results

Following mask regeneration:
1. **Dataset Verification Tool (`verify_dataset.py`):**
   - Verified all 115 samples in `MFVLR_Dataset/metadata/all.csv`.
   - **Status:** **`PASS`**
   - **Errors:** 0
   - **Warnings:** 0
2. **Visual Grids Updated:**
   - LatTrans: [`MFVLR_Dataset/logs/verification_samples/lattrans_visual_grid.png`](file:///c:/Ổ đĩa D/MFVLR/MFVLR_Dataset/logs/verification_samples/lattrans_visual_grid.png)
   - IAFaces: [`MFVLR_Dataset/logs/verification_samples/iafaces_visual_grid.png`](file:///c:/Ổ đĩa D/MFVLR/MFVLR_Dataset/logs/verification_samples/iafaces_visual_grid.png)
   - FaceSwapper: [`MFVLR_Dataset/logs/verification_samples/faceswapper_visual_grid.png`](file:///c:/Ổ đĩa D/MFVLR/MFVLR_Dataset/logs/verification_samples/faceswapper_visual_grid.png)
