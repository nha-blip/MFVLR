# MFVLR Reproduction: Phase 4 Completion Report

**Date:** 2026-09-11  
**Project:** MFVLR Reproduction (Paper: *MFVLR: Multi-domain Fine-grained Vision-Language Reconstruction for Generalizable Diffusion Face Forgery Detection and Localization*, arXiv:2605.10071v1)  
**Task:** Phase 4 (Vision Decoder, Appearance Decoder, Mask Decoder, Shared U-Net Decoder Trunk, Residual Generation, and VD Unit Tests)

---

## 1. Phase 4 Status

**STATUS: PASS**

All components of the Vision Decoder (VD) side, including the shared U-Net decoder trunk, Appearance Decoder (AD), Mask Decoder (MD), and residual generation ($I_r = |I_{\text{pre}} - I|$), have been implemented, verified, and unit-tested with 100% pass rate (34/34 tests passing across the complete test suite). No Phase 5 components (VIM, FLT, LE, LD, losses, or training loops) have been implemented.

---

## 2. Files Created

1. `models/vision/unet_decoder.py` – `UNetDecoderTrunk` and `UpsampleBlock` with 4-stage upsampling ($14 \to 28 \to 56 \to 112 \to 224$), optional multi-scale skip concatenation, and output channels 64.
2. `models/vision/appearance_decoder.py` – `AppearanceDecoder` and `AppearanceHead` producing $I_{\text{pre}} \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ with Sigmoid activation.
3. `models/vision/mask_decoder.py` – `MaskDecoder` and `MaskHead` producing raw logits $M_{\text{pre}} \in \mathbb{R}^{B \times 2 \times 224 \times 224}$.
4. `models/vision/vision_decoder.py` – `VisionDecoder` integrating the shared trunk with AD, MD, and residual generation ($I_r = |I_{\text{pre}} - I|$).
5. `tests/test_vision_decoder.py` – 6 unit tests verifying shapes, $I_r \ge 0$, true AD/MD parameter sharing, separate convolution heads, and gradient backpropagation.
6. `docs/phase4_report.md` – This report.

---

## 3. Files Modified

1. `models/vision/__init__.py` – Exported all new vision decoder classes (`UNetDecoderTrunk`, `UpsampleBlock`, `AppearanceDecoder`, `AppearanceHead`, `MaskDecoder`, `MaskHead`, `VisionDecoder`).
2. `models/__init__.py` – Exported Vision Decoder classes.
3. `docs/reproduction_notes.md` – Updated phase status to **PHASE 4 COMPLETE**, updated decision ledger with Phase 4 artifacts, and set Phase 5 stop condition.

---

## 4. Exact Decoder Architecture

```text
Local Feature I_loc: [B, 1024, 14, 14]  (+ Optional Skips from UNetEncoder)
        │
        ▼
[ Shared U-Net Decoder Trunk ]
  ├── Stage 1: Upsample 2x (14 -> 28) + Concat s3 [512, 28, 28] ──► ConvBlock (1024 -> 512)
  ├── Stage 2: Upsample 2x (28 -> 56) + Concat s2 [256, 56, 56] ──► ConvBlock (512 -> 256)
  ├── Stage 3: Upsample 2x (56 -> 112) + Concat s1 [128, 112, 112] ──► ConvBlock (256 -> 128)
  └── Stage 4: Upsample 2x (112 -> 224) + Concat s0 [64, 224, 224] ──► ConvBlock (128 -> 64)
        │
        ▼
Shared Decoded Features: [B, 64, 224, 224]
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
[ Appearance Head (AD) ]                  [ Mask Head (MD) ]
Conv2d(64 -> 3, kernel=3, pad=1)          Conv2d(64 -> 2, kernel=3, pad=1)
        │                                         │
     Sigmoid                                 (Raw Logits)
        │                                         │
        ▼                                         ▼
I_pre: [B, 3, 224, 224] in [0, 1]         M_pre: [B, 2, 224, 224]
```

---

## 5. Decoder Spatial and Channel Flow

| Stage | Input Shape | Upsample Operation | Skip Channels | Output Channels | Output Spatial Size |
|---|---|---|---|---|---|
| Bottleneck ($I_{\text{loc}}$) | `[B, 1024, 14, 14]` | None | N/A | 1024 | $14 \times 14$ |
| Up Stage 1 | `[B, 1024, 14, 14]` | Bilinear $2\times$ $\to 28 \times 28$ | $+512$ ($s_3$) | 512 | $28 \times 28$ |
| Up Stage 2 | `[B, 512, 28, 28]` | Bilinear $2\times$ $\to 56 \times 56$ | $+256$ ($s_2$) | 256 | $56 \times 56$ |
| Up Stage 3 | `[B, 256, 56, 56]` | Bilinear $2\times$ $\to 112 \times 112$ | $+128$ ($s_1$) | 128 | $112 \times 112$ |
| Up Stage 4 | `[B, 128, 112, 112]` | Bilinear $2\times$ $\to 224 \times 224$ | $+64$ ($s_0$) | 64 | $224 \times 224$ |
| AD Head | `[B, 64, 224, 224]` | Conv3x3 + Sigmoid | None | 3 | $224 \times 224$ |
| MD Head | `[B, 64, 224, 224]` | Conv3x3 (Raw logits) | None | 2 | $224 \times 224$ |

---

## 6. Skip Connection Implementation

- **Source:** Skip features are produced by `UNetEncoder` during image encoding:
  - $s_0 \in \mathbb{R}^{B \times 64 \times 224 \times 224}$
  - $s_1 \in \mathbb{R}^{B \times 128 \times 112 \times 112}$
  - $s_2 \in \mathbb{R}^{B \times 256 \times 56 \times 56}$
  - $s_3 \in \mathbb{R}^{B \times 512 \times 28 \times 28}$
- **Fusion:** In `UpsampleBlock`, the upsampled tensor is concatenated along the channel dimension with the corresponding skip feature: $\text{cat}([x_{\text{up}}, s], \text{dim}=1)$, then passed through double $3 \times 3$ convolutions with GroupNorm and GELU.
- **Fallback:** If skips are `None`, `UpsampleBlock` zero-pads the skip channel dimension, ensuring compatibility with inputs where skips are omitted.
- **Classification:** Documented explicitly as `ASSUMPTION_FROM_PAPER_GAP` since the paper does not specify skip topology.

---

## 7. Paper-Specified Decisions (`PAPER_SPECIFIED`)

1. **Decoder Inputs:** Local appearance forgery feature $I_{\text{loc}} \in \mathbb{R}^{B \times 1024 \times 14 \times 14}$.
2. **Appearance Output Shape:** $I_{\text{pre}} \in \mathbb{R}^{B \times 3 \times 224 \times 224}$.
3. **Mask Output Shape:** $M_{\text{pre}} \in \mathbb{R}^{B \times 2 \times 224 \times 224}$ ($f=2$ categories).
4. **Parameter Sharing:** The U-Net decoder in Mask Decoder (MD) adopts the **exact same network and weights** as the U-Net decoder in Appearance Decoder (AD) (Section III-C).
5. **Separate Heads:** AD has its own appearance reconstruction convolution head; MD has its own manipulation localization convolution head.
6. **Residual Formulation:**
   $$I_r = |I_{\text{pre}} - I| \in \mathbb{R}^{B \times 3 \times 224 \times 224}, \quad I_r \ge 0$$

---

## 8. Assumptions Introduced (`ASSUMPTION_FROM_PAPER_GAP`)

| Decision / Gap | Implementation Choice | Rationale |
|---|---|---|
| **Decoder Topology** | 4-stage upsampler: channels `[512, 256, 128, 64]`, GroupNorm (32 groups), GELU | Matches Phase 3 encoder channel schedule and outputs 64-channel $224 \times 224$ feature map. |
| **Skip Fusion Strategy** | Bilinear $2\times$ upsample + channel concatenation + double conv residual block | Standard U-Net decoding architecture. |
| **Appearance Activation** | `Sigmoid()` on AD head | Maps reconstructed RGB pixels to $[0, 1]$ matching the normalized input image range. |
| **Mask Head Output** | Raw 2-channel logits (no softmax or sigmoid) | Standard PyTorch formulation for subsequent `nn.CrossEntropyLoss` in Phase 8 without loss of numerical precision. |

---

## 9. Explanation and Proof of AD/MD True Weight Sharing

To strictly satisfy the paper requirement (*"The UNet decoder adopts the same network and weights as those in AD, to decrease the number of parameters"*, Section III-C):

1. `VisionDecoder` instantiates exactly **one** `UNetDecoderTrunk` instance (`self.decoder_trunk`).
2. `self.appearance_decoder = AppearanceDecoder(decoder_trunk=self.decoder_trunk, ...)`
3. `self.mask_decoder = MaskDecoder(decoder_trunk=self.decoder_trunk, ...)`
4. **Verification in `tests/test_vision_decoder.py`:**
   - `assert vd.ad.decoder_trunk is vd.md.decoder_trunk` (identical module instance).
   - `assert all(p1 is p2 for p1, p2 in zip(vd.ad.decoder_trunk.parameters(), vd.md.decoder_trunk.parameters()))` (identical parameter objects).
   - `assert vd.ad.head is not vd.md.head` (separate convolution heads).
   - Backpropagation through joint loss verified gradients properly accumulate into the single shared trunk.

---

## 10. Appearance Output Behavior

- **Tensor Name:** $I_{\text{pre}}$
- **Shape:** `[B, 3, 224, 224]`
- **Value Range:** $[0.0, 1.0]$ strictly enforced by Sigmoid activation.
- **Finite Check:** Verified in tests with zero NaN or Inf values.

---

## 11. Mask Output Behavior

- **Tensor Name:** $M_{\text{pre}}$
- **Shape:** `[B, 2, 224, 224]`
- **Channel 0:** Logits for class 0 (Real / Unmanipulated).
- **Channel 1:** Logits for class 1 (Fake / Manipulated).
- **Activation:** Raw unnormalized logits (no softmax inside model).
- **Finite Check:** Verified in tests with zero NaN or Inf values.

---

## 12. Residual Implementation

- **Formula:**
  $$I_r = \operatorname{abs}(I_{\text{pre}} - I) \in \mathbb{R}^{B \times 3 \times 224 \times 224}$$
- **Method:** Implemented via `VisionDecoder.compute_residual(i_pre, orig_image)` and evaluated automatically in `VisionDecoder.forward(i_loc, orig_image=...)`.
- **Properties:** Verified non-negative ($I_r \ge 0$), finite, and exact element-wise absolute difference.

---

## 13. Parameter Count Breakdown

```text
Vision Decoder Total Parameters: 13,586,885 (13.59M)
Vision Decoder Trainable Parameters: 13,586,885

Breakdown:
  ├── Shared U-Net Decoder Trunk: 13,584,000 (13.58M)
  │     ├── Up Stage 1 (1024+512 -> 512): 9,439,232
  │     ├── Up Stage 2 (512+256 -> 256): 2,949,632
  │     ├── Up Stage 3 (256+128 -> 128): 983,296
  │     └── Up Stage 4 (128+64 -> 64): 211,840
  ├── Appearance Reconstruction Head (AD): 1,731 (Conv 64 -> 3)
  └── Mask Localization Head (MD): 1,154 (Conv 64 -> 2)

Note: AD and MD share 100% of the 13,584,000 trunk parameters (0 duplicate parameters).
```

---

## 14. Commands Executed

```bash
# 1. Run Phase 4 Vision Decoder tests
python -m pytest tests/test_vision_decoder.py -v

# 2. Run complete test suite (Phase 2 + Phase 3 + Phase 4)
python -m pytest -v

# 3. Parameter count verification
python -c "from models.vision.vision_decoder import VisionDecoder; vd = VisionDecoder(); print(sum(p.numel() for p in vd.parameters()))"
```

---

## 15. Exact Test Results

```text
============================= test session starts =============================
platform win32 -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Paper
collected 34 items

tests/test_config.py::test_mfvlr_config_loading PASSED                   [  2%]
tests/test_config.py::test_dataset_config_loading PASSED                 [  5%]
tests/test_dummy_dataset.py::test_dummy_dataset_length PASSED            [  8%]
tests/test_dummy_dataset.py::test_dummy_dataset_item_structure PASSED    [ 11%]
tests/test_dummy_dataset.py::test_dummy_dataloader_batching PASSED       [ 14%]
tests/test_imports.py::test_import_utils PASSED                          [ 17%]
tests/test_imports.py::test_import_datasets PASSED                       [ 20%]
tests/test_imports.py::test_import_packages PASSED                       [ 23%]
tests/test_mask_generator.py::test_real_mask_all_zeros PASSED            [ 26%]
tests/test_mask_generator.py::test_efs_mask_all_ones PASSED              [ 29%]
tests/test_mask_generator.py::test_am_fs_mask_thresholding PASSED        [ 32%]
tests/test_mve.py::test_unet_encoder_output_shape PASSED                 [ 35%]
tests/test_mve.py::test_unet_encoder_skips PASSED                        [ 38%]
tests/test_mve.py::test_image_transformer_structure PASSED               [ 41%]
tests/test_mve.py::test_image_encoder_forward PASSED                     [ 44%]
tests/test_mve.py::test_mve_true_weight_sharing PASSED                   [ 47%]
tests/test_mve.py::test_mve_full_forward_and_fusion PASSED               [ 50%]
tests/test_mve.py::test_mve_gradient_backpropagation PASSED              [ 52%]
tests/test_prompt_generator.py::test_real_prompts PASSED                 [ 55%]
tests/test_prompt_generator.py::test_fake_ddpm_prompts PASSED            [ 58%]
tests/test_prompt_generator.py::test_fake_stylegan3_prompts PASSED       [ 61%]
tests/test_prompt_generator.py::test_fake_fslsd_prompts PASSED           [ 64%]
tests/test_prompt_generator.py::test_fake_diffae_prompts PASSED          [ 67%]
tests/test_utils.py::test_seed_determinism PASSED                        [ 70%]
tests/test_utils.py::test_classification_metrics PASSED                  [ 73%]
tests/test_utils.py::test_localization_metrics PASSED                    [ 76%]
tests/test_utils.py::test_checkpoint_save_and_load PASSED                [ 79%]
tests/test_utils.py::test_visualization_save PASSED                      [ 82%]
tests/test_vision_decoder.py::test_unet_decoder_trunk_output_shape PASSED [ 85%]
tests/test_vision_decoder.py::test_appearance_decoder_output PASSED      [ 88%]
tests/test_vision_decoder.py::test_mask_decoder_output PASSED            [ 91%]
tests/test_vision_decoder.py::test_vision_decoder_true_weight_sharing PASSED [ 94%]
tests/test_vision_decoder.py::test_vision_decoder_residual_generation PASSED [ 97%]
tests/test_vision_decoder.py::test_vision_decoder_backpropagation PASSED [100%]

============================= 34 passed in 11.10s =============================
```

---

## 16. Warnings, Errors, Skipped Tests, or Unresolved Issues

- **Errors during execution:** 0
- **Failures:** 0
- **Skipped tests:** 0
- **Warnings:** 0
- **Unresolved issues:** None.

---

## 17. Deviations from Reproduction Prompt

- **Deviations:** None.
- All structural, configuration, mathematical, and pipeline constraints outlined in `MFVLR_Codex_Reproduction_Prompt.md` for Phase 4 were strictly followed.

---

## 18. Unresolved Issues

None. All Phase 4 unit tests and regression tests pass with 100% success.

---

## 19. Ready for Phase 5 Confirmation

**YES.** The Vision Decoder (VD), shared U-Net Decoder Trunk, Appearance Decoder (AD), Mask Decoder (MD), and residual calculation ($I_r = |I_{\text{pre}} - I|$) are fully implemented, verified, and unit-tested.

The repository is completely ready to proceed to **PHASE 5** (Vision Injection Module VIM implementation per Eq. 4–11 and `tests/test_vim.py`).
