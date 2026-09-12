# Phase 12 Final Report: Comprehensive Paper-Faithfulness & Reproducibility Audit

**Status:** A. PAPER-FAITHFUL IMPLEMENTATION READY FOR EXPERIMENTAL REPRODUCTION  
**Date:** 2026-09-12  
**Authoritative Source:** *MFVLR: Multi-domain Fine-grained Vision-Language Reconstruction for Generalizable Diffusion Face Forgery Detection and Localization* (arXiv:2605.10071v1)  
**Test Suite Pass Rate:** **93 / 93 tests passed (100%) in 86.38s (0:01:26)**

---

## 1. Executive Summary

This final audit report concludes the multi-phase implementation of the MFVLR framework. Every mathematical equation (Eq. 1 through Eq. 27), architectural parameter sharing requirement, tensor dimension constraint, and optimization protocol specified in the paper has been verified through strict unit, integration, and end-to-end smoke tests.

All unspecified engineering details (e.g. tokenizer merge tables, exact template strings, manifest formats, VIM head count $r=8$, FFN hidden dimension $2048$, and total epoch count) have been explicitly isolated, classified as `ASSUMPTION_FROM_PAPER_GAP`, and documented in the paper traceability ledger. No pretrained weights have been secretly loaded, no ImageNet normalizations were introduced, and no paper benchmark numbers have been fabricated.

The codebase is classified as **Status A: Paper-faithful implementation ready for experimental reproduction**.

---

## 2. Sources Audited

1. `paper/2605.10071v1.pdf` — **Authoritative Source of Truth**
2. `docs/paper_spec.md` — Extracted formal specification
3. `docs/reproduction_notes.md` — Complete decision and architecture ledger
4. `MFVLR_Codex_Reproduction_Prompt.md` — Reproduction requirements and phase plan
5. `docs/phase1_report.md` through `docs/phase11_report.md` — Historical phase reports
6. All repository code files and 23 test suites

---

## 3. Files Inspected and Modified During Phase 12

### Files Inspected:
- All 15 model files in `models/` (vision, language, heads, losses, mfvlr.py)
- All 8 dataset files in `datasets/`
- All 6 utility files in `utils/`
- All 3 top-level CLI scripts (`train.py`, `evaluate.py`, `infer.py`)
- All 23 test suites in `tests/`

### Files Modified / Created in Phase 12:
- [`models/mfvlr.py`](file:///d:/Paper/models/mfvlr.py): Made `predict_fake_prob(fake_class_index=1)` accept configurable `fake_class_index` in `MFVLROutput` and `MFVLRInferenceOutput`.
- [`docs/paper_code_traceability.md`](file:///d:/Paper/docs/paper_code_traceability.md): Created comprehensive Eq. (1)-(27) paper-to-code traceability matrix with hybrid `PAPER_SPECIFIED + ASSUMPTION_FROM_PAPER_GAP` classifications.
- [`README.md`](file:///d:/Paper/README.md): Created exhaustive, transparent repository documentation and usage guide.
- [`docs/phase12_final_audit.md`](file:///d:/Paper/docs/phase12_final_audit.md): This comprehensive final audit report.

---

## 4. Equation-to-Code Traceability Summary

A complete line-by-line equation audit was conducted in [`docs/paper_code_traceability.md`](file:///d:/Paper/docs/paper_code_traceability.md). Key findings include:

| Component | Paper Equations | Implementation Modules | Verification Status |
| :--- | :--- | :--- | :--- |
| **Local Vision & Image Transformer** | Eq. (1)-(2) | `models/vision/unet_encoder.py`, `transformer.py`, `mve.py` | `VERIFIED` |
| **Language Positional & LE Depth** | Eq. (3) | `models/language/embeddings.py`, `language_encoder.py` | `VERIFIED` |
| **Vision Injection Module (VIM)** | Eq. (4)-(11) | `models/language/vim.py` | `VERIFIED` |
| **Language Decoder & Causal Cross-MHA** | Eq. (12)-(16) | `models/language/language_decoder.py` | `VERIFIED` |
| **Appearance Reconstruction Loss ($L_{\text{ar}}$)** | Eq. (17) | `models/losses/appearance_reconstruction_loss.py` | `VERIFIED` (Strictly MSE) |
| **Localization Loss ($L_{\text{fl}}$)** | Eq. (18) | `models/losses/localization_loss.py` | `VERIFIED` (2-class CE) |
| **KL Divergence Loss ($L_{\text{kl}}$)** | Eq. (19) | `models/losses/kl_loss.py` | `VERIFIED` ($\tau=0.5, P(T_l) \to Q(T_{\text{lpre}})$) |
| **Cross-Modal Contrastive Loss ($L_{\text{cmc}}$)** | Eq. (20)-(23) | `models/losses/cmc_loss.py` | `VERIFIED` (Unnormalized dot product, trainable $\tau$) |
| **Language Reconstruction Loss ($L_{\text{lr}}$)** | Eq. (24)-(25) | `models/losses/language_reconstruction_loss.py` | `VERIFIED` ($s=49408, n=308$) |
| **Forgery Detection Loss ($L_{\text{fd}}$)** | Eq. (26) | `models/losses/detection_loss.py` | `VERIFIED` (2-class CE) |
| **Multi-Task Total Loss ($L$)** | Eq. (27) | `models/losses/total_loss.py` | `VERIFIED` (All $\lambda_k = 1.0$) |

---

## 5. Architectural Audits

### 5.1 Multi-domain Vision Encoder (MVE)
- Input: $I \in \mathbb{R}^{B \times 3 \times 224 \times 224}$.
- U-Net Encoder: Produces $I_{\text{loc}} \in \mathbb{R}^{B \times 1024 \times 14 \times 14}$.
- Image Transformer: $B=4$ Transformer blocks, sequence length $197$ ($14 \times 14 = 196$ spatial patches + 1 learnable CLS token), embedding dimension $d=512$.
- Global Feature: $I_g \in \mathbb{R}^{B \times 512}$ extracted from class token at index 0.
- Residual Feature: $I_{\text{rg}} \in \mathbb{R}^{B \times 512}$ extracted from RE applied to residual image $I_r$.
- Fusion: $I_v = I_g + I_{\text{rg}} \in \mathbb{R}^{B \times 512}$ via strict element-wise sum (no concatenation, no averaging).
- **Weight Sharing:** `model.mve.re is model.mve.ie` verified with identical parameter data pointers (`data_ptr()`).

### 5.2 Vision Decoder (VD)
- Shared Trunk: 4-stage residual upsampling U-Net decoder with skip connection integration.
- Appearance Head: Convolutional reconstruction mapping trunk features to $I_{\text{pre}} \in \mathbb{R}^{B \times 3 \times 224 \times 224}$.
- Mask Head: Convolutional localization mapping trunk features to $M_{\text{pre}} \in \mathbb{R}^{B \times 2 \times 224 \times 224}$.
- **Trunk Sharing:** `ad.decoder_trunk is md.decoder_trunk` verified with identical parameter data pointers.

### 5.3 Residual Domain
- Residual Calculation: $I_r = |I_{\text{pre}} - I| \in \mathbb{R}^{B \times 3 \times 224 \times 224}$.
- Verified: Strictly element-wise absolute difference; NO squared residual, NO hidden frequency transform, NO unpapered normalization.

### 5.4 Vision Injection Module (VIM)
- Implements Eq. (4)-(11) with $r=8$ attention heads (`ASSUMPTION_FROM_PAPER_GAP`).
- Input Query $Q$: Language features projected to $[B, r, n, d/r]$.
- Visual Key $K$ and Value $V$: Singleton visual token $I_v \in \mathbb{R}^{B \times 1 \times 512}$ projected to $[B, r, 1, d/r]$.
- Singleton Softmax Property: Dot-product attention matrix $A = \text{softmax}(Q K^T / \sqrt{d_k}) \in \mathbb{R}^{B \times r \times n \times 1}$ has key length 1; the softmax output along key dimension is identically $1.0$. This behavior is preserved without artificial alteration.

### 5.5 Language Encoder (LE)
- Vocabulary: $s = 49,408$, Token sequence length: $n = 308$, Embed dim: $d = 512$.
- Depth: $E = 12$ Language Encoder blocks.
- Block Architecture: Pre-LN MHA $\to$ VIM($I_v$) $\to$ Pre-LN FFN with residual connections.
- Output: $T_{\text{hig}}^e \in \mathbb{R}^{B \times 308 \times 512}$ (direct output of 12th block, no final LayerNorm).
- Global Language Token: $T_l = T_{\text{hig}}^e[:, -1, :] \in \mathbb{R}^{B \times 512}$ (strictly the LAST token).

### 5.6 Language Decoder (LD)
- Depth: $D = 7$ Language Decoder blocks.
- Shifted Input: Prepends learnable BOS token and drops the last token to maintain sequence length $n=308$.
- Block Architecture: Pre-LN Masked MHA (strict causal upper-triangular mask) $\to$ Pre-LN Cross-MHA (Query from decoder, Key/Value from complete $T_{\text{hig}}^e$) $\to$ Pre-LN FFN $\to$ VIM($I_v$).
- Vocabulary Projection: $T_{\text{pre}} = T_{\text{rec}}^d W_{\text{voc}}^T \in \mathbb{R}^{B \times 308 \times 49408}$.
- **Weight Tying:** `flt.decoder.token_embedding.weight is flt.encoder.embeddings.token_embed.weight` verified.

---

## 6. Multi-Task Losses Audit

1. **$L_{\text{fd}}$ (Eq. 26):** Cross-entropy on 2-class logits $y_{\text{pre}} \in \mathbb{R}^{B \times 2}$.
2. **$L_{\text{lr}}$ (Eq. 24-25):** Cross-entropy on vocabulary logits $T_{\text{pre}} \in \mathbb{R}^{B \times 308 \times 49408}$ against ground-truth tokens $T_{\text{tok}}$.
3. **$L_{\text{ar}}$ (Eq. 17):** Strictly Mean Squared Error (MSE) $L_{\text{ar}} = \frac{1}{HW} \sum \|I_{\text{pre}} - I\|_2^2$, NOT L1 loss.
4. **$L_{\text{fl}}$ (Eq. 18):** 2-class pixel-wise cross-entropy on $M_{\text{pre}} \in \mathbb{R}^{B \times 2 \times 224 \times 224}$ against binary mask target $M_{\text{gt}} \in \{0, 1\}^{224 \times 224}$.
5. **$L_{\text{kl}}$ (Eq. 19):** Kullback-Leibler divergence with softened probabilities ($\tau=0.5$) in the exact direction $D_{\text{KL}}(P(T_l) \parallel Q(T_{\text{lpre}}))$.
6. **$L_{\text{cmc}}$ (Eq. 20-23):** Cross-modal contrastive loss using unnormalized dot-product similarity, trainable temperature $\tau$ (initialized to $0.07$), and bidirectional symmetric average $\frac{1}{2}(L_{v2l} + L_{l2v})$.
7. **Total Loss (Eq. 27):** $\sum_{k} \lambda_k L_k$ with all $\lambda_k = 1.0$.

---

## 7. Ground-Truth Mask Generation Audit

Implements the exact paper procedure (Section III-E, Eq. 18):
- **Real Image:** Ground truth mask $M_{\text{gt}} = \mathbf{0}_{224 \times 224}$.
- **Entire Face Synthesis (EFS):** Ground truth mask $M_{\text{gt}} = \mathbf{1}_{224 \times 224}$.
- **Attribute Manipulation (AM) / Face Swapping (FS):**
  1. Compute absolute pixel-wise difference in RGB channels: $|I_{\text{fake}} - I_{\text{source}}|$.
  2. Convert difference into grayscale.
  3. Divide by 255 to yield a map in $[0, 1]$.
  4. Apply threshold of $0.1$: $\mathbb{I}(\Delta_{\text{norm}} > 0.1)$.
- **Validation Safeguard:** Missing source image for AM/FS samples raises an explicit `ValueError` rather than producing an incorrect all-zero mask.

---

## 8. Parameter Sharing & Parameter Count Audit

### Exact Parameter Identity Audit:
- `IE is RE`: **TRUE** (`data_ptr: 6576631265280 == 6576631265280`)
- `AD trunk is MD trunk`: **TRUE** (`data_ptr: 6576849027328 == 6576849027328`)
- `W_voc encoder is decoder`: **TRUE** (`data_ptr: 6576924524800 == 6576924524800`)

### Parameter Count Breakdown:
- **Unique Model Parameters:** **159,466,439** (~159.47M)
- **Trainable Model Parameters:** **159,466,439** (100% trainable from scratch)
- **Loss Module Trainable Parameters:** **1** (`cmc_loss.log_tau`)
- **Total Optimization Parameters:** **159,466,440**
- *Submodule Breakdown:*
  - MVE (IE/RE shared + Image Transformer): 32,784,384
  - Vision Decoder (Shared trunk + AD/MD heads): 13,586,885
  - FLT (LE + LD + tied vocab): 112,831,488
    - Language Encoder (12 blocks + VIM): 75,890,688
    - Language Decoder (7 blocks + VIM): 62,237,696
  - Adapter: 262,656
  - Detection Head: 1,026

*Note on Parameter Count:* Because the paper does not specify the internal U-Net channel count or FFN expansion ratios, these counts reflect the standard reproduction architecture.

---

## 9. Training vs. Inference & Label Mapping Audits

### Training vs. Inference:
- **Training Forward:** Evaluates multi-modal graph (image + prompt tokens) and computes all 6 loss objectives.
- **Inference Forward (`forward_image_only`):** Evaluates visual branch only ($I \to \text{IE} \to \text{VD} \to I_r \to \text{RE} \to I_v \to \text{Detection Head}$). FLT, LE, LD, VIM language path, and tokenizer are completely bypassed. Mathematical equivalence between full-model vision branch and image-only path is verified by unit tests.

### Label Mapping & AUC Integrity:
- Configurable `real_class_index` (default: 0) and `fake_class_index` (default: 1).
- AUC continuously evaluates the positive class column: $\text{softmax}(y_{\text{pre}})[:, \text{fake\_class\_index}]$.
- Tested with inverted mapping (`fake_class_index = 0`, `real_class_index = 1`), proving AUC calculates $100\%$ without inversion.

---

## 10. Training From Scratch Audit

- The paper specifies that MFVLR is trained from scratch.
- Verified that zero pretrained weights (e.g. ImageNet, CLIP, torchvision backbones) are downloaded or loaded during model initialization.
- All weights initialize from scratch via standard PyTorch initialization.

---

## 11. Final Ledgers

### PAPER_SPECIFIED Decisions:
1. Input image resolution: $224 \times 224 \times 3$.
2. Local appearance feature dimension: $I_{\text{loc}} \in \mathbb{R}^{B \times 1024 \times 14 \times 14}$.
3. Global feature dimension: $d = 512$.
4. Image Transformer depth: $B = 4$ blocks, sequence length 197.
5. Language Encoder depth: $E = 12$ blocks.
6. Language Decoder depth: $D = 7$ blocks.
7. Text sequence length: $n = 308$ tokens.
8. Text vocabulary size: $s = 49,408$ tokens.
9. Weight sharing between Image Encoder and Residual Encoder ($\text{RE} \equiv \text{IE}$).
10. Shared decoder trunk between Appearance Decoder and Mask Decoder.
11. Tied vocabulary embedding and projection matrix $W_{\text{voc}}^T$.
12. Visual feature additive fusion: $I_v = I_g + I_{\text{rg}}$.
13. Global language token: $T_l$ extracted strictly from the LAST token of $T_{\text{hig}}^e$.
14. Vision Injection Module (Eq. 4-11) preserving singleton visual key/value token.
15. Mask ground truth rules: Real $= 0$, EFS $= 1$, AM/FS difference threshold $> 0.1$.
16. Appearance reconstruction loss: Strictly Mean Squared Error (Eq. 17).
17. KL divergence loss: $D_{\text{KL}}(P(T_l) \parallel Q(T_{\text{lpre}}))$ with $\tau=0.5$ (Eq. 19).
18. CMC loss: Unnormalized dot product, trainable $\tau$ initialized to $0.07$, bidirectional average (Eq. 20-23).
19. Equal multi-task loss weighting: All $\lambda_k = 1.0$.
20. Optimization parameters: Adam optimizer ($\text{lr}=10^{-4}, \text{weight decay}=10^{-3}$), StepLR ($\text{step}=15, \gamma=0.1$), batch size $b=8$.
21. Image-only inference for testing/evaluation.
22. Trained from scratch without pretrained weights.

### ASSUMPTION_FROM_PAPER_GAP Decisions:
1. **Float RGB $[0, 1]$ Scaling:** Reconstruction-compatible engineering choice aligning with Appearance Decoder output $I_{\text{pre}} \in [0, 1]$ and residual $I_r = |I_{\text{pre}} - I|$.
2. **Augmentation Disabled by Default:** Baseline assumption; absence in paper does not prove prohibition.
3. **VIM Attention Head Count & Dimension:** $r=8$ heads and $d/r=64$ per-head dimension.
4. **Feed-Forward Hidden Dimension:** 2048 (expansion factor 4) across Image Transformer, LE, LD, and VIM FFN sublayers.
5. **Transformer Architecture Details:** 8 attention heads in Image Transformer, LE, and LD; Pre-LN topology; GELU activation; dropout $= 0.0$.
6. **Prompt Packing Strategy ($4 \times 77 = 308$):** Padding/concatenation of L1-L4 prompt levels.
7. **Prompt Template Strings:** Standard templates (`"A photo of..."`) used for L1-L4 metadata.
8. **Exact Tokenizer Vocabulary & Merge Rules:** Deterministic offline hash/BPE token mapping in $[0, 49407]$ used in lieu of unreleased vocabulary files.
9. **Special Token IDs:** $\text{BOS}=49406, \text{EOS}=49407, \text{PAD}=0$.
10. **Grayscale Luminance Weights:** Standard ITU-R BT.601 weights $(0.299, 0.587, 0.114)$ for mask difference conversion.
11. **Manifest File Formats:** Supported `.jsonl`, `.json`, `.csv` with relative path resolution.
12. **Configurable Class Indices:** `real_class_index` and `fake_class_index` configurable in code/CLI.
13. **Total Training Epoch Count:** Configurable via CLI/YAML (`epochs: null` default).
14. **U-Net Topology:** 4-stage residual downsampling trunk $[128, 256, 512, 1024]$ and GroupNorm (32 groups).

### IMPLEMENTATION_ERROR Ledger:
- **Count:** **0 errors**. No contradictions with the academic paper exist in the codebase.

---

## 12. Verification & Test Execution Results

### 1. PyTest Full Suite Execution
- **Command:** `python -m pytest -v`
- **Output:**
  ```text
  ======================== 93 passed in 86.38s (0:01:26) ========================
  ```
- **Breakdown:** 93 passed, 0 failed, 0 errors, 0 skipped across 23 test modules.

### 2. Dry-Run Execution
- **Command:** `python train.py --config configs/mfvlr.yaml --dry-run --device cpu`
- **Output:**
  ```text
  [MFVLR_Train] [INFO] Using device: cpu
  [MFVLR_Train] [INFO] Train dataset size: 16, Val dataset size: 8
  [MFVLR_Train] [INFO] Executing DRY-RUN smoke test (1 training batch)...
  [MFVLR_Train] [INFO] Dry-run training step completed successfully. Losses: {'total_loss': 718.4789, 'loss_fd': 2.9971, 'loss_lr': 513.8177, 'loss_cmc': 193.0825, 'loss_fl': 0.7431, 'loss_ar': 0.0925, 'loss_kl': 7.7458}
  [MFVLR_Train] [INFO] Executing dry-run image-only evaluation batch...
  [MFVLR_Train] [INFO] Dry-run evaluation metrics: {'acc': 50.0, 'auc': 18.75, 'miou': 32.7667, 'iou_class_0': 27.5974, 'iou_class_1': 37.9360}
  [MFVLR_Train] [INFO] DRY-RUN test PASSED. Exiting.
  ```
- **Exit Code:** `0` (Success).

---

## 13. Final Statement on Benchmark Reproduction

**Explicit Disclaimer:**  
This implementation provides a paper-faithful reproduction implementation of MFVLR with documented assumptions for unspecified architectural and engineering details as described in *arXiv:2605.10071v1*. However, paper benchmark numbers (e.g. cross-generator AUC/mIoU tables on GenFace) have **NOT** yet been demonstrated because long-running real dataset training has intentionally not been executed in accordance with the Phase 12 stop condition.

The repository is now fully verified, documented, and ready for experimental reproduction on GenFace.
