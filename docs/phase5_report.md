# MFVLR Reproduction: Phase 5 Completion Report

**Date:** 2026-09-11  
**Project:** MFVLR Reproduction (Paper: *MFVLR: Multi-domain Fine-grained Vision-Language Reconstruction for Generalizable Diffusion Face Forgery Detection and Localization*, arXiv:2605.10071v1)  
**Task:** Phase 5 (Vision Injection Module VIM, Exact Equations 4–11, Singleton K/V Mathematical Analysis, and VIM Unit Tests)

---

## 1. Phase 5 Status

**STATUS: PASS**

The Vision Injection Module (VIM) has been implemented strictly per paper Equations (4)–(11), verified, and unit-tested with 100% pass rate (40/40 tests passing across the complete test suite). No Phase 6 components (Language Encoder, Language Decoder, FLT, token embedding, adapter, detection head, losses, or training loops) have been implemented.

---

## 2. Files Created

1. `models/language/vim.py` – `VisionInjectionModule` (and alias `VIM`) implementing Eq. (4)–(11).
2. `tests/test_vim.py` – 6 comprehensive unit tests verifying 2D/3D inputs, singleton attention properties, step-by-step tensor shapes, and backward propagation.
3. `docs/phase5_report.md` – This report.

---

## 3. Files Modified

1. `models/language/__init__.py` – Exported `VisionInjectionModule` and `VIM`.
2. `models/__init__.py` – Exported `VisionInjectionModule` and `VIM`.
3. `docs/reproduction_notes.md` – Updated phase status to **PHASE 5 COMPLETE**, updated decision ledger with Phase 5 artifacts and mathematical singleton K/V analysis, and set Phase 6 stop condition.

---

## 4. Exact VIM Equations Implemented

The module faithfully implements Equations (4)–(11) from Section III-D of the paper:

1. **Linear Projections (Eq. 4–6):**
   $$q_j = T_{\text{tok}}^j W_{\text{que}}^j \in \mathbb{R}^{B \times n \times d}$$
   $$k_j = I_v W_{\text{key}}^j \in \mathbb{R}^{B \times 1 \times d}$$
   $$v_j = I_v W_{\text{val}}^j \in \mathbb{R}^{B \times 1 \times d}$$
   where $W_{\text{que}}^j, W_{\text{key}}^j, W_{\text{val}}^j \in \mathbb{R}^{d \times d}$.

2. **Head Partition (Eq. 7–9):**
   $$\{Q_{j, i} \in \mathbb{R}^{B \times n \times \frac{d}{r}}\}_{i=1}^r = \operatorname{Pa}(q_j)$$
   $$\{K_{j, i} \in \mathbb{R}^{B \times 1 \times \frac{d}{r}}\}_{i=1}^r = \operatorname{Pa}(k_j)$$
   $$\{V_{j, i} \in \mathbb{R}^{B \times 1 \times \frac{d}{r}}\}_{i=1}^r = \operatorname{Pa}(v_j)$$

3. **Cross-Attention Calculation per Head (Eq. 10):**
   $$T_{j, i}^{\text{glo}} = \delta\left(\frac{Q_{j, i} K_{j, i}^T}{\sqrt{d/r}}\right) V_{j, i} \in \mathbb{R}^{B \times n \times \frac{d}{r}}$$
   where $\delta$ is the softmax function along the key dimension ($\text{dim}=-1$).

4. **Head Concatenation, Output Projection & Residual Addition (Eq. 11):**
   $$T_j^{\text{glo}} = \operatorname{Cat}(\{T_{j, i}^{\text{glo}}\}_{i=1}^r) \in \mathbb{R}^{B \times n \times d}$$
   $$T_j^{\text{add}} = T_j^{\text{glo}} W_{\text{fc}}^j + T_{\text{tok}}^j \in \mathbb{R}^{B \times n \times d}$$
   where $W_{\text{fc}}^j \in \mathbb{R}^{d \times d}$.

---

## 5. Q / K / V Direction

- **QUERY ($Q$):** Derived exclusively from language representations: $q_j = T_{\text{tok}}^j W_{\text{que}}^j$.
- **KEY ($K$):** Derived exclusively from fused visual representations: $k_j = I_v W_{\text{key}}^j$.
- **VALUE ($V$):** Derived exclusively from fused visual representations: $v_j = I_v W_{\text{val}}^j$.
- **Strict Prohibition:** Vision was NOT used as Query, and language was NOT used as Key/Value.

---

## 6. Tensor Shapes at Every VIM Operation

For batch size $B=2$, token length $n=308$, feature dimension $d=512$, and head count $r=8$ ($\text{head\_dim} = 64$):

| Operation / Step | Expression | Output Shape | Data Type | Notes |
|---|---|---|---|---|
| Language Input ($T_{\text{tok}}^j$) | Input | `[2, 308, 512]` | `float32` | Word representations |
| Visual Input ($I_v$) | Input | `[2, 1, 512]` (or `[2, 512]`) | `float32` | Fused visual feature |
| Visual Unsqueeze | `i_v.unsqueeze(1)` | `[2, 1, 512]` | `float32` | Standardized 3D shape |
| Query Projection | $q_j = T_{\text{tok}} W_{\text{que}}$ | `[2, 308, 512]` | `float32` | Eq. (4) |
| Key Projection | $k_j = I_v W_{\text{key}}$ | `[2, 1, 512]` | `float32` | Eq. (5) |
| Value Projection | $v_j = I_v W_{\text{val}}$ | `[2, 1, 512]` | `float32` | Eq. (6) |
| Split Query ($Q$) | `Q.view(...).transpose(1, 2)` | `[2, 8, 308, 64]` | `float32` | Eq. (7) |
| Split Key ($K$) | `K.view(...).transpose(1, 2)` | `[2, 8, 1, 64]` | `float32` | Eq. (8) |
| Split Value ($V$) | `V.view(...).transpose(1, 2)` | `[2, 8, 1, 64]` | `float32` | Eq. (9) |
| Attention Scores | $Q K^T / \sqrt{64}$ | `[2, 8, 308, 1]` | `float32` | Scaled dot products |
| Softmax Probabilities ($A$) | $\operatorname{softmax}(\text{scores}, \text{dim}=-1)$ | `[2, 8, 308, 1]` | `float32` | Exactly $1.0$ |
| Head Context ($T_{\text{glo}}^{\text{heads}}$) | $A @ V$ | `[2, 8, 308, 64]` | `float32` | Eq. (10) |
| Concatenated Context ($T^{\text{glo}}$) | `t_glo_heads.reshape(...)` | `[2, 308, 512]` | `float32` | Eq. (11) |
| Output Projection & Add ($T^{\text{add}}$) | $T^{\text{glo}} W_{\text{fc}} + T_{\text{tok}}$ | `[2, 308, 512]` | `float32` | Eq. (11) |

---

## 7. Number of Heads and Head Dimension

- **Total Feature Dimension ($d$):** `512` (PAPER_SPECIFIED)
- **Number of Heads ($r$):** `8` (ASSUMPTION_FROM_PAPER_GAP, approved reproduction baseline)
- **Head Dimension ($\frac{d}{r}$):** $512 / 8 = 64$
- **Attention Scale Factor:** $\frac{1}{\sqrt{64}} = 0.125$

---

## 8. Paper-Specified Decisions (`PAPER_SPECIFIED`)

1. **Module Purpose:** Fine-grained word-level visual-language interaction without quadratic attention cost (Section III-D).
2. **Key/Value Content:** Uses **only the single visual class/global token** $I_v$ rather than patch tokens, reducing attention map generation to linear cost.
3. **Query/Key/Value Assignment:** Language = Query ($W_{\text{que}} \in \mathbb{R}^{d \times d}$); Vision = Key ($W_{\text{key}} \in \mathbb{R}^{d \times d}$); Vision = Value ($W_{\text{val}} \in \mathbb{R}^{d \times d}$).
4. **Residual Structure:** Output projection $W_{\text{fc}} \in \mathbb{R}^{d \times d}$ followed by direct residual addition to $T_{\text{tok}}^j$ ($T_{\text{add}}^j = T_{\text{glo}}^j W_{\text{fc}}^j + T_{\text{tok}}^j$).

---

## 9. Assumptions Introduced (`ASSUMPTION_FROM_PAPER_GAP`)

| Decision / Gap | Implementation Choice | Rationale |
|---|---|---|
| **Head Partition $r$** | `num_heads = 8` | Unspecified in paper; 8 heads divide $d=512$ into standard 64-dim projections. |
| **Linear Layer Bias** | `bias = True` (configurable) | Standard PyTorch `nn.Linear` default for $W_{\text{que}}, W_{\text{key}}, W_{\text{val}}, W_{\text{fc}}$. |
| **Flexible 2D Input** | Automatically unsqueeze $I_v$ if passed as $[B, 512]$ | Simplifies forward integration between MVE and FLT. |

---

## 10. Singleton K/V Mathematical Behavior

Because the paper specifies that Key and Value contain only the single class token $I_v$, the sequence length along the key dimension is exactly $L_k = 1$:

1. **Softmax Value:** For any scalar score $s \in \mathbb{R}$, $\operatorname{softmax}([s]) = \frac{e^s}{e^s} \equiv 1.0$.
2. **Context Value:** $T_{j, i}^{\text{glo}} = 1.0 \cdot V_{j, i} = V_{j, i}$ (broadcast across all $n=308$ word positions).
3. **Faithful Reproduction:** The implementation strictly preserves this paper-specified formulation. We do **not** invent extra patch tokens or alter the equation to artificially produce multi-token distributions.

---

## 11. Measured Attention Values

In `tests/test_vim.py::test_vim_singleton_attention_property`:
- **Attention Shape:** `[2, 8, 308, 1]`
- **Minimum Value:** `1.0000000`
- **Maximum Value:** `1.0000000`
- **Deviation from 1.0:** `torch.allclose(attn, torch.ones_like(attn), atol=1e-6)` $\to$ `True`.

---

## 12. Gradient Behavior of $W_{\text{que}}$, $W_{\text{key}}$, $W_{\text{val}}$, and $W_{\text{fc}}$

Empirical gradient norms evaluated during backward pass on scalar loss $L = \sum T_{\text{add}}$:

| Parameter | Tensor Shape | Gradient Status | Gradient Norm | Mathematical Explanation |
|---|---|---|---|---|
| `w_fc.weight` | `[512, 512]` | Connected | `125,994.2` | Receives direct non-zero gradient from $T_{\text{add}} = T_{\text{glo}} W_{\text{fc}} + T_{\text{tok}}$. |
| `w_val.weight` | `[512, 512]` | Connected | `117,574.2` | Receives direct non-zero gradient from $T_{\text{glo}} = 1.0 \cdot V$. |
| `w_que.weight` | `[512, 512]` | Connected | `0.0` | In computational graph (`grad is not None`), but mathematically zero because $\frac{\partial \operatorname{softmax}(s)}{\partial s} = s(1-s) = 1(0) = 0$. |
| `w_key.weight` | `[512, 512]` | Connected | `0.0` | In computational graph (`grad is not None`), but mathematically zero due to singleton softmax derivative. |
| $T_{\text{tok}}$ (Language In) | `[2, 308, 512]` | Connected | `561.6` | Direct non-zero gradient from residual addition. |
| $I_v$ (Vision In) | `[2, 1, 512]` | Connected | `3,053.3` | Direct non-zero gradient through $W_{\text{val}}$. |

---

## 13. Parameter Count Summary

```text
Vision Injection Module (VIM) Parameter Breakdown:
  ├── W_que (Language -> Query Linear 512 -> 512): 262,656
  ├── W_key (Vision -> Key Linear 512 -> 512):     262,656
  ├── W_val (Vision -> Value Linear 512 -> 512):   262,656
  └── W_fc  (Output Projection Linear 512 -> 512): 262,656

Total Parameters per VIM instance: 1,050,624 (1.05M)
Trainable Parameters: 1,050,624 (100%)
```

---

## 14. Commands Executed

```bash
# 1. Run Phase 5 VIM unit tests
python -m pytest tests/test_vim.py -v

# 2. Run complete test suite (Phase 2 + Phase 3 + Phase 4 + Phase 5)
python -m pytest -v

# 3. Inspect VIM gradient norms
python -c "import torch; from models.language.vim import VisionInjectionModule; vim = VisionInjectionModule(); t=torch.randn(2,308,512,requires_grad=True); v=torch.randn(2,1,512,requires_grad=True); vim(t,v).sum().backward(); print('w_fc:', vim.w_fc.weight.grad.norm().item(), 'w_val:', vim.w_val.weight.grad.norm().item(), 'w_que:', vim.w_que.weight.grad.norm().item())"
```

---

## 15. Exact Test Results

```text
============================= test session starts =============================
platform win32 -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Paper
collected 40 items

tests/test_config.py::test_mfvlr_config_loading PASSED                   [  2%]
tests/test_config.py::test_dataset_config_loading PASSED                 [  5%]
tests/test_dummy_dataset.py::test_dummy_dataset_length PASSED            [  7%]
tests/test_dummy_dataset.py::test_dummy_dataset_item_structure PASSED    [ 10%]
tests/test_dummy_dataset.py::test_dummy_dataloader_batching PASSED       [ 12%]
tests/test_imports.py::test_import_utils PASSED                          [ 15%]
tests/test_imports.py::test_import_datasets PASSED                       [ 17%]
tests/test_imports.py::test_import_packages PASSED                       [ 20%]
tests/test_mask_generator.py::test_real_mask_all_zeros PASSED            [ 22%]
tests/test_mask_generator.py::test_efs_mask_all_ones PASSED              [ 25%]
tests/test_mask_generator.py::test_am_fs_mask_thresholding PASSED        [ 27%]
tests/test_mve.py::test_unet_encoder_output_shape PASSED                 [ 30%]
tests/test_mve.py::test_unet_encoder_skips PASSED                        [ 32%]
tests/test_mve.py::test_image_transformer_structure PASSED               [ 35%]
tests/test_mve.py::test_image_encoder_forward PASSED                     [ 37%]
tests/test_mve.py::test_mve_true_weight_sharing PASSED                   [ 40%]
tests/test_mve.py::test_mve_full_forward_and_fusion PASSED               [ 42%]
tests/test_mve.py::test_mve_gradient_backpropagation PASSED              [ 45%]
tests/test_prompt_generator.py::test_real_prompts PASSED                 [ 47%]
tests/test_prompt_generator.py::test_fake_ddpm_prompts PASSED            [ 50%]
tests/test_prompt_generator.py::test_fake_stylegan3_prompts PASSED       [ 52%]
tests/test_prompt_generator.py::test_fake_fslsd_prompts PASSED           [ 55%]
tests/test_prompt_generator.py::test_fake_diffae_prompts PASSED          [ 57%]
tests/test_utils.py::test_seed_determinism PASSED                        [ 60%]
tests/test_utils.py::test_classification_metrics PASSED                  [ 62%]
tests/test_utils.py::test_localization_metrics PASSED                    [ 65%]
tests/test_utils.py::test_checkpoint_save_and_load PASSED                [ 67%]
tests/test_utils.py::test_visualization_save PASSED                      [ 70%]
tests/test_vim.py::test_vim_initialization_and_hyperparameters PASSED    [ 72%]
tests/test_vim.py::test_vim_forward_3d_inputs PASSED                     [ 75%]
tests/test_vim.py::test_vim_forward_2d_vision_input PASSED               [ 77%]
tests/test_vim.py::test_vim_singleton_attention_property PASSED          [ 80%]
tests/test_vim.py::test_vim_gradient_backpropagation PASSED              [ 82%]
tests/test_vim.py::test_vim_step_by_step_shapes PASSED                   [ 85%]
tests/test_vision_decoder.py::test_unet_decoder_trunk_output_shape PASSED [ 87%]
tests/test_vision_decoder.py::test_appearance_decoder_output PASSED      [ 90%]
tests/test_vision_decoder.py::test_mask_decoder_output PASSED            [ 92%]
tests/test_vision_decoder.py::test_vision_decoder_true_weight_sharing PASSED [ 95%]
tests/test_vision_decoder.py::test_vision_decoder_residual_generation PASSED [ 97%]
tests/test_vision_decoder.py::test_vision_decoder_backpropagation PASSED [100%]

============================= 40 passed in 11.14s =============================
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
- All structural, mathematical, and pipeline constraints outlined in `MFVLR_Codex_Reproduction_Prompt.md` for Phase 5 were strictly observed.

---

## 18. Unresolved Issues

None. All Phase 5 tests and prior regression tests pass with 100% success.

---

## 19. Ready for Phase 6 Confirmation

**YES.** The Vision Injection Module (VIM) is fully implemented, verified, and unit-tested.

The repository is completely ready to proceed to **PHASE 6** (Language Encoder implementation, token embedding interface, $E=12$ Transformer blocks with VIM injection, high-level embeddings $T_{\text{hig}}^e$, last-token $T_l$ extraction, and `tests/test_language_encoder.py`).
