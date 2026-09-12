# MFVLR Reproduction: Phase 2 Completion Report

**Date:** 2026-09-11  
**Project:** MFVLR Reproduction (Paper: *MFVLR: Multi-domain Fine-grained Vision-Language Reconstruction for Generalizable Diffusion Face Forgery Detection and Localization*, arXiv:2605.10071v1)  
**Task:** Phase 2 (Repository Scaffolding, Configuration System, Dummy Dataset, Utilities, and Import/Data Tests)

---

## 1. Phase 2 Status

**STATUS: PASS**

All required components of Phase 2 have been implemented, verified, and unit-tested with 100% pass rate. No model backbones (MVE, IE, RE, VD, FLT, VIM), losses, or training loops have been implemented yet, preserving the Phase 2 boundary.

---

## 2. All Files Created

The following 28 files were created during Phase 2:

1. `README.md`
2. `requirements.txt`
3. `docs/architecture.md`
4. `docs/phase2_report.md` (this report)
5. `configs/mfvlr.yaml`
6. `configs/dataset.yaml`
7. `datasets/__init__.py`
8. `datasets/transforms.py`
9. `datasets/prompt_generator.py`
10. `datasets/mask_generator.py`
11. `datasets/dummy_dataset.py`
12. `datasets/genface.py`
13. `models/__init__.py`
14. `models/vision/__init__.py`
15. `models/language/__init__.py`
16. `losses/__init__.py`
17. `utils/__init__.py`
18. `utils/seed.py`
19. `utils/metrics.py`
20. `utils/checkpoint.py`
21. `utils/logger.py`
22. `utils/visualization.py`
23. `tests/__init__.py`
24. `tests/test_imports.py`
25. `tests/test_config.py`
26. `tests/test_dummy_dataset.py`
27. `tests/test_prompt_generator.py`
28. `tests/test_mask_generator.py`
29. `tests/test_utils.py`

---

## 3. All Files Modified

1. `docs/reproduction_notes.md` – Updated phase status to **PHASE 2 COMPLETE**, updated the decision ledger with Phase 2 artifacts, and set the Phase 3 stop condition.

---

## 4. Repository Structure After Phase 2

```text
MFVLR/
├── README.md
├── requirements.txt
├── paper/
│   └── 2605.10071v1.pdf
├── docs/
│   ├── paper_spec.md
│   ├── architecture.md
│   ├── reproduction_notes.md
│   └── phase2_report.md
├── configs/
│   ├── mfvlr.yaml
│   └── dataset.yaml
├── datasets/
│   ├── __init__.py
│   ├── dummy_dataset.py
│   ├── genface.py
│   ├── mask_generator.py
│   ├── prompt_generator.py
│   └── transforms.py
├── models/
│   ├── __init__.py
│   ├── vision/
│   │   └── __init__.py
│   └── language/
│       └── __init__.py
├── losses/
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── logger.py
│   ├── metrics.py
│   ├── seed.py
│   └── visualization.py
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_dummy_dataset.py
    ├── test_imports.py
    ├── test_mask_generator.py
    ├── test_prompt_generator.py
    └── test_utils.py
```

---

## 5. Configuration Files Created and Key Values

### `configs/mfvlr.yaml`
Contains all paper-specified constants and explicit assumption parameters:

- **Model Dimensions (PAPER_SPECIFIED):**
  - `image_size`: `224`
  - `in_channels`: `3`
  - `local_channels`: `1024` ($c = 1024$)
  - `local_height`: `14` ($h = 14$)
  - `local_width`: `14` ($w = 14$)
  - `embed_dim`: `512` ($d = 512$)
  - `image_transformer_blocks`: `4` ($B = 4$)
  - `language_encoder_blocks`: `12` ($E = 12$)
  - `language_decoder_blocks`: `7` ($D = 7$)
  - `vocab_size`: `49408` ($s = 49,408$)
  - `max_text_tokens`: `308` ($n = 308$)
  - `num_classes`: `2` ($f = 2$)
  - `mask_channels`: `2` ($f = 2$ for $M_{\text{pre}}$)

- **Backbone Assumptions (ASSUMPTION_FROM_PAPER_GAP):**
  - `image_attention_heads`: `8`
  - `language_attention_heads`: `8`
  - `vim_heads`: `8` ($r = 8$)
  - `dim_feedforward`: `2048` ($4 \times d$)
  - `dropout`: `0.0`
  - `activation`: `"gelu"`
  - `unet_encoder_channels`: `[128, 256, 512, 1024]`
  - `unet_decoder_channels`: `[512, 256, 128, 64]`

- **Loss Coefficients (PAPER_SPECIFIED):**
  - `lambda_fd`: `1.0`
  - `lambda_lr`: `1.0`
  - `lambda_cmc`: `1.0`
  - `lambda_fl`: `1.0`
  - `lambda_ar`: `1.0`
  - `lambda_kl`: `1.0`
  - `kl_temperature`: `0.5`

- **CMC Parameters (PAPER_SPECIFIED):**
  - `initial_temperature`: `0.07` ($\tau$)
  - `trainable_temperature`: `true`
  - `similarity`: `"dot_product"` (no default cosine/L2 normalization)
  - `l2_normalize`: `false`

- **Training Settings (PAPER_SPECIFIED & ASSUMPTIONS):**
  - `batch_size`: `8` ($b = 8$)
  - `optimizer`: `"adam"`
  - `learning_rate`: `1.0e-4`
  - `weight_decay`: `1.0e-3`
  - `scheduler.step_size`: `15` (reduce LR by $10\times$ every 15 epochs)
  - `scheduler.gamma`: `0.1`
  - `epochs`: `null` (not specified in paper; user-supplied)

### `configs/dataset.yaml`
- `dataset_name`: `"genface"`
- `image_size`: `224`
- `mask.gt_threshold`: `0.1` (PAPER_SPECIFIED)
- `mask.grayscale_weights`: `[0.299, 0.587, 0.114]` (ASSUMPTION_FROM_PAPER_GAP)
- `prompt.levels`: `[1, 2, 3, 4]`

---

## 6. DummyMFVLRDataset Specification

The `DummyMFVLRDataset` in `datasets/dummy_dataset.py` generates synthetic samples adhering strictly to the required schema:

| Field Name | Description | Shape | Data Type | Value Range |
|---|---|---|---|---|
| `image` | Synthetic input appearance image | `[3, 224, 224]` | `torch.float32` | $[0.0, 1.0]$ |
| `label` | One-hot binary classification label | `[2]` | `torch.float32` | `[1.0, 0.0]` (Real) or `[0.0, 1.0]` (Fake) |
| `label_idx` | Integer class index | `()` | `torch.int64` | `0` (Real) or `1` (Fake) |
| `mask` | Ground-truth localization mask | `[224, 224]` | `torch.float32` | $\{0.0, 1.0\}$ |
| `tokens` | Token IDs for hierarchical prompt | `[308]` | `torch.int64` | $[0, 49407]$ |
| `prompt` | Concatenated L1–L4 prompt text | scalar string | `str` | Text |
| `prompts_dict` | Individual L1–L4 level prompt strings | dictionary | `dict` | Keys: `'L1'`, `'L2'`, `'L3'`, `'L4'`, `'full_prompt'` |
| `generator` | Named generator | scalar string | `str` | e.g. `'DDPM'`, `'DiffFace'`, `'StyleGAN3'`, `'Real'` |
| `forgery_type` | Manipulation category | scalar string | `str` | `'EFS'`, `'FS'`, `'AM'`, `'Real'` |
| `source_image` | Source image tensor for mask computation | `[3, 224, 224]` | `torch.float32` | $[0.0, 1.0]$ |
| `has_source` | Boolean indicating source presence | scalar bool | `bool` / `torch.bool` | `True` for AM/FS, `False` for Real/EFS |

---

## 7. Utilities Implemented

1. **`utils/seed.py` (`set_seed`):**
   - Sets random seed across Python `random`, NumPy `np.random`, PyTorch CPU (`torch.manual_seed`), and CUDA (`torch.cuda.manual_seed_all`).
   - Configures cuDNN determinism: `deterministic=True`, `benchmark=False`.

2. **`utils/metrics.py`:**
   - `compute_classification_metrics`: Computes Accuracy (`ACC`) and Area Under ROC Curve (`AUC`) in percentage $[0, 100]$.
   - `compute_localization_metrics`: Computes Mean Intersection over Union (`mIoU`) across class 0 (unmanipulated) and class 1 (manipulated), plus individual class IoUs in percentage $[0, 100]$.

3. **`utils/checkpoint.py`:**
   - `save_checkpoint`: Saves training state dict, optimizer state, epoch, and optionally tracks `best_model.pt`.
   - `load_checkpoint`: Restores model weights, optimizer, and scheduler states with device mapping.

4. **`utils/logger.py` (`setup_logger`):**
   - Configures standardized logging to both stdout console and an optional log file (`train.log`).

5. **`utils/visualization.py`:**
   - `tensor_to_pil`: Converts float tensors $[C, H, W]$ in $[0, 1]$ to PIL RGB/L images.
   - `save_visualization`: Exports horizontal multi-panel comparison images combining Image, Reconstruction ($I_{\text{pre}}$), Residual ($I_r$), Ground-Truth Mask ($M$), and Predicted Mask ($M_{\text{pre}}$).

---

## 8. All Assumptions (`ASSUMPTION_FROM_PAPER_GAP`) Introduced in Phase 2

| Decision / Gap | Phase 2 Implementation | Rationale |
|---|---|---|
| **Random Seed** | `seed: 42` | Needed for reproducible execution; exposed in YAML config. |
| **DataLoader Collation** | Zero tensor placeholder for `source_image` + `has_source: bool` flag | Standard PyTorch `default_collate` requires homogeneous types across batch dictionaries. |
| **Grayscale Luminance Formula** | $0.299 R + 0.587 G + 0.114 B$ | Standard ITU-R BT.601 conversion formula before dividing by 255 and thresholding at 0.1. |
| **Mask Threshold Comparator** | `value > 0.1` | Strictly evaluates values greater than 0.1 as positive manipulation mask. |
| **Prompt Packing Structure** | `4x77_concat` | 4 prompt levels illustrated in Fig. 6 concatenated to reach $n=308$ token length ($4 \times 77$). |
| **Metric Epsilon** | $\epsilon = 10^{-7}$ in `compute_localization_metrics` | Avoids division by zero when union of class is empty. |
| **Preprocessing Pipeline** | Resize to $224 \times 224$ and convert to float32 $[0, 1]$ | Minimal transformation preserving raw image geometry and range for appearance reconstruction. |
| **Training Epoch Count** | `epochs: null` in config | Paper does not state total training epochs; runner must specify via CLI or YAML. |

---

## 9. Commands Executed

```bash
# 1. Environment verification
python -c "import torch; print(torch.__version__)"

# 2. Dependency installation in environment
python -m pip install pytest

# 3. Unit test execution
python -m pytest -v
```

---

## 10. All Tests Executed

21 automated unit tests across 6 test modules:

1. `tests/test_config.py::test_mfvlr_config_loading`
2. `tests/test_config.py::test_dataset_config_loading`
3. `tests/test_dummy_dataset.py::test_dummy_dataset_length`
4. `tests/test_dummy_dataset.py::test_dummy_dataset_item_structure`
5. `tests/test_dummy_dataset.py::test_dummy_dataloader_batching`
6. `tests/test_imports.py::test_import_utils`
7. `tests/test_imports.py::test_import_datasets`
8. `tests/test_imports.py::test_import_packages`
9. `tests/test_mask_generator.py::test_real_mask_all_zeros`
10. `tests/test_mask_generator.py::test_efs_mask_all_ones`
11. `tests/test_mask_generator.py::test_am_fs_mask_thresholding`
12. `tests/test_prompt_generator.py::test_real_prompts`
13. `tests/test_prompt_generator.py::test_fake_ddpm_prompts`
14. `tests/test_prompt_generator.py::test_fake_stylegan3_prompts`
15. `tests/test_prompt_generator.py::test_fake_fslsd_prompts`
16. `tests/test_prompt_generator.py::test_fake_diffae_prompts`
17. `tests/test_utils.py::test_seed_determinism`
18. `tests/test_utils.py::test_classification_metrics`
19. `tests/test_utils.py::test_localization_metrics`
20. `tests/test_utils.py::test_checkpoint_save_and_load`
21. `tests/test_utils.py::test_visualization_save`

---

## 11. Exact Test Results

```text
============================= test session starts =============================
platform win32 -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Paper
collected 21 items

tests/test_config.py::test_mfvlr_config_loading PASSED                   [  4%]
tests/test_config.py::test_dataset_config_loading PASSED                 [  9%]
tests/test_dummy_dataset.py::test_dummy_dataset_length PASSED            [ 14%]
tests/test_dummy_dataset.py::test_dummy_dataset_item_structure PASSED    [ 19%]
tests/test_dummy_dataset.py::test_dummy_dataloader_batching PASSED       [ 23%]
tests/test_imports.py::test_import_utils PASSED                          [ 28%]
tests/test_imports.py::test_import_datasets PASSED                       [ 33%]
tests/test_imports.py::test_import_packages PASSED                       [ 38%]
tests/test_mask_generator.py::test_real_mask_all_zeros PASSED            [ 42%]
tests/test_mask_generator.py::test_efs_mask_all_ones PASSED              [ 47%]
tests/test_mask_generator.py::test_am_fs_mask_thresholding PASSED        [ 52%]
tests/test_prompt_generator.py::test_real_prompts PASSED                 [ 57%]
tests/test_prompt_generator.py::test_fake_ddpm_prompts PASSED            [ 61%]
tests/test_prompt_generator.py::test_fake_stylegan3_prompts PASSED       [ 66%]
tests/test_prompt_generator.py::test_fake_fslsd_prompts PASSED           [ 71%]
tests/test_prompt_generator.py::test_fake_diffae_prompts PASSED          [ 76%]
tests/test_utils.py::test_seed_determinism PASSED                        [ 80%]
tests/test_utils.py::test_classification_metrics PASSED                  [ 85%]
tests/test_utils.py::test_localization_metrics PASSED                    [ 90%]
tests/test_utils.py::test_checkpoint_save_and_load PASSED                [ 95%]
tests/test_utils.py::test_visualization_save PASSED                      [100%]

============================= 21 passed in 6.08s ==============================
```

---

## 12. Warnings, Errors, Skipped Tests, or Unresolved Issues

- **Errors during execution:** 0
- **Failures:** 0
- **Skipped tests:** 0
- **Warnings:** 0
- **Unresolved issues:** None.

---

## 13. Deviations from `MFVLR_Codex_Reproduction_Prompt.md`

- **Deviations:** None.
- All structural, configuration, mathematical, and pipeline constraints outlined in the prompt were strictly preserved.

---

## 14. Ready for Phase 3 Confirmation

**YES.** The repository scaffolding, configuration system, dataset abstractions, and utility foundation are fully in place and verified.

The repository is completely ready to proceed to **PHASE 3** (MVE: U-Net encoder, image tokenization, Image Transformer, shared Residual Encoder, and `tests/test_mve.py`).
