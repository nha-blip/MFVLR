# MFVLR Codex Reproduction Prompt

You are a senior deep-learning research engineer specializing in PyTorch, Computer Vision, Vision-Language Models, Transformers, U-Net, Contrastive Learning, Face Forgery Detection, Image Manipulation Localization, and academic-paper reproduction.

Your task is to reproduce the model described in:

**MFVLR: Multi-domain Fine-grained Vision-Language Reconstruction for Generalizable Diffusion Face Forgery Detection and Localization**

## AVAILABLE SOURCE MATERIAL

Before writing any implementation code, read:

1. `paper/2605.10071v1.pdf`
2. `docs/paper_spec.md`

The original PDF is the **ultimate source of truth**. `docs/paper_spec.md` is an extracted implementation specification prepared from the paper. If `paper_spec.md` conflicts with the paper, follow the PDF and document the conflict.

Do **not** search for or assume an unreleased official implementation. The goal is a **paper reproduction**, not exact recovery of the authors' private source code.

## 0. MOST IMPORTANT RULE

Never silently invent missing architecture details.

For every implementation decision, classify it as either:

- `PAPER_SPECIFIED`
- `ASSUMPTION_FROM_PAPER_GAP`

When the paper does not specify something, add a code comment:

```python
# ASSUMPTION_FROM_PAPER_GAP:
# Paper does not specify ...
# Reproduction chooses ...
```

Also document the assumption in `docs/reproduction_notes.md`.

Typical unspecified details may include exact U-Net topology, channel schedule, skip connections, normalization, activations, Transformer attention-head count, FFN dimension, dropout, tokenizer implementation, vocabulary/checkpoint, exact L1-L4 prompt packing, initialization, mask activation/reduction, padding handling, preprocessing, augmentation, total epochs, and inference threshold.

## 1. IMPLEMENTATION TARGET

Use Python 3.10+, PyTorch, torchvision, timm only if needed, transformers/tokenizer libraries only if justified, numpy, Pillow/OpenCV where appropriate, scikit-learn, PyYAML, and tqdm.

Requirements: CUDA support, AMP, reproducible seeds, configurable architecture/dataset paths/hyperparameters, checkpoint save/load, modular code, unit tests, and no hard-coded absolute paths. Target single-GPU first, but keep the design DDP-friendly.

## 2. REQUIRED PROJECT STRUCTURE

Create approximately:

```text
MFVLR/
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── inference.py
├── paper/
│   └── 2605.10071v1.pdf
├── docs/
│   ├── paper_spec.md
│   ├── architecture.md
│   └── reproduction_notes.md
├── configs/
│   ├── mfvlr.yaml
│   └── dataset.yaml
├── datasets/
│   ├── __init__.py
│   ├── genface.py
│   ├── transforms.py
│   ├── prompt_generator.py
│   ├── mask_generator.py
│   └── dummy_dataset.py
├── models/
│   ├── __init__.py
│   ├── mfvlr.py
│   ├── adapter.py
│   ├── classifier.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── unet_encoder.py
│   │   ├── image_transformer.py
│   │   ├── image_encoder.py
│   │   ├── residual_encoder.py
│   │   ├── mve.py
│   │   ├── appearance_decoder.py
│   │   ├── mask_decoder.py
│   │   └── vision_decoder.py
│   └── language/
│       ├── __init__.py
│       ├── embeddings.py
│       ├── vim.py
│       ├── language_encoder.py
│       ├── language_decoder.py
│       └── flt.py
├── losses/
│   ├── __init__.py
│   ├── appearance_reconstruction.py
│   ├── localization_loss.py
│   ├── kl_alignment.py
│   ├── cmc_loss.py
│   ├── language_reconstruction.py
│   ├── detection_loss.py
│   └── total_loss.py
├── utils/
│   ├── seed.py
│   ├── metrics.py
│   ├── checkpoint.py
│   ├── logger.py
│   └── visualization.py
└── tests/
    ├── test_mve.py
    ├── test_vision_decoder.py
    ├── test_vim.py
    ├── test_language_encoder.py
    ├── test_language_decoder.py
    ├── test_losses.py
    └── test_full_forward.py
```

Minor structural changes are allowed only when they improve clarity.

## 3. PAPER-SPECIFIED GLOBAL CONSTANTS

Use the main configuration from the paper:

- Input image: `3×224×224`
- Image Transformer blocks: `B = 4`
- Language Encoder blocks: `E = 12`
- Language Decoder blocks: `D = 7`
- Vocabulary size: `s = 49,408`
- Categories: `f = 2`
- Feature dimension: `d = 512`
- Text token count: `n = 308`
- Batch size: `b = 8`
- Local feature channels: `c = 1024`
- Local feature size: `h = w = 14`
- `I_loc ∈ R^(1024×14×14)`
- Visual token count: `196 + 1 class token = 197`
- CMC temperature: trainable, initialized at `0.07`
- KL softmax temperature: `0.5`
- Mask threshold: `0.1`
- Optimizer: Adam
- LR: `1e-4`
- Weight decay: `1e-3`
- Scheduler: reduce LR by factor 10 every 15 epochs

Do not invent the total epoch count.

## 4. OVERALL MFVLR ARCHITECTURE

MFVLR contains:

1. Multi-domain Vision Encoder (MVE)
2. Vision Decoder (VD)
3. Fine-grained Language Transformer (FLT)

Training inputs: image `I`, text `T`, detection label `y`, localization mask `M`.

### Image flow

```text
I
↓
Image Encoder (IE)
├─> I_loc
│    ↓
│   Vision Decoder
│   ├─> I_pre
│   └─> M_pre
│
└─> I_g

I_r = |I_pre - I|
↓
Residual Encoder (same architecture AND same weights as IE)
↓
I_rg

I_v = I_rg + I_g
├─> Adapter -> T_lpre
└─> MLP Head -> y_pre
```

### Text flow

```text
Hierarchical prompt T
↓
Tokenizer
↓
Token embeddings
↓
T_low^e
↓
Language Encoder + VIM(I_v)
↓
T_hig^e
├─> last token -> T_l
└─> Language Decoder + T_low^e + I_v
    ↓
    T_rec^d
    ↓
    vocabulary projection
    ↓
    T_pre
```

## 5. MULTI-DOMAIN VISION ENCODER

MVE contains Image Encoder (IE) and Residual Encoder (RE).

### 5.1 Image Encoder

Input `I ∈ R^(3×224×224)`.

U-Net encoder produces `I_loc ∈ R^(1024×14×14)`.

For the Transformer path:

```text
I_loc
-> flatten
-> project to d=512
-> append learnable class token
-> I_tok ∈ R^(197×512)
-> + learnable positional embedding P_i
-> B=4 Transformer blocks
-> I_TE
-> class token
-> I_g ∈ R^(1×512)
```

Implement Eq. (1) and Eq. (2) from the paper.

The exact U-Net topology, attention-head count, FFN width, activation, dropout, and unspecified normalization details must be marked as assumptions.

## 6. VISION DECODER

VD contains Appearance Decoder (AD) and Mask Decoder (MD), both receiving `I_loc`.

### 6.1 Appearance Decoder

`I_pre = AD(I_loc)`, expected output `I_pre ∈ R^(3×224×224)`.

Paper: U-Net decoder + appearance reconstruction convolution module.

### 6.2 Mask Decoder

`M_pre = MD(I_loc)`, with `M_pre ∈ R^(2×224×224)`.

Critical requirement: the U-Net decoder in MD uses the **same network and same weights** as AD's U-Net decoder. Implement true parameter sharing, not two independent identical decoders.

## 7. APPEARANCE RESIDUAL

Paper explicitly defines:

```python
I_r = torch.abs(I_pre - I)
```

Expected shape: `3×224×224`.

Do not replace it with signed subtraction or squared residual.

## 8. RESIDUAL ENCODER

RE has the same architecture **and same weights** as IE.

Do not instantiate an independent encoder. Use actual shared parameters.

```text
I_g  = shared_IE(I)
I_rg = shared_IE(I_r)
I_v  = I_rg + I_g
```

`I_v ∈ R^(1×512)`.

Add a test proving parameter identity/sharing.

## 9. HIERARCHICAL TEXT PROMPTS

Implement the Fine-grained Text Generator behavior as far as the paper supports.

L1 authenticity: real/fake.

L2 manipulation category: real face, entire synthesized face, identity swapped face, attribute manipulated face.

L3 source family: diffusion-based model or GAN-based model.

L4 source generator, e.g. `The source generative model of this photo is DDPM`.

Generators shown/described include DDPM, LatDiff, CollDiff, StyleGAN3, DiffFace, FSLSD, FaceSwapper, Diffae, LatentTransformer, and IA-FaceS.

The paper does not clearly define how L1-L4 are packed into the single `n=308` sequence. Make packing configurable and document the selected strategy as an assumption.

## 10. TOKENIZATION AND LANGUAGE EMBEDDINGS

Tokenize `T` to `T_tok^e ∈ R^n`, `n=308`.

Vocabulary matrix: `W_voc ∈ R^(49408×512)`.

Token embeddings: `T_low^e ∈ R^(308×512)`.

Then:

```text
T_1^tra = T_low^e + P_e
```

with `P_e ∈ R^(308×512)`.

Tokenizer implementation, vocab file, padding token, BOS/EOS, truncation, and exact prompt packing are not fully specified. Keep tokenization behind a clean interface and document assumptions.

## 11. LANGUAGE ENCODER

LE has `E=12` blocks.

Input `T_1^tra ∈ R^(n×d)`, output `T_hig^e ∈ R^(n×d)`.

Per LE block:

```text
T_tok^j = MHA_j^e(LN_j^e(T_tra_j)) + T_tra_j
T_add^j = T_glo^j W_fc^j + T_tok^j
T_tra_(j+1) = FF_j^e(T_add^j) + T_add^j
```

VIM is applied between the MHA step and the FF step.

Global language embedding:

```text
T_l = LAST TOKEN of T_hig^e
```

`T_l ∈ R^(1×512)`.

Do not replace this with mean pooling or an invented CLS token.

## 12. VISION INJECTION MODULE (VIM)

Implement Eq. (4)-(11) exactly.

Inputs: language `T_tok^j ∈ R^(n×d)` and visual `I_v ∈ R^(1×d)`.

### Q/K/V

```text
q_j = T_tok^j W_que^j
k_j = I_v W_key^j
v_j = I_v W_val^j
```

Therefore:

- Language = Query
- Vision = Key + Value

Do not reverse them.

Partition into `r` heads:

```text
Q_j,i ∈ R^(n×d/r)
K_j,i ∈ R^(1×d/r)
V_j,i ∈ R^(1×d/r)
```

The paper does not specify `r`; expose `vim.num_heads` in config and mark the chosen value as an assumption.

Per-head cross-attention:

```text
T_glo_(j,i) = softmax(Q_(j,i) K_(j,i)^T / sqrt(d/r)) V_(j,i)
```

Concatenate heads:

```text
T_glo^j = Cat(T_glo_(j,1), ..., T_glo_(j,r))
```

Then:

```text
T_add^j = T_glo^j W_fc^j + T_tok^j
```

Use the visual global/class feature `I_v` as K/V, not all patch tokens.

## 13. LANGUAGE DECODER

LD has `D=7` blocks and reconstructs the original text.

Inputs: `T_low^e`, `T_hig^e`, `I_v`.

Prepare decoder input by inserting/using a begin token, removing the final token so length remains `n`, then adding decoder positional embeddings `P_d`. BOS/padding details are not fully specified and must be documented.

Per LD block, preserve exact paper order:

```text
T_tj^mmha = MMHA_j^d(T_tj^tra) + T_tj^tra
T_tj^mha  = MHA_j^d(T_tj^mmha, T_hig^e) + T_tj^mmha
T_tj^ff   = FF_j^d(T_tj^mha) + T_tj^mha
T_t(j+1)^tra = VIM_j^d(T_tj^ff, I_v) + T_tj^ff
```

MMHA must be causal.

After D blocks: `T_rec^d ∈ R^(n×d)`.

Vocabulary projection:

```text
T_pre = T_rec^d W_voc^T
```

so `T_pre ∈ R^(n×49408)`.

## 14. ADAPTER

Map `I_v ∈ R^(1×512)` through a minimal adapter / FC mapping to `T_lpre ∈ R^(1×512)`.

`T_lpre` is a predicted global language feature derived from visual features. It is not an actual token, prompt, or sentence.

The exact adapter topology is not fully specified; keep it minimal and document assumptions.

## 15. FORGERY DETECTION HEAD

Feed `I_v` into an MLP/FC head to produce `y_pre` for real/fake classification.

The paper does not fully specify the classifier internals. Use the smallest reasonable implementation and mark assumptions.

## 16. LOSS FUNCTIONS

Implement all six losses:

```text
L = L_fd + L_lr + L_cmc + L_fl + L_ar + L_kl
```

Paper shows coefficient 1 for all terms. Default YAML weights must therefore all be `1.0`.

## 17. APPEARANCE RECONSTRUCTION LOSS

Eq. (17):

```text
L_ar = (1/b) Σ_u (I^u - I_pre^u)^2
```

Pixel/channel reduction details are unspecified. Choose a reasonable numerically stable reduction and document it as an assumption.

## 18. FORGERY LOCALIZATION LOSS

Eq. (18):

```text
L_fl = (1/b) Σ_u -(M^u)^T log(M_pre^u)
```

Ground-truth mask creation for AM/FS:

1. fake image
2. corresponding source image
3. absolute RGB pixel difference
4. grayscale
5. divide by 255
6. threshold at 0.1

Real image mask = 0. Entire synthesized face mask = 1.

Paper ambiguity: `M_pre ∈ R^(2×224×224)` while `M ∈ R^(224×224)`. Do not hide this. Choose a consistent formulation and document it as `ASSUMPTION_FROM_PAPER_GAP`.

## 19. KL FEATURE ALIGNMENT LOSS

```text
I_v -> Adapter -> T_lpre
T_hig^e -> last token -> T_l
```

Both are `1×512`.

KL temperature = `0.5`.

Define:

```text
P = softmax(T_l / 0.5)
Q = softmax(T_lpre / 0.5)
```

Implement Eq. (19) direction exactly:

```text
L_kl = D_KL(P || Q)
```

Do not reverse it.

## 20. CROSS-MODAL CONTRASTIVE LOSS

For batch pairs `{(I_v^u, T_l^u)}` build the full `B×B` similarity matrix. Diagonal pairs are positives; off-diagonal are negatives according to the one-hot pairing target.

### Critical similarity rule

The paper explicitly states `sim(·)` performs a **dot product**. Therefore default implementation must be:

```python
similarity = I_v @ T_l.T
```

Do not silently replace this with cosine similarity or L2 normalization. If normalized cosine is offered as an experimental option, it must be off by default and clearly marked as an assumption/experimental variant.

### Trainable temperature

`tau` is trainable and initialized to `0.07`.

Use a numerically safe positive parameterization if desired, but document it.

### Vision -> Language

```text
logits_v2l = similarity_matrix / tau
S_v2l = softmax(logits_v2l, dim=language_dimension)
```

Targets correspond to `[0,1,...,B-1]`.

### Language -> Vision

```text
logits_l2v = similarity_matrix.T / tau
```

Then:

```text
L_cmc = (L_v2l + L_l2v) / 2
```

Match Eq. (20)-(23); do not simply import a generic CLIP loss without checking equivalence.

## 21. LANGUAGE RECONSTRUCTION LOSS

Paper:

```text
T_pre = T_rec^d W_voc^T
```

Use a numerically stable token-level cross-entropy equivalent to Eq. (25). Padding/ignore-index handling is not specified and must be documented. Ensure causal shifting is correct.

## 22. FORGERY DETECTION LOSS

Use a numerically stable two-class cross-entropy equivalent to Eq. (26):

```text
y_pre = MLP(I_v)
```

No label smoothing unless explicitly experimental.

## 23. TOTAL LOSS

Return:

```python
{
    "total": ...,
    "fd": ...,
    "lr": ...,
    "cmc": ...,
    "fl": ...,
    "ar": ...,
    "kl": ...,
}
```

## 24. FULL TRAINING FORWARD

Implement `model.forward_train(...)` and expose optional intermediate representations:

```text
I_loc, I_pre, I_r, I_g, I_rg, I_v,
T_low, T_hig, T_l, T_lpre, T_rec,
text_logits, mask_logits, detection_logits
```

Avoid retaining unnecessary intermediates in normal training mode.

## 25. INFERENCE MUST NOT REQUIRE TEXT

At test time, no text prompt is required.

```text
I
-> IE
-> I_loc + I_g
-> VD
-> I_pre + M_pre
-> I_r = |I_pre - I|
-> shared IE/RE
-> I_rg
-> I_v = I_g + I_rg
-> MLP
-> y_pre
```

Return detection logits, mask, and optionally reconstruction.

Do not invoke tokenizer, LE, LD, VIM language path, Adapter-for-KL, CMC, or language reconstruction during standard inference.

Implement separate `forward_train(...)` and `forward_inference(images)` methods.

## 26. DATASET

Implement a GenFace-compatible dataset interface returning conceptually:

```python
{
    "image": I,
    "label": y,
    "mask": M,
    "prompt": T,
    "generator": generator_name,
    "forgery_type": forgery_type,
    "source_image": optional_source,
}
```

Do not invent a GenFace folder layout. Make mapping configurable. If real dataset structure is unavailable, create a `DummyMFVLRDataset` for unit tests only.

## 27. MASK GENERATOR

Implement a separate mask-generation utility using the paper's absolute-difference -> grayscale -> /255 -> threshold 0.1 pipeline. Real masks are zeros; entire synthesized masks are ones. Unspecified grayscale details must be documented.

## 28. METRICS

Implement paper metrics:

- Detection: ACC, AUC
- Localization: mIoU

Do not claim AP or other metrics are part of the paper unless found in the source.

## 29. CONFIGURATION FILE

Create `configs/mfvlr.yaml` with paper-specified defaults and explicit assumption fields. Example:

```yaml
model:
  image_size: 224
  local_channels: 1024
  local_height: 14
  local_width: 14
  embed_dim: 512
  image_transformer_blocks: 4
  language_encoder_blocks: 12
  language_decoder_blocks: 7
  vocab_size: 49408
  max_text_tokens: 308
  num_classes: 2

  # ASSUMPTION_FROM_PAPER_GAP
  image_attention_heads: null
  language_attention_heads: null
  vim_heads: null

prompt:
  levels: [1, 2, 3, 4]
  # ASSUMPTION_FROM_PAPER_GAP
  packing_strategy: null

loss:
  lambda_fd: 1.0
  lambda_lr: 1.0
  lambda_cmc: 1.0
  lambda_fl: 1.0
  lambda_ar: 1.0
  lambda_kl: 1.0
  kl_temperature: 0.5

cmc:
  initial_temperature: 0.07
  trainable_temperature: true
  similarity: dot_product
  l2_normalize: false

training:
  batch_size: 8
  optimizer: adam
  learning_rate: 1.0e-4
  weight_decay: 1.0e-3
  scheduler:
    type: step
    step_size: 15
    gamma: 0.1

  # NOT SPECIFIED BY PAPER
  epochs: null

  amp: true
  num_workers: 4
  seed: 42
```

Engineering defaults such as `num_workers`, `seed`, and AMP behavior must not be presented as paper-specified.

## 30. REPRODUCTION NOTES

Create `docs/reproduction_notes.md` with:

```text
| Component | Paper specifies? | Implementation | Reason/Assumption |
```

Include U-Net encoder/decoder, IE/RE sharing, AD/MD decoder sharing, image Transformer, all attention-head counts, FF dimensions, activations, dropout, tokenizer, vocabulary, L1-L4 packing, BOS/EOS/PAD, positional/class-token initialization, mask activation, L_fl formulation, L_ar reduction, language reconstruction padding, CMC similarity and temperature, KL, adapter, classifier, preprocessing, augmentation, epochs, checkpoint selection, and inference threshold.

## 31. ARCHITECTURE DOCUMENTATION

Create `docs/architecture.md` showing training and inference data flow, including:

```text
Image: I -> IE -> I_loc -> VD -> I_pre -> residual -> shared RE -> I_rg -> + I_g -> I_v
Text: Prompt -> tokens -> T_low -> LE + VIM(I_v) -> T_hig -> T_l
Reconstruction: T_low + T_hig + I_v -> LD -> T_rec -> T_pre
Alignment: I_v -> Adapter -> T_lpre <-> KL <-> T_l
Contrastive: I_v <-> T_l via CMC
Detection: I_v -> MLP -> Real/Fake
Localization: I_loc -> MD -> M_pre
Inference: image only
```

## 32. REQUIRED UNIT TESTS

Implement tests for:

1. IE: `[2,3,224,224] -> I_loc [2,1024,14,14], I_g [2,512]`
2. VD: `I_pre [2,3,224,224], M_pre [2,2,224,224]`
3. Residual shape and non-negativity
4. Actual IE/RE parameter sharing
5. Actual AD/MD decoder sharing
6. VIM: language `[2,308,512]`, vision `[2,1,512]`, output `[2,308,512]`
7. LE: `T_hig [2,308,512]`, `T_l [2,512]`, and verify last-token extraction
8. LD: `T_rec [2,308,512]`, text logits `[2,308,49408]`, causal masking
9. Adapter: `[2,512] -> [2,512]`
10. CMC: similarity matrix `[B,B]`, trainable tau, no default L2 normalization
11. KL finite scalar
12. All six losses finite
13. Full training forward
14. `total_loss.backward()` with no unexpected NaN/Inf gradients
15. Image-only inference with no text input

## 33. DEBUGGING FEATURES

Optional debug mode should log major tensor shapes, CMC similarity matrix, trainable `tau`, each of the six losses, and total loss. Detach only copies used for logging.

## 34. TRAINING SCRIPT

`train.py` must support:

```bash
python train.py --config configs/mfvlr.yaml
```

Include config loading, dataset/DataLoader, model, optimizer, scheduler, AMP, forward_train, six losses, backward, optimizer step, validation, checkpointing, resume, and logging.

Do not automatically launch long training during implementation.

## 35. EVALUATION

`evaluate.py` must load checkpoints and compute ACC, AUC, and mIoU using image-only inference.

## 36. INFERENCE SCRIPT

Support:

```bash
python inference.py --image path/to/image.jpg --checkpoint path/to/model.pt
```

Output detection score/logits, predicted class, localization mask, and optionally reconstructed appearance image. If threshold is unspecified by the paper, expose it as config/CLI and document that it is not paper-specified.

## 37. DO NOT MAKE THESE MISTAKES

Do not:

1. Replace dot-product CMC with cosine by default.
2. L2-normalize CMC features unless explicitly experimental.
3. Reverse VIM Q/K/V: Q=language, K/V=I_v.
4. Use all image patch tokens as VIM K/V.
5. Replace last-token `T_l` with mean pooling.
6. Give IE and RE independent weights.
7. Give AD and MD independent decoder weights.
8. Require text during inference.
9. Reverse KL direction.
10. Invent loss weights.
11. Claim unspecified U-Net details are paper-authentic.
12. Claim an unspecified tokenizer is paper-authentic.
13. Claim unspecified attention-head counts are from the paper.
14. Invent an epoch count and call it paper-specified.
15. Silently use pretrained weights because standard implementations do.
16. Treat `pretrained=False` as a complete interpretation of "trained from scratch".
17. Hide ambiguities.
18. Optimize/simplify the architecture before obtaining a faithful baseline.

## 38. DEVELOPMENT PROCESS

Work phase-by-phase.

### PHASE 1
Read the PDF and `docs/paper_spec.md`. Verify the specification. Create/update `docs/reproduction_notes.md`. Stop and report paper-specified details, required assumptions, and planned architecture. Do not implement the model yet if critical contradictions exist.

### PHASE 2
Create repository skeleton, config system, dummy dataset, and utilities. Run import tests.

### PHASE 3
Implement U-Net encoder assumption, image tokenization, image Transformer, Image Encoder, shared Residual Encoder, and MVE. Run `tests/test_mve.py` and stop until it passes.

### PHASE 4
Implement shared U-Net decoder, Appearance Decoder, Mask Decoder, and residual generation. Run `tests/test_vision_decoder.py`.

### PHASE 5
Implement VIM exactly from Eq. (4)-(11). Run `tests/test_vim.py` and inspect tensor shapes.

### PHASE 6
Implement token embedding interface, Language Encoder, `T_hig`, and last-token `T_l`. Run `tests/test_language_encoder.py`.

### PHASE 7
Implement shifted decoder input, causal MMHA, encoder-decoder MHA, FF, VIM, Language Decoder, and vocabulary projection. Run `tests/test_language_decoder.py`.

### PHASE 8
Implement Adapter, Detection head, all six losses, and total loss. Run `tests/test_losses.py`.

### PHASE 9
Integrate `MFVLR.forward_train()`. Run a full dummy batch and `total_loss.backward()`; verify gradients.

### PHASE 10
Implement `forward_inference()` and confirm no text is required.

### PHASE 11
Implement `train.py`, `evaluate.py`, and `inference.py`.

### PHASE 12
Perform a complete code-vs-paper audit and create `docs/final_reproduction_audit.md` with:

```text
| Component | Paper | Code | Exact/Assumption | Notes |
```

## 39. IMPORTANT STOP CONDITIONS

If a missing paper detail materially changes architecture or mathematics:

1. identify it
2. inspect `paper_spec.md`
3. inspect the original PDF again
4. if still unspecified, mark `ASSUMPTION_FROM_PAPER_GAP`
5. choose the smallest reasonable implementation
6. make it configurable
7. document it

If two interpretations are equally plausible, implement the simpler baseline and document the alternative.

## 40. FIRST ACTION NOW

Start with **PHASE 1 ONLY**.

Read:

```text
paper/2605.10071v1.pdf
docs/paper_spec.md
```

Then inspect the existing repository and create/update `docs/reproduction_notes.md`.

Report:

1. Fully paper-specified architecture details
2. Details requiring assumptions
3. Proposed assumption for each missing detail
4. Exact tensor flow through MVE
5. Exact tensor flow through FLT
6. Exact VIM Q/K/V
7. Exact six losses
8. Expected tensor shapes
9. Potential implementation risks

Do **not** implement MFVLR yet. Wait until PHASE 1 analysis is complete before proceeding to PHASE 2.
