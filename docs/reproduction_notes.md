# MFVLR reproduction notes

## Phase status and sources

**Status: PHASE 11 COMPLETE.**
- Scaffolding, configuration system, DummyMFVLRDataset, utilities, and tests verified (Phase 2).
- Multi-domain Vision Encoder (MVE) verified (Phase 3).
- Vision Decoder (VD) verified (Phase 4).
- Vision Injection Module (VIM) implementing Eq. (4)-(11) verified (Phase 5).
- Language Encoder (LE) with E=12 blocks, token embeddings, positional embeddings, VIM injection, T_hig^e, and last-token T_l verified (Phase 6).
- Language Decoder (LD) with D=7 blocks, shifted BOS input, causal MMHA, cross-attention using full T_hig^e, VIM injection, T_rec^d, and tied W_voc^T vocabulary projection verified (Phase 7).
- FLT wrapper, Adapter, Detection Head, all six losses (L_fd, L_lr, L_cmc, L_fl, L_ar, L_kl), and total loss verified (Phase 8).
- Full End-to-End MFVLR Model Orchestration, loss integration, weight sharing preservation (IE/RE, AD/MD trunk, W_voc), exact residual/fusion equality verification, and full forward/backward pass verified (Phase 9).
- Phase 10 training orchestration, Adam optimizer (lr=1e-4, wd=1e-3, including CMC log_tau), StepLR (step_size=15, gamma=0.1), image-only inference path (forward_image_only, FLT bypass verified), evaluation pipeline (ACC, AUC, mIoU), and full checkpoint save/load integration verified (84/84 tests passing) (Phase 10).
- Phase 11 generic manifest dataset interface, tokenizer boundary ($n=308, s=49408$), configurable label mapping, train.py (with --dry-run and --resume), evaluate.py (image-only), infer.py (single image with optional mask export), and complete test suite (93/93 tests passing in 89.43s) verified (Phase 11).

Sources inspected:

1. `paper/2605.10071v1.pdf` - the authoritative source (18 pages).
2. `docs/paper_spec.md` - the extracted specification.
3. `MFVLR_Codex_Reproduction_Prompt.md` - reproduction constraints and phase plan.

### Source reconciliation

**PAPER_SPECIFIED:** No material conflict was found between the PDF and `paper_spec.md` for architecture, tensor shapes, sharing requirements, equations, training/inference separation, or reported hyperparameters. The PDF remains authoritative.

**ASSUMPTION_FROM_PAPER_GAP:** The printed CMC equations use a free `m` index in their numerators, and the mask loss pairs a single-channel target `M` with `M_pre ∈ R^(2×224×224)`. These are paper ambiguities, not conflicts with the spec. Any code must expose and document the selected numerically stable interpretation.

## PAPER_SPECIFIED: architecture plan

MFVLR has three components:

1. **Multi-domain Vision Encoder (MVE):** Image Encoder (IE) + Residual Encoder (RE).
2. **Vision Decoder (VD):** Appearance Decoder (AD) + Mask Decoder (MD).
3. **Fine-grained Language Transformer (FLT):** Language Encoder (LE) + Language Decoder (LD), both using Vision Injection Modules (VIMs).

At training time, the inputs are image `I`, hierarchical prompt(s) `T`, detection label `y`, and localization mask `M`. At standard inference, the model is image-only: MVE + VD + detection MLP; no tokenizer, LE, LD, VIM language path, adapter, CMC, KL, or language reconstruction is invoked.

### MVE tensor flow

**PAPER_SPECIFIED:**

```text
I: [batch, 3, 224, 224]
  -> IE U-Net encoder
  -> I_loc: [batch, 1024, 14, 14]
      -> VD
      -> flattened/projected image tokens + learnable class token + P_i
      -> I_tok / I_1^tra: [batch, 197, 512]
      -> B=4 image Transformer blocks
      -> class token I_g: [batch, 1, 512]

I_loc -> AD -> I_pre: [batch, 3, 224, 224]
I_loc -> MD -> M_pre: [batch, 2, 224, 224]
I_r = |I_pre - I|: [batch, 3, 224, 224]
I_r -> RE (same architecture AND same weights as IE) -> I_rg: [batch, 1, 512]
I_v = I_rg + I_g: [batch, 1, 512]
I_v -> adapter -> T_lpre: [batch, 1, 512]
I_v -> MLP head -> y_pre: two-class detection logits (exact output layout not stated)
```

The paper equations are:

```math
I_{tok}=\operatorname{App}(\operatorname{Proj}(\operatorname{Flat}(I_{loc})))
\in \mathbb{R}^{(hw+1)\times d},\qquad
I^{tra}_1=I_{tok}+P_i. \tag{1}
```

```math
\operatorname{TE}(I^{tra}_1)=
\operatorname{TB}^{i}_{B}\circ\operatorname{TB}^{i}_{B-1}\circ\cdots\circ
\operatorname{TB}^{i}_{1}(I^{tra}_1)=I_{TE}. \tag{2}
```

`I_g` is the class token of `I_TE`; `I_v=I_{rg}+I_g`.

**PAPER_SPECIFIED sharing constraints:** RE must be the IE architecture **with the same weights**, not an independent copy. MD's U-Net decoder must be AD's U-Net decoder **with the same network and weights**, while AD and MD have their respective reconstruction/localization convolution modules.

### VD and residual flow

**PAPER_SPECIFIED:**

```math
I_{pre}=\operatorname{AD}(I_{loc}),\qquad
M_{pre}=\operatorname{MD}(I_{loc})\in\mathbb{R}^{f\times224\times224},\qquad
I_r=|I_{pre}-I|,
```

where `f=2`. AD is a U-Net decoder followed by an appearance reconstruction module with a convolutional layer. MD is a U-Net decoder followed by a manipulation-localization module with a convolutional layer.

### FLT tensor flow

**PAPER_SPECIFIED:**

```text
T
  -> tokenizer -> T_tok^e: [batch, 308]
  -> vocabulary embedding W_voc: [49408, 512]
  -> T_low^e: [batch, 308, 512]
  -> + P_e -> T_1^tra: [batch, 308, 512]
  -> LE, E=12 blocks with VIM(I_v)
  -> T_hig^e: [batch, 308, 512]
  -> final sequence token -> T_l: [batch, 1, 512]

T_low^e shifted with a begin token, final token removed, + P_d
  -> T_t1^tra: [batch, 308, 512]
  -> LD, D=7 blocks with causal MMHA, MHA(T_hig^e), FF, VIM(I_v)
  -> T_rec^d: [batch, 308, 512]
  -> T_pre = T_rec^d W_voc^T: [batch, 308, 49408]
```

LE uses `E=12` Transformer blocks and takes the last token, not a mean pool or an invented language class token, as global language feature:

```math
T^{tra}_1=T^e_{low}+P_e,
```

```math
\operatorname{LE}(T^{tra}_1)=
\operatorname{TB}^{e}_{E}\circ\operatorname{TB}^{e}_{E-1}\circ\cdots\circ
\operatorname{TB}^{e}_{1}(T^{tra}_1)=T^e_{hig}. \tag{3}
```

For LE block `j`:

```math
T^j_{tok}=\operatorname{MHA}^{e}_{j}(\operatorname{LN}^{e}_{j}(T^{tra}_{j}))+T^{tra}_{j},
```

```math
T^j_{add}=T^j_{glo}W^j_{fc}+T^j_{tok},\qquad
T^{tra}_{j+1}=\operatorname{FF}^{e}_{j}(T^j_{add})+T^j_{add}.
```

LD uses `D=7` Transformer blocks:

```math
\operatorname{LD}(T^{tra}_{t1},T^e_{hig},I_v)=
\operatorname{TB}^{d}_{D}\circ\cdots\circ
\operatorname{TB}^{d}_{1}(T^{tra}_{t1},T^e_{hig},I_v)=T^d_{rec}. \tag{12}
```

For LD block `j`, the paper order is fixed:

```math
T^{mmha}_{tj}=\operatorname{MMHA}^{d}_{j}(T^{tra}_{tj})+T^{tra}_{tj}, \tag{13}
```

```math
T^{mha}_{tj}=\operatorname{MHA}^{d}_{j}(T^{mmha}_{tj},T^e_{hig})+T^{mmha}_{tj}, \tag{14}
```

```math
T^{ff}_{tj}=\operatorname{FF}^{d}_{j}(T^{mha}_{tj})+T^{mha}_{tj}, \tag{15}
```

```math
T^{tra}_{t(j+1)}=\operatorname{VIM}^{d}_{j}(T^{ff}_{tj},I_v)+T^{ff}_{tj}. \tag{16}
```

Thus VIM is between MHA and FF in LE, and after MMHA, MHA, and FF in LD.

### Hierarchical prompts

**PAPER_SPECIFIED:** For each image, GenFace hierarchy produces L1-L4 text prompts.

| Level | Exact form / alternatives shown in the paper |
|---|---|
| L1 | `A photo of a real face` or `A photo of a fake face` |
| L2 | real, entire synthesized, identity swapped, or attribute manipulated face |
| L3 | real; a diffusion-based model; or a GAN-based model |
| L4 | real; or `The source generative model of this photo is [generator]` |

Examples shown/described include DDPM, LatDiff, CollDiff, StyleGAN3, DiffFace, FSLSD, FaceSwapper, Diffae, LatentTransformer, and IA-FaceS. The paper uses EFS, AM, and FS categories. It reports that L1-L4 together is the main prompt configuration, and only L1-L2 are used for the FF++ cross-dataset experiment because that data has limited labels.

## PAPER_SPECIFIED: exact VIM Q/K/V

VIM accepts word features `T_tok^j` and the single global fused visual feature `I_v`. Language provides the query; vision provides both key and value. All three projection matrices are `R^(d×d)`:

```math
q_j=T^j_{tok}W^j_{que}, \tag{4}
```

```math
k_j=I_vW^j_{key}, \tag{5}
```

```math
v_j=I_vW^j_{val}. \tag{6}
```

For `r` heads:

```math
\{Q_{j,i}\in\mathbb{R}^{n\times d/r}\}_{i=1}^{r}=\operatorname{Pa}(q_j), \tag{7}
```

```math
\{K_{j,i}\in\mathbb{R}^{1\times d/r}\}_{i=1}^{r}=\operatorname{Pa}(k_j),\qquad
\{V_{j,i}\in\mathbb{R}^{1\times d/r}\}_{i=1}^{r}=\operatorname{Pa}(v_j). \tag{8--9}
```

```math
T^{glo}_{j,i}=\delta\left(\frac{Q_{j,i}K^T_{j,i}}{\sqrt{d/r}}\right)V_{j,i}, \tag{10}
```

```math
T^j_{glo}=\operatorname{Cat}(\{T^{glo}_{j,i}\}_{i=1}^{r})\in\mathbb{R}^{n\times d},\qquad
T^j_{add}=T^j_{glo}W^j_{fc}+T^j_{tok}. \tag{11}
```

**PAPER_SPECIFIED:** Only `I_v` (the visual class/global feature), not all image/patch tokens, is used as VIM K/V. The paper explicitly motivates this as linear rather than quadratic attention-map cost.

## PAPER_SPECIFIED & ASSUMPTION Audit: All Six Losses & Total Objective

All six loss terms appear in the paper with unit coefficient (Eq. 27):

```math
\mathcal{L} = \lambda_{fd}\mathcal{L}_{fd} + \lambda_{lr}\mathcal{L}_{lr} + \lambda_{cmc}\mathcal{L}_{cmc} + \lambda_{fl}\mathcal{L}_{fl} + \lambda_{ar}\mathcal{L}_{ar} + \lambda_{kl}\mathcal{L}_{kl}, \tag{27}
```
where the paper sets $\lambda_{fd} = \lambda_{lr} = \lambda_{cmc} = \lambda_{fl} = \lambda_{ar} = \lambda_{kl} = 1.0$.

### Detailed Per-Loss Audit

#### 1. Forgery Detection Loss ($\mathcal{L}_{fd}$)
- **Exact Paper Equation (Eq. 26):**
  $$\mathcal{L}_{fd} = \frac{1}{b}\sum_{u=1}^b -(y^u)^T \log(y_{pre}^u)$$
- **Tensor Inputs:** Predicted detection logits/probabilities $y_{pre}$, target detection label $y$.
- **Tensor Shapes:** $y_{pre} \in \mathbb{R}^{B \times 2}$, target $y \in \{0, 1\}^B$ (or one-hot $[B, 2]$).
- **Reduction:** Mean over batch dimension $b$.
- **Temperature:** N/A.
- **Direction / Order:** Standard cross-entropy between target label $y$ and prediction $y_{pre}$.
- **Status Breakdown:**
  - `PAPER_SPECIFIED`: 2-class formulation, one-hot ground-truth $y \in \{[0, 1]^T, [1, 0]^T\}$, batch mean reduction.
  - `ASSUMPTION_FROM_PAPER_GAP`: Detection head outputs unnormalized logits; loss computed via `F.cross_entropy(y_pre, y_target)` for numerical stability.

#### 2. Language Reconstruction Loss ($\mathcal{L}_{lr}$)
- **Exact Paper Equations (Eq. 24 & 25):**
  $$T_{pre} = T_{rec}^d W_{voc}^T \in \mathbb{R}^{n \times s} \tag{24}$$
  $$\mathcal{L}_{lr} = \frac{1}{b}\sum_{u=1}^b \sum_{x=1}^n -(T_{gt}^{u,x})^T \log(T_{pre}^{u,x}) \tag{25}$$
- **Tensor Inputs:** Reconstructed language logits $T_{pre}$, target token IDs/one-hot $T_{gt}$.
- **Tensor Shapes:** $T_{pre} \in \mathbb{R}^{B \times 308 \times 49408}$, $T_{gt} \in \{0, \dots, 49407\}^{B \times 308}$ (or one-hot $[B, 308, 49408]$).
- **Reduction:** Sum over sequence length $n = 308$, mean over batch $b$ (or token mean reduction).
- **Temperature:** N/A.
- **Direction / Order:** Cross-entropy between ground-truth prompt tokens $T_{gt}$ and predicted vocabulary logits $T_{pre}$.
- **Status Breakdown:**
  - `PAPER_SPECIFIED`: Linear projection using transposed vocabulary embedding $W_{voc}^T$, vocabulary size $s = 49408$, sequence length $n = 308$, batch mean reduction.
  - `ASSUMPTION_FROM_PAPER_GAP`: Optional `ignore_index` for pad tokens; `F.cross_entropy` applied directly over flattened `[B * n, s]` logits without duplicating vocabulary tensors.

#### 3. Appearance Reconstruction Loss ($\mathcal{L}_{ar}$)
- **Exact Paper Equation (Eq. 17):**
  $$\mathcal{L}_{ar} = \frac{1}{b}\sum_{u=1}^b (I^u - I_{pre}^u)^2 \tag{17}$$
- **Tensor Inputs:** Reconstructed appearance image $I_{pre}$, input image $I$.
- **Tensor Shapes:** $I_{pre} \in \mathbb{R}^{B \times 3 \times 224 \times 224}$, $I \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ in $[0, 1]$.
- **Reduction:** Mean squared error over spatial and channel dimensions, averaged over batch $b$.
- **Temperature:** N/A.
- **Direction / Order:** Symmetric MSE $\|I - I_{pre}\|_2^2$.
- **Status Breakdown:**
  - `PAPER_SPECIFIED`: Squared difference between $I$ and $I_{pre}$, batch mean reduction.
  - `ASSUMPTION_FROM_PAPER_GAP`: Spatial and channel reduction via mean (`reduction="mean"`).

#### 4. Forgery Localization Loss ($\mathcal{L}_{fl}$)
- **Exact Paper Equation (Eq. 18):**
  $$\mathcal{L}_{fl} = \frac{1}{b}\sum_{u=1}^b -(M^u)^T \log(M_{pre}^u) \tag{18}$$
- **Tensor Inputs:** Predicted manipulation mask logits $M_{pre}$, ground-truth binary mask $M$.
- **Tensor Shapes:** $M_{pre} \in \mathbb{R}^{B \times 2 \times 224 \times 224}$, $M \in \{0, 1\}^{B \times 224 \times 224}$.
- **Reduction:** Mean over pixels and batch dimension $b$.
- **Temperature:** N/A.
- **Direction / Order:** 2-class pixel-wise cross-entropy between target binary mask $M$ and prediction $M_{pre}$.
- **Status Breakdown:**
  - `PAPER_SPECIFIED`: Two-class mask logits ($f = 2$), mask spatial dimension $224 \times 224$, pixel-wise negative log-likelihood.
  - `ASSUMPTION_FROM_PAPER_GAP`: $M_{pre}$ output as raw logits; loss evaluated via `F.cross_entropy(M_pre, M.long())`.

#### 5. KL Semantic Alignment Loss ($\mathcal{L}_{kl}$)
- **Exact Paper Equation (Eq. 19):**
  $$\mathcal{L}_{kl} = \frac{1}{b}\sum_{u=1}^b \delta(T_l^u)^T \log\left(\frac{\delta(T_l^u)}{\delta(T_{lpre}^u)}\right) \tag{19}$$
  where $\delta(z) = \operatorname{softmax}(z / \tau_{kl})$ with $\tau_{kl} = 0.5$.
- **Tensor Inputs:** Global language feature $T_l$ (target/teacher distribution $P$), adapter predicted language feature $T_{lpre}$ (predicted/student distribution $Q$).
- **Tensor Shapes:** $T_l \in \mathbb{R}^{B \times 512}$, $T_{lpre} \in \mathbb{R}^{B \times 512}$.
- **Reduction:** Batch mean reduction: $\frac{1}{b}\sum_{u=1}^b D_{KL}(P_u \parallel Q_u)$.
- **Temperature:** $\tau_{kl} = 0.5$ (PAPER_SPECIFIED).
- **Direction / Order:** Strictly $D_{KL}(P \parallel Q) = \sum P \log(P / Q) = \sum P (\log P - \log Q)$ where $P = \operatorname{softmax}(T_l / 0.5)$ and $Q = \operatorname{softmax}(T_{lpre} / 0.5)$.
- **Status Breakdown:**
  - `PAPER_SPECIFIED`: Exact direction $T_l \parallel T_{lpre}$, temperature $\tau_{kl} = 0.5$, batch mean reduction.
  - `ASSUMPTION_FROM_PAPER_GAP`: Numerically stable implementation via `F.kl_div(F.log_softmax(T_lpre / 0.5, dim=-1), F.softmax(T_l / 0.5, dim=-1), reduction="batchmean")`.

#### 6. Cross-Modal Contrastive Loss ($\mathcal{L}_{cmc}$)
- **Exact Paper Equations (Eq. 20–23):**
  $$S_{v2l}^u(I_v, T_l) = \frac{\exp(\operatorname{sim}(I_v^u, T_l^u)/\tau)}{\sum_{m=1}^b \exp(\operatorname{sim}(I_v^u, T_l^m)/\tau)} \tag{20}$$
  $$S_{l2v}^u(T_l, I_v) = \frac{\exp(\operatorname{sim}(T_l^u, I_v^u)/\tau)}{\sum_{m=1}^b \exp(\operatorname{sim}(T_l^u, I_v^m)/\tau)} \tag{21}$$
  $$\mathcal{L}_{v2l} = \frac{1}{b}\sum_{u=1}^b -\log(S_{v2l}^u(I_v, T_l)), \quad \mathcal{L}_{l2v} = \frac{1}{b}\sum_{u=1}^b -\log(S_{l2v}^u(T_l, I_v)) \tag{22-23}$$
  $$\mathcal{L}_{cmc} = \frac{\mathcal{L}_{v2l} + \mathcal{L}_{l2v}}{2}$$
- **Tensor Inputs:** Fused visual global feature $I_v$, global language feature $T_l$.
- **Tensor Shapes:** $I_v \in \mathbb{R}^{B \times 512}$, $T_l \in \mathbb{R}^{B \times 512}$.
- **Similarity:** $\operatorname{sim}(a, b) = a \cdot b^T$ (**strictly DOT PRODUCT, NO L2 normalization**).
- **Temperature:** Trainable parameter $\tau$, initialized at $\tau = 0.07$ (PAPER_SPECIFIED).
- **Reduction:** Mean over batch $b$.
- **Direction / Order:** Bidirectional average: $(\mathcal{L}_{v2l} + \mathcal{L}_{l2v}) / 2$.
- **Status Breakdown:**
  - `PAPER_SPECIFIED`: Unnormalized dot-product similarity matrix $S = I_v T_l^T \in \mathbb{R}^{B \times B}$, trainable temperature initialized to $0.07$, diagonal pairing targets, symmetric bidirectional average.
  - `ASSUMPTION_FROM_PAPER_GAP`: Trainable temperature parameterization as `log_tau` parameter ($\tau = \exp(\log\tau)$) initialized to $\log(0.07)$ to ensure positivity; categorical cross-entropy reduction.

## PAPER_SPECIFIED: reported constants and training facts

| Item | Value |
|---|---:|
| Input `I` / reconstructed image / residual | `3×224×224` |
| `I_loc` | `1024×14×14` |
| Image token sequence | `197×512` (`14×14 + 1` class token) |
| `I_g`, `I_rg`, `I_v`, `T_l`, `T_lpre` | `1×512` |
| `M` | `224×224` |
| `M_pre` | `2×224×224` |
| `T_low^e`, `T_hig^e`, `T_rec^d` | `308×512` |
| `T_pre` and `T_gt` | `308×49,408` |
| Image Transformer blocks `B` | `4` |
| LE blocks `E` | `12` |
| LD blocks `D` | `7` |
| Vocabulary `s` | `49,408` |
| Feature width `d` | `512` |
| Text length `n` | `308` |
| Detection categories `f` | `2` |
| Batch size `b` | `8` |
| CMC temperature | trainable, initial `0.07` |
| KL softmax temperature | `0.5` |
| Ground-truth mask threshold | `0.1` |
| Optimizer | Adam, LR `1e-4`, weight decay `1e-3` |
| Scheduler | learning rate divided by 10 every 15 epochs |
| Framework/hardware reported | PyTorch / Tesla V100 |

For AM/FS masks, paper-specified generation is: fake/source absolute RGB difference -> grayscale -> divide by 255 -> threshold `0.1`. Real masks are all zero; entire synthesized face masks are all one. MFVLR is stated to train from scratch; no pretrained initialization is reported. Paper metrics are ACC and AUC for detection, and mean class-wise IoU for localization.

## ASSUMPTION_FROM_PAPER_GAP: decision ledger

The following are **proposed Phase 2 defaults only**. They are not paper-authentic claims, will be labeled in code/configuration as `ASSUMPTION_FROM_PAPER_GAP`, and remain configurable.

| Component | Paper specifies? | Planned implementation after confirmation | Reason / assumption |
|---|---|---|---|
| IE/RE U-Net encoder topology | No; only U-Net encoder and output `1024×14×14` | Four downsampling stages `224→112→56→28→14`, with candidate channels `128,256,512,1024` and retained skip features | The stated spatial output fixes total stride 16 but not topology/channels. |
| Encoder convolution block / norm / activation | No | Two 3×3 convolutions per stage, GroupNorm, GELU | Small stable baseline; all are configurable assumptions. |
| AD/MD shared decoder trunk | Sharing is yes; topology is no | One decoder-trunk module with four upsampling stages and shared skip inputs; distinct final 3-channel reconstruction and 2-channel mask heads | Preserves required true decoder sharing while keeping specified distinct convolution modules. |
| Image Transformer details | Block count and width are yes; details are no | Pre-norm self-attention, 8 heads, FF width `2048`, GELU, dropout `0.0` | `512` is divisible by 8; the rest is a minimal configurable baseline. |
| LE/LD attention and FF details | Block counts, ordering, and width are yes; details are no | 8 heads, FF width `2048`, GELU, dropout `0.0`; LD cross-MHA uses decoder states as Q and `T_hig^e` as K/V | Standard minimal Transformer interpretation. The cross-MHA Q/K/V assignment is not formally defined by the paper. |
| VIM head count `r` | No | `r=8` configuration default | Required to partition `d=512`; choice is not reported. |
| VIM projections in LD | LD has VIM, but its separate Q/K/V notation/tying is not stated | Give each LD VIM its own projections and apply the same Eq. 4-11 operation to `T_tj^ff` | Follows named `VIM_j^d` in Eq. 16 without assuming LE/LD weight tying. |
| Single-K/V VIM behavior | Equations are yes; their practical consequence is not discussed | Preserve exactly one K/V position; do not add patch tokens or alter attention | With one K/V, per-word softmax is 1, so attention output is value-derived. Faithfulness takes priority over modifying the equation. |
| FTG/tokenizer implementation | No exact tokenizer; paper cites CLIP and fixes `s=49408` | Tokenizer interface with a CLIP-compatible BPE tokenizer having vocabulary size 49,408 | Vocabulary size and CLIP citation motivate this choice, but do not uniquely specify it. |
| L1-L4 packing | No | Candidate default: tokenize L1, L2, L3, L4 individually to 77 tokens and concatenate in L1→L4 order to form 308 tokens; make selectable | `308 = 4×77` and the figure depicts four prompt lines. This is a strong inference, not an explicit paper statement. |
| BOS/EOS/PAD IDs and padding | No | Use the selected CLIP-compatible tokenizer's defined special IDs; right-pad; retain token mask | The paper specifies neither IDs nor padding behavior. |
| Positional/class-token initialization | Learnable tokens/positions are yes; initialization is no | Truncated normal, std `0.02` | Conventional configurable initialization only. |
| Train-from-scratch initialization | Train from scratch yes; distribution is no | No pretrained weights; initialize modules with documented PyTorch/TruncNormal defaults | `pretrained=False` alone does not make an unreported initialization paper-specified. |
| Appearance output range | No | RGB inputs scaled to `[0,1]`; AD output uses sigmoid | Lets `I_pre` share input range for residual/MSE; activation is an assumption. |
| Mask activation and `L_fl` | `M` is one channel, `M_pre` has two channels; formulation unspecified | MD returns two raw logits; convert binary `M` to class indices `{0,1}` and use pixelwise two-class cross-entropy / log-softmax equivalent | Matches output shape and Eq. 18 most directly, but is not fully defined by paper. |
| `L_ar` reduction | Only batch factor is printed | Mean squared error over batch, channels, and pixels | A numerically stable scalar reduction is required; exact spatial/channel reduction is absent. |
| `L_lr` padding reduction | No | Token cross entropy with the chosen pad ID as `ignore_index`; no label smoothing | Prevents padding from training the decoder; not specified. |
| CMC implementation | Dot product, diagonal pairing, trainable `tau=0.07` are yes | Full `[B,B]` unnormalized dot-product matrix; two cross-entropies; parameterize `tau=exp(log_tau)` initialized at `0.07` | Preserves paper mathematics and gives a positive numerical temperature. The parameterization is an assumption. |
| KL implementation | Direction and `0.5` are yes; numerical reduction is no | `P=softmax(T_l/0.5)`, `Q=softmax(T_lpre/0.5)`, compute `D_KL(P || Q)` with batch-mean reduction | Exact direction preserved; framework reduction is an assumption. |
| Adapter | Only an FC layer is specified | One `Linear(512,512)` | Smallest topology consistent with paper. |
| Detection classifier | MLP head containing FC is specified; width/depth is no | One `Linear(512,2)` on squeezed `I_v` | Smallest two-class implementation. |
| Residual preprocessing | Absolute residual is specified; further processing is no | Feed raw nonnegative `abs(I_pre-I)` into shared IE with no frequency/normalization branch | Avoids inventing residual transforms. |
| Grayscale conversion for masks | RGB difference, grayscale, `/255`, threshold are yes; formula is no | RGB luminance weights `(0.299, 0.587, 0.114)`; threshold `>0.1` | A conventional grayscale formula/comparator is needed but not reported. |
| Dataset layout and source pairing | No | Dataset manifest/config maps image, label, generator/type, optional source image; fail clearly if AM/FS source is required but missing | The prompt prohibits inventing a GenFace folder layout. |
| Image preprocessing | Input size is yes; process is no | RGB conversion, resize to 224×224, float `[0,1]`; no default normalization | Minimal reconstruction-compatible baseline; configurable. |
| Augmentation | No | Disabled by default | No paper support for a particular augmentation. |
| Epoch count / validation split / sampler | No | `epochs: null`, supplied explicitly by experiment owner; split/sampler fully configurable | The paper only specifies optimizer and LR schedule. |
| Checkpoint selection | No | If validation exists, select highest validation AUC and log ACC/mIoU; record rule in run metadata | A selection criterion is required but not paper-stated. |
| Inference detection rule | No | Return raw logits/probabilities; default class is `argmax`; optional threshold is explicit config, default `0.5` only when requested | Paper supplies no decision threshold. |
| Inference mask rule / mIoU | Output shape is yes; postprocess is no | Default binary mask is 2-channel `argmax`; calculate class-wise IoU then mean | Consistent with two-class logits and stated mIoU, but thresholding details are absent. |
| AMP, seed, workers, DDP | No | Engineering configuration only, never described as paper hyperparameters | Required by reproduction prompt but not by the paper. |

## Implementation risks requiring tests or explicit review

1. **VIM degeneracy risk:** Eq. 10 has exactly one K/V token per head. Softmax over a length-one key dimension is identically 1, so the cross-attention output is value-only and independent of query/key scores. Adding patch K/V tokens would change the paper; preserving one token is faithful but should be unit-tested and called out.
2. **Localization-loss ambiguity:** `M ∈ R^(224×224)` versus `M_pre ∈ R^(2×224×224)` leaves channel expansion, activation, and reduction unspecified. The selected CE interpretation must be isolated in `localization_loss.py`, marked as an assumption, and tested for finite gradients.
3. **CMC notation ambiguity:** The printed Eq. 20-21 numerator index is not fully disambiguated. The required reproduction interpretation is full batch `[B,B]` dot-product logits with diagonal positive targets, without L2 normalization. Test both directional cross-entropies and trainable positive temperature.
4. **Weight-sharing risk:** RE must be the same IE object/parameters and not a deep copy. AD and MD must reuse the same decoder trunk/parameters. Tests must assert parameter identity, not merely numerical equality at construction time.
5. **Prompt/tokenizer risk:** `s=49,408` and `n=308` constrain but do not fully define the tokenizer or packing. The proposed 4×77 packing is plausible but must be configurable and separately logged in every run.
6. **Reconstruction-range risk:** `I_pre` scale directly affects both `L_ar` and `I_r`. Input scale, AD activation, and residual behavior must be configured consistently and logged.
7. **Training-from-scratch risk:** No pretrained weights, initialization scheme, epoch count, validation protocol, sampling, or augmentation is stated. Report all operational choices as reproduction settings rather than paper results.
8. **Memory/compute risk:** `[B,308,49,408]` language logits are large (for the paper batch size 8, about 1.21e8 values before gradients). AMP and careful loss computation will be engineering aids, but must not alter model mathematics.
9. **Image-path reuse ambiguity:** The paper describes image encoding before reconstruction and MVE processing after residual generation but does not state whether `I_g` is cached or recomputed. Planned baseline: compute IE(I) once, retain `I_loc`/`I_g`, and apply the shared IE to `I_r`; this is an assumption to make explicit in code/comments.
10. **Evaluation risk:** The paper reports ACC, AUC, and mean class-wise IoU, but not data split definitions or output thresholds. Evaluation configuration must record labels, source-mask availability, split, and post-processing.

## Phase 2, 3, 4, 5 & 6 Implementation Summary & Verification

Phase 2 completed artifacts:
- Configuration system: `configs/mfvlr.yaml` and `configs/dataset.yaml` with explicit paper parameters and gap assumptions.
- Dataset utilities: `FineGrainedTextGenerator` (L1-L4 prompt hierarchy), `generate_ground_truth_mask` (Eq. 18 difference pipeline), `DummyMFVLRDataset`, and `GenFaceDataset`.
- Utilities: `set_seed`, `compute_classification_metrics` (ACC, AUC), `compute_localization_metrics` (mIoU), `save_checkpoint`, `load_checkpoint`, `setup_logger`, `save_visualization`.

Phase 3 completed artifacts:
- `models/vision/unet_encoder.py`: `UNetEncoder` extracting $I_{\text{loc}} \in \mathbb{R}^{B \times 1024 \times 14 \times 14}$ and multi-scale skip connections.
- `models/vision/image_transformer.py`: `ImageTransformer` and `ImageTransformerBlock` with $B = 4$ blocks, 8 heads, 2048 feed-forward width, and GELU activation.
- `models/vision/image_encoder.py`: `ImageEncoder` implementing Eq. 1 and Eq. 2 ($I_{\text{loc}} \to \text{flat/proj} \to \text{CLS/pos\_embed} \to \text{Transformer} \to I_g$).
- `models/vision/residual_encoder.py`: `ResidualEncoder` wrapping the shared `ImageEncoder` instance to ensure parameter identity.
- `models/vision/mve.py`: `MultiDomainVisionEncoder` fusing appearance ($I_g$) and residual ($I_{rg}$) features via addition ($I_v = I_g + I_{rg}$).
- `tests/test_mve.py`: 7 unit tests.

Phase 4 completed artifacts:
- `models/vision/unet_decoder.py`: `UNetDecoderTrunk` with 4-stage upsampling ($14 \to 28 \to 56 \to 112 \to 224$), optional multi-scale skip concatenation, and output channels 64.
- `models/vision/appearance_decoder.py`: `AppearanceDecoder` combining shared decoder trunk with `AppearanceHead` (Conv 64 $\to$ 3 + Sigmoid) producing $I_{\text{pre}} \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ in $[0, 1]$.
- `models/vision/mask_decoder.py`: `MaskDecoder` combining shared decoder trunk with `MaskHead` (Conv 64 $\to$ 2) producing raw logits $M_{\text{pre}} \in \mathbb{R}^{B \times 2 \times 224 \times 224}$.
- `models/vision/vision_decoder.py`: `VisionDecoder` managing true parameter-shared trunk, AD, MD, and residual generation ($I_r = |I_{\text{pre}} - I|$).
- `tests/test_vision_decoder.py`: 6 unit tests.

Phase 5 completed artifacts:
- `models/language/vim.py`: `VisionInjectionModule` (and `VIM` alias) implementing Eq. (4)-(11) with Q=Language ($T_{\text{tok}}^j$), K/V=Vision ($I_v$), 8 heads (head_dim=64), scale $1/\sqrt{64}$, output projection $W_{\text{fc}}$, and residual addition ($T_{\text{add}}^j = T_{\text{glo}}^j W_{\text{fc}}^j + T_{\text{tok}}^j$).
- Mathematical Singleton K/V property: Since K/V has sequence length 1 ($I_v$), softmax over the singleton key dimension evaluates identically to 1.0 ($A = [1.0]$). Gradients flow to $W_{\text{val}}$, $W_{\text{fc}}$, $T_{\text{tok}}$, and $I_v$ directly; gradients from softmax back to $Q$ and $K$ are mathematically zero due to singleton softmax derivative ($s(1-s) = 0$).
- `tests/test_vim.py`: 6 unit tests verifying forward shapes, 2D/3D visual input support, singleton attention probabilities, step-by-step tensor shapes, and backward propagation.

Phase 6 completed artifacts:
- `models/language/embeddings.py`: `LanguageEmbeddings` wrapping `nn.Embedding(49408, 512)` ($T_{\text{low}}^e$) and learnable positional embedding $P_e \in \mathbb{R}^{1 \times 308 \times 512}$ ($T_1^{\text{tra}} = T_{\text{low}}^e + P_e$).
- `models/language/language_encoder.py`: `LanguageEncoderBlock` (Pre-LN MHA $\to$ VIM $\to$ Pre-LN FFN) and `LanguageEncoder` with $E=12$ blocks, defining $T_{\text{hig}}^e \in \mathbb{R}^{B \times 308 \times 512}$ strictly as the direct output of the 12th LE block without any final LayerNorm per Eq. (3), and global language representation $T_l = T_{\text{hig}}^e[:, -1, :] \in \mathbb{R}^{B \times 512}$ (strictly the LAST token).
- `tests/test_language_encoder.py`: 7 unit tests verifying token embeddings, $E=12$ block structure, independent VIM instances, tensor preservation, last-token extraction identity (asserting NOT mean/max/first token), singleton K/V behavior, and gradient backpropagation.
- Complete test suite: 47/47 unit tests passing across Phase 2, Phase 3, Phase 4, Phase 5, and Phase 6.

Phase 7 completed artifacts:
- `models/language/language_decoder.py`: `LanguageDecoderBlock` (Pre-LN MMHA $\to$ Pre-LN Cross-MHA($T_{\text{hig}}^e$) $\to$ Pre-LN FFN $\to$ VIM) and `LanguageDecoder` with $D=7$ blocks, shifted input with learnable BOS, decoder positional embedding $P_d \in \mathbb{R}^{1 \times 308 \times 512}$, direct output $T_{\text{rec}}^d \in \mathbb{R}^{B \times 308 \times 512}$ (no final LayerNorm), and tied vocabulary projection $T_{\text{pre}} = T_{\text{rec}}^d W_{\text{voc}}^T \in \mathbb{R}^{B \times 308 \times 49408}$ via `LanguageDecoderOutput`.
- `tests/test_language_decoder.py`: 7 unit tests verifying shifted input preparation (prepend BOS, drop last token), $D=7$ block structure, independent VIM instances, causal attention masking, cross-attention with complete $T_{\text{hig}}^e$, true weight tying with $W_{\text{voc}}^T$, and gradient backpropagation.
- Complete test suite: 54/54 unit tests passing across Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, and Phase 7.

Phase 8 completed artifacts:
- `models/language/flt.py`: `FineGrainedLanguageTransformer` (FLT) integrating Language Encoder ($E=12$) and Language Decoder ($D=7$) with true weight tying $W_{\text{voc}}^T$.
- `models/heads/adapter.py`: `Adapter` mapping visual global feature $I_v \in \mathbb{R}^{B \times 512}$ to predicted language feature $T_{\text{lpre}} \in \mathbb{R}^{B \times 512}$.
- `models/heads/detection_head.py`: `DetectionHead` producing 2-class raw logits $y_{\text{pre}} \in \mathbb{R}^{B \times 2}$ from $I_v$.
- `models/losses/`: Modular loss package implementing all 6 paper objectives and weighted total loss:
  - `detection_loss.py`: $L_{\text{fd}}$ (Eq. 26)
  - `language_reconstruction_loss.py`: $L_{\text{lr}}$ (Eq. 24-25)
  - `appearance_reconstruction_loss.py`: $L_{\text{ar}}$ (Eq. 17)
  - `localization_loss.py`: $L_{\text{fl}}$ (Eq. 18)
  - `kl_loss.py`: $L_{\text{kl}}$ (Eq. 19) with $\tau = 0.5$ and exact $D_{\text{KL}}(P(T_l) \parallel Q(T_{\text{lpre}}))$ direction
  - `cmc_loss.py`: $L_{\text{cmc}}$ (Eq. 20-23) with unnormalized dot product, trainable $\tau = 0.07$, and bidirectional average
  - `total_loss.py`: `MFVLRLoss` (Eq. 27) computing weighted multi-task loss with unit weights
- `tests/test_flt.py`, `tests/test_heads.py`, `tests/test_losses.py`: 13 unit tests verifying FLT, heads, and all six losses.
- Complete test suite: 67/67 unit tests passing across Phase 2 through Phase 8.

Phase 9 completed artifacts:
- `models/mfvlr.py`: `MFVLR` and `MFVLROutput` orchestrating complete end-to-end forward pass composing MVE, Vision Decoder, FLT, Adapter, and Detection Head with structured output container exposing all 14 intermediate/final representations:
  - $I_{\text{loc}} \in \mathbb{R}^{B \times 1024 \times 14 \times 14}$
  - $I_g \in \mathbb{R}^{B \times 512}$
  - $I_{\text{pre}} \in \mathbb{R}^{B \times 3 \times 224 \times 224}$
  - $M_{\text{pre}} \in \mathbb{R}^{B \times 2 \times 224 \times 224}$
  - $I_r = |I_{\text{pre}} - I| \in \mathbb{R}^{B \times 3 \times 224 \times 224}$
  - $I_{rg} \in \mathbb{R}^{B \times 512}$
  - $I_v = I_g + I_{rg} \in \mathbb{R}^{B \times 512}$
  - $y_{\text{pre}} \in \mathbb{R}^{B \times 2}$
  - $T_{\text{lpre}} \in \mathbb{R}^{B \times 512}$
  - $T_{\text{low}} \in \mathbb{R}^{B \times 308 \times 512}$
  - $T_{\text{hig}} \in \mathbb{R}^{B \times 308 \times 512}$
  - $T_l \in \mathbb{R}^{B \times 512}$
  - $T_{\text{rec}} \in \mathbb{R}^{B \times 308 \times 512}$
  - $T_{\text{pre}} \in \mathbb{R}^{B \times 308 \times 49408}$
- Verified full autograd connectivity (no detaches, no NumPy conversion in forward).
- Verified parameter sharing & tying integrity:
  - `model.mve.re is model.mve.ie`
  - `model.vision_decoder.appearance_decoder.decoder_trunk is model.vision_decoder.mask_decoder.decoder_trunk`
  - `model.flt.decoder.token_embedding.weight is model.flt.encoder.embeddings.token_embed.weight`
- Full backward gradient flow confirmed reaching Vision (IE conv, Transformer, AD head, MD head, shared trunk), Language (tied token embed, pos embed, LE self-attn, LE FFN, LE VIM $W_{\text{val}}$, LE VIM $W_{\text{fc}}$, LD MMHA, LD Cross-MHA, LD FFN, LD VIM $W_{\text{val}}$, LD VIM $W_{\text{fc}}$), Heads (Adapter, Detection Head), and Loss parameter (CMC $\log \tau$).
- `tests/test_mfvlr_integration.py`: 5 integration tests.
- Complete test suite: 73/73 tests passing across Phase 2 through Phase 9.

Phase 10 completed artifacts:
- `models/mfvlr.py`:
  - `forward_image_only(image)`: Implemented image-only inference graph ($I \to \text{IE} \to \text{VD} \to I_r \to \text{RE} \to I_v \to \text{Detection Head} \to y_{\text{pre}}, M_{\text{pre}}$) with FLT bypass proof.
  - `MFVLRInferenceOutput`: Structured container with prediction helper methods (`predict_class()`, `predict_fake_prob()`, `predict_mask()`).
- `utils/trainer.py`:
  - `create_optimizer`: Adam (not AdamW), $\text{lr}=10^{-4}$, $\text{weight\_decay}=10^{-3}$, incorporating model parameters and trainable loss parameter ($\log \tau$).
  - `create_scheduler`: StepLR ($\text{step\_size}=15$, $\gamma=0.1$).
  - `train_step`: Executing full forward, multi-task loss computation, backward pass, and optimizer step, returning detached scalar loss dict.
  - `train_one_epoch`: Epoch-level iteration over DataLoader aggregating mean losses.
- `utils/evaluator.py`:
  - `evaluate`: Evaluation pipeline strictly executing image-only forward pass, computing ACC, continuous AUC, and mIoU.
- `utils/checkpoint.py`:
  - `create_checkpoint_state`, `save_checkpoint`, `load_checkpoint`: Full training state persistence restoring model weights, trainable loss parameter (`cmc_loss.log_tau`), optimizer state, scheduler state, and metadata.
- `tests/test_training.py`, `tests/test_inference.py`, `tests/test_evaluation.py`: 11 unit tests.
- Complete test suite: 84/84 tests passing across Phase 2 through Phase 10 in 69.36s.

Phase 11 completed artifacts:
- `datasets/tokenizer.py`: `MFVLRTokenizer` deterministic tokenizer abstraction enforcing $n=308$ tokens, vocabulary bounds $[0, 49407]$, and offline token encoding (`tokenize` functional interface).
- `datasets/mask_generator.py`: Updated `generate_ground_truth_mask` to enforce strict validation raising explicit `ValueError` when source image is missing for manipulated (AM/FS) samples.
- `datasets/manifest_dataset.py`: `MFVLRDataset` supporting generic JSONL, JSON, and CSV manifests with configurable `real_class_index` (default: 0) and `fake_class_index` (default: 1), hierarchical prompt generation, mask generation, and path resolution relative to `dataset_root`.
- `utils/metrics.py` & `utils/evaluator.py`: Updated classification metric evaluation to accept configurable `positive_label` / `fake_class_index` so continuous AUC strictly evaluates fake class probability column `softmax(y_pre)[:, fake_class_index]` without hardcoded class 1 assumptions.
- `train.py`: Runnable training script supporting `--config`, `--resume`, `--dry-run`, `--epochs`, `--batch-size`, `--lr`, `--device`, and `--output-dir`.
- `evaluate.py`: Runnable evaluation entry point executing paper-specified image-only inference (`forward_image_only(image)`), loading checkpoints, and computing ACC, AUC, and mIoU.
- `infer.py`: Single-image inference CLI executing image-only forward pass, reporting logits, class probabilities, predicted semantic label, manipulated area %, and optional `--output-mask` localization map export.
- `configs/mfvlr.yaml`: Updated with manifest dataset entries, configurable class indices, learning rate and scheduler aliases.
- `tests/test_dataset_manifest.py`, `tests/test_label_mapping.py`, `tests/test_cli.py`: 9 new unit tests verifying manifest loading, mask validation, tokenizer contract, inverted label mappings, dry-run smoke test, evaluation CLI, and single-image inference.
- Complete test suite: 93/93 tests passing across all 11 phases in 89.43s.

## Phase 12 Stop Condition

No real GenFace dataset download/training, long-running benchmark training, or result fabrication have been executed. Execution stops here awaiting user confirmation to proceed to PHASE 12.
