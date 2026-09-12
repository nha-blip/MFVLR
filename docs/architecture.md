# MFVLR Architecture Documentation

This document describes the architectural data flow, tensor shapes, module interfaces, and training vs. inference separation for MFVLR.

---

## 1. Overall System Overview

MFVLR consists of three primary components:
1. **Multi-domain Vision Encoder (MVE)**: Extracts appearance features ($I_g$) and residual features ($I_{rg}$) using shared weights.
2. **Vision Decoder (VD)**: Reconstructs appearance image ($I_{\text{pre}}$) and predicts forgery mask ($M_{\text{pre}}$) using a shared decoder trunk.
3. **Fine-grained Language Transformer (FLT)**: Injects global visual features into hierarchical language prompts ($T$) using Vision Injection Modules (VIM) and reconstructs text tokens ($T_{\text{pre}}$).

---

## 2. Training Data Flow

```mermaid
graph TD
    subgraph Vision Pathway
        I[Input Image I: 3x224x224] --> IE[Image Encoder IE]
        IE --> I_loc[Local Feature I_loc: 1024x14x14]
        IE --> I_g[Global Appearance Feature I_g: 1x512]
        
        I_loc --> VD_Trunk[Shared U-Net Decoder Trunk]
        VD_Trunk --> AD_Head[AD Head] --> I_pre[Predicted Appearance I_pre: 3x224x224]
        VD_Trunk --> MD_Head[MD Head] --> M_pre[Predicted Mask M_pre: 2x224x224]
        
        I --> ResGen[Residual Gen: |I_pre - I|]
        I_pre --> ResGen
        ResGen --> I_r[Residual Image I_r: 3x224x224]
        
        I_r --> RE[Residual Encoder RE - Shared IE Weights]
        RE --> I_rg[Global Residual Feature I_rg: 1x512]
        
        I_g --> Fuse((+))
        I_rg --> Fuse
        Fuse --> I_v[Fused Visual Feature I_v: 1x512]
    end

    subgraph Heads
        I_v --> Adapter[Adapter Linear 512->512] --> T_lpre[Predicted Language Feature T_lpre: 1x512]
        I_v --> MLP[Detection Head Linear 512->2] --> y_pre[Detection Logits: 2]
    end

    subgraph Language Pathway
        T[Hierarchical Prompt T] --> Tok[Tokenizer] --> T_low[T_low: 308x512]
        T_low --> LE[Language Encoder: 12 Blocks with VIM]
        I_v -.->|VIM K/V| LE
        LE --> T_hig[High-level Embeddings T_hig: 308x512]
        T_hig --> T_l[Global Language Feature T_l: 1x512 - Last Token]
        
        T_low --> Shift[Shift + BOS] --> T_t1[T_t1: 308x512]
        T_t1 --> LD[Language Decoder: 7 Blocks with MMHA + MHA + VIM]
        T_hig -.->|Cross MHA| LD
        I_v -.->|VIM K/V| LD
        LD --> T_rec[Reconstructed Text T_rec: 308x512]
        T_rec --> VocabProj[Vocab Proj W_voc^T] --> T_pre[Token Logits: 308x49408]
    end

    subgraph Losses
        I & I_pre --> L_ar[L_ar: Appearance Reconstruction]
        M_pre & M[GT Mask M] --> L_fl[L_fl: Forgery Localization]
        T_lpre & T_l --> L_kl[L_kl: Feature Alignment KL]
        I_v & T_l --> L_cmc[L_cmc: Cross-Modal Contrastive]
        T_pre & T_gt[GT Tokens] --> L_lr[L_lr: Language Reconstruction]
        y_pre & y[GT Label y] --> L_fd[L_fd: Forgery Detection]
    end
```

---

## 3. Inference Data Flow (Image-Only)

At test time, **no text prompt is required or processed**. The FLT, adapter, tokenizer, and language losses are completely bypassed:

```mermaid
graph TD
    I[Input Image I: 3x224x224] --> IE[Image Encoder IE]
    IE --> I_loc[Local Feature I_loc: 1024x14x14]
    IE --> I_g[Global Appearance Feature I_g: 1x512]
    
    I_loc --> VD[Vision Decoder]
    VD --> I_pre[Reconstructed Image I_pre: 3x224x224]
    VD --> M_pre[Predicted Localization Mask Logits M_pre: 2x224x224]
    
    I --> ResGen[|I_pre - I|]
    I_pre --> ResGen
    ResGen --> I_r[Residual Image I_r: 3x224x224]
    
    I_r --> RE[Residual Encoder RE - Shared Weights]
    RE --> I_rg[Global Residual Feature I_rg: 1x512]
    
    I_g --> Fuse((+))
    I_rg --> Fuse
    Fuse --> I_v[Fused Visual Feature I_v: 1x512]
    
    I_v --> MLP[Detection Head Linear 512->2]
    MLP --> y_pre[Real/Fake Detection Logits]
```

---

## 4. Vision Injection Module (VIM) Details

VIM enables word-level visual-language interaction using the single fused class token $I_v$:

```math
q_j = T_{\text{tok}}^j W_{\text{que}}^j \in \mathbb{R}^{n \times d}, \quad k_j = I_v W_{\text{key}}^j \in \mathbb{R}^{1 \times d}, \quad v_j = I_v W_{\text{val}}^j \in \mathbb{R}^{1 \times d}
```

- Head partition ($r=8$): $Q_{j, i} \in \mathbb{R}^{n \times 64}, K_{j, i} \in \mathbb{R}^{1 \times 64}, V_{j, i} \in \mathbb{R}^{1 \times 64}$.
- Attention calculation:
  $$T_{j, i}^{\text{glo}} = \operatorname{softmax}\left(\frac{Q_{j, i} K_{j, i}^T}{\sqrt{64}}\right) V_{j, i} \in \mathbb{R}^{n \times 64}$$
- Head concatenation and residual projection:
  $$T_j^{\text{add}} = \operatorname{Cat}(T_{j, 1}^{\text{glo}}, \dots, T_{j, r}^{\text{glo}}) W_{\text{fc}}^j + T_{\text{tok}}^j \in \mathbb{R}^{n \times d}$$

---

## 5. Summary of Mathematical Losses

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fd}} + \mathcal{L}_{\text{lr}} + \mathcal{L}_{\text{cmc}} + \mathcal{L}_{\text{fl}} + \mathcal{L}_{\text{ar}} + \mathcal{L}_{\text{kl}}$$

1. $\mathcal{L}_{\text{ar}} = \frac{1}{B} \sum_{u=1}^B \operatorname{mean}((I^u - I_{\text{pre}}^u)^2)$
2. $\mathcal{L}_{\text{fl}} = \frac{1}{B} \sum_{u=1}^B \operatorname{CrossEntropy}(M_{\text{pre}}^u, M^u)$
3. $\mathcal{L}_{\text{kl}} = \frac{1}{B} \sum_{u=1}^B D_{\text{KL}}(\operatorname{softmax}(T_l^u / 0.5) \parallel \operatorname{softmax}(T_{l\text{pre}}^u / 0.5))$
4. $\mathcal{L}_{\text{cmc}} = \frac{1}{2}(\mathcal{L}_{v2l} + \mathcal{L}_{l2v})$ where logits $= \frac{I_v T_l^T}{\tau}$
5. $\mathcal{L}_{\text{lr}} = \frac{1}{B} \sum_{u=1}^B \sum_{x=1}^n \operatorname{CrossEntropy}(T_{\text{pre}}^{u, x}, T_{\text{gt}}^{u, x})$
6. $\mathcal{L}_{\text{fd}} = \frac{1}{B} \sum_{u=1}^B \operatorname{CrossEntropy}(y_{\text{pre}}^u, y^u)$
