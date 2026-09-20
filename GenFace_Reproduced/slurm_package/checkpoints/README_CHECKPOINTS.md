# MFVLR / GenFace-Reproduced: Checkpoints Placement & Verification Guide

This document describes where to place official checkpoints on the remote SLURM server and how they are verified.

> **CRITICAL RULE:**  
> Missing checkpoint $\implies$ **FAIL** that generator.  
> SHA256 mismatch $\implies$ **FAIL** that generator.  
> Never silently use another checkpoint.

---

## 1. Directory Structure on Server

Place checkpoints under `$CHECKPOINT_ROOT` (defined in `configs/server.env`):

```
$CHECKPOINT_ROOT/
├── EFS/
│   ├── StyleGAN3/
│   │   └── stylegan3-r-ffhq-1024x1024.pkl
│   ├── DDPM/
│   │   └── google_ddpm_celebahq_256/
│   ├── LatDiff/
│   │   └── ffhq_ldm_kl_f4.ckpt
│   └── CollDiff/
│       └── colldiff_ffhq.ckpt
├── AM/
│   ├── IAFaces/
│   │   └── iafaces-celebahq-256.pth
│   ├── LatTrans/
│   │   ├── psp_ffhq_encode.pt
│   │   └── tnet_logs_001/
│   │       ├── tnet_0.pth.tar
│   │       ├── ...
│   │       └── tnet_39.pth.tar
│   └── DiffAE/
│       ├── ffhq1024_diffae.ckpt
│       └── celeba_cls.ckpt
└── FS/
    └── FaceSwapper/
        ├── faceswapper.ckpt
        ├── model_ir_se50.pth
        └── wing.ckpt
```

---

## 2. Checkpoint Verification Table

| Generator | Checkpoint Path | Expected SHA256 | Approx Size | Official Source |
| :--- | :--- | :--- | :---: | :--- |
| **StyleGAN3** | `EFS/StyleGAN3/stylegan3-r-ffhq-1024x1024.pkl` | `a5879a7384ff82173ea51f930e386928e44f80077c449c394fcc7ae9cf71bc8f` | 364 MB | NVIDIA NGC StyleGAN3 release |
| **IAFaces** | `AM/IAFaces/iafaces-celebahq-256.pth` | `b08552b6c1adcc372b665e005cf1a4fa3a50112ba747391060e268db41b6cab2` | 335 MB | Official IA-FaceS repo |
| **LatTrans** | `AM/LatTrans/psp_ffhq_encode.pt` | `f786ac5cf18fcdbc5c57c9f3f12cbedca69381fbdf5694dd132294e3133749d7` | 1.14 GB | Official pixel2style2pixel repo |
| **FaceSwapper** | `FS/FaceSwapper/faceswapper.ckpt` | `74d781083057091ab8e74b9e86ce2c84a5b66c72dd1f218290e0aeaa0d87b4ee` | 391 MB | Official FaceSwapper release |
| **DDPM** | `EFS/DDPM/google_ddpm_celebahq_256` | `6b81d77cb077d018788417c88b4ddc89a7444c1dcae9e0350a4ec9ec51787c8d` | 455 MB | Hugging Face google/ddpm-celebahq-256 |
| **LatDiff** | `EFS/LatDiff/ffhq_ldm_kl_f4.ckpt` | `4224fbe29fa9be7e6f83ec55ee988cf2332616f73587e974e370a256df2e6dc3` | 1.37 GB | CompVis LDM FFHQ release |
| **CollDiff** | `EFS/CollDiff/colldiff_ffhq.ckpt` | `3a4f89d38c64119d67b2d56e7e59c00ab2f5a6b0c2049e49a8fe10ca1c15f9b4` | 1.37 GB | Official Collaborative Diffusion repo |
| **DiffAE** | `AM/DiffAE/ffhq1024_diffae.ckpt` | `9746fb9ae9c4d9ad67f8921e51b3117fe4ee06b23b3fd8959d571871a37c4db6` | 613 MB | Official DiffAE release |
| **DiffAE Cls** | `AM/DiffAE/celeba_cls.ckpt` | `b0b2e3e566cf2c78e38d72e70e9aeb1777085c88b6932a392ddb86653e163b27` | 3.5 MB | Official DiffAE CelebA classifier |

---

## 3. Automated Verification at Preflight

The preflight job (`slurm/00_preflight.sbatch`) will automatically:
1. Scan `$CHECKPOINT_ROOT` for each checkpoint file.
2. Compute the SHA256 digest of each checkpoint.
3. Compare against `configs/checkpoints.yaml`.
4. Report any missing files or digest discrepancies before any generation job is permitted to launch.
