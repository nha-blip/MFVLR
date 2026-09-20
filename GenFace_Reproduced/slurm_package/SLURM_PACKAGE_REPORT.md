# MFVLR / GenFace-Reproduced: SLURM Package Technical Audit & Readiness Report

> **Creation Date:** September 20, 2026  
> **Deployment Target:** Separate Linux SLURM Server with NVIDIA A100 GPUs (40GB / 80GB)  
> **Production Pool Target:** **375,000 Fake Images** across 8 Verified Generators  
> **Package Location:** `GenFace_Reproduced/slurm_package/`  
> **Status:** **COMPLETE & PORTABLE — READY FOR SERVER UPLOAD**

---

## 1. Executive Summary

A self-contained, fully portable SLURM deployment package has been created in `GenFace_Reproduced/slurm_package/`. It encapsulates all necessary configurations, deterministic pairing and quota manifests, generation entry points, preflight/benchmark tools, and SLURM array job scripts required to execute the 375,000-image GenFace-Reproduced production campaign on a remote Linux cluster equipped with NVIDIA A100 GPUs.

* **Zero Local Execution:** No A100 benchmarks or SLURM jobs were executed on the local development machine.
* **Zero Path Leakage:** Every script and configuration has been audited and confirmed free of Windows drive paths (`C:`, `D:`, `\\`). All filesystem paths dynamically resolve from `$PROJECT_ROOT`, `$DATA_ROOT`, `$CHECKPOINT_ROOT`, and `$OUTPUT_ROOT`.
* **Complete Hardware Isolation:** Blocked generators (`DiffFace`, `FSLSD`) remain strictly blocked without substitution. The local pilot dataset (`MFVLR_Dataset/`, 115 samples) and pre-production dataset (`GenFace_Reproduced/preproduction_500/`, 3,880 samples) remain completely untouched and verified.

---

## 2. Directory Structure of the Portable Package

```
slurm_package/
├── README.md                          # Comprehensive Linux server deployment guide
├── SLURM_PACKAGE_REPORT.md            # This report
├── env/
│   ├── environment.yml                # Conda environment definition for Linux + CUDA 11.8 + A100
│   ├── requirements.txt               # Pinned pip dependencies
│   └── setup_env.sh                   # Automated environment build & test script
├── configs/
│   ├── server.env.example             # Server-specific paths, partition, and account template
│   ├── checkpoints.yaml               # Checkpoint registry with expected paths and SHA256 digests
│   ├── datasets.yaml                  # Pristine dataset specifications (CelebA, CelebA-HQ, FFHQ)
│   ├── production_manifest.yaml       # Master production manifest (375k pool, frozen parameters)
│   ├── stylegan3.yaml
│   ├── iafaces.yaml
│   ├── lattrans.yaml
│   ├── faceswapper.yaml
│   ├── ddpm.yaml
│   ├── latdiff.yaml
│   ├── colldiff.yaml
│   └── diffae.yaml
├── manifests/
│   ├── faceswapper_pairs_30k.csv      # 30,000 CelebA bidirectional swap pairs
│   ├── lattrans_quota_60k.csv         # 60,000 samples across 40 CelebA T-Nets
│   └── iafaces_quota_5k.csv           # 5,000 samples across 3 facial components
├── scripts/
│   ├── generate.py                    # Unified resumable generation worker with atomic writes
│   ├── preflight_check.py             # System, GPU, disk, dataset, and checkpoint SHA256 verifier
│   ├── benchmark_a100.py              # Empirical A100 benchmark (50-100 samples/gen, VRAM, throughput)
│   ├── merge_metadata.py              # Shard aggregator and completeness validator
│   └── verify_production.py           # Comprehensive production verifier (read-only)
├── slurm/
│   ├── 00_preflight.sbatch            # Preflight check job (1 GPU, checks A100, SHA256, paths)
│   ├── 01_benchmark_a100.sbatch       # A100 benchmark job (1 A100, measures real throughput)
│   ├── generate_stylegan3.sbatch      # Array: 50 tasks, chunk 1,000 (50k target)
│   ├── generate_iafaces.sbatch        # Array: 10 tasks, chunk 500 (5k target)
│   ├── generate_lattrans.sbatch       # Array: 60 tasks, chunk 1,000 (60k target)
│   ├── generate_faceswapper.sbatch    # Array: 30 tasks, chunk 1,000 (30k target)
│   ├── generate_ddpm.sbatch           # Array: 100 tasks, chunk 500 (50k target)
│   ├── generate_latdiff.sbatch        # Array: 120 tasks, chunk 500 (60k target)
│   ├── generate_colldiff.sbatch       # Array: 200 tasks, chunk 250 (50k target)
│   ├── generate_diffae.sbatch         # Array: 280 tasks, chunk 250 (70k target)
│   ├── verify_production.sbatch       # Post-generation verification job
│   └── submit_all.sh                  # Staged submission script (--dry-run / --submit)
├── checkpoints/
│   └── README_CHECKPOINTS.md          # Checkpoint download instructions, sizes, and SHA256 hashes
└── logs/                              # Placeholder directory for cluster logs
```

---

## 3. Production Targets & SLURM Array Architecture

The 375,000 fake pool is partitioned into non-overlapping deterministic sample ranges across 850 SLURM array tasks:

| Generator | Category | Architecture | Target Pool | Chunk Size | Number of Tasks | Walltime / Task | Memory / Task | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **StyleGAN3** | EFS | GAN | 50,000 | 1,000 | 50 | 00:20:00 | 16 GB | 819 MB |
| **IAFaces** | AM | GAN | 5,000 | 500 | 10 | 00:15:00 | 16 GB | 375 MB |
| **LatTrans** | AM | GAN | 60,000 | 1,000 | 60 | 00:20:00 | 20 GB | 2,273 MB |
| **FaceSwapper** | FS | GAN | 30,000 | 1,000 | 30 | 00:25:00 | 20 GB | 514 MB |
| **DDPM** | EFS | Diffusion | 50,000 | 500 | 100 | 01:00:00 | 16 GB | 803 MB |
| **LatDiff** | EFS | Diffusion | 60,000 | 500 | 120 | 00:45:00 | 16 GB | 2,721 MB |
| **CollDiff** | EFS | Diffusion | 50,000 | 250 | 200 | 01:00:00 | 24 GB | 4,471 MB |
| **DiffAE** | AM | Diffusion | 70,000 | 250 | 280 | 00:45:00 | 24 GB | 1,646 MB |
| **TỔNG** | | | **375,000** | | **850** | | | |

* **Array Index Calculation:**
  $$\text{START\_IDX} = \text{SLURM\_ARRAY\_TASK\_ID} \times \text{CHUNK\_SIZE}$$
  $$\text{END\_IDX} = \min(\text{START\_IDX} + \text{CHUNK\_SIZE}, \text{TARGET\_COUNT})$$
* **Concurrency Throttling:** Configured to `%4` by default (adjustable in `server.env` via `MAX_CONCURRENT_TASKS` based on available A100 GPU quota).

---

## 4. Checkpoint Requirements & Integrity Protocol

| Generator | Expected Path on Server | Expected SHA256 Digest | Size |
| :--- | :--- | :--- | :---: |
| **StyleGAN3** | `checkpoints/EFS/StyleGAN3/stylegan3-r-ffhq-1024x1024.pkl` | `a5879a7384ff82173ea51f930e386928e44f80077c449c394fcc7ae9cf71bc8f` | 364 MB |
| **IAFaces** | `checkpoints/AM/IAFaces/iafaces-celebahq-256.pth` | `b08552b6c1adcc372b665e005cf1a4fa3a50112ba747391060e268db41b6cab2` | 335 MB |
| **LatTrans** | `checkpoints/AM/LatTrans/psp_ffhq_encode.pt` | `f786ac5cf18fcdbc5c57c9f3f12cbedca69381fbdf5694dd132294e3133749d7` | 1.14 GB |
| **LatTrans T-Nets**| `checkpoints/AM/LatTrans/tnet_logs_001/tnet_{0..39}.pth.tar` | 40 individual checkpoints (listed in `checkpoints.yaml`) | ~40 MB |
| **FaceSwapper** | `checkpoints/FS/FaceSwapper/faceswapper.ckpt` | `74d781083057091ab8e74b9e86ce2c84a5b66c72dd1f218290e0aeaa0d87b4ee` | 391 MB |
| **FaceSwapper Arc**| `checkpoints/FS/FaceSwapper/model_ir_se50.pth` | `951aeaf8242a3cf134c4fcba13c9e378385bbd195972824ff433eb1f9429188d` | 166 MB |
| **FaceSwapper Wng**| `checkpoints/FS/FaceSwapper/wing.ckpt` | `8e7a68e7d23d8c19eb9e7f82782e4e1a06a2879a95393c8378546522c09191e4` | 89 MB |
| **DDPM** | `checkpoints/EFS/DDPM/google_ddpm_celebahq_256` | `6b81d77cb077d018788417c88b4ddc89a7444c1dcae9e0350a4ec9ec51787c8d` | 455 MB |
| **LatDiff** | `checkpoints/EFS/LatDiff/ffhq_ldm_kl_f4.ckpt` | `4224fbe29fa9be7e6f83ec55ee988cf2332616f73587e974e370a256df2e6dc3` | 1.37 GB |
| **CollDiff** | `checkpoints/EFS/CollDiff/colldiff_ffhq.ckpt` | `3a4f89d38c64119d67b2d56e7e59c00ab2f5a6b0c2049e49a8fe10ca1c15f9b4` | 1.37 GB |
| **DiffAE** | `checkpoints/AM/DiffAE/ffhq1024_diffae.ckpt` | `9746fb9ae9c4d9ad67f8921e51b3117fe4ee06b23b3fd8959d571871a37c4db6` | 613 MB |
| **DiffAE Cls** | `checkpoints/AM/DiffAE/celeba_cls.ckpt` | `b0b2e3e566cf2c78e38d72e70e9aeb1777085c88b6932a392ddb86653e163b27` | 3.5 MB |

* **Audit Rule:** Missing checkpoint or SHA256 digest discrepancy immediately causes the preflight job to fail that generator.

---

## 5. Source Dataset Requirements

* **CelebA:** Required for FaceSwapper. Needs at least 30,000 aligned images in `$DATA_ROOT/CelebA/images/` to fulfill the 30,000 bidirectional swap pairing manifest (`faceswapper_pairs_30k.csv`).
* **CelebA-HQ:** Required for IAFaces, LatTrans, and DDPM. At least 30,000 images in `$DATA_ROOT/CelebA-HQ/images/`.
* **FFHQ:** Required for StyleGAN3, LatDiff, CollDiff, and DiffAE. At least 70,000 images in `$DATA_ROOT/FFHQ/images/`.

---

## 6. Local Quality & Syntax Verification Results

Before finalizing, all package contents underwent rigorous automated static inspection:

1. **Python Syntax Compilation (`py_compile`):**
   * `scripts/benchmark_a100.py`: **PASS**
   * `scripts/generate.py`: **PASS**
   * `scripts/merge_metadata.py`: **PASS**
   * `scripts/preflight_check.py`: **PASS**
   * `scripts/verify_production.py`: **PASS**
2. **Bash Syntax Compilation (`bash -n`):**
   * `env/setup_env.sh`: **PASS**
   * `slurm/submit_all.sh`: **PASS**
   * `slurm/00_preflight.sbatch`: **PASS**
   * `slurm/01_benchmark_a100.sbatch`: **PASS**
   * `slurm/generate_stylegan3.sbatch`: **PASS**
   * `slurm/generate_iafaces.sbatch`: **PASS**
   * `slurm/generate_lattrans.sbatch`: **PASS**
   * `slurm/generate_faceswapper.sbatch`: **PASS**
   * `slurm/generate_ddpm.sbatch`: **PASS**
   * `slurm/generate_latdiff.sbatch`: **PASS**
   * `slurm/generate_colldiff.sbatch`: **PASS**
   * `slurm/generate_diffae.sbatch`: **PASS**
   * `slurm/verify_production.sbatch`: **PASS**
3. **YAML Configuration Validation:**
   * 11 YAML files (`configs/*.yaml`) parsed and verified: **100% PASS**.
4. **Manifest Row Counts:**
   * `faceswapper_pairs_30k.csv`: **30,000 records (PASS)**
   * `lattrans_quota_60k.csv`: **60,000 records (PASS)**
   * `iafaces_quota_5k.csv`: **5,000 records (PASS)**
5. **Portable Path Audit:**
   * Audited for Windows drive letters (`C:\`, `D:\`, `c:/`, `d:/`): **0 occurrences found (100% PASS)**.

---

## 7. Known Unknown Server Settings (To Be Populated in `server.env`)

The user must fill in cluster-specific variables upon uploading to the server:
* `PROJECT_ROOT`: Absolute path to `slurm_package/` on the cluster.
* `DATA_ROOT`: Absolute path to pristine datasets directory.
* `CHECKPOINT_ROOT`: Absolute path to checkpoints directory.
* `OUTPUT_ROOT`: Absolute path where generated data and metadata will be saved.
* `SLURM_PARTITION`: Cluster GPU partition name (e.g. `gpu_a100`, `batch_gpu`).
* `SLURM_ACCOUNT`: Cluster billing/project account (if required by cluster).
* `SLURM_QOS`: Cluster Quality of Service (if required).

---

## 8. Final Production Readiness Table

| STT | Generator | Category | Architecture | Target Pool | Chunk Size | Array Tasks | Checkpoint Ready | Manifest Ready | SLURM Ready | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **StyleGAN3** | EFS | GAN | 50,000 | 1,000 | 50 | Verified | N/A | Ready | **PRODUCTION_READY** |
| 2 | **IAFaces** | AM | GAN | 5,000 | 500 | 10 | Verified | 5k Quota | Ready | **PRODUCTION_READY** |
| 3 | **LatTrans** | AM | GAN | 60,000 | 1,000 | 60 | Verified | 60k Quota | Ready | **PRODUCTION_READY** |
| 4 | **FaceSwapper** | FS | GAN | 30,000 | 1,000 | 30 | Verified | 30k Pairs | Ready | **PRODUCTION_READY** |
| 5 | **DDPM** | EFS | Diffusion | 50,000 | 500 | 100 | Verified | N/A | Ready | **PRODUCTION_READY** |
| 6 | **LatDiff** | EFS | Diffusion | 60,000 | 500 | 120 | Verified | N/A | Ready | **PRODUCTION_READY** |
| 7 | **CollDiff** | EFS | Diffusion | 50,000 | 250 | 200 | Verified | N/A | Ready | **PRODUCTION_READY** |
| 8 | **DiffAE** | AM | Diffusion | 70,000 | 250 | 280 | Verified | N/A | Ready | **PRODUCTION_READY** |
| 9 | **DiffFace** | FS | Diffusion | 0 | — | — | Missing | N/A | N/A | **BLOCKED** |
| 10 | **FSLSD** | FS | GAN | 0 | — | — | Missing | N/A | N/A | **BLOCKED** |
| **TỔNG** | | | | **375,000** | | **850** | | | | **READY (8) / BLOCKED (2)** |

---

## 9. Strict Stop Protocol

> [!IMPORTANT]
> **Execution is STOPPED.**
> * The portable SLURM package is fully assembled in `GenFace_Reproduced/slurm_package/`.
> * No local benchmarks or SLURM jobs have been submitted.
> * No bulk generation has been initiated.
> * Existing pilot and pre-production datasets remain 100% untouched.
> * All subsequent actions will occur manually on the remote A100 SLURM server.
