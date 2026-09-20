# MFVLR / GenFace-Reproduced: A100 SLURM Deployment & Execution Guide

This package provides everything required to deploy, benchmark, and execute the large-scale **GenFace-Reproduced** dataset generation campaign (**375,000 fake face images** across 8 verified generators) on an NVIDIA A100 Linux cluster managed by SLURM.

All SLURM execution tasks have been consolidated into a **single unified SLURM file**: `slurm/run_dataset.slurm`, modeled directly after the project's standard `train_cospy.slurm`.

---

## 1. Quick Architecture Overview

```
slurm_package/
├── README.md                      # This guide
├── SLURM_PACKAGE_REPORT.md        # Technical audit & readiness report
├── env/
│   ├── environment.yml            # Conda environment definition for Linux + CUDA 11.8 + A100
│   ├── requirements.txt           # Pinned pip requirements
│   └── setup_env.sh               # Automated environment installation script
├── configs/
│   ├── server.env.example         # Cluster paths & SLURM partition template
│   ├── checkpoints.yaml           # Checkpoint registry with SHA256 digests
│   ├── datasets.yaml              # Pristine dataset locations (CelebA, CelebA-HQ, FFHQ)
│   ├── production_manifest.yaml   # Master production manifest (375k pool)
│   └── *.yaml                     # Per-generator configurations (8 generators)
├── manifests/
│   ├── faceswapper_pairs_30k.csv  # 30,000 CelebA bidirectional swap pairs
│   ├── lattrans_quota_60k.csv     # 60,000 samples across 40 CelebA T-Nets
│   └── iafaces_quota_5k.csv       # 5,000 samples across 3 facial components
├── scripts/
│   ├── generate.py                # Resumable generation worker with atomic writes
│   ├── preflight_check.py         # Hardware, CUDA, disk, and checkpoint SHA256 checker
│   ├── benchmark_a100.py          # Real empirical benchmark for 1 A100 GPU
│   ├── merge_metadata.py          # Merges per-task shards into master CSV
│   └── verify_production.py       # Read-only dataset validator (all 375k samples)
├── slurm/
│   ├── run_dataset.slurm          # Unified SLURM script for all actions (preflight, benchmark, generators, verify)
│   └── submit_all.sh              # Master submission orchestrator (--dry-run / --submit)
└── checkpoints/
    └── README_CHECKPOINTS.md      # Detailed checkpoint sources, sizes, and SHA256 hashes
```

---

## 2. Step-by-Step Deployment Instructions

### Step 1: Upload Package to the Cluster

On your local machine, use `rsync` or `scp` to upload `slurm_package` to your scratch/project directory on the SLURM server:

```bash
rsync -avzP GenFace_Reproduced/slurm_package/ user@cluster.example.edu:/scratch/user/MFVLR/slurm_package/
```

SSH into the cluster:

```bash
ssh user@cluster.example.edu
cd /scratch/user/MFVLR/slurm_package
```

---

### Step 2: Configure Server Environment

Copy the template `configs/server.env.example` to `configs/server.env`:

```bash
cp configs/server.env.example configs/server.env
nano configs/server.env
```

Edit `configs/server.env` with your cluster's absolute paths:

```bash
PROJECT_ROOT=/scratch/user/MFVLR/slurm_package
DATA_ROOT=/scratch/datasets/MFVLR_Sources
CHECKPOINT_ROOT=/scratch/models/MFVLR_Checkpoints
OUTPUT_ROOT=/scratch/user/MFVLR/production_output
CONDA_ENV=mfvlr_a100
SLURM_PARTITION=gpu_a100
SLURM_ACCOUNT=my_project_account
SLURM_QOS=normal
MAX_CONCURRENT_TASKS=4
```

---

### Step 3: Place Checkpoints and Pristine Datasets

1. **Checkpoints:** Place official model checkpoints under `$CHECKPOINT_ROOT` as described in [`checkpoints/README_CHECKPOINTS.md`](checkpoints/README_CHECKPOINTS.md):
   * `EFS/StyleGAN3/stylegan3-r-ffhq-1024x1024.pkl`
   * `AM/IAFaces/iafaces-celebahq-256.pth`
   * `AM/LatTrans/psp_ffhq_encode.pt` and `tnet_logs_001/`
   * `FS/FaceSwapper/faceswapper.ckpt`, `model_ir_se50.pth`, `wing.ckpt`
   * `EFS/DDPM/google_ddpm_celebahq_256`
   * `EFS/LatDiff/ffhq_ldm_kl_f4.ckpt`
   * `EFS/CollDiff/colldiff_ffhq.ckpt`
   * `AM/DiffAE/ffhq1024_diffae.ckpt` and `celeba_cls.ckpt`

2. **Pristine Datasets:** Place source face images under `$DATA_ROOT`:
   * `CelebA/images/` (at least 30,000 images `000001.jpg` to `030000.jpg`)
   * `CelebA-HQ/images/` (at least 30,000 images)
   * `FFHQ/images/` (at least 70,000 images)

---

### Step 4: Create Conda Environment

Run the automated setup script to build the environment:

```bash
bash env/setup_env.sh mfvlr_a100
```

Verify that the environment activates cleanly and detects CUDA:

```bash
conda activate mfvlr_a100
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

### Step 5: Run Preflight Check (STEP A)

Submit the preflight job using the unified SLURM file to verify hardware, CUDA, disk space, and checkpoint SHA256 digests:

```bash
sbatch slurm/run_dataset.slurm preflight
```

Monitor the preflight output:

```bash
tail -f logs/genface_prod_*.out
```

Ensure all checkpoints report **`PASS`** and GPU model is confirmed as NVIDIA A100.

---

### Step 6: Run Empirical A100 Benchmark (STEP B)

Submit the benchmark job to measure exact throughput and peak VRAM on an A100 GPU:

```bash
sbatch slurm/run_dataset.slurm benchmark
```

Review results:

```bash
cat A100_BENCHMARK_REPORT.md
cat benchmark_a100.json
```

---

### Step 7: Single-Task Smoke Test (STEP C)

Before submitting all 850 array tasks, run exactly **ONE task (Task 0)** for each generator to verify cluster execution:

```bash
sbatch --array=0-0 slurm/run_dataset.slurm stylegan3
sbatch --array=0-0 slurm/run_dataset.slurm iafaces
sbatch --array=0-0 slurm/run_dataset.slurm lattrans
sbatch --array=0-0 slurm/run_dataset.slurm faceswapper
sbatch --array=0-0 slurm/run_dataset.slurm ddpm
sbatch --array=0-0 slurm/run_dataset.slurm latdiff
sbatch --array=0-0 slurm/run_dataset.slurm colldiff
sbatch --array=0-0 slurm/run_dataset.slurm diffae
```

Inspect task logs in `logs/` to confirm that images, masks, and metadata shards are written cleanly without errors.

---

### Step 8: Full Production Submission (STEP D & E)

**Option 1: Using the Master Submission Orchestrator (`submit_all.sh`)**
1. Test submission dry-run:
   ```bash
   ./slurm/submit_all.sh --dry-run
   ```

2. When ready, launch full generation (prompts for confirmation):
   ```bash
   ./slurm/submit_all.sh --submit
   ```

**Option 2: Single All-in-One Master Job Array**
Submit all 850 tasks across all 8 generators in one single command:
```bash
sbatch --array=0-849%4 slurm/run_dataset.slurm all
```

---

## 3. Monitoring & Cluster Management

* **Check active jobs:**
  ```bash
  squeue -u $USER
  ```
* **Inspect job resource usage & efficiency:**
  ```bash
  sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS,AllocTRES
  ```
* **Cancel specific generator or task:**
  ```bash
  scancel <JOB_ID>
  scancel <JOB_ID>_<TASK_ID>
  ```
* **Cancel all jobs:**
  ```bash
  scancel -u $USER
  ```

---

## 4. Failure Recovery & Resumability

If an array task fails (e.g. due to node failure or walltime limit):
1. Review the error log in `logs/genface_prod_<job_id>_<task_id>.err`.
2. Re-submit only the failed task index:
   ```bash
   sbatch --array=<TASK_ID>-<TASK_ID> slurm/run_dataset.slurm <generator>
   ```
3. The generation worker automatically verifies existing images and masks, skipping already completed samples without redundant computation.

---

## 5. Post-Generation Aggregation & Verification

Once all generation arrays complete:
1. Merge metadata shards and validate the entire dataset:
   ```bash
   sbatch slurm/run_dataset.slurm verify
   ```
2. Monitor verification progress:
   ```bash
   tail -f logs/genface_prod_*.out
   ```
3. The verifier asserts:
   * Exactly 375,000 valid samples across the 8 generators.
   * 0 missing indices.
   * 0 duplicate sample IDs or SHA256 hashes.
   * 100% RGB $224 \times 224$ PNG images.
   * 100% uint8 $\{0, 255\}$ $224 \times 224$ PNG masks conforming to MFVLR BT.601 luminance difference formula.
   * Source and target pairing integrity for AM and FS.
   * All 26 metadata fields populated with `split = UNASSIGNED`.
