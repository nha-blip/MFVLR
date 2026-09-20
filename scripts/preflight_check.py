#!/usr/bin/env python3
"""
MFVLR / GenFace-Reproduced: SLURM Server Preflight Health & Checkpoint Verifier
Runs on the SLURM server (requesting 1 GPU) to audit hardware, software,
dataset paths, and checkpoint SHA256 integrity prior to large-scale generation.
"""

import os
import sys
import platform
import shutil
import hashlib
import subprocess
from pathlib import Path
import yaml

# Resolve roots
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data")).resolve()
CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", PROJECT_ROOT / "checkpoints")).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "output")).resolve()

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def check_system():
    print("================================================================================")
    print("=== MFVLR / GenFace-Reproduced: Preflight System Audit ===")
    print("================================================================================")
    print(f"Node Hostname:    {platform.node()}")
    print(f"OS Platform:      {platform.platform()}")
    print(f"Python Version:   {sys.version.split()[0]}")
    print(f"Working Dir:      {Path.cwd()}")
    print(f"PROJECT_ROOT:     {PROJECT_ROOT}")
    print(f"DATA_ROOT:        {DATA_ROOT}")
    print(f"CHECKPOINT_ROOT:  {CHECKPOINT_ROOT}")
    print(f"OUTPUT_ROOT:      {OUTPUT_ROOT}")

    # Check disk space on output root
    try:
        total, used, free = shutil.disk_usage(OUTPUT_ROOT.parent if not OUTPUT_ROOT.exists() else OUTPUT_ROOT)
        print(f"Output Disk Free: {free / (1024**3):.2f} GB (Total: {total / (1024**3):.2f} GB)")
    except Exception as e:
        print(f"Output Disk Free: UNKNOWN ({e})")

def check_gpu():
    print("\n--- GPU & CUDA Audit ---")
    try:
        import torch
        print(f"PyTorch Version:  {torch.__version__}")
        print(f"CUDA Available:   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Device 0:     {gpu_name}")
            print(f"Total VRAM:       {vram_gb:.2f} GB")
            print(f"Compute Cap:      {torch.cuda.get_device_capability(0)}")

            if "A100" not in gpu_name:
                print(">>> [WARNING] Detected GPU does NOT appear to be an NVIDIA A100!")
                print(">>> Pre-production and array resource allocations are tuned for A100 (40GB/80GB).")
            else:
                print(">>> [PASS] Confirmed NVIDIA A100 hardware.")
        else:
            print(">>> [CRITICAL ERROR] CUDA is not available to PyTorch!")
            return False
    except ImportError:
        print(">>> [CRITICAL ERROR] PyTorch is not installed in the current environment!")
        return False
    return True

def check_checkpoints():
    print("\n--- Checkpoint Audit & SHA256 Integrity ---")
    ckpt_yaml_path = PROJECT_ROOT / "configs" / "checkpoints.yaml"
    if not ckpt_yaml_path.exists():
        print(f">>> [FAIL] Checkpoint registry not found at: {ckpt_yaml_path}")
        return False

    with open(ckpt_yaml_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)["checkpoints"]

    all_pass = True
    for gen, info in registry.items():
        rel_path = info["server_path"]
        exp_sha = info["sha256"]
        full_path = CHECKPOINT_ROOT / rel_path

        if not full_path.exists():
            print(f"[{gen:<12}] MISSING: {full_path}")
            all_pass = False
            continue

        actual_sha = compute_sha256(full_path)
        if actual_sha.lower() != exp_sha.lower():
            print(f"[{gen:<12}] SHA256 MISMATCH! Exp: {exp_sha[:12]}... Got: {actual_sha[:12]}...")
            all_pass = False
        else:
            print(f"[{gen:<12}] PASS (SHA256: {actual_sha[:12]}..., Size: {full_path.stat().st_size / (1024**2):.1f} MB)")

    return all_pass

def check_datasets():
    print("\n--- Pristine Source Datasets Audit ---")
    data_yaml_path = PROJECT_ROOT / "configs" / "datasets.yaml"
    if not data_yaml_path.exists():
        print(f">>> [FAIL] Dataset registry not found at: {data_yaml_path}")
        return False

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)["datasets"]

    all_pass = True
    for name, info in registry.items():
        rel_p = info["server_relative_path"]
        full_p = DATA_ROOT / rel_p
        req_count = info.get("required_sample_count", 0)

        if not full_p.exists():
            print(f"[{name:<12}] MISSING: {full_p}")
            all_pass = False
            continue

        file_count = len(list(full_p.glob("*.*")))
        if file_count < req_count:
            print(f"[{name:<12}] WARNING: Found {file_count:,} files, expected >= {req_count:,} in {full_p}")
        else:
            print(f"[{name:<12}] PASS ({file_count:,} images present in {full_p})")

    return all_pass

def check_manifests():
    print("\n--- Production Manifests Audit ---")
    manifest_dir = PROJECT_ROOT / "manifests"
    manifests = [
        ("faceswapper_pairs_30k.csv", 30000),
        ("lattrans_quota_60k.csv", 60000),
        ("iafaces_quota_5k.csv", 5000),
    ]
    all_pass = True
    for fname, exp_count in manifests:
        p = manifest_dir / fname
        if not p.exists():
            print(f"[{fname:<28}] MISSING: {p}")
            all_pass = False
            continue
        with open(p, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f) - 1
        if lines != exp_count:
            print(f"[{fname:<28}] COUNT MISMATCH: Found {lines:,}, expected {exp_count:,}")
            all_pass = False
        else:
            print(f"[{fname:<28}] PASS ({lines:,} records)")
    return all_pass

def main():
    check_system()
    gpu_ok = check_gpu()
    ckpt_ok = check_checkpoints()
    data_ok = check_datasets()
    mani_ok = check_manifests()

    print("\n================================================================================")
    print(f"Preflight Result: GPU={gpu_ok}, Checkpoints={ckpt_ok}, Datasets={data_ok}, Manifests={mani_ok}")
    if gpu_ok and ckpt_ok and data_ok and mani_ok:
        print(">>> PREFLIGHT AUDIT 100% PASS: Ready for A100 Benchmarking & Production!")
        sys.exit(0)
    else:
        print(">>> PREFLIGHT AUDIT FAILED: Please fix above issues before submitting production jobs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
