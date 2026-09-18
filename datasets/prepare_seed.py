"""Automated downloader and manifest builder for Mengieong/SEED_balanced dataset.

Handles downloading from HuggingFace, extraction of zip archives,
label mapping for sequential edits (L=0 Real, L>=1 Fake),
balanced sampling (50% Real / 50% Fake), and JSONL manifest generation for MFVLR.
"""

import argparse
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Fix Windows cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SEED_HF_REPO = "Mengieong/SEED_balanced"
SEED_FILES = [
    "SEED_subset_1.zip",
    "SEED_subset_2.zip",
    "SEED_subset_3.zip",
]


def download_seed_dataset(dest_dir: Path) -> None:
    """Download SEED_balanced dataset zip files directly with progress and resume support."""
    print(f"Downloading {SEED_HF_REPO} archives to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        for fname in SEED_FILES:
            dest_file = dest_dir / fname
            if dest_file.exists() and dest_file.stat().st_size > 10 * 1024 * 1024:
                print(f"[Skip] {fname} already downloaded ({dest_file.stat().st_size / (1024*1024):.1f} MB).")
                continue
            print(f"\n--- Downloading {fname} ---")
            hf_hub_download(
                repo_id=SEED_HF_REPO,
                filename=fname,
                repo_type="dataset",
                local_dir=str(dest_dir),
            )
            print(f"[OK] Downloaded {fname} successfully.")
    except ImportError:
        print("[Error] huggingface_hub is not installed. Run: pip install huggingface_hub")
        print("Alternatively, download manually via CLI:")
        print(f"  huggingface-cli download {SEED_HF_REPO} --repo-type dataset --local-dir {dest_dir}")
        sys.exit(1)


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip archive with progress bar."""
    print(f"Extracting {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
            zip_ref.extract(member, extract_to)
    print(f"[OK] Extracted {zip_path.name}.")


def find_and_extract_all_zips(data_dir: Path) -> None:
    """Find and extract all SEED subset zip archives."""
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Search for zips in data_dir and subdirectories
    zip_files = list(data_dir.glob("*.zip")) + list(data_dir.glob("**/*.zip"))
    unique_zips = {p.resolve(): p for p in zip_files}

    if not unique_zips:
        print(f"[Warning] No .zip files found in {data_dir}. Checking if already extracted...")
        return

    for zip_path in unique_zips.values():
        folder_name = zip_path.stem
        target_dir = images_dir / folder_name
        if target_dir.exists() and len(list(target_dir.glob("*"))) > 0:
            print(f"[Skip] {folder_name} already extracted.")
        else:
            extract_zip(zip_path, target_dir)


import ast


def parse_metadata_line(line_str: str) -> Optional[tuple]:
    """Parse a single metadata line from txt/csv/tsv files supporting SEED semicolon format."""
    line_str = line_str.strip()
    if not line_str or line_str.startswith("#"):
        return None

    # Handle semicolon delimiter (standard in SEED benchmark: path;[0, 0, 0, 0])
    if ";" in line_str:
        parts = line_str.split(";")
        img_raw = parts[0].strip()
        target_raw = parts[1].strip() if len(parts) > 1 else ""
        prompt_raw = parts[2].strip() if len(parts) > 2 else ""
    elif "\t" in line_str:
        parts = line_str.split("\t")
        img_raw = parts[0].strip()
        target_raw = parts[1].strip() if len(parts) > 1 else ""
        prompt_raw = parts[2].strip() if len(parts) > 2 else ""
    elif "," in line_str:
        if "[" in line_str:
            idx = line_str.index("[")
            img_raw = line_str[:idx].rstrip(",").strip()
            target_raw = line_str[idx:].strip()
            prompt_raw = ""
        else:
            parts = line_str.split(",")
            img_raw = parts[0].strip()
            target_raw = parts[1].strip() if len(parts) > 1 else ""
            prompt_raw = parts[2].strip() if len(parts) > 2 else ""
    else:
        if "[" in line_str:
            idx = line_str.index("[")
            img_raw = line_str[:idx].strip()
            target_raw = line_str[idx:].strip()
            prompt_raw = ""
        else:
            parts = line_str.split()
            img_raw = parts[0].strip()
            target_raw = parts[1].strip() if len(parts) > 1 else ""
            prompt_raw = ""

    if not img_raw:
        return None

    # Parse target_raw into sequence length / label
    seq_len = None
    vec = None
    if target_raw.startswith("[") and target_raw.endswith("]"):
        try:
            vec = ast.literal_eval(target_raw)
            if isinstance(vec, (list, tuple)):
                # If vector of ints (e.g., [0, 0, 0, 0] for Real, [1, 3, 0, 0] for Fake)
                if len(vec) == 0:
                    seq_len = 0
                elif all(isinstance(x, (int, float)) for x in vec):
                    non_zeros = [x for x in vec if int(x) != 0]
                    seq_len = len(non_zeros)
                else:
                    seq_len = len(vec)
        except Exception:
            pass

    if seq_len is None and target_raw:
        if target_raw.isdigit():
            seq_len = int(target_raw)
        elif target_raw.lower() in ("real", "orig", "original", "clean", "0"):
            seq_len = 0
        elif target_raw.lower() in ("fake", "manipulated", "am", "1"):
            seq_len = 1

    rec = {
        "seq_len": seq_len,
        "label": 0 if (seq_len == 0) else (1 if (seq_len is not None and seq_len > 0) else None),
        "is_fake": (seq_len > 0) if (seq_len is not None) else None,
    }
    if vec is not None:
        rec["sequence_vector"] = vec
    if prompt_raw:
        rec["prompt"] = prompt_raw

    return img_raw, rec


def parse_seed_samples(data_dir: Path) -> List[Dict]:
    """Scan extracted images and index files to construct metadata records."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images_dir = data_dir / "images" if (data_dir / "images").exists() else data_dir

    all_samples: List[Dict] = []
    meta_records: Dict[str, Dict] = {}

    # 1. Search for all metadata / index files in dataset directory
    meta_files = (
        list(data_dir.glob("**/*.json"))
        + list(data_dir.glob("**/*.jsonl"))
        + list(data_dir.glob("**/*.csv"))
        + list(data_dir.glob("**/*.tsv"))
        + list(data_dir.glob("**/*.txt"))
    )

    valid_meta_files = [mf for mf in meta_files if "manifest" not in mf.name and not mf.name.endswith(".zip")]
    print(f"Found {len(valid_meta_files)} potential metadata/index files in {data_dir}:")
    for mf in valid_meta_files[:10]:
        print(f"  - {mf.relative_to(data_dir).as_posix()} ({mf.stat().st_size / 1024:.1f} KB)")

    sample_raw_records = []

    for mf in valid_meta_files:
        try:
            if mf.suffix in (".json", ".jsonl"):
                with open(mf, "r", encoding="utf-8") as f:
                    if mf.suffix == ".json":
                        try:
                            data = json.load(f)
                        except Exception:
                            continue
                        if isinstance(data, dict):
                            for k, v in data.items():
                                if isinstance(v, dict):
                                    meta_records[os.path.basename(k)] = v
                                    meta_records[Path(k).stem] = v
                                    meta_records[k] = v
                                    if len(sample_raw_records) < 3:
                                        sample_raw_records.append((k, v))
                                else:
                                    meta_records[os.path.basename(k)] = {"label": v}
                                    meta_records[Path(k).stem] = {"label": v}
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    img_name = item.get("file_name", item.get("filename", item.get("image", item.get("img", item.get("image_path", "")))))
                                    if img_name:
                                        meta_records[os.path.basename(img_name)] = item
                                        meta_records[Path(img_name).stem] = item
                                        meta_records[img_name] = item
                                        if len(sample_raw_records) < 3:
                                            sample_raw_records.append((img_name, item))
                    else:  # .jsonl
                        for line in f:
                            line_str = line.strip()
                            if not line_str:
                                continue
                            try:
                                rec = json.loads(line_str)
                            except Exception:
                                continue
                            if isinstance(rec, dict):
                                img_name = rec.get("file_name", rec.get("filename", rec.get("image", rec.get("img", rec.get("image_path", rec.get("image_name", ""))))))
                                if img_name:
                                    meta_records[os.path.basename(img_name)] = rec
                                    meta_records[Path(img_name).stem] = rec
                                    meta_records[img_name] = rec
                                    if len(sample_raw_records) < 3:
                                        sample_raw_records.append((img_name, rec))

            elif mf.suffix in (".csv", ".tsv", ".txt"):
                with open(mf, "r", encoding="utf-8") as f:
                    for line in f:
                        res = parse_metadata_line(line)
                        if res is None:
                            continue
                        img_raw, rec = res
                        img_k = os.path.basename(img_raw)
                        stem_k = Path(img_k).stem

                        # Register under all path resolutions
                        meta_records[img_k] = rec
                        meta_records[stem_k] = rec
                        meta_records[img_raw] = rec
                        try:
                            # Relative path from dataset root through metadata parent directory
                            rel_parent = mf.parent.relative_to(data_dir).as_posix()
                            meta_records[f"{rel_parent}/{img_raw}"] = rec
                        except Exception:
                            pass

                        if len(sample_raw_records) < 3:
                            sample_raw_records.append((img_raw, rec))
        except Exception as e:
            print(f"[Notice] Could not parse metadata file {mf.name}: {e}")

    if meta_records:
        print(f"Loaded metadata entries for {len(meta_records)} index keys.")
        if sample_raw_records:
            print(f"Sample raw metadata record: key={sample_raw_records[0][0]}, content={json.dumps(sample_raw_records[0][1])[:300]}")

    # 2. Collect all image files
    all_image_paths = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    print(f"Found {len(all_image_paths)} image files in {images_dir}.")
    if len(all_image_paths) > 0:
        print(f"Sample image paths: {[p.relative_to(data_dir).as_posix() for p in all_image_paths[:5]]}")

    seq_len_counts: Dict[int, int] = {}

    def extract_sequence_length(meta_dict: Dict) -> Optional[int]:
        """Extract sequence edit length from dictionary using multiple heuristics."""
        if not meta_dict:
            return None

        # 1. Direct explicit integer seq_len check
        if "seq_len" in meta_dict and meta_dict["seq_len"] is not None:
            val = meta_dict["seq_len"]
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str) and val.strip().isdigit():
                return int(val.strip())

        # 2. Check boolean is_fake / is_real
        if "is_real" in meta_dict and meta_dict["is_real"] is not None:
            val = meta_dict["is_real"]
            if isinstance(val, bool):
                return 0 if val else 1
            if str(val).lower() in ("true", "1", "yes"):
                return 0
            if str(val).lower() in ("false", "0", "no"):
                return 1

        if "is_fake" in meta_dict and meta_dict["is_fake"] is not None:
            val = meta_dict["is_fake"]
            if isinstance(val, bool):
                return 1 if val else 0
            if str(val).lower() in ("false", "0", "no"):
                return 0
            if str(val).lower() in ("true", "1", "yes"):
                return 1

        # Check sequence field
        for seq_key in ["sequence", "model_sequence", "attr_sequence", "attribute_sequence", "edit_sequence", "operations", "instructions", "edits"]:
            if seq_key in meta_dict:
                seq_val = meta_dict[seq_key]
                if seq_val is None:
                    return 0
                if isinstance(seq_val, (list, tuple)):
                    return len(seq_val)
                if isinstance(seq_val, (int, float)):
                    return int(seq_val)
                if isinstance(seq_val, str):
                    seq_str = seq_val.strip()
                    if seq_str in ("", "[]", "()", "{}", "none", "None", "null"):
                        return 0
                    if seq_str.startswith("[") and seq_str.endswith("]"):
                        try:
                            parsed = ast.literal_eval(seq_str)
                            if isinstance(parsed, (list, tuple)):
                                return len(parsed)
                        except Exception:
                            pass
                    if seq_str.isdigit():
                        return int(seq_str)
                    # Comma separated list of edits
                    items = [x.strip() for x in seq_str.split(",") if x.strip()]
                    return len(items)

        # Check integer fields
        for num_key in ["seq_len", "length", "edit_length", "num_edits", "steps", "step", "edit_order", "order", "level", "depth"]:
            if num_key in meta_dict:
                val = meta_dict[num_key]
                if isinstance(val, (int, float)):
                    return int(val)
                if isinstance(val, str) and val.strip().isdigit():
                    return int(val.strip())

        # Check label field
        if "label" in meta_dict:
            val = meta_dict["label"]
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str):
                v_str = val.strip().lower()
                if v_str in ("0", "real", "orig", "original", "clean"):
                    return 0
                if v_str in ("1", "fake", "manipulated", "am"):
                    return 1
                if v_str.isdigit():
                    return int(v_str)

        # Check model / generator / prompt
        for gen_key in ["generator", "editor", "model", "method"]:
            if gen_key in meta_dict:
                gen_val = str(meta_dict[gen_key]).strip().lower()
                if gen_val in ("none", "null", "real", "orig", "original", ""):
                    return 0

        if "prompt" in meta_dict and str(meta_dict["prompt"]).strip().lower() in ("", "none", "null", "real"):
            return 0

        return None

    for img_path in tqdm(all_image_paths, desc="Parsing image records"):
        rel_path = img_path.relative_to(data_dir).as_posix()
        fname = img_path.name
        stem = img_path.stem.lower()
        path_str = rel_path.lower()
        parent_name = img_path.parent.name.lower()
        path_parts = [p.lower() for p in img_path.relative_to(data_dir).parts]

        # 3. Lookup in metadata records using full relative path, filename, stem, and subpath
        meta = (
            meta_records.get(rel_path)
            or meta_records.get(fname)
            or meta_records.get(stem)
            or meta_records.get("/".join(path_parts[-2:]))
            or meta_records.get("/".join(path_parts[-3:]))
            or {}
        )

        if not meta:
            sidecar_json = img_path.with_suffix(".json")
            if sidecar_json.exists():
                try:
                    with open(sidecar_json, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

        # 4. Determine sequence edit length L (0: Real, >=1: Fake)
        seq_len = extract_sequence_length(meta)

        if seq_len is None:
            # Pattern matching on path/filename components
            # Level 0 (Real) indicators
            if any(k in path_parts for k in ["real", "original", "orig", "clean", "raw", "source", "ffhq", "celeba", "celebamask", "l0", "length_0", "len_0", "step_0", "edit_0"]):
                seq_len = 0
            elif stem.endswith("_0") or stem.endswith("-0") or "_l0_" in stem or stem.startswith("0_") or stem.startswith("real_") or "_real" in stem:
                seq_len = 0
            # Level 1..4 (Fake) indicators
            elif any(k in path_parts for k in ["length_1", "len_1", "step_1", "edit_1", "l1"]):
                seq_len = 1
            elif stem.endswith("_1") or stem.endswith("-1") or "_l1_" in stem or "step1" in stem or "edit1" in stem:
                seq_len = 1
            elif any(k in path_parts for k in ["length_2", "len_2", "step_2", "edit_2", "l2"]):
                seq_len = 2
            elif stem.endswith("_2") or stem.endswith("-2") or "_l2_" in stem or "step2" in stem or "edit2" in stem:
                seq_len = 2
            elif any(k in path_parts for k in ["length_3", "len_3", "step_3", "edit_3", "l3"]):
                seq_len = 3
            elif stem.endswith("_3") or stem.endswith("-3") or "_l3_" in stem or "step3" in stem or "edit3" in stem:
                seq_len = 3
            elif any(k in path_parts for k in ["length_4", "len_4", "step_4", "edit_4", "l4"]):
                seq_len = 4
            elif stem.endswith("_4") or stem.endswith("-4") or "_l4_" in stem or "step4" in stem or "edit4" in stem:
                seq_len = 4
            else:
                # Fallback: check if path contains explicit model names known to be fake
                if any(m in path_str for m in ["sd3", "sd15", "sd21", "sdxl", "flux", "pixart", "deepfake"]):
                    seq_len = 1
                else:
                    seq_len = 0 if ("real" in path_str or "ffhq" in path_str) else 1

        is_fake = (int(seq_len) > 0)
        manip_type = "AM" if is_fake else "Real"

        seq_len_counts[int(seq_len)] = seq_len_counts.get(int(seq_len), 0) + 1

        record = {
            "image_path": rel_path,
            "label": 1 if is_fake else 0,
            "is_fake": is_fake,
            "manipulation_type": manip_type,
            "seq_len": int(seq_len),
            "generator": "Diffusion" if is_fake else None,
            "family": "diffusion" if is_fake else None,
        }
        if "prompt" in meta:
            record["prompt"] = meta["prompt"]
        if "source_image" in meta:
            record["source_image_path"] = meta["source_image"]
        if "sequence" in meta:
            record["sequence"] = meta["sequence"]

        all_samples.append(record)

    print(f"Sequence length breakdown: {dict(sorted(seq_len_counts.items()))}")
    return all_samples


def build_balanced_manifests(
    data_dir: Path,
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    balance_train: bool = True,
    seed: int = 42,
) -> None:
    """Split samples into train/val/test and balance the training set (50% Real / 50% Fake)."""
    random.seed(seed)

    real_samples = [s for s in samples if not s["is_fake"]]
    fake_samples = [s for s in samples if s["is_fake"]]

    print(f"Total collected: {len(samples)} ({len(real_samples)} Real, {len(fake_samples)} Fake)")

    if not real_samples or not fake_samples:
        print("[Warning] Dataset does not have both Real and Fake samples!")
        random.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        n_val = int(len(samples) * val_ratio)
        train_set = samples[:n_train]
        val_set = samples[n_train:n_train + n_val]
        test_set = samples[n_train + n_val:]
    else:
        random.shuffle(real_samples)
        random.shuffle(fake_samples)

        # Split real samples
        n_real_train = int(len(real_samples) * train_ratio)
        n_real_val = int(len(real_samples) * val_ratio)
        real_train = real_samples[:n_real_train]
        real_val = real_samples[n_real_train:n_real_train + n_real_val]
        real_test = real_samples[n_real_train + n_real_val:]

        # Split fake samples
        n_fake_train = int(len(fake_samples) * train_ratio)
        n_fake_val = int(len(fake_samples) * val_ratio)
        fake_train = fake_samples[:n_fake_train]
        fake_val = fake_samples[n_fake_train:n_fake_train + n_fake_val]
        fake_test = fake_samples[n_fake_train + n_fake_val:]

        # Balance training set to 50% Real / 50% Fake
        if balance_train and len(real_train) > 0:
            target_fake_count = len(real_train)
            if len(fake_train) > target_fake_count:
                by_seq: Dict[int, List[Dict]] = {}
                for s in fake_train:
                    by_seq.setdefault(s.get("seq_len", 1), []).append(s)

                balanced_fake = []
                per_seq = max(1, target_fake_count // max(1, len(by_seq)))
                for seq_l, group in by_seq.items():
                    random.shuffle(group)
                    balanced_fake.extend(group[:per_seq])

                if len(balanced_fake) < target_fake_count:
                    remaining = [s for s in fake_train if s not in balanced_fake]
                    random.shuffle(remaining)
                    balanced_fake.extend(remaining[:target_fake_count - len(balanced_fake)])

                fake_train = balanced_fake[:target_fake_count]
                print(f"[Balance] Subsampled Fake train samples to {len(fake_train)} to match Real ({len(real_train)})")

        train_set = real_train + fake_train
        val_set = real_val + fake_val
        test_set = real_test + fake_test

        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)

    def write_jsonl(records: List[Dict], filepath: Path):
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        n_r = sum(1 for r in records if not r["is_fake"])
        n_f = sum(1 for r in records if r["is_fake"])
        print(f"Saved {filepath.name}: {len(records)} samples ({n_r} Real, {n_f} Fake)")

    write_jsonl(train_set, data_dir / "train_manifest.jsonl")
    write_jsonl(val_set, data_dir / "val_manifest.jsonl")
    write_jsonl(test_set, data_dir / "test_manifest.jsonl")


def update_config(config_path: Path, data_dir: Path) -> None:
    """Update configs/mfvlr.yaml with SEED_balanced dataset paths and balanced loss weights."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        rel_root = data_dir.relative_to(Path.cwd()).as_posix()
    except Exception:
        rel_root = "data/SEED_balanced"

    cfg.setdefault("dataset", {})
    cfg["dataset"]["root"] = rel_root
    cfg["dataset"]["train_manifest"] = f"{rel_root}/train_manifest.jsonl"
    cfg["dataset"]["val_manifest"] = f"{rel_root}/val_manifest.jsonl"
    cfg["dataset"]["test_manifest"] = f"{rel_root}/test_manifest.jsonl"
    cfg["dataset"]["real_class_index"] = 0
    cfg["dataset"]["fake_class_index"] = 1

    # Optimal balanced loss weights
    cfg.setdefault("loss", {})
    cfg["loss"]["lambda_fd"] = 5.0
    cfg["loss"]["lambda_fl"] = 2.0
    cfg["loss"]["lambda_lr"] = 0.1
    cfg["loss"]["lambda_cmc"] = 1.0
    cfg["loss"]["lambda_ar"] = 1.0
    cfg["loss"]["lambda_kl"] = 1.0

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] Updated {config_path.name} with SEED dataset paths and balanced loss weights.")


def main():
    parser = argparse.ArgumentParser(description="Prepare Mengieong/SEED_balanced dataset for MFVLR.")
    parser.add_argument("--data-dir", type=str, default="data/SEED_balanced", help="Dataset directory.")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if already downloaded.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip zip extraction if already done.")
    parser.add_argument("--no-balance", action="store_true", help="Do not balance 50/50 Real/Fake in train set.")
    parser.add_argument("--update-config", action="store_true", default=True, help="Update configs/mfvlr.yaml.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download if requested
    if not args.skip_download:
        download_seed_dataset(data_dir)

    # 2. Extract zips
    if not args.skip_extract:
        find_and_extract_all_zips(data_dir)

    # 3. Parse and build manifests
    print("\n--- Generating Balanced Manifests ---")
    samples = parse_seed_samples(data_dir)
    if samples:
        build_balanced_manifests(
            data_dir=data_dir,
            samples=samples,
            balance_train=not args.no_balance,
        )

        # 4. Update config
        if args.update_config:
            candidates = [
                Path("configs/mfvlr.yaml").resolve(),
                Path(__file__).resolve().parent.parent / "configs" / "mfvlr.yaml",
                Path(__file__).resolve().parent / "configs" / "mfvlr.yaml",
            ]
            for cp in candidates:
                if cp.exists():
                    update_config(cp, data_dir)
                    break
    else:
        print("[Notice] No image samples found yet. Run with download enabled or place dataset zips in data_dir.")


if __name__ == "__main__":
    main()
