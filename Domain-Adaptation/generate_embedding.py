# generate_embeddings.py
# Usage examples:
# python generate_embeddings.py --source_domains source_domains.yaml --lora_library_path catseg/loradb/
# or
# python generate_embeddings.py --source_domains_file source_domains.yaml --lora_library_path catseg/loradb/

from pathlib import Path
import os
import yaml
import argparse
from utils import get_domain_args
from embedding import EmbeddingManager
from tqdm import tqdm
import numpy as np

DETECTRON2_DATASET_PATH = os.getenv("DETECTRON2_DATASETS", "")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CLIP centroid embeddings for domains.")
    parser.add_argument("--source_domains", type=str, required=False, help="YAML file with list of domain names (top-level list).")
    parser.add_argument("--source_domains_file", type=str, required=False, help="(alias) YAML file with list of domain names (top-level list).")
    parser.add_argument("--lora_library_path", type=str, required=True, help="Path where domain statistics will be stored (per-domain subfolders).")
    parser.add_argument("--force_recompute", action="store_true", help="Force recompute and overwrite existing statistics.")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints.")
    args = parser.parse_args()

    # Accept either --source_domains or --source_domains_file for backward compatibility
    source_yaml = args.source_domains or args.source_domains_file
    if source_yaml is None:
        parser.error("Provide --source_domains or --source_domains_file pointing to the YAML file listing source domains.")
    return args, Path(source_yaml), Path(args.lora_library_path)

def load_source_domains(yaml_path: Path):
    if not yaml_path.exists():
        raise FileNotFoundError(f"Source domains YAML not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Expected: top-level list like: ["acdc-fog-", "cs-normal", ...] or mapping; handle both
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # if mapping, use keys
        return list(data.keys())
    else:
        raise ValueError("Unsupported YAML format for source domains. Provide a list or mapping.")

def safe_mean_embeddings(emb_list):
    """
    emb_list: list of numpy arrays, each of shape (1, D) or (D,)
    returns mean vector shape (D,)
    """
    if not emb_list:
        return None
    # stack properly
    arrs = []
    for e in emb_list:
        a = np.asarray(e)
        if a.ndim == 2 and a.shape[0] == 1:
            a = a.reshape(-1)
        arrs.append(a)
    stacked = np.stack(arrs, axis=0)  # shape (N, D)
    return np.mean(stacked, axis=0)

def main():
    args, source_yaml, lora_library_path = parse_args()

    source_domains = load_source_domains(source_yaml)
    lora_library_path = Path(lora_library_path)
    lora_library_path.mkdir(parents=True, exist_ok=True)

    embedding_manager = EmbeddingManager()  # uses ClipEmbeddingModel by default (your file)
    print("EmbeddingManager loaded. Beginning domain embedding generation...")

    for domain_name in source_domains:
        print(f"\n> Processing domain: {domain_name}")

        # get domain args (train path etc.)
        domain_args = get_domain_args(domain_name, "train", get_cofing_only=True)
        train_dataset_path = Path(domain_args.train_dataset_path)
        print(f"  train dataset path: {train_dataset_path}")

        if not train_dataset_path.exists():
            print(f"  ERROR: train path does not exist: {train_dataset_path}. Skipping domain.")
            continue

        domain_dir = lora_library_path / domain_name
        domain_dir.mkdir(parents=True, exist_ok=True)

        # statistics file path (keeps same naming as embedding.calculate_statistics uses)
        suffix = "_statistics.npz"
        statistics_path = domain_dir / f"{domain_name}{suffix}"
        if statistics_path.exists() and not args.force_recompute:
            print(f"  Statistics already exist at {statistics_path}. Skipping (use --force_recompute to overwrite).")
            continue

        print(f"  Calculating embeddings for dataset: {train_dataset_path}")
        # embed dataset (this uses your EmbeddingManager.embed_dataset which returns list of embeddings)
        dataset_embeddings = embedding_manager.embed_dataset(train_dataset_path, debug=args.debug)

        if not dataset_embeddings:
            print(f"  WARNING: No embeddings returned for domain {domain_name}. Skipping.")
            continue

        # compute mean robustly (handles shapes like (1,D) or (D,))
        train_average_embedding = safe_mean_embeddings(dataset_embeddings)
        if train_average_embedding is None:
            print(f"  ERROR: failed to compute mean embedding for {domain_name}")
            continue

        # Save to npz
        try:
            np.savez(
                statistics_path,
                train_average_embedding=train_average_embedding,
            )
            print(f"  Saved statistics to {statistics_path}")
        except Exception as e:
            print(f"  ERROR saving statistics: {e}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
