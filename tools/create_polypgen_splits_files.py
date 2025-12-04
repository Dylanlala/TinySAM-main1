import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from typing import Dict, List, Tuple

from polypgen_dataset import PolypGenDataset


def _dataset_to_records(ds: PolypGenDataset) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for img_path, mask_path, group in ds.samples:
        records.append({
            "image": os.path.abspath(img_path),
            "mask": os.path.abspath(mask_path),
            "group": group,
        })
    return records


def create_fixed_splits(root_dir: str, out_dir: str = "polypgen_splits") -> None:
    os.makedirs(out_dir, exist_ok=True)

    train_ds = PolypGenDataset(root_dir, split="train")
    val_ds = PolypGenDataset(root_dir, split="val")
    test_ds = PolypGenDataset(root_dir, split="test")

    splits = {
        "train": _dataset_to_records(train_ds),
        "val": _dataset_to_records(val_ds),
        "test": _dataset_to_records(test_ds),
        "meta": {
            "root_dir": os.path.abspath(root_dir),
            "note": "Created with center-based grouping in PolypGenDataset to avoid leakage.",
        },
    }

    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    # Also write plain lists for convenience
    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        img_list = [os.path.abspath(p[0]) for p in ds.samples]
        mask_list = [os.path.abspath(p[1]) for p in ds.samples]
        with open(os.path.join(out_dir, f"{split_name}_images.txt"), "w") as f:
            f.write("\n".join(img_list))
        with open(os.path.join(out_dir, f"{split_name}_masks.txt"), "w") as f:
            f.write("\n".join(mask_list))

    print(f"Saved fixed splits to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="PolypGen root dir (PolypGen2021_MultiCenterData_v3)")
    parser.add_argument("--out_dir", type=str, default="polypgen_splits",
                        help="Output directory for split files")
    args = parser.parse_args()

    create_fixed_splits(args.root_dir, args.out_dir)


