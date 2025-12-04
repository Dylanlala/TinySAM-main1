import os
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def mask_to_yolo_bboxes(mask_path: str) -> List[Tuple[float, float, float, float]]:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    mask = (mask > 128).astype(np.uint8)
    if mask.max() == 0:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    boxes: List[Tuple[float, float, float, float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 5:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # YOLO normalized xywh
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        boxes.append((cx, cy, nw, nh))
    return boxes


def write_yolo_label(label_path: str, boxes: List[Tuple[float, float, float, float]], class_id: int = 0) -> None:
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        for (cx, cy, w, h) in boxes:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def convert_from_split_files(splits_dir: str, out_root: str, include_splits: List[str] = ["train", "val"]) -> None:
    splits_path = Path(splits_dir) / "splits.json"
    assert splits_path.exists(), f"splits file not found: {splits_path}"
    with open(splits_path, "r") as f:
        splits = json.load(f)

    images_dir = Path(out_root) / "images"
    labels_dir = Path(out_root) / "labels"
    for split in include_splits:
        os.makedirs(images_dir / split, exist_ok=True)
        os.makedirs(labels_dir / split, exist_ok=True)

        for rec in splits[split]:
            img_path = Path(rec["image"]).resolve()
            mask_path = Path(rec["mask"]).resolve()

            # symlink/copy image
            dst_img = images_dir / split / img_path.name
            if not dst_img.exists():
                try:
                    os.symlink(img_path, dst_img)
                except Exception:
                    import shutil
                    shutil.copy2(img_path, dst_img)

            # write label
            boxes = mask_to_yolo_bboxes(str(mask_path))
            dst_label = labels_dir / split / (img_path.stem + ".txt")
            write_yolo_label(str(dst_label), boxes)

    # write a basic data.yaml for YOLO training
    data_yaml = Path(out_root) / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            "path: .",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: polyp",
        ])
    )
    print(f"YOLO dataset prepared at: {Path(out_root).resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", type=str, default="polypgen_splits")
    parser.add_argument("--out_root", type=str, default="datasets/polypgen_yolo")
    args = parser.parse_args()

    convert_from_split_files(args.splits_dir, args.out_root)



