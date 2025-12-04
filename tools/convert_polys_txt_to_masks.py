#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def is_image_file(filename: str) -> bool:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    return any(filename.lower().endswith(ext) for ext in exts)


def read_boxes_from_txt(txt_path: Path) -> np.ndarray:
    # Expected format examples:
    # L1: 0 or 1 (flag)
    # L2+: x1 y1 x2 y2 (pixel coords), multiple lines possible
    try:
        with open(txt_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return np.zeros((0, 4), dtype=int)

    if len(lines) == 0:
        return np.zeros((0, 4), dtype=int)

    boxes = []
    for ln in lines[1:]:  # skip first flag line
        parts = ln.split()
        if len(parts) >= 4:
            try:
                x1, y1, x2, y2 = map(int, parts[:4])
                boxes.append([x1, y1, x2, y2])
            except Exception:
                continue
    if len(boxes) == 0 and len(lines) >= 5:
        # fallback: sometimes the file may not have the first flag
        try:
            x1, y1, x2, y2 = map(int, lines[:4])
            boxes.append([x1, y1, x2, y2])
        except Exception:
            pass
    return np.array(boxes, dtype=int) if boxes else np.zeros((0, 4), dtype=int)


def generate_mask_for_image(img_path: Path, txt_path: Path, out_path: Path) -> Optional[Path]:
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = read_boxes_from_txt(txt_path)
    for (x1, y1, x2, y2) in boxes:
        x1c = max(0, min(w - 1, x1))
        y1c = max(0, min(h - 1, y1))
        x2c = max(0, min(w - 1, x2))
        y2c = max(0, min(h - 1, y2))
        if x2c > x1c and y2c > y1c:
            mask[y1c:y2c, x1c:x2c] = 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)
    return out_path


def convert_annotations(images_root: Path, ann_root: Path, masks_root: Path) -> None:
    total = 0
    converted = 0
    missing = 0
    for dirpath, _, filenames in os.walk(images_root):
        for fname in filenames:
            if not is_image_file(fname):
                continue
            total += 1
            img_path = Path(dirpath) / fname
            rel = img_path.relative_to(images_root)
            txt_path = ann_root / rel.with_suffix('.txt')
            out_mask_path = masks_root / rel.with_suffix('.png')
            if not txt_path.exists():
                missing += 1
                continue
            if generate_mask_for_image(img_path, txt_path, out_mask_path) is not None:
                converted += 1
    print(f"Done. total images: {total}, converted: {converted}, missing txt: {missing}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Convert Polys YOLO .txt to binary mask PNGs")
    parser.add_argument('--images', type=str, required=True, help='Path to Polys/Test/Images')
    parser.add_argument('--annotations', type=str, required=True, help='Path to Polys/Test/Annotations')
    parser.add_argument('--out-masks', type=str, required=True, help='Output masks root directory')
    args = parser.parse_args()

    convert_annotations(Path(args.images), Path(args.annotations), Path(args.out_masks))


if __name__ == '__main__':
    main()


