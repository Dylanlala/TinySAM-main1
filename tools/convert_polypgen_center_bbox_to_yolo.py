import os
import json
from pathlib import Path
from typing import List, Tuple

import cv2


def resolve_annotation_paths(img_path: Path) -> Tuple[Path, Path, str]:
    """
    Given an image path, resolve the expected bbox txt path and mask path, and return a tag:
    - tag == "center": for .../data_Ck/images_Ck/xxx.jpg -> bbox_Ck/xxx.txt and masks_Ck/xxx_mask.jpg
    - tag == "sequence": for .../sequenceData/.../images_seqXX/xxx.jpg -> bbox_seqXX/xxx.txt and masks_seqXX/xxx_mask.jpg
    """
    parent = img_path.parent  # images_Ck or images_seqXX
    stem = img_path.stem

    if parent.name.startswith("images_C"):
        center = parent.name.split("images_")[-1]  # Ck
        base = parent.parent  # data_Ck
        # Official files may be saved as '<stem>_mask.txt'. Prefer plain '.txt' if exists, else fallback.
        bbox_plain = base / f"bbox_{center}" / f"{stem}.txt"
        bbox_mask = base / f"bbox_{center}" / f"{stem}_mask.txt"
        bbox_txt = bbox_plain if bbox_plain.exists() else bbox_mask
        mask_path = base / f"masks_{center}" / f"{stem}_mask.jpg"
        return bbox_txt, mask_path, "center"

    if parent.name.startswith("images_seq"):
        seq_dir = parent.parent  # .../seqXX
        suffix = parent.name.split("images_")[-1]  # seqXX
        bbox_plain = seq_dir / f"bbox_{suffix}" / f"{stem}.txt"
        bbox_mask = seq_dir / f"bbox_{suffix}" / f"{stem}_mask.txt"
        bbox_txt = bbox_plain if bbox_plain.exists() else bbox_mask
        mask_path = seq_dir / f"masks_{suffix}" / f"{stem}_mask.jpg"
        return bbox_txt, mask_path, "sequence"

    raise ValueError(f"Unexpected image folder name: {parent.name}")


def xyxy_to_yolo(cx1: float, cy1: float, cx2: float, cy2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = cx1, cy1, cx2, cy2
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def read_bbox_lines(txt_path: Path) -> List[Tuple[str, float, float, float, float]]:
    lines: List[Tuple[str, float, float, float, float]] = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x1, y1, x2, y2 = parts
            try:
                lines.append((cls, float(x1), float(y1), float(x2), float(y2)))
            except Exception:
                continue
    return lines


def write_yolo_label(dst_label: Path, records: List[Tuple[int, float, float, float, float]]) -> None:
    os.makedirs(dst_label.parent, exist_ok=True)
    with open(dst_label, "w") as f:
        for cid, cx, cy, ww, hh in records:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")


def mask_to_rects(mask_path: Path) -> List[Tuple[float, float, float, float]]:
    """Fallback: derive bbox from mask's external contours (xyxy in pixels)."""
    import numpy as np
    if not mask_path.exists():
        return []
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return []
    m = (m > 128).astype("uint8")
    if m.max() == 0:
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: List[Tuple[float, float, float, float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 5:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        rects.append((float(x), float(y), float(x + bw), float(y + bh)))
    return rects


def convert_using_center_bbox(
    splits_dir: str,
    out_root: str,
    include_splits: list[str] = ["train", "val"],
    with_masks: bool = False,
    official_only: bool = False,
    require_mask: bool = False,
) -> None:
    splits_path = Path(splits_dir) / "splits.json"
    assert splits_path.exists(), f"splits file not found: {splits_path}"
    with open(splits_path, "r") as f:
        splits = json.load(f)

    out_root_path = Path(out_root)
    images_root = out_root_path / "images"
    labels_root = out_root_path / "labels"
    masks_root = out_root_path / "masks"

    for split in include_splits:
        for rec in splits[split]:
            img_path = Path(rec["image"]).resolve()

            # Locate bbox txt and potential mask (supports center and sequence) early
            src_bbox, mask_path, tag = resolve_annotation_paths(img_path)
            has_bbox = src_bbox.exists()
            has_mask = mask_path.exists()

            # Enforce constraints BEFORE creating any files
            if split in ("train", "val"):
                if official_only and not has_bbox:
                    continue
                if require_mask and not has_mask:
                    continue

            # Read image size for normalization
            im = cv2.imread(str(img_path))
            if im is None:
                # skip unreadable image entirely
                continue
            ih, iw = im.shape[:2]

            # Now safe to create image symlink/copy
            dst_img = images_root / split / img_path.name
            os.makedirs(dst_img.parent, exist_ok=True)
            if not dst_img.exists():
                try:
                    os.symlink(img_path, dst_img)
                except Exception:
                    import shutil
                    shutil.copy2(img_path, dst_img)

            yolo_records: List[Tuple[int, float, float, float, float]] = []

            if has_bbox:
                for cls, x1, y1, x2, y2 in read_bbox_lines(src_bbox):
                    class_id = 0  # polyp
                    cx, cy, ww, hh = xyxy_to_yolo(x1, y1, x2, y2, iw, ih)
                    if ww > 0 and hh > 0:
                        yolo_records.append((class_id, cx, cy, ww, hh))
            else:
                # Fallback to mask-derived boxes (useful for sequenceData lacking bbox)
                rects = mask_to_rects(mask_path)
                for x1, y1, x2, y2 in rects:
                    class_id = 0
                    cx, cy, ww, hh = xyxy_to_yolo(x1, y1, x2, y2, iw, ih)
                    if ww > 0 and hh > 0:
                        yolo_records.append((class_id, cx, cy, ww, hh))

            # Always write label file. If no boxes -> empty file (negative sample)
            dst_label = labels_root / split / (img_path.stem + ".txt")
            write_yolo_label(dst_label, yolo_records)

            # optionally symlink masks for later segmentation eval convenience
            if with_masks and has_mask:
                dst_mask = masks_root / split / mask_path.name
                os.makedirs(dst_mask.parent, exist_ok=True)
                if not dst_mask.exists():
                    try:
                        os.symlink(mask_path, dst_mask)
                    except Exception:
                        import shutil
                        shutil.copy2(mask_path, dst_mask)

    # write data.yaml
    data_yaml = out_root_path / "data.yaml"
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: polyp",
    ]
    if "test" in include_splits:
        lines.insert(2, "test: images/test")
    data_yaml.write_text("\n".join(lines))

    print(f"YOLO dataset (from center bbox) prepared at: {out_root_path.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", type=str, default="polypgen_splits")
    parser.add_argument("--out_root", type=str, default="datasets/polypgen_yolo")
    parser.add_argument("--include_test", action="store_true", help="Also build test images/labels (for eval only)")
    parser.add_argument("--with_masks", action="store_true", help="Symlink masks under out_root/masks/<split> for later seg eval")
    parser.add_argument("--official_only", action="store_true", help="Use only official bbox; skip samples without bbox (train/val)")
    parser.add_argument("--require_mask", action="store_true", help="Require masks to exist for samples (train/val)")
    args = parser.parse_args()

    splits = ["train", "val"]
    if args.include_test:
        splits.append("test")
    convert_using_center_bbox(
        args.splits_dir,
        args.out_root,
        include_splits=splits,
        with_masks=args.with_masks,
        official_only=args.official_only,
        require_mask=args.require_mask,
    )


