import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def split_dataset(files: List[Path], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    return train_files, val_files, test_files


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mask_to_yolo_box(mask: np.ndarray) -> List[List[float]]:
    # Single class (polyp). If any foreground exists, compute bbox from contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    if not contours:
        return boxes
    h, w = mask.shape[:2]
    for cnt in contours:
        if cv2.contourArea(cnt) <= 0:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = (x + x + bw) / 2.0 / w
        cy = (y + y + bh) / 2.0 / h
        nw = bw / w
        nh = bh / h
        boxes.append([0, cx, cy, nw, nh])
    return boxes


def polygon_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    # Returns list of Nx2 polygons (float32)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) <= 1.0:
            continue
        cnt = cnt.squeeze(1)
        if cnt.ndim != 2 or cnt.shape[0] < 3:
            continue
        polys.append(cnt.astype(np.float32))
    return polys


def write_yolo_detection(label_path: Path, boxes: List[List[float]]) -> None:
    if not boxes:
        label_path.write_text("")
        return
    lines = ["{} {:.6f} {:.6f} {:.6f} {:.6f}".format(int(c), x, y, w, h) for c, x, y, w, h in boxes]
    label_path.write_text("\n".join(lines) + "\n")


def write_yolo_segmentation(label_path: Path, polys: List[np.ndarray], img_w: int, img_h: int) -> None:
    # YOLOv8/11 segmentation: class_id x1 y1 x2 y2 ... normalized
    lines = []
    for poly in polys:
        if poly.shape[0] < 3:
            continue
        x = poly[:, 0] / img_w
        y = poly[:, 1] / img_h
        coords = np.stack([x, y], axis=1).reshape(-1)
        coords = np.clip(coords, 0.0, 1.0)
        line = "0 " + " ".join([f"{v:.6f}" for v in coords.tolist()])
        lines.append(line)
    if not lines:
        label_path.write_text("")
    else:
        label_path.write_text("\n".join(lines) + "\n")


def process_split(split_name: str, files: List[Path], images_dir: Path, masks_dir: Path, out_root: Path, keep_empty: bool) -> None:
    out_images = out_root / "images" / split_name
    out_labels_det = out_root / "labels_det" / split_name
    out_labels_seg = out_root / "labels_seg" / split_name
    out_masks = out_root / "masks" / split_name
    ensure_dir(out_images)
    ensure_dir(out_labels_det)
    ensure_dir(out_labels_seg)
    ensure_dir(out_masks)

    for img_path in tqdm(files, desc=f"{split_name}"):
        name = img_path.name
        mask_path = masks_dir / name
        if not mask_path.exists():
            # Skip if no mask
            continue

        # Read mask and image size
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_bin = (mask > 127).astype(np.uint8) * 255

        # Detection labels (bbox per connected component)
        det_boxes = mask_to_yolo_box(mask_bin)
        if not det_boxes and not keep_empty:
            # Optionally drop empty (negative) samples
            continue

        # Segmentation labels (polygon per component)
        seg_polys = polygon_from_mask(mask_bin)

        # Write outputs
        # Copy image
        cv2.imwrite(str(out_images / name), img)
        # Save mask aligned
        cv2.imwrite(str(out_masks / name), mask_bin)

        # Detection label path mirrors image stem with .txt
        label_det_path = (out_labels_det / (img_path.stem + ".txt"))
        label_seg_path = (out_labels_seg / (img_path.stem + ".txt"))
        write_yolo_detection(label_det_path, det_boxes)
        write_yolo_segmentation(label_seg_path, seg_polys, w, h)


def write_yaml(yaml_path: Path, out_root: Path) -> None:
    content = f"""
path: {out_root}
train: images/train
val: images/val
test: images/test

names:
  0: polyp
""".strip() + "\n"
    yaml_path.write_text(content)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Kvasir-SEG for YOLO detection and segmentation")
    parser.add_argument("--src", type=str, default=str(Path("Kvasir-SEG").resolve()), help="Source Kvasir-SEG directory with images/ and masks/")
    parser.add_argument("--out", type=str, default=str(Path("datasets/kvasir_yolo").resolve()), help="Output dataset root")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Val ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--keep-empty", action="store_true", help="Keep empty (negative) samples; otherwise drop")
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    images_dir = src / "images"
    masks_dir = src / "masks"
    out_root = Path(args.out)
    ensure_dir(out_root)

    files = list_images(images_dir)
    if len(files) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    train_files, val_files, test_files = split_dataset(files, args.train, args.val, args.seed)

    process_split("train", train_files, images_dir, masks_dir, out_root, keep_empty=args.keep_empty)
    process_split("val", val_files, images_dir, masks_dir, out_root, keep_empty=args.keep_empty)
    process_split("test", test_files, images_dir, masks_dir, out_root, keep_empty=args.keep_empty)

    # Write YAMLs for detection and segmentation (same splits, different labels dirs)
    write_yaml(out_root / "kvasir_yolo_det.yaml", out_root)
    write_yaml(out_root / "kvasir_yolo_seg.yaml", out_root)

    print(f"Done. Output at: {out_root}")
    print("Labels (detection): labels_det/{train,val,test}")
    print("Labels (segmentation): labels_seg/{train,val,test}")


if __name__ == "__main__":
    main()

import os
import shutil
import random
from typing import List, Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_image_mask_pairs(images_dir: str, masks_dir: str) -> List[Tuple[str, str]]:
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    pairs = []
    for img_name in image_files:
        mask_path = os.path.join(masks_dir, img_name)
        if os.path.isfile(mask_path):
            pairs.append((os.path.join(images_dir, img_name), mask_path))
    return pairs


def mask_to_yolo_bboxes(mask: np.ndarray,
                        merge_all: bool = True,
                        min_area_ratio: float = 0.0005,
                        min_side_ratio: float = 0.01,
                        morph_kernel: int = 3,
                        morph_iter: int = 2) -> List[Tuple[float, float, float, float]]:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Morphology to reduce fragmentation and fill small holes
    if morph_kernel > 0 and morph_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k, iterations=morph_iter)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k, iterations=max(1, morph_iter // 2))

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bin_mask.shape

    min_area = max(1.0, min_area_ratio * (h * w))
    min_side_w = max(1.0, min_side_ratio * w)
    min_side_h = max(1.0, min_side_ratio * h)

    boxes_xyxy: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < min_side_w or bh < min_side_h:
            continue
        boxes_xyxy.append((x, y, x + bw, y + bh))

    if len(boxes_xyxy) == 0:
        return []

    if merge_all:
        xs1 = min(b[0] for b in boxes_xyxy)
        ys1 = min(b[1] for b in boxes_xyxy)
        xs2 = max(b[2] for b in boxes_xyxy)
        ys2 = max(b[3] for b in boxes_xyxy)
        bw = xs2 - xs1
        bh = ys2 - ys1
        cx = xs1 + bw / 2.0
        cy = ys1 + bh / 2.0
        return [(cx / w, cy / h, bw / w, bh / h)]

    boxes: List[Tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        boxes.append((cx / w, cy / h, bw / w, bh / h))
    return boxes


def write_yolo_label(label_path: str, boxes: List[Tuple[float, float, float, float]], class_id: int = 0) -> None:
    if len(boxes) == 0:
        # Write empty file to indicate no objects
        open(label_path, "w").close()
        return
    lines = [f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in boxes]
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def split_dataset(pairs: List[Tuple[str, str]], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    return train_pairs, val_pairs, test_pairs


def prepare_kvasir_yolo(
    kvasir_root: str,
    out_root: str,
    img_exts=(".jpg", ".jpeg", ".png"),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    images_dir = os.path.join(kvasir_root, "images")
    masks_dir = os.path.join(kvasir_root, "masks")
    assert os.path.isdir(images_dir), f"Missing images dir: {images_dir}"
    assert os.path.isdir(masks_dir), f"Missing masks dir: {masks_dir}"

    pairs = find_image_mask_pairs(images_dir, masks_dir)
    if len(pairs) == 0:
        raise RuntimeError("No image-mask pairs found.")

    train_pairs, val_pairs, test_pairs = split_dataset(pairs, train_ratio, val_ratio, seed)

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        img_out = os.path.join(out_root, "images", split_name)
        lbl_out = os.path.join(out_root, "labels", split_name)
        ensure_dir(img_out)
        ensure_dir(lbl_out)
        for img_path, mask_path in split_pairs:
            img_name = os.path.basename(img_path)
            base, _ = os.path.splitext(img_name)
            dst_img = os.path.join(img_out, img_name)
            shutil.copy2(img_path, dst_img)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            boxes = mask_to_yolo_bboxes(mask)
            label_path = os.path.join(lbl_out, f"{base}.txt")
            write_yolo_label(label_path, boxes, class_id=0)

    print(f"Prepared YOLO dataset at: {out_root}")
    print(f"Counts -> train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")


def write_dataset_yaml(yaml_path: str, dataset_root: str) -> None:
    content = (
        f"path: {dataset_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names: ['polyp']\n"
        f"nc: 1\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Wrote dataset yaml: {yaml_path}")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    kvasir_root = os.path.join(repo_root, "Kvasir-SEG")
    out_root = os.path.join(repo_root, "datasets", "kvasir_yolo")
    ensure_dir(out_root)
    prepare_kvasir_yolo(kvasir_root=kvasir_root, out_root=out_root)
    yaml_path = os.path.join(repo_root, "kvasir_yolo.yaml")
    write_dataset_yaml(yaml_path=yaml_path, dataset_root=out_root)



