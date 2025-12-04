import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_coco(coco_json_path: Path) -> Tuple[Dict[int, Dict], List[Dict], Dict[int, str]]:
    with coco_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data.get("images", [])}
    anns = data.get("annotations", [])
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    return images, anns, categories


def xywh_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    # clamp to image bounds in case of small negatives
    x = max(0.0, x)
    y = max(0.0, y)
    w = max(0.0, w)
    h = max(0.0, h)
    cx = x + w / 2.0
    cy = y + h / 2.0
    # normalize
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def write_yolo_labels(
    images: Dict[int, Dict],
    anns: List[Dict],
    categories: Dict[int, str],
    split_images_dir: Path,
    labels_out_dir: Path,
    class_name_to_id: Dict[str, int],
) -> None:
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    # group annotations by image_id
    imgid_to_anns: Dict[int, List[Dict]] = {}
    for ann in anns:
        imgid_to_anns.setdefault(ann["image_id"], []).append(ann)

    # iterate over images in split folder to only create labels for existing images
    image_files = {p.name: p for p in split_images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}}
    file_name_to_imgid = {img.get("file_name"): img_id for img_id, img in images.items()}

    num_written = 0
    num_missing = 0
    for file_name, image_path in image_files.items():
        img_id = file_name_to_imgid.get(file_name)
        if img_id is None:
            num_missing += 1
            continue
        img_info = images[img_id]
        img_w = int(img_info["width"])
        img_h = int(img_info["height"])
        lines: List[str] = []
        for ann in imgid_to_anns.get(img_id, []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            cat_id = ann["category_id"]
            cat_name = categories.get(cat_id, str(cat_id))
            class_id = class_name_to_id.get(cat_name)
            if class_id is None:
                # unseen category name; skip robustly
                continue
            bbox = ann["bbox"]  # COCO xywh
            x, y, w, h = xywh_to_yolo(bbox, img_w, img_h)
            # filter degenerate boxes
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        # write label file (empty file if no objects, as YOLO expects)
        label_path = labels_out_dir / (image_path.stem + ".txt")
        with label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        num_written += 1
    print(f"Wrote {num_written} label files to {labels_out_dir} ({num_missing} images had no matching entries in COCO JSON).")


def main():
    # dataset root (change via environment variable if needed)
    root = Path(os.environ.get("ALL_COCO_ROOT", "") or "/home/huangmanling/huangmanling/yolo_sam/TinySAM-main/all_coco").resolve()
    ann_dir = root / "annotations"
    train_json = ann_dir / "train_coco_format.json"
    val_json = ann_dir / "val_coco_format.json"
    test_json = ann_dir / "test_coco_format.json"

    # discover classes from train json
    images_train, _, categories = load_coco(train_json)
    class_names_sorted = [categories[k] for k in sorted(categories.keys())]
    class_name_to_id = {name: idx for idx, name in enumerate(class_names_sorted)}
    print(f"Detected classes: {class_names_sorted}")

    # convert train/val/test
    splits = [
        ("train", train_json),
        ("val", val_json),
        ("test", test_json),
    ]
    for split_name, json_path in splits:
        if not json_path.exists():
            print(f"Skip {split_name}: {json_path} not found.")
            continue
        images, anns, _ = load_coco(json_path)
        split_images_dir = root / split_name
        labels_out_dir = root / "labels" / split_name
        write_yolo_labels(images, anns, categories, split_images_dir, labels_out_dir, class_name_to_id)

    # also write out a minimal data yaml suggestion
    data_yaml = root / "all_coco_yolo.yaml"
    yaml_content = [
        f"path: {root}",
        "train: train",
        "val: val",
        "test: test",
        f"names: {class_names_sorted}",
        "",
    ]
    data_yaml.write_text("\n".join(yaml_content), encoding="utf-8")
    print(f"Wrote data yaml to: {data_yaml}")


if __name__ == "__main__":
    main()









