import os
import json
from typing import Dict, List, Tuple


def collect_single_frame_samples(root_dir: str) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    for center_id in range(1, 7):
        center_dir = os.path.join(root_dir, f"data_C{center_id}")
        images_dir = os.path.join(center_dir, f"images_C{center_id}")
        masks_dir = os.path.join(center_dir, f"masks_C{center_id}")
        bbox_dir = os.path.join(center_dir, f"bbox_C{center_id}")

        if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir) and os.path.isdir(bbox_dir)):
            continue

        for filename in os.listdir(images_dir):
            if not filename.endswith(".jpg"):
                continue
            image_path = os.path.join(images_dir, filename)
            mask_name = filename.replace(".jpg", "_mask.jpg")
            mask_path = os.path.join(masks_dir, mask_name)
            yolo_name = filename.replace(".jpg", "_mask.txt")
            yolo_path = os.path.join(bbox_dir, yolo_name)

            if os.path.isfile(mask_path) and os.path.isfile(yolo_path):
                samples.append({
                    "image": image_path,
                    "mask": mask_path,
                    "yolo": yolo_path,
                    "group": f"C{center_id}"
                })
    return samples


def collect_sequence_samples(root_dir: str) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    positive_dir = os.path.join(root_dir, "sequenceData", "positive")
    if not os.path.isdir(positive_dir):
        return samples

    seq_dirs = sorted([d for d in os.listdir(positive_dir) if d.startswith("seq")], key=lambda x: int(x[3:]))
    for seq_dir in seq_dirs:
        seq_path = os.path.join(positive_dir, seq_dir)
        images_dir = os.path.join(seq_path, f"images_{seq_dir}")
        masks_dir = os.path.join(seq_path, f"masks_{seq_dir}")
        bbox_dir = os.path.join(seq_path, f"bbox_{seq_dir}")

        if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir) and os.path.isdir(bbox_dir)):
            continue

        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
        for img_file in image_files:
            image_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file.replace(".jpg", "_mask.jpg"))
            yolo_path = os.path.join(bbox_dir, img_file.replace(".jpg", "_mask.txt"))
            if os.path.isfile(mask_path) and os.path.isfile(yolo_path):
                samples.append({
                    "image": image_path,
                    "mask": mask_path,
                    "yolo": yolo_path,
                    "group": f"seq_{seq_dir}"
                })
    return samples


def write_list_file(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    # Dataset root (v3)
    root_dir = \
        "/home/huangmanling/huangmanling/TinySAM-main/PolypGen/PolypGens/PolypGen2021_MultiCenterData_v3"
    out_dir = "/home/huangmanling/huangmanling/TinySAM-main/polypgen_splits_fixed"
    os.makedirs(out_dir, exist_ok=True)

    # Train = all single-frame C1..C6 (with both mask and txt)
    train_samples = collect_single_frame_samples(root_dir)
    # Test = all positive sequences (with both mask and txt)
    test_samples = collect_sequence_samples(root_dir)
    # Val = empty to avoid any leakage by design
    val_samples: List[Dict[str, str]] = []

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    # Write consolidated JSON
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    # Also write flat text lists for convenience (YOLO and TinySAM)
    def to_lines(records: List[Dict[str, str]], key: str) -> List[str]:
        return [rec[key] for rec in records if key in rec]

    write_list_file(os.path.join(out_dir, "train_images.txt"), to_lines(train_samples, "image"))
    write_list_file(os.path.join(out_dir, "train_masks.txt"), to_lines(train_samples, "mask"))
    write_list_file(os.path.join(out_dir, "train_yolo.txt"), to_lines(train_samples, "yolo"))

    write_list_file(os.path.join(out_dir, "test_images.txt"), to_lines(test_samples, "image"))
    write_list_file(os.path.join(out_dir, "test_masks.txt"), to_lines(test_samples, "mask"))
    write_list_file(os.path.join(out_dir, "test_yolo.txt"), to_lines(test_samples, "yolo"))

    write_list_file(os.path.join(out_dir, "val_images.txt"), to_lines(val_samples, "image"))
    write_list_file(os.path.join(out_dir, "val_masks.txt"), to_lines(val_samples, "mask"))
    write_list_file(os.path.join(out_dir, "val_yolo.txt"), to_lines(val_samples, "yolo"))

    print("Fixed splits generated:")
    print(f"  Train: {len(train_samples)} samples (C1..C6 single-frames)")
    print(f"  Val:   {len(val_samples)} samples (empty by design)")
    print(f"  Test:  {len(test_samples)} samples (positive sequences)")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()


