#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure project root on sys.path for local 'tinysam' imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinysam.build_sam import sam_model_registry
from tinysam.predictor import SamPredictor


def is_image_file(filename: str) -> bool:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    return any(filename.lower().endswith(ext) for ext in exts)


def try_read_image(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return img
    except Exception:
        return None


def find_image_and_mask_pairs(images_root: Path, ann_root: Optional[Path]) -> Tuple[List[Path], dict]:
    image_list: List[Path] = []
    mask_lookup = {}
    if ann_root is None:
        # images only
        for dirpath, _, filenames in os.walk(images_root):
            for fname in filenames:
                if is_image_file(fname):
                    image_list.append(Path(dirpath) / fname)
        return sorted(image_list), mask_lookup

    for dirpath, _, filenames in os.walk(images_root):
        for fname in filenames:
            if not is_image_file(fname):
                continue
            img_path = Path(dirpath) / fname
            rel = img_path.relative_to(images_root)
            mask_dir = ann_root / rel.parent
            mask_base = rel.stem
            mpath: Optional[Path] = None
            for suffix in [".png", ".jpg", ".tif", ".tiff", ".bmp"]:
                c1 = mask_dir / f"{mask_base}{suffix}"
                if c1.exists():
                    mpath = c1
                    break
                c2 = mask_dir / f"{mask_base}_mask{suffix}"
                if c2.exists():
                    mpath = c2
                    break
            if mpath is None:
                txt = mask_dir / f"{mask_base}.txt"
                if txt.exists():
                    mpath = txt
            image_list.append(img_path)
            if mpath is not None:
                mask_lookup[img_path] = mpath
    return sorted(image_list), mask_lookup


def read_boxes_and_negative_flag_from_txt(txt_path: Path, img_shape: Tuple[int, int, int]) -> Tuple[List[Tuple[int, int, int, int]], bool]:
    try:
        with open(txt_path, 'r') as f:
            raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not raw_lines:
            return [], False
        first = raw_lines[0]
        is_negative = first == '0'
        lines = raw_lines[1:] if first in ('0', '1') else raw_lines
        boxes: List[Tuple[int, int, int, int]] = []
        h, w = img_shape[:2]
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                x1 = max(0, int(float(parts[0])))
                y1 = max(0, int(float(parts[1])))
                x2 = min(w, int(float(parts[2])))
                y2 = min(h, int(float(parts[3])))
                if x1 < x2 and y1 < y2:
                    boxes.append((x1, y1, x2, y2))
        return boxes, is_negative
    except Exception:
        return [], False


def build_mask_from_boxes(img_shape: Tuple[int, int, int], boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    h, w = img_shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = 255
    return m


class TinySAMEvaluatorNoTxt:
    def __init__(self, sam_weights: str, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load_sam(sam_weights)

    def _load_sam(self, sam_weights: str) -> None:
        self.sam = sam_model_registry["vit_t"](checkpoint=sam_weights).to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

    @torch.no_grad()
    def predict_mask_with_grid(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, int]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        h, w = image_bgr.shape[:2]
        grid_x = np.linspace(w * 0.25, w * 0.75, num=3, dtype=int)
        grid_y = np.linspace(h * 0.25, h * 0.75, num=3, dtype=int)
        points = np.array([[x, y] for y in grid_y for x in grid_x])
        labels = np.ones(len(points), dtype=np.int32)
        masks, _, _ = self.sam_predictor.predict(point_coords=points, point_labels=labels)
        if len(masks) > 0:
            combined = np.max(masks.astype(np.uint8), axis=0)
            return combined, len(points)
        return np.zeros((h, w), dtype=np.uint8), len(points)

    def _compute_metrics_numpy(self, gt_bin: np.ndarray, pr_bin: np.ndarray) -> dict:
        try:
            from skimage.measure import label, regionprops
        except ImportError:
            return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}
        gt_label = label(gt_bin)
        pr_label = label(pr_bin)
        gt_regions = regionprops(gt_label)
        pr_regions = regionprops(pr_label)
        if len(gt_regions) == 0 and len(pr_regions) == 0:
            return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}
        image_dice, image_iou, image_tp = [], [], 0
        for tr in gt_regions:
            true_mask_i = (gt_label == tr.label)
            best_iou = 0.0
            best_dice = 0.0
            ts = true_mask_i.sum()
            for pr in pr_regions:
                pred_mask_i = (pr_label == pr.label)
                ps = pred_mask_i.sum()
                if ps == 0 or ts == 0:
                    continue
                inter = np.logical_and(true_mask_i, pred_mask_i).sum()
                union = np.logical_or(true_mask_i, pred_mask_i).sum()
                iou = inter / union if union > 0 else 0.0
                dice = (2 * inter) / (ts + ps) if (ts + ps) > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_dice = dice
            if best_iou > 0.5:
                image_dice.append(best_dice)
                image_iou.append(best_iou)
                image_tp += 1
        image_fp = max(0, len(pr_regions) - image_tp)
        image_fn = max(0, len(gt_regions) - image_tp)
        precision = image_tp / (image_tp + image_fp) if (image_tp + image_fp) > 0 else 0.0
        recall = image_tp / (image_tp + image_fn) if (image_tp + image_fn) > 0 else 0.0
        avg_dice = float(np.mean(image_dice)) if image_dice else 0.0
        avg_iou = float(np.mean(image_iou)) if image_iou else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"dice": avg_dice, "iou": avg_iou, "precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": image_tp, "fp": image_fp, "fn": image_fn}

    def evaluate(self, images_root: Path, ann_root: Optional[Path]) -> dict:
        image_list, mask_lookup = find_image_and_mask_pairs(images_root, ann_root)
        if len(image_list) == 0:
            print("No images found! Please check the image directory path.")
            return {"error": "No images found", "summary": {}}

        broken_log = []
        results = []
        total_time = 0.0
        total_detections = 0

        # Positive/Negative accumulators
        pos_dices: List[float] = []
        pos_ious: List[float] = []
        pos_precisions: List[float] = []
        pos_recalls: List[float] = []
        pos_f1s: List[float] = []
        pos_tp_sum = 0
        pos_fp_sum = 0
        pos_fn_sum = 0

        neg_images = 0
        neg_tn_images = 0
        neg_fp_images = 0

        print(f"Processing {len(image_list)} images...")
        for idx, img_path in enumerate(sorted(image_list)):
            img = try_read_image(img_path)
            if img is None:
                broken_log.append(str(img_path))
                continue

            start = time.time()
            pred_mask, detections = self.predict_mask_with_grid(img)  # NEVER use txt prompts
            elapsed = time.time() - start
            total_time += elapsed
            total_detections += detections

            metrics = {}
            if ann_root:
                gt_path = mask_lookup.get(img_path)
                if gt_path and gt_path.exists():
                    if gt_path.suffix.lower() in [".png", ".jpg", ".tif", ".tiff", ".bmp"]:
                        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                        if gt_mask is None:
                            broken_log.append(str(gt_path))
                            continue
                        gt_bin = (gt_mask > 128).astype(np.uint8)
                    elif gt_path.suffix.lower() == ".txt":
                        boxes, is_negative = read_boxes_and_negative_flag_from_txt(gt_path, img.shape)
                        if is_negative:
                            gt_bin = np.zeros(img.shape[:2], dtype=np.uint8)
                        else:
                            gt_bin = (build_mask_from_boxes(img.shape, boxes) > 0).astype(np.uint8)
                    else:
                        gt_bin = None

                    if gt_bin is not None:
                        gt_h, gt_w = gt_bin.shape
                        pr_h, pr_w = pred_mask.shape
                        if pr_h != gt_h or pr_w != gt_w:
                            pred_resized = cv2.resize(pred_mask, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
                        else:
                            pred_resized = pred_mask
                        pr_bin = (pred_resized > 0).astype(np.uint8)

                        is_negative_img = (gt_bin.sum() == 0)
                        if is_negative_img:
                            neg_images += 1
                            from skimage.measure import label
                            has_pred_region = label(pr_bin).max() > 0
                            if has_pred_region:
                                neg_fp_images += 1
                                metrics = {"specificity": 0.0, "dice": 0.0, "iou": 0.0}
                            else:
                                neg_tn_images += 1
                                metrics = {"specificity": 1.0, "dice": 1.0, "iou": 1.0}
                        else:
                            metrics = self._compute_metrics_numpy(gt_bin, pr_bin)
                            pos_tp_sum += metrics.get("tp", 0)
                            pos_fp_sum += metrics.get("fp", 0)
                            pos_fn_sum += metrics.get("fn", 0)
                            pos_dices.append(metrics.get("dice", 0.0))
                            pos_ious.append(metrics.get("iou", 0.0))
                            pos_precisions.append(metrics.get("precision", 0.0))
                            pos_recalls.append(metrics.get("recall", 0.0))
                            if "f1" in metrics:
                                pos_f1s.append(metrics.get("f1", 0.0))

            results.append({
                "image": str(img_path.relative_to(images_root)),
                "time_sec": elapsed,
                "detections": detections,
                "metrics": metrics,
            })

        num = len(results)
        if num == 0:
            return {"error": "No images processed", "summary": {}}

        avg_time = total_time / num
        fps_avg_latency = 1.0 / avg_time if avg_time > 0 else 0.0
        fps_throughput = num / total_time if total_time > 0 else 0.0

        metrics_list = [r["metrics"] for r in results if r["metrics"]]
        mdice_vals = [m.get("dice", 0.0) for m in metrics_list]
        miou_vals = [m.get("iou", 0.0) for m in metrics_list]
        m_dice = float(np.mean(mdice_vals)) if mdice_vals else 0.0
        m_iou = float(np.mean(miou_vals)) if miou_vals else 0.0

        pos_avg = {
            "dice": float(np.mean(pos_dices)) if pos_dices else 0.0,
            "iou": float(np.mean(pos_ious)) if pos_ious else 0.0,
            "precision": float(np.mean(pos_precisions)) if pos_precisions else 0.0,
            "recall": float(np.mean(pos_recalls)) if pos_recalls else 0.0,
            "f1": float(np.mean(pos_f1s)) if pos_f1s else 0.0,
            "tp": int(pos_tp_sum),
            "fp": int(pos_fp_sum),
            "fn": int(pos_fn_sum),
        }
        specificity = (neg_tn_images / (neg_tn_images + neg_fp_images)) if (neg_tn_images + neg_fp_images) > 0 else 0.0

        summary = {
            "total_images": num,
            "skipped_broken": len(broken_log),
            "total_time_sec": total_time,
            "avg_time_per_image_sec": avg_time,
            "fps_avg_latency": fps_avg_latency,
            "fps_throughput": fps_throughput,
            "total_detections": total_detections,
            "avg_detections": total_detections / num if num > 0 else 0.0,
            "avg_dice": pos_avg["dice"],
            "avg_iou": pos_avg["iou"],
            "mDice": m_dice,
            "mIoU": m_iou,
            "avg_precision": pos_avg["precision"],
            "avg_recall": pos_avg["recall"],
            "avg_f1": pos_avg["f1"],
            "specificity": float(specificity),
            "positive_metrics": pos_avg,
        }

        return {"summary": summary, "results": results, "broken_images": broken_log}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TinySAM official weights on Polys/Test without txt prompts")
    parser.add_argument("--test-images", type=str, required=True, help="Path to Polys/Test/Images directory")
    parser.add_argument("--test-annotations", type=str, required=False, help="Path to Polys/Test/Annotations directory (for GT only)")
    parser.add_argument("--sam-weights", type=str, required=True, help="Path to official TinySAM weights, e.g., weights/tinysam_xxx.pth")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="polys_tinysam_official_no_txt_report.json")
    args = parser.parse_args()

    images_root = Path(args.test_images)
    ann_root = Path(args.test_annotations) if args.test_annotations else None

    if not images_root.exists():
        print(f"Test Images directory not found: {images_root}")
        sys.exit(1)
    if ann_root and not ann_root.exists():
        print(f"Annotations directory not found: {ann_root}")
        sys.exit(1)

    evaluator = TinySAMEvaluatorNoTxt(sam_weights=args.sam_weights, device=args.device)
    report = evaluator.evaluate(images_root, ann_root)

    if "error" in report:
        print(f"❌ Error: {report['error']}")
        return

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    print("\n" + "="*70)
    print("🚀 TinySAM Evaluation (Official Weights, No TXT Prompts)")
    print("="*70)
    print(f"📊 Images evaluated: {s['total_images']} (skipped broken: {s['skipped_broken']})")
    print(f"⏱️  Total time: {s['total_time_sec']:.2f}s | Avg time/img: {s['avg_time_per_image_sec']:.4f}s")
    print(f"🚀 FPS (avg latency): {s['fps_avg_latency']:.2f} | FPS (throughput): {s['fps_throughput']:.2f}")
    print(f"📌 mDice: {s.get('mDice', 0.0):.4f} | mIoU: {s.get('mIoU', 0.0):.4f} | Specificity: {s.get('specificity', 0.0):.4f}")
    print(f"📈 Positives (avg): Dice={s['avg_dice']:.4f}, IoU={s['avg_iou']:.4f}, Precision={s['avg_precision']:.4f}, Recall={s['avg_recall']:.4f}, F1={s['avg_f1']:.4f}")
    print(f"💾 Report saved to: {args.out}")
    print("="*70)


if __name__ == "__main__":
    main()


