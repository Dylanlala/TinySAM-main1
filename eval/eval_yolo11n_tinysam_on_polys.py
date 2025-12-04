#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch

# Ensure project root on sys.path for local 'tinysam' imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinysam.build_sam import sam_model_registry
from tinysam.predictor import SamPredictor

# Ultralytics
REPO_ROOT = os.path.dirname(__file__)
LOCAL_ULTRA = os.path.join(REPO_ROOT, "../ultralyticss_new")
if os.path.isdir(LOCAL_ULTRA) and LOCAL_ULTRA not in sys.path:
    sys.path.insert(0, LOCAL_ULTRA)
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(f"Failed to import YOLO from local ultralyticss_new at {LOCAL_ULTRA}: {e}")



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


def find_image_and_mask_pairs(images_root: Path, ann_root: Path) -> List[Tuple[Path, Optional[Path]]]:
    """Find image and annotation file pairs, supporting both image masks and YOLO format txt files"""
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for dirpath, _, filenames in os.walk(images_root):
        for fname in filenames:
            if not is_image_file(fname):
                continue
            img_path = Path(dirpath) / fname
            # Mirror subdir under ann_root
            rel = img_path.relative_to(images_root)
            mask_dir = ann_root / rel.parent
            mask_base = rel.stem
            
            # First try to find image format mask files
            gt_mask_path: Optional[Path] = None
            for suffix in [".png", ".jpg", ".tif", ".tiff", ".bmp"]:
                candidate = mask_dir / f"{mask_base}{suffix}"
                if candidate.exists():
                    gt_mask_path = candidate
                    break
                candidate2 = mask_dir / f"{mask_base}_mask{suffix}"
                if candidate2.exists():
                    gt_mask_path = candidate2
                    break
            
            # If no image mask found, try to find YOLO format txt files
            if gt_mask_path is None:
                txt_candidate = mask_dir / f"{mask_base}.txt"
                if txt_candidate.exists():
                    gt_mask_path = txt_candidate
            
            pairs.append((img_path, gt_mask_path))
    return pairs


class YOLODetector:
    """YOLO detector for polyp detection"""
    def __init__(self, weights_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        if YOLO is None:
            raise ImportError("ultralytics package is required for YOLO detection")
        
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in image and return list of detections with boxes and confidence scores"""
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id
                    })
        
        return detections


class TinySAMEvaluator:
    def __init__(
        self,
        sam_weights: str,
        yolo_weights: Optional[str] = None,
        device: Optional[str] = None,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        min_conf_for_sam: float = 0.3,  # Minimum confidence to pass detection to SAM
        use_min_conf_filter: bool = True,  # Whether to use the confidence filter
        save_visualizations: bool = False,
        visualization_dir: Optional[str] = None,
        visualization_limit: int = 0,
        empty_txt_as_negative: bool = False,
        box_scale: float = 1.0,
        use_point_prompt: bool = True,
        use_labels_as_prompts: bool = False,
        labels_are_yolo_normalized: bool = True,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.min_conf_for_sam = min_conf_for_sam
        self.use_min_conf_filter = use_min_conf_filter
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir
        self.visualization_limit = max(0, int(visualization_limit or 0))
        self._saved_viz = 0
        self.empty_txt_as_negative = empty_txt_as_negative
        self.box_scale = float(box_scale) if box_scale is not None else 1.0
        self.use_point_prompt = bool(use_point_prompt)
        self.use_labels_as_prompts = bool(use_labels_as_prompts)
        self.labels_are_yolo_normalized = bool(labels_are_yolo_normalized)
        
        # Load models
        self._load_sam(sam_weights)
        self._load_yolo(yolo_weights, yolo_conf, yolo_iou)
        
        print(f"YOLO-SAM evaluator initialized:")
        print(f"  - Using confidence filter: {use_min_conf_filter}")
        if use_min_conf_filter:
            print(f"  - Minimum confidence for SAM: {min_conf_for_sam}")
        if self.save_visualizations:
            out_dir = self.visualization_dir or "visualizations"
            os.makedirs(out_dir, exist_ok=True)
            self.visualization_dir = out_dir

    def _load_sam(self, sam_weights: str) -> None:
        """Load TinySAM model"""
        self.sam = sam_model_registry["vit_t"](checkpoint=sam_weights).to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

    def _load_yolo(self, yolo_weights: Optional[str], conf: float, iou: float) -> None:
        """Load YOLO detector if weights are provided"""
        self.yolo = None
        if yolo_weights:
            try:
                self.yolo = YOLODetector(yolo_weights, conf, iou)
                print(f"YOLO detector loaded with weights: {yolo_weights}")
            except Exception as e:
                print(f"Failed to load YOLO detector: {e}")
                self.yolo = None

    @torch.no_grad()
    def predict_mask_with_yolo_sam(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, int, List[Dict]]:
        """
        Use YOLO to detect polyps and then SAM to segment them
        
        Returns:
            combined_mask: Combined segmentation mask
            detection_count: Number of detections used
            yolo_detections: List of YOLO detections with confidence scores
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        
        combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        detection_count = 0
        yolo_detections = []
        
        if self.yolo is not None:
            # Get YOLO detections
            detections = self.yolo.detect(image_rgb)
            yolo_detections = detections.copy()
            
            h_img, w_img = image_bgr.shape[:2]
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                
                # Apply confidence filter if enabled
                if self.use_min_conf_filter and conf < self.min_conf_for_sam:
                    continue
                
                # Optionally expand the box
                if self.box_scale and self.box_scale != 1.0:
                    bw = x2 - x1
                    bh = y2 - y1
                    cx_mid = (x1 + x2) * 0.5
                    cy_mid = (y1 + y2) * 0.5
                    new_w = bw * self.box_scale
                    new_h = bh * self.box_scale
                    x1 = max(0, int(round(cx_mid - new_w * 0.5)))
                    x2 = min(w_img - 1, int(round(cx_mid + new_w * 0.5)))
                    y1 = max(0, int(round(cy_mid - new_h * 0.5)))
                    y2 = min(h_img - 1, int(round(cy_mid + new_h * 0.5)))
                    if x2 <= x1 or y2 <= y1:
                        continue

                # Use center point optionally and box as prompts for SAM
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=(np.array([[cx, cy]]) if self.use_point_prompt else None),
                    point_labels=(np.array([1]) if self.use_point_prompt else None),
                    box=np.array([x1, y1, x2, y2])
                )
                
                if len(masks) > 0:
                    combined_mask = np.logical_or(combined_mask, masks[0]).astype(np.uint8)
                    detection_count += 1
        else:
            # Fallback to grid prompts if YOLO is not available
            h, w = image_bgr.shape[:2]
            grid_x = np.linspace(w * 0.25, w * 0.75, num=3, dtype=int)
            grid_y = np.linspace(h * 0.25, h * 0.75, num=3, dtype=int)
            points = np.array([[x, y] for y in grid_y for x in grid_x])
            labels = np.ones(len(points), dtype=np.int32)
            masks, _, _ = self.sam_predictor.predict(point_coords=points, point_labels=labels)
            if len(masks) > 0:
                combined_mask = np.max(masks.astype(np.uint8), axis=0)
                detection_count = len(points)
        
        return combined_mask, detection_count, yolo_detections

    @torch.no_grad()
    def predict_mask_with_boxes(self, image_bgr: np.ndarray, boxes_list: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, int]:
        """Use provided pixel boxes as prompts for SAM (optionally with center point)."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)

        combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        detection_count = 0

        for x1, y1, x2, y2 in boxes_list:
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            masks, _, _ = self.sam_predictor.predict(
                point_coords=(np.array([[cx, cy]]) if self.use_point_prompt else None),
                point_labels=(np.array([1]) if self.use_point_prompt else None),
                box=np.array([x1, y1, x2, y2])
            )
            if len(masks) > 0:
                combined_mask = np.logical_or(combined_mask, masks[0]).astype(np.uint8)
                detection_count += 1

        return combined_mask, detection_count

    def _parse_boxes_from_txt(self, txt_path: Path, img_shape: Tuple[int, int, int]) -> Tuple[List[Tuple[int, int, int, int]], bool]:
        """Read boxes from txt. Supports two formats:
        - Pixel coords per line: x1 y1 x2 y2 (optionally first line 0/1 negative flag)
        - YOLO normalized per line: class cx cy w h in [0,1] (if labels_are_yolo_normalized=True)
        Returns: (boxes_list, is_negative)
        """
        try:
            with open(txt_path, 'r') as f:
                raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception:
            return [], False

        if not raw_lines:
            return [], False

        h, w = img_shape[:2]
        is_negative = False
        first = raw_lines[0]
        lines = raw_lines
        # Optional first-line flag 0/1 for negativity (pixel-coord convention)
        if first in ('0', '1'):
            is_negative = first == '0'
            lines = raw_lines[1:]

        boxes: List[Tuple[int, int, int, int]] = []
        for line in lines:
            parts = line.split()
            # Try YOLO normalized format: class cx cy w h
            if self.labels_are_yolo_normalized and len(parts) >= 5:
                try:
                    # Accept either with class id or without
                    if len(parts) == 5:
                        cx, cy, bw, bh = map(float, parts[0:5])
                    else:
                        _cls = float(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                    # Values expected in [0,1]
                    cx_px = cx * w
                    cy_px = cy * h
                    bw_px = bw * w
                    bh_px = bh * h
                    x1 = max(0, int(round(cx_px - bw_px * 0.5)))
                    y1 = max(0, int(round(cy_px - bh_px * 0.5)))
                    x2 = min(w - 1, int(round(cx_px + bw_px * 0.5)))
                    y2 = min(h - 1, int(round(cy_px + bh_px * 0.5)))
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))
                    continue
                except Exception:
                    pass
            # Fallback: pixel coords x1 y1 x2 y2
            if len(parts) >= 4:
                try:
                    x1 = max(0, int(float(parts[0])))
                    y1 = max(0, int(float(parts[1])))
                    x2 = min(w - 1, int(float(parts[2])))
                    y2 = min(h - 1, int(float(parts[3])))
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))
                except Exception:
                    continue

        return boxes, is_negative

    def _compute_metrics_numpy(self, gt_bin: np.ndarray, pr_bin: np.ndarray) -> dict:
        """Compute performance metrics using connected component analysis"""
        try:
            from skimage.measure import label, regionprops
        except ImportError:
            print("Warning: skimage not available, falling back to pixel-level metrics")
            return self._compute_metrics_pixel_level(gt_bin, pr_bin)
        
        # Label connected components
        gt_label = label(gt_bin)
        pr_label = label(pr_bin)
        
        # Get region properties
        gt_regions = regionprops(gt_label)
        pr_regions = regionprops(pr_label)
        
        if len(gt_regions) == 0 and len(pr_regions) == 0:
            return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        
        # Polyp-level metrics: match connected components
        image_dice, image_iou, image_tp, image_fp, image_fn = [], [], 0, 0, 0
        
        # For each ground truth region, find the best matching prediction region
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
                
                # Calculate intersection and union
                inter = np.logical_and(true_mask_i, pred_mask_i).sum()
                union = np.logical_or(true_mask_i, pred_mask_i).sum()
                
                iou = inter / union if union > 0 else 0.0
                dice = (2 * inter) / (ts + ps) if (ts + ps) > 0 else 0.0
                
                if iou > best_iou:
                    best_iou = iou
                    best_dice = dice
            
            # Only count matches with IoU > 0.5
            if best_iou > 0.5:
                image_dice.append(best_dice)
                image_iou.append(best_iou)
                image_tp += 1
            else:
                image_fn += 1
        
        # Calculate false positives (extra prediction regions)
        image_fp = max(0, len(pr_regions) - image_tp)
        
        # Calculate precision and recall
        precision = image_tp / (image_tp + image_fp) if (image_tp + image_fp) > 0 else 0.0
        recall = image_tp / (image_tp + image_fn) if (image_tp + image_fn) > 0 else 0.0
        
        # Calculate average Dice and IoU
        avg_dice = float(np.mean(image_dice)) if image_dice else 0.0
        avg_iou = float(np.mean(image_iou)) if image_iou else 0.0
        
        # Calculate F1 score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "dice": avg_dice,
            "iou": avg_iou,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": image_tp,
            "fp": image_fp,
            "fn": image_fn,
            "gt_regions": len(gt_regions),
            "pred_regions": len(pr_regions)
        }
    
    def _compute_metrics_pixel_level(self, gt_bin: np.ndarray, pr_bin: np.ndarray) -> dict:
        """Pixel-level performance metrics (fallback method)"""
        gt_flat = gt_bin.astype(np.uint8).flatten()
        pr_flat = pr_bin.astype(np.uint8).flatten()
        
        if gt_flat.size == 0 or pr_flat.size == 0:
            return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        
        tp = int(np.sum((gt_flat == 1) & (pr_flat == 1)))
        fp = int(np.sum((gt_flat == 0) & (pr_flat == 1)))
        fn = int(np.sum((gt_flat == 1) & (pr_flat == 0)))
        tn = int(np.sum((gt_flat == 0) & (pr_flat == 0)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        return {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

    def evaluate(self, images_root: Path, ann_root: Optional[Path]) -> dict:
        """Evaluate YOLO+TinySAM on the dataset"""
        # Find image and annotation pairs
        pairs = find_image_and_mask_pairs(images_root, ann_root) if ann_root else []
        image_list: List[Path] = []
        mask_lookup = {}
        missing_masks: List[str] = []
        
        if ann_root:
            for img_path, mpath in pairs:
                if mpath is None or not mpath.exists():
                    missing_masks.append(str(img_path))
                    continue
                image_list.append(img_path)
                mask_lookup[img_path] = mpath
        else:
            # Find all images if no annotation directory is provided
            print(f"Searching for images in: {images_root}")
            for dirpath, _, filenames in os.walk(images_root):
                for fname in filenames:
                    if is_image_file(fname):
                        img_path = Path(dirpath) / fname
                        if img_path.exists() and img_path.stat().st_size > 0:
                            image_list.append(img_path)
        
        print(f"Total images found: {len(image_list)}")
        
        if len(image_list) == 0:
            print("No images found! Please check the image directory path.")
            return {"error": "No images found", "summary": {}}

        broken_log = []
        results = []
        total_time = 0.0
        total_detections = 0
        total_yolo_detections = 0

        # Performance metrics accumulators (overall)
        dices = []
        ious = []
        precisions = []
        recalls = []
        f1s = []
        tps = []
        fps = []
        fns = []

        # Positive/Negative accumulators (specificity tracking like eval_tinysam_on_polys.py)
        pos_dices = []
        pos_ious = []
        pos_precisions = []
        pos_recalls = []
        pos_f1s = []
        pos_tp_sum = 0
        pos_fp_sum = 0
        pos_fn_sum = 0

        neg_images = 0
        neg_tn_images = 0
        neg_fp_images = 0

        print(f"Processing {len(image_list)} images...")
        
        for idx, img_path in enumerate(sorted(image_list)):
            if idx % 10 == 0:
                print(f"Processing {idx+1}/{len(image_list)}: {img_path.name}")
                
            img = try_read_image(img_path)
            if img is None:
                broken_log.append(str(img_path))
                continue

            start = time.time()
            
            # Predict mask
            pred_mask = None
            detection_count = 0
            yolo_detections = []
            used_label_boxes = False

            # If using labels as prompts and we have a matching txt, parse boxes and use them
            if ann_root:
                gt_path = mask_lookup.get(img_path)
                if gt_path and gt_path.exists() and gt_path.suffix.lower() == ".txt" and self.use_labels_as_prompts:
                    boxes_list, _neg = self._parse_boxes_from_txt(gt_path, img.shape)
                    if boxes_list:
                        pred_mask, detection_count = self.predict_mask_with_boxes(img, boxes_list)
                        yolo_detections = [{"bbox": list(b), "confidence": None, "class_id": None} for b in boxes_list]
                        used_label_boxes = True

            # Fallback to YOLO detector if not using labels or no boxes
            if pred_mask is None:
                pred_mask, detection_count, yolo_detections = self.predict_mask_with_yolo_sam(img)
                total_yolo_detections += len(yolo_detections)
            
            elapsed = time.time() - start
            total_time += elapsed
            total_detections += detection_count

            metrics = {}
            is_negative = False
            gt_mask_from_boxes = None
            if ann_root:
                gt_path = mask_lookup.get(img_path)
                if gt_path and gt_path.exists():
                    # Handle different annotation formats
                    if gt_path.suffix.lower() in [".png", ".jpg", ".tif", ".tiff", ".bmp"]:
                        # Image mask format
                        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                        if gt_mask is not None:
                            gt_bin = (gt_mask > 128).astype(np.uint8)
                            
                            # Ensure prediction mask matches GT size
                            gt_h, gt_w = gt_bin.shape
                            pred_h, pred_w = pred_mask.shape
                            
                            if pred_h != gt_h or pred_w != gt_w:
                                pred_resized = cv2.resize(pred_mask, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
                            else:
                                pred_resized = pred_mask
                            
                            pr_bin = (pred_resized > 0).astype(np.uint8)
                            
                            # Compute metrics
                            metrics = self._compute_metrics_numpy(gt_bin, pr_bin)

                            # Positive accumulators (image assumed positive when mask has any fg)
                            if gt_bin.sum() > 0:
                                pos_tp_sum += metrics.get("tp", 0)
                                pos_fp_sum += metrics.get("fp", 0)
                                pos_fn_sum += metrics.get("fn", 0)
                                pos_dices.append(metrics.get("dice", 0.0))
                                pos_ious.append(metrics.get("iou", 0.0))
                                pos_precisions.append(metrics.get("precision", 0.0))
                                pos_recalls.append(metrics.get("recall", 0.0))
                                if "f1" in metrics:
                                    pos_f1s.append(metrics.get("f1", 0.0))
                        else:
                            broken_log.append(str(gt_path))
                            continue
                    elif gt_path.suffix.lower() == ".txt":
                        # Build GT from txt: support YOLO-normalized or pixel coords
                        try:
                            h, w = img.shape[:2]
                            boxes_list, is_negative = self._parse_boxes_from_txt(gt_path, img.shape)

                            gt_mask = np.zeros((h, w), dtype=np.uint8)
                            if not is_negative and boxes_list:
                                for (x1, y1, x2, y2) in boxes_list:
                                    if x2 > x1 and y2 > y1:
                                        gt_mask[y1:y2, x1:x2] = 255

                            gt_bin = (gt_mask > 0).astype(np.uint8)

                            # Resize pred to GT
                            gt_h, gt_w = gt_bin.shape
                            pred_h, pred_w = pred_mask.shape
                            if pred_h != gt_h or pred_w != gt_w:
                                pred_resized = cv2.resize(pred_mask, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
                            else:
                                pred_resized = pred_mask
                            pr_bin = (pred_resized > 0).astype(np.uint8)

                            # Negative images: specificity via presence of predicted regions
                            if is_negative:
                                neg_images += 1
                                try:
                                    from skimage.measure import label
                                    pr_label = label(pr_bin)
                                    has_pred_region = pr_label.max() > 0
                                except Exception:
                                    has_pred_region = pr_bin.sum() > 0
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
                        except Exception as e:
                            print(f"Error processing annotation file {gt_path}: {e}")
                            broken_log.append(str(gt_path))
                            continue
                    else:
                        broken_log.append(str(gt_path))
                        continue

            # Accumulate metrics for summary
            if metrics:
                dices.append(metrics.get("dice", 0))
                ious.append(metrics.get("iou", 0))
                precisions.append(metrics.get("precision", 0))
                recalls.append(metrics.get("recall", 0))
                f1s.append(metrics.get("f1", 0))
                tps.append(metrics.get("tp", 0))
                fps.append(metrics.get("fp", 0))
                fns.append(metrics.get("fn", 0))

            results.append({
                "image": str(img_path.relative_to(images_root)),
                "time_sec": elapsed,
                "detections": detection_count,
                "yolo_detections": len(yolo_detections),
                "yolo_detections_info": yolo_detections,
                "metrics": metrics,
            })

            # Save visualization if requested
            if self.save_visualizations and (self.visualization_limit == 0 or self._saved_viz < self.visualization_limit):
                try:
                    vis = img.copy()
                    # Draw YOLO boxes
                    for det in yolo_detections:
                        x1, y1, x2, y2 = det['bbox']
                        conf = det.get('confidence', 0.0)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis, f"{conf:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    # Overlay SAM mask (red)
                    mask_vis = (pred_mask > 0).astype(np.uint8) * 255
                    if mask_vis.shape[:2] != vis.shape[:2]:
                        mask_vis = cv2.resize(mask_vis, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
                    colored = np.zeros_like(vis)
                    colored[:, :, 2] = mask_vis
                    vis = cv2.addWeighted(vis, 1.0, colored, 0.4, 0)

                    rel = Path(img_path).relative_to(images_root)
                    out_path = Path(self.visualization_dir) / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), vis)
                    self._saved_viz += 1
                except Exception as _e:
                    # Do not block evaluation on visualization errors
                    pass

        num = len(results)
        if num == 0:
            print("No images were successfully processed!")
            return {"error": "No images processed", "summary": {}}

        # Calculate summary statistics
        avg_time = total_time / num
        fps_avg_latency = 1.0 / avg_time if avg_time > 0 else 0.0
        fps_throughput = num / total_time if total_time > 0 else 0.0

        # Calculate average metrics
        avg_dice = float(np.mean(dices)) if dices else 0.0
        avg_iou = float(np.mean(ious)) if ious else 0.0
        avg_precision = float(np.mean(precisions)) if precisions else 0.0
        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        avg_f1 = float(np.mean(f1s)) if f1s else 0.0
        total_tp = sum(tps) if tps else 0
        total_fp = sum(fps) if fps else 0
        total_fn = sum(fns) if fns else 0
        
        # Calculate mDice and mIoU (mean of all image-level Dice and IoU scores)
        mDice = float(np.mean(dices)) if dices else 0.0
        mIoU = float(np.mean(ious)) if ious else 0.0

        # Negative specificity summary
        specificity = (neg_tn_images / (neg_tn_images + neg_fp_images)) if (neg_tn_images + neg_fp_images) > 0 else 0.0
        pos_avg = {
            "dice": float(np.mean(pos_dices)) if pos_dices else 0.0,
            "iou": float(np.mean(pos_ious)) if pos_ious else 0.0,
            "precision": float(np.mean(pos_precisions)) if pos_precisions else 0.0,
            "recall": float(np.mean(pos_recalls)) if pos_recalls else 0.0,
            "f1": float(np.mean(pos_f1s)) if pos_f1s else 0.0,
            "tp": int(sum(tps)) if tps else int(total_tp),
            "fp": int(sum(fps)) if fps else int(total_fp),
            "fn": int(sum(fns)) if fns else int(total_fn),
        }
        neg_avg = {
            "num_negative_images": int(neg_images),
            "tn_images": int(neg_tn_images),
            "fp_images": int(neg_fp_images),
            "specificity": float(specificity),
        }

        summary = {
            "total_images": num,
            "skipped_broken": len(broken_log),
            "total_time_sec": total_time,
            "avg_time_per_image_sec": avg_time,
            "fps_avg_latency": fps_avg_latency,
            "fps_throughput": fps_throughput,
            "total_detections": total_detections,
            "total_yolo_detections": total_yolo_detections,
            "avg_detections": total_detections / num if num > 0 else 0.0,
            "avg_yolo_detections": total_yolo_detections / num if num > 0 else 0.0,
            "avg_dice": avg_dice,
            "avg_iou": avg_iou,
            "mDice": mDice,
            "mIoU": mIoU,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "min_conf_for_sam": self.min_conf_for_sam,
            "use_min_conf_filter": self.use_min_conf_filter,
            "specificity": float(specificity),
            "positive_metrics": pos_avg,
            "negative_metrics": neg_avg,
        }

        return {
            "summary": summary,
            "results": results,
            "broken_images": broken_log,
            "missing_masks": missing_masks,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate YOLO+TinySAM on Polys dataset")
    parser.add_argument("--test-images", type=str, required=True, help="Path to test images directory")
    parser.add_argument("--test-annotations", type=str, required=False, help="Path to test annotations directory")
    parser.add_argument("--sam-weights", type=str, default="results_ldpolyvideo/best_model.pth", 
                       help="Path to TinySAM weights")
    parser.add_argument("--yolo-weights", type=str, required=True, help="Path to YOLO weights")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--out", type=str, default="yolo_tinysam_eval_report.json", 
                       help="Output JSON report filename")
    parser.add_argument("--broken-log", type=str, default="broken_images.txt", 
                       help="Filename for broken images log")
    parser.add_argument("--yolo-conf", type=float, default=0.25, 
                       help="YOLO confidence threshold for detection")
    parser.add_argument("--yolo-iou", type=float, default=0.45, 
                       help="YOLO IoU threshold for NMS")
    parser.add_argument("--min-conf-sam", type=float, default=0.3, 
                       help="Minimum confidence to pass detection to SAM")
    parser.add_argument("--no-conf-filter", action="store_true", 
                       help="Disable confidence filtering (use all YOLO detections)")
    parser.add_argument("--save-viz", action="store_true", 
                       help="Save visualizations with YOLO boxes and SAM masks")
    parser.add_argument("--viz-dir", type=str, default="results_ldpolyvideo/visualizations", 
                       help="Directory to save visualizations")
    parser.add_argument("--viz-max", type=int, default=0, 
                       help="Max number of visualizations to save (0 = no limit)")
    parser.add_argument("--empty-txt-as-negative", action="store_true", 
                       help="Treat empty annotation txt as negative samples for specificity")
    parser.add_argument("--box-scale", type=float, default=1.0, 
                       help="Scale factor to expand YOLO boxes before SAM prompting (e.g., 1.2)")
    parser.add_argument("--no-sam-point", action="store_true", 
                       help="Disable point prompt; use only box for SAM")
    parser.add_argument("--use-labels-as-prompts", action="store_true",
                       help="Use annotation txt boxes as SAM prompts (bypass YOLO detector)")
    parser.add_argument("--labels-are-yolo-normalized", action="store_true",
                       help="Treat txt labels as YOLO normalized cx cy w h (default true)")
    
    args = parser.parse_args()

    images_root = Path(args.test_images)
    ann_root = Path(args.test_annotations) if args.test_annotations else None

    if not images_root.exists():
        print(f"Test Images directory not found: {images_root}")
        sys.exit(1)
    if ann_root and not ann_root.exists():
        print(f"Annotations directory not found: {ann_root}")
        sys.exit(1)

    # Initialize evaluator
    evaluator = TinySAMEvaluator(
        sam_weights=args.sam_weights,
        yolo_weights=args.yolo_weights,
        device=args.device,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        min_conf_for_sam=args.min_conf_sam,
        use_min_conf_filter=not args.no_conf_filter,
        save_visualizations=args.save_viz,
        visualization_dir=args.viz_dir,
        visualization_limit=args.viz_max,
        empty_txt_as_negative=args.empty_txt_as_negative,
        box_scale=args.box_scale,
        use_point_prompt=not args.no_sam_point,
        use_labels_as_prompts=args.use_labels_as_prompts,
        labels_are_yolo_normalized=True if args.labels_are_yolo_normalized else True,
    )

    # Run evaluation
    report = evaluator.evaluate(images_root, ann_root)

    # Check for errors
    if "error" in report:
        print(f"❌ Error: {report['error']}")
        if "summary" in report and report["summary"]:
            # Save report even if there was an error but we have some results
            pass
        else:
            print("No summary available. Exiting.")
            return

    # Save detailed report
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    if report.get("broken_images"):
        with open(args.broken_log, "w") as f:
            for p in report["broken_images"]:
                f.write(p + "\n")

    # Print performance summary
    s = report["summary"]
    print("\n" + "="*70)
    print("🚀 YOLO+TinySAM Evaluation Report")
    print("="*70)
    print(f"📊 Images evaluated: {s['total_images']} (skipped broken: {s['skipped_broken']})")
    print(f"⏱️  Total time: {s['total_time_sec']:.2f}s | Avg time/img: {s['avg_time_per_image_sec']:.4f}s")
    print(f"🚀 FPS (avg latency): {s['fps_avg_latency']:.2f} | FPS (throughput): {s['fps_throughput']:.2f}")
    print(f"📌 YOLO detections: {s['total_yolo_detections']} | SAM segmentations: {s['total_detections']}")
    print(f"📈 Performance Metrics:")
    print(f"    mDice: {s['mDice']:.4f} | mIoU: {s['mIoU']:.4f}")
    print(f"    Dice: {s['avg_dice']:.4f} | IoU: {s['avg_iou']:.4f}")
    print(f"    Precision: {s['avg_precision']:.4f} | Recall: {s['avg_recall']:.4f}")
    print(f"    F1: {s['avg_f1']:.4f}")
    print(f"    TP: {s['total_tp']} | FP: {s['total_fp']} | FN: {s['total_fn']}")
    if 'specificity' in s:
        print(f"    Specificity: {s['specificity']:.4f}")
    print(f"🔧 Configuration:")
    print(f"    Confidence filter: {s['use_min_conf_filter']}")
    if s['use_min_conf_filter']:
        print(f"    Min confidence for SAM: {s['min_conf_for_sam']}")
    print(f"💾 Report saved to: {args.out}")
    print("="*70)


if __name__ == "__main__":
    main()