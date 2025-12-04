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

try:
    from ultralytics import YOLO  # optional
except Exception:
    YOLO = None


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
    """查找图像和标注文件对，支持YOLO格式的txt文件"""
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for dirpath, _, filenames in os.walk(images_root):
        for fname in filenames:
            if not is_image_file(fname):
                continue
            img_path = Path(dirpath) / fname
            # mirror subdir under ann_root
            rel = img_path.relative_to(images_root)
            mask_dir = ann_root / rel.parent
            mask_base = rel.stem
            
            # 首先尝试查找图像格式的掩码文件
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
            
            # 如果没有找到图像格式的掩码，尝试查找YOLO格式的txt文件
            if gt_mask_path is None:
                txt_candidate = mask_dir / f"{mask_base}.txt"
                if txt_candidate.exists():
                    gt_mask_path = txt_candidate
            
            pairs.append((img_path, gt_mask_path))
    return pairs


def create_mask_from_yolo_txt(txt_path: Path, img_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """从自定义边界框格式的txt文件创建掩码"""
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None
        
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 第一行是类别ID，第二行是边界框坐标
        if len(lines) >= 2:
            try:
                # 解析边界框坐标: x1 y1 x2 y2
                coords = lines[1].strip().split()
                if len(coords) >= 4:
                    x1 = max(0, int(float(coords[0])))
                    y1 = max(0, int(float(coords[1])))
                    x2 = min(w, int(float(coords[2])))
                    y2 = min(h, int(float(coords[3])))
                    
                    # 确保坐标有效
                    if x1 < x2 and y1 < y2:
                        # 在掩码上绘制矩形区域
                        mask[y1:y2, x1:x2] = 255
                        print(f"    Created mask for bbox: ({x1}, {y1}) to ({x2}, {y2})")
                    else:
                        print(f"    Invalid bbox coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                else:
                    print(f"    Insufficient coordinates in line: {lines[1].strip()}")
            except (ValueError, IndexError) as e:
                print(f"    Error parsing coordinates: {e}")
                return None
        
        return mask
    except Exception as e:
        print(f"Error reading annotation {txt_path}: {e}")
        return None


class TinySAMEvaluator:
    def __init__(
        self,
        sam_weights: str,
        yolo_weights: Optional[str] = None,
        device: Optional[str] = None,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load_sam(sam_weights)
        self._load_yolo(yolo_weights, yolo_conf, yolo_iou)

    def _load_sam(self, sam_weights: str) -> None:
        self.sam = sam_model_registry["vit_t"](checkpoint=sam_weights).to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

    def _load_yolo(self, yolo_weights: Optional[str], conf: float, iou: float) -> None:
        self.yolo = None
        self.yolo_conf = conf
        self.yolo_iou = iou
        if yolo_weights:
            if YOLO is None:
                raise RuntimeError("ultralytics not available but yolo_weights was provided")
            self.yolo = YOLO(yolo_weights)

    @torch.no_grad()
    def predict_mask_with_sam(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, int]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)

        combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        detection_count = 0

        if self.yolo is not None:
            yolo_results = self.yolo(image_rgb, conf=self.yolo_conf, iou=self.yolo_iou)
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=np.array([[cx, cy]]),
                    point_labels=np.array([1]),
                    box=np.array([x1, y1, x2, y2])
                )
                if len(masks) > 0:
                    combined_mask = np.logical_or(combined_mask, masks[0]).astype(np.uint8)
                    detection_count += 1
        else:
            # Fallback: sparse grid prompts across the image center region
            h, w = image_bgr.shape[:2]
            grid_x = np.linspace(w * 0.25, w * 0.75, num=3, dtype=int)
            grid_y = np.linspace(h * 0.25, h * 0.75, num=3, dtype=int)
            points = np.array([[x, y] for y in grid_y for x in grid_x])
            labels = np.ones(len(points), dtype=np.int32)
            masks, _, _ = self.sam_predictor.predict(point_coords=points, point_labels=labels)
            if len(masks) > 0:
                combined_mask = np.max(masks.astype(np.uint8), axis=0)
                detection_count = len(points)

        return combined_mask, detection_count

    def _compute_metrics_numpy(self, gt_bin: np.ndarray, pr_bin: np.ndarray) -> dict:
        """使用息肉级连通域匹配方法计算性能指标，参考训练脚本的逻辑"""
        try:
            from skimage.measure import label, regionprops
            print(f"    Using connected component analysis...")
        except ImportError:
            print("Warning: skimage not available, falling back to pixel-level metrics")
            return self._compute_metrics_pixel_level(gt_bin, pr_bin)
        
        # 使用连通域标记
        gt_label = label(gt_bin)
        pr_label = label(pr_bin)
        
        # 获取连通域属性
        gt_regions = regionprops(gt_label)
        pr_regions = regionprops(pr_label)
        
        print(f"    GT regions: {len(gt_regions)}, Pred regions: {len(pr_regions)}")
        
        if len(gt_regions) == 0 and len(pr_regions) == 0:
            print("    No regions found, returning zero metrics")
            return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        
        # 息肉级指标：逐样本按连通域匹配
        image_dice, image_iou, image_tp, image_fp, image_fn = [], [], 0, 0, 0
        
        # 对每个真实息肉区域，找到最佳匹配的预测区域
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
                
                # 计算交集和并集
                inter = np.logical_and(true_mask_i, pred_mask_i).sum()
                union = np.logical_or(true_mask_i, pred_mask_i).sum()
                
                iou = inter / union if union > 0 else 0.0
                dice = (2 * inter) / (ts + ps) if (ts + ps) > 0 else 0.0
                
                if iou > best_iou:
                    best_iou = iou
                    best_dice = dice
            
            # 只有IoU > 0.5的匹配才计入统计（参考训练脚本）
            if best_iou > 0.5:
                image_dice.append(best_dice)
                image_iou.append(best_iou)
                image_tp += 1
                print(f"    Matched region: IoU={best_iou:.4f}, Dice={best_dice:.4f}")
            else:
                image_fn += 1
                print(f"    Unmatched region: IoU={best_iou:.4f}")
        
        # 计算假阳性（多余的预测区域）
        image_fp = max(0, len(pr_regions) - image_tp)
        
        # 计算精确率和召回率
        precision = image_tp / (image_tp + image_fp) if (image_tp + image_fp) > 0 else 0.0
        recall = image_tp / (image_tp + image_fn) if (image_tp + image_fn) > 0 else 0.0
        
        # 计算平均Dice和IoU
        avg_dice = float(np.mean(image_dice)) if image_dice else 0.0
        avg_iou = float(np.mean(image_iou)) if image_iou else 0.0
        
        # 计算F1分数
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"    Final metrics: TP={image_tp}, FP={image_fp}, FN={image_fn}")
        print(f"    Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
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
        """像素级性能指标计算（备用方法）"""
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

    def _read_boxes_from_txt(self, txt_path: Path, img_shape: Tuple[int, int, int]) -> Tuple[List[Tuple[int, int, int, int]], bool]:
        """从txt文件读取边界框坐标，并识别阴性样本（首行0表示无息肉）。

        返回: (boxes_list, is_negative)
        """
        try:
            with open(txt_path, 'r') as f:
                raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            
            if not raw_lines:
                return [], False
            
            # 首行可能是标志位：0=阴性，1=阳性
            first = raw_lines[0]
            is_negative = first == '0'
            # 如果首行是0或1，则剩余行才是框；否则所有行为框
            lines = raw_lines[1:] if first in ('0', '1') else raw_lines
            
            boxes: List[Tuple[int, int, int, int]] = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 4:
                    # 解析边界框坐标: x1 y1 x2 y2
                    x1 = max(0, int(float(parts[0])))
                    y1 = max(0, int(float(parts[1])))
                    x2 = min(img_shape[1], int(float(parts[2])))
                    y2 = min(img_shape[0], int(float(parts[3])))
                    if x1 < x2 and y1 < y2:
                        boxes.append((x1, y1, x2, y2))
            return boxes, is_negative
        except Exception as e:
            print(f"Error reading boxes from {txt_path}: {e}")
            return [], False
    
    def predict_mask_with_boxes(self, image_bgr: np.ndarray, boxes_list: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, int]:
        """使用边界框提示进行SAM预测（参考训练脚本）"""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        
        combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        detection_count = 0
        
        for x1, y1, x2, y2 in boxes_list:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            masks, _, _ = self.sam_predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                box=np.array([x1, y1, x2, y2])
            )
            
            if len(masks) > 0:
                combined_mask = np.logical_or(combined_mask, masks[0]).astype(np.uint8)
                detection_count += 1
        
        return combined_mask, detection_count

    def evaluate(self, images_root: Path, ann_root: Optional[Path]) -> dict:
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
            # 改进的图像文件查找逻辑
            print(f"Searching for images in: {images_root}")
            for dirpath, _, filenames in os.walk(images_root):
                for fname in filenames:
                    if is_image_file(fname):
                        img_path = Path(dirpath) / fname
                        # 检查文件是否可读
                        if img_path.exists() and img_path.stat().st_size > 0:
                            image_list.append(img_path)
                            if len(image_list) % 100 == 0:
                                print(f"Found {len(image_list)} images so far...")
        
        print(f"Total images found: {len(image_list)}")
        
        if len(image_list) == 0:
            print("No images found! Please check the image directory path.")
            return {"error": "No images found", "summary": {}}

        broken_log = []
        results = []
        total_time = 0.0
        total_detections = 0

        # Positive/Negative accumulators
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
            
            # 从标注文件获取边界框提示，并构建GT掩码（适配LDPolypVideo txt）
            boxes_list = []
            gt_mask_from_boxes = None
            is_negative = False
            if ann_root:
                gt_path = mask_lookup.get(img_path)
                if gt_path and gt_path.exists():
                    if gt_path.suffix.lower() == ".txt":
                        # 从txt文件读取边界框和阴性标志
                        boxes, is_negative = self._read_boxes_from_txt(gt_path, img.shape)
                        if boxes:
                            boxes_list = boxes
                        # 阴性样本：GT掩码全零；阳性样本：用所有boxes绘制GT掩码
                        h, w = img.shape[:2]
                        m = np.zeros((h, w), dtype=np.uint8)
                        if not is_negative and boxes:
                            for (x1, y1, x2, y2) in boxes:
                                x1c = max(0, min(w - 1, x1))
                                y1c = max(0, min(h - 1, y1))
                                x2c = max(0, min(w - 1, x2))
                                y2c = max(0, min(h - 1, y2))
                                if x2c > x1c and y2c > y1c:
                                    m[y1c:y2c, x1c:x2c] = 255
                        gt_mask_from_boxes = m
            
            # 使用边界框提示进行预测（如果可用）
            if boxes_list:
                pred_mask, detections = self.predict_mask_with_boxes(img, boxes_list)
            else:
                pred_mask, detections = self.predict_mask_with_sam(img)
            
            elapsed = time.time() - start
            total_time += elapsed
            total_detections += detections

            metrics = {}
            if ann_root:
                gt_path = mask_lookup.get(img_path)
                if gt_path and gt_path.exists():
                    # 根据文件扩展名判断是图像还是LDPolypVideo格式的txt
                    if gt_path.suffix.lower() in [".png", ".jpg", ".tif", ".tiff", ".bmp"]:
                        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                        if gt_mask is not None:
                            # 确保ground truth mask是二值的
                            gt_bin = (gt_mask > 128).astype(np.uint8)
                            
                            # 关键修复：确保预测掩码和GT掩码具有相同的尺寸
                            gt_h, gt_w = gt_bin.shape
                            pred_h, pred_w = pred_mask.shape
                            
                            # 如果尺寸不匹配，调整预测掩码到GT尺寸
                            if pred_h != gt_h or pred_w != gt_w:
                                pred_resized = cv2.resize(pred_mask, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
                            else:
                                pred_resized = pred_mask
                            
                            # 二值化预测掩码
                            pr_bin = (pred_resized > 0).astype(np.uint8)
                            
                            # 调试信息：显示掩码尺寸
                            if idx < 5:
                                print(f"    Mask sizes: GT={gt_bin.shape}, Pred={pred_mask.shape}, Resized={pred_resized.shape}")
                                print(f"    GT sum: {gt_bin.sum()}, Pred sum: {pr_bin.sum()}")
                            
                            # 计算性能指标
                            metrics = self._compute_metrics_numpy(gt_bin, pr_bin)
                            
                            # 调试信息
                            if idx < 5:  # 只对前5张图片输出调试信息
                                print(f"  Debug {img_path.name}: GT sum={gt_bin.sum()}, Pred sum={pr_bin.sum()}")
                                print(f"    GT regions: {metrics.get('gt_regions', 0)}, Pred regions: {metrics.get('pred_regions', 0)}")
                                print(f"    TP: {metrics.get('tp', 0)}, FP: {metrics.get('fp', 0)}, FN: {metrics.get('fn', 0)}")
                                print(f"    Dice: {metrics.get('dice', 0):.4f}, IoU: {metrics.get('iou', 0):.4f}")
                                print(f"    Precision: {metrics.get('precision', 0):.4f}, Recall: {metrics.get('recall', 0):.4f}")
                                if 'f1' in metrics:
                                    print(f"    F1: {metrics.get('f1', 0):.4f}")
                        else:
                            broken_log.append(str(gt_path))
                            continue
                    elif gt_path.suffix.lower() == ".txt":
                        # 从LDPolypVideo的txt文件创建掩码（支持阴性样本）
                        gt_mask = gt_mask_from_boxes
                        if gt_mask is not None:
                            gt_bin = (gt_mask > 0).astype(np.uint8)
                            gt_h, gt_w = gt_bin.shape
                            pred_h, pred_w = pred_mask.shape
                            if pred_h != gt_h or pred_w != gt_w:
                                pred_resized = cv2.resize(pred_mask, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
                            else:
                                pred_resized = pred_mask
                            pr_bin = (pred_resized > 0).astype(np.uint8)
                            if idx < 5:
                                print(f"    Mask sizes: GT={gt_bin.shape}, Pred={pred_mask.shape}, Resized={pred_resized.shape}")
                                print(f"    GT sum: {gt_bin.sum()}, Pred sum: {pr_bin.sum()}")
                            # 阴性样本统计特异性；阳性样本按连通域计算
                            if is_negative:
                                neg_images += 1
                                # 预测是否有任何连通域
                                from skimage.measure import label
                                pr_label = label(pr_bin)
                                has_pred_region = pr_label.max() > 0
                                if has_pred_region:
                                    neg_fp_images += 1
                                    metrics = {"specificity": 0.0, "dice": 0.0, "iou": 0.0}
                                else:
                                    neg_tn_images += 1
                                    metrics = {"specificity": 1.0, "dice": 1.0, "iou": 1.0}
                            else:
                                metrics = self._compute_metrics_numpy(gt_bin, pr_bin)
                                # 累计正样本指标
                                pos_tp_sum += metrics.get("tp", 0)
                                pos_fp_sum += metrics.get("fp", 0)
                                pos_fn_sum += metrics.get("fn", 0)
                                pos_dices.append(metrics.get("dice", 0.0))
                                pos_ious.append(metrics.get("iou", 0.0))
                                pos_precisions.append(metrics.get("precision", 0.0))
                                pos_recalls.append(metrics.get("recall", 0.0))
                                if "f1" in metrics:
                                    pos_f1s.append(metrics.get("f1", 0.0))
                            
                            # 调试信息
                            if idx < 5:  # 只对前5张图片输出调试信息
                                print(f"  Debug {img_path.name}: GT sum={gt_bin.sum()}, Pred sum={pr_bin.sum()}")
                                print(f"    GT regions: {metrics.get('gt_regions', 0)}, Pred regions: {metrics.get('pred_regions', 0)}")
                                print(f"    TP: {metrics.get('tp', 0)}, FP: {metrics.get('fp', 0)}, FN: {metrics.get('fn', 0)}")
                                print(f"    Dice: {metrics.get('dice', 0):.4f}, IoU: {metrics.get('iou', 0):.4f}")
                                print(f"    Precision: {metrics.get('precision', 0):.4f}, Recall: {metrics.get('recall', 0):.4f}")
                                if 'f1' in metrics:
                                    print(f"    F1: {metrics.get('f1', 0):.4f}")
                        else:
                            broken_log.append(str(gt_path))
                            continue
                    else:
                        broken_log.append(str(gt_path))
                        continue

            results.append({
                "image": str(img_path.relative_to(images_root)),
                "time_sec": elapsed,
                "detections": detections,
                "metrics": metrics,
            })

        num = len(results)
        if num == 0:
            print("No images were successfully processed!")
            return {"error": "No images processed", "summary": {}}

        avg_time = total_time / num
        fps_avg_latency = 1.0 / avg_time if avg_time > 0 else 0.0
        fps_throughput = num / total_time if total_time > 0 else 0.0

        # 计算平均性能指标（分别统计阳性和阴性，再给出综合）
        metrics_list = [r["metrics"] for r in results if r["metrics"]]
        # mDice/mIoU（image-level mean over all images; negatives counted as 1 when perfectly empty, 0 otherwise）
        mdice_vals = [m["dice"] for m in metrics_list if "dice" in m]
        miou_vals = [m["iou"] for m in metrics_list if "iou" in m]
        m_dice = float(np.mean(mdice_vals)) if mdice_vals else 0.0
        m_iou = float(np.mean(miou_vals)) if miou_vals else 0.0
        # 正样本平均
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
        # 阴性样本特异性
        specificity = (neg_tn_images / (neg_tn_images + neg_fp_images)) if (neg_tn_images + neg_fp_images) > 0 else 0.0
        neg_avg = {
            "num_negative_images": int(neg_images),
            "tn_images": int(neg_tn_images),
            "fp_images": int(neg_fp_images),
            "specificity": float(specificity),
        }
        # 综合：对正样本的dice/iou/precision/recall/f1取均值；报告阴性的specificity
        avg_dice = pos_avg["dice"]
        avg_iou = pos_avg["iou"]
        avg_precision = pos_avg["precision"]
        avg_recall = pos_avg["recall"]
        avg_f1 = pos_avg["f1"]

        # 计算平均检测数量
        avg_detections = total_detections / num if num > 0 else 0.0

        summary = {
            "total_images": num,
            "skipped_broken": len(broken_log),
            "total_time_sec": total_time,
            "avg_time_per_image_sec": avg_time,
            "fps_avg_latency": fps_avg_latency,
            "fps_throughput": fps_throughput,
            "total_detections": total_detections,
            "avg_detections": avg_detections,
            "avg_dice": avg_dice,
            "avg_iou": avg_iou,
            "mDice": m_dice,
            "mIoU": m_iou,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
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

    parser = argparse.ArgumentParser(description="Evaluate TinySAM on Polys/Test with improved metrics")
    parser.add_argument("--test-images", type=str, required=True, help="Path to Polys/Test/Images directory")
    parser.add_argument("--test-annotations", type=str, required=False, help="Path to Polys/Test/Annotations directory")
    parser.add_argument("--sam-weights", type=str, default="../results_ldpolyvideo/best_model.pth")
    parser.add_argument("--yolo-weights", type=str, default=None, help="Optional YOLO weights for box prompts")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="polys_tinysam_eval_report.json")
    parser.add_argument("--broken-log", type=str, default="broken_images.txt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    args = parser.parse_args()

    images_root = Path(args.test_images)
    ann_root = Path(args.test_annotations) if args.test_annotations else None

    if not images_root.exists():
        print(f"Test Images directory not found: {images_root}")
        sys.exit(1)
    if ann_root and not ann_root.exists():
        print(f"Annotations directory not found: {ann_root}")
        sys.exit(1)

    evaluator = TinySAMEvaluator(
        sam_weights=args.sam_weights,
        yolo_weights=args.yolo_weights,
        device=args.device,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
    )

    report = evaluator.evaluate(images_root, ann_root)

    # 检查是否有错误
    if "error" in report:
        print(f"❌ Error: {report['error']}")
        if "summary" in report and report["summary"]:
            # 如果有summary，仍然可以保存报告
            pass
        else:
            print("No summary available. Exiting.")
            return

    # 保存详细报告
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    if report.get("broken_images"):
        with open(args.broken_log, "w") as f:
            for p in report["broken_images"]:
                f.write(p + "\n")

    # 打印详细的性能报告
    s = report["summary"]
    print("\n" + "="*70)
    print("🚀 TinySAM Evaluation Report (Polys/Test)")
    print("="*70)
    print(f"📊 Images evaluated: {s['total_images']} (skipped broken: {s['skipped_broken']})")
    print(f"⏱️  Total time: {s['total_time_sec']:.2f}s | Avg time/img: {s['avg_time_per_image_sec']:.4f}s")
    print(f"🚀 FPS (avg latency): {s['fps_avg_latency']:.2f} | FPS (throughput): {s['fps_throughput']:.2f}")
    print(f"📌 mDice: {s.get('mDice', 0.0):.4f} | mIoU: {s.get('mIoU', 0.0):.4f} | Specificity: {s.get('specificity', 0.0):.4f}")
    print(f"🎯 Total detections: {s['total_detections']} | Avg detections: {s['avg_detections']:.2f}")
    print(f"📈 Performance Metrics:")
    print(f"    Dice: {s['avg_dice']:.4f} | IoU: {s['avg_iou']:.4f}")
    print(f"    Precision: {s['avg_precision']:.4f} | Recall: {s['avg_recall']:.4f}")
    print(f"    F1: {s['avg_f1']:.4f}")
    print(f"💾 Report saved to: {args.out}")
    print("="*70)


if __name__ == "__main__":
    main()


