#!/usr/bin/env python3
"""
使用GT bbox作为prompt，评估TinySAM在测试集上的分割性能
"""

import os
import sys
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# 添加路径
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from tinysam.build_sam import sam_model_registry
from tinysam.predictor import SamPredictor
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


def parse_yolo_label(txt_path, img_width, img_height):
    """解析YOLO格式标签，返回像素坐标的bbox列表"""
    boxes = []
    if not txt_path.exists():
        return boxes
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                # YOLO格式: class_id cx cy w h (归一化)
                cx = float(parts[1]) * img_width
                cy = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height
                
                # 转换为 xmin, ymin, xmax, ymax
                xmin = max(0, int(cx - w / 2))
                ymin = max(0, int(cy - h / 2))
                xmax = min(img_width - 1, int(cx + w / 2))
                ymax = min(img_height - 1, int(cy + h / 2))
                
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
    
    return boxes


def calculate_metrics(pred_mask, gt_mask):
    """计算分割指标"""
    # 转换为二值mask
    pred_flat = (pred_mask > 128).astype(np.uint8).flatten()
    gt_flat = (gt_mask > 128).astype(np.uint8).flatten()
    
    # 如果两者都是空，返回完美分数
    if gt_flat.sum() == 0 and pred_flat.sum() == 0:
        return {
            'dice': 1.0,
            'iou': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0
        }
    
    # 计算指标
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def evaluate_tinysam_with_gt_bbox(
    sam_weights,
    test_images_dir,
    test_labels_dir,
    test_masks_dir,
    output_dir,
    sam_model_type="vit_t",
    device="cuda",
    use_center_point=True,
):
    """
    使用GT bbox作为prompt，评估TinySAM在测试集上的性能
    
    Args:
        sam_weights: TinySAM模型权重路径
        test_images_dir: 测试图片目录
        test_labels_dir: 测试标签目录（包含GT bbox）
        test_masks_dir: GT mask目录
        output_dir: 输出目录
        sam_model_type: TinySAM模型类型
        device: 设备
        use_center_point: 是否使用中心点作为额外prompt
    """
    print("=" * 60)
    print("使用GT bbox评估TinySAM分割性能")
    print("=" * 60)
    
    # 检查路径
    test_images_path = Path(test_images_dir)
    test_labels_path = Path(test_labels_dir)
    test_masks_path = Path(test_masks_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not test_images_path.exists():
        print(f"Error: 测试图片目录不存在: {test_images_dir}")
        return
    
    if not test_labels_path.exists():
        print(f"Error: 测试标签目录不存在: {test_labels_dir}")
        return
    
    if not test_masks_path.exists():
        print(f"Error: GT mask目录不存在: {test_masks_dir}")
        return
    
    # 加载TinySAM模型
    print(f"\n加载TinySAM模型: {sam_weights}")
    sam_model = sam_model_registry[sam_model_type](checkpoint=sam_weights)
    sam_model.to(device)
    sam_predictor = SamPredictor(sam_model)
    print(f"模型加载完成，使用设备: {device}")
    
    # 获取所有测试图片
    image_files = sorted(list(test_images_path.glob("*.jpg")))
    if not image_files:
        image_files = sorted(list(test_images_path.glob("*.png")))
    
    if not image_files:
        print(f"Error: 在 {test_images_dir} 中未找到图片文件")
        return
    
    print(f"\n找到 {len(image_files)} 张测试图片")
    print(f"使用中心点prompt: {use_center_point}")
    print("\n开始评估...")
    
    # 存储每张图片的结果
    all_metrics = []
    per_image_results = []
    
    # 处理每张图片
    for img_file in tqdm(image_files, desc="处理图片"):
        img_name = img_file.stem
        
        # 读取原图
        image_bgr = cv2.imread(str(img_file))
        if image_bgr is None:
            print(f"Warning: 无法读取图片 {img_file}")
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_bgr.shape[:2]
        
        # 读取GT bbox
        label_file = test_labels_path / f"{img_name}.txt"
        gt_boxes = parse_yolo_label(label_file, w, h)
        
        if len(gt_boxes) == 0:
            # 没有GT bbox，跳过
            print(f"Warning: {img_name} 没有GT bbox，跳过")
            continue
        
        # 读取GT mask
        mask_file = test_masks_path / f"{img_name}.jpg"
        if not mask_file.exists():
            mask_file = test_masks_path / f"{img_name}.png"
        
        if not mask_file.exists():
            print(f"Warning: {img_name} 没有GT mask，跳过")
            continue
        
        gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"Warning: 无法读取GT mask {mask_file}")
            continue
        
        # 调整GT mask尺寸以匹配原图
        if gt_mask.shape[:2] != (h, w):
            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # TinySAM分割
        sam_predictor.set_image(image_rgb)
        
        # 合并所有GT bbox的预测mask
        combined_pred_mask = np.zeros((h, w), dtype=np.uint8)
        
        for bbox in gt_boxes:
            x1, y1, x2, y2 = bbox
            
            # 使用bbox和中心点作为prompt
            if use_center_point:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                point_coords = np.array([[cx, cy]])
                point_labels = np.array([1])
            else:
                point_coords = None
                point_labels = None
            
            box = np.array([x1, y1, x2, y2])
            
            # 预测mask
            pred_masks, scores, logits = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
            )
            
            if len(pred_masks) > 0:
                mask = pred_masks[0].astype(np.uint8) * 255
                # 合并到总mask中
                combined_pred_mask = np.maximum(combined_pred_mask, mask)
        
        # 计算指标
        metrics = calculate_metrics(combined_pred_mask, gt_mask)
        all_metrics.append(metrics)
        
        per_image_results.append({
            'image_name': img_name,
            'num_gt_boxes': len(gt_boxes),
            'metrics': metrics
        })
    
    # 计算平均指标
    if len(all_metrics) == 0:
        print("Error: 没有有效的评估结果")
        return
    
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
    }
    
    # 保存结果
    report = {
        'evaluation_time': datetime.now().isoformat(),
        'model_info': {
            'sam_weights': str(sam_weights),
            'sam_model_type': sam_model_type,
            'use_center_point': use_center_point,
        },
        'dataset_info': {
            'test_images_dir': str(test_images_dir),
            'test_labels_dir': str(test_labels_dir),
            'test_masks_dir': str(test_masks_dir),
            'total_images': len(image_files),
            'valid_evaluations': len(all_metrics),
        },
        'summary': {
            'average_metrics': avg_metrics,
        },
        'per_image_results': per_image_results,
    }
    
    report_file = output_path / "evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    print(f"总图片数: {len(image_files)}")
    print(f"有效评估数: {len(all_metrics)}")
    print("\n平均指标:")
    print(f"  Dice (F1):     {avg_metrics['dice']:.4f} ({avg_metrics['dice']*100:.2f}%)")
    print(f"  IoU:           {avg_metrics['iou']:.4f} ({avg_metrics['iou']*100:.2f}%)")
    print(f"  Precision:     {avg_metrics['precision']:.4f} ({avg_metrics['precision']*100:.2f}%)")
    print(f"  Recall:        {avg_metrics['recall']:.4f} ({avg_metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:      {avg_metrics['f1']:.4f} ({avg_metrics['f1']*100:.2f}%)")
    print(f"\n详细报告保存在: {report_file}")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用GT bbox作为prompt，评估TinySAM在测试集上的分割性能"
    )
    parser.add_argument(
        "--sam-weights",
        type=str,
        required=True,
        help="TinySAM模型权重路径"
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default="Kvasir-SEG/test/images",
        help="测试图片目录"
    )
    parser.add_argument(
        "--test-labels",
        type=str,
        default="Kvasir-SEG/test/labels",
        help="测试标签目录（包含GT bbox）"
    )
    parser.add_argument(
        "--test-masks",
        type=str,
        default="Kvasir-SEG/masks",
        help="GT mask目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results_gt_bbox",
        help="输出目录"
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_t",
        help="TinySAM模型类型"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--no-center-point",
        action="store_true",
        help="不使用中心点作为prompt（仅使用bbox）"
    )
    
    args = parser.parse_args()
    
    evaluate_tinysam_with_gt_bbox(
        sam_weights=args.sam_weights,
        test_images_dir=args.test_images,
        test_labels_dir=args.test_labels,
        test_masks_dir=args.test_masks,
        output_dir=args.output_dir,
        sam_model_type=args.sam_model_type,
        device=args.device,
        use_center_point=not args.no_center_point,
    )

