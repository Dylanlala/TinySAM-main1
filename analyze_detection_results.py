#!/usr/bin/env python3
"""
分析YOLO检测结果，统计TP、FP、FN等指标
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import cv2
from typing import List, Tuple, Dict

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个bbox的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y2_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def parse_yolo_label(txt_path: Path, img_width: int, img_height: int) -> List[List[float]]:
    """解析YOLO格式的标签文件，返回像素坐标的bbox列表"""
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
                xmin = cx - w / 2
                ymin = cy - h / 2
                xmax = cx + w / 2
                ymax = cy + h / 2
                
                boxes.append([xmin, ymin, xmax, ymax])
    
    return boxes


def analyze_detections(
    predictions_json: str,
    test_labels_dir: str,
    test_images_dir: str,
    iou_threshold: float = 0.5
) -> Dict:
    """
    分析检测结果
    
    Args:
        predictions_json: YOLO预测结果JSON文件（COCO格式）
        test_labels_dir: 测试标签目录
        test_images_dir: 测试图片目录（用于获取图片尺寸和文件名）
        iou_threshold: IoU阈值，用于判断TP/FP
    """
    # 读取预测结果（COCO格式）
    with open(predictions_json, 'r') as f:
        pred_data = json.load(f)
    
    test_labels_path = Path(test_labels_dir)
    test_images_path = Path(test_images_dir)
    
    # 获取所有测试图片，建立image_id到文件名的映射
    test_images = sorted(list(test_images_path.glob("*.jpg")))
    image_id_to_name = {i: img.name for i, img in enumerate(test_images)}
    
    # 按image_id分组预测结果
    from collections import defaultdict
    preds_by_image = defaultdict(list)
    for pred in pred_data:
        image_id = pred['image_id']
        # COCO格式: bbox是[x, y, width, height]
        x, y, w, h = pred['bbox']
        # 转换为x1, y1, x2, y2
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        preds_by_image[image_id].append({
            'bbox': [x1, y1, x2, y2],
            'confidence': pred['score']
        })
    
    # 统计信息
    total_gt_polyps = 0
    total_pred_polyps = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # 每张图片的详细统计
    image_stats = []
    
    # 处理每张图片
    for image_id, image_name in image_id_to_name.items():
        image_stem = Path(image_name).stem
        
        # 获取图片尺寸
        img = cv2.imread(str(test_images_path / image_name))
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        
        # 读取GT标签
        gt_label_path = test_labels_path / f"{image_stem}.txt"
        gt_boxes = parse_yolo_label(gt_label_path, img_width, img_height)
        total_gt_polyps += len(gt_boxes)
        
        # 获取预测结果
        pred_boxes = preds_by_image.get(image_id, [])
        total_pred_polyps += len(pred_boxes)
        
        # 匹配GT和预测
        matched_gt = set()
        matched_pred = set()
        tp_count = 0
        
        # 对每个预测框，找最佳匹配的GT框
        for pred_idx, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box['bbox'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp_count += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
        
        # 计算FP和FN
        fp_count = len(pred_boxes) - len(matched_pred)
        fn_count = len(gt_boxes) - len(matched_gt)
        
        total_tp += tp_count
        total_fp += fp_count
        total_fn += fn_count
        
        # 记录每张图片的统计
        image_stats.append({
            'image': image_name,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'tp': tp_count,
            'fp': fp_count,
            'fn': fn_count,
            'precision': tp_count / len(pred_boxes) if len(pred_boxes) > 0 else 0.0,
            'recall': tp_count / len(gt_boxes) if len(gt_boxes) > 0 else 0.0,
        })
    
    # 计算总体指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        'summary': {
            'total_gt_polyps': total_gt_polyps,
            'total_pred_polyps': total_pred_polyps,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'detection_rate': total_tp / total_gt_polyps if total_gt_polyps > 0 else 0.0,
        },
        'per_image_stats': image_stats
    }
    
    return results


def print_analysis_report(results: Dict):
    """打印分析报告"""
    summary = results['summary']
    image_stats = results['per_image_stats']
    
    print("\n" + "="*70)
    print("检测结果详细分析")
    print("="*70)
    print(f"\n【总体统计】")
    print(f"  测试集中实际息肉数（GT）: {summary['total_gt_polyps']}")
    print(f"  模型检测到的息肉数: {summary['total_pred_polyps']}")
    print(f"  成功检测到的息肉数（TP）: {summary['total_tp']}")
    print(f"  误检的息肉数（FP）: {summary['total_fp']}")
    print(f"  漏检的息肉数（FN）: {summary['total_fn']}")
    
    print(f"\n【性能指标】")
    print(f"  精确率（Precision）: {summary['precision']:.4f} ({summary['precision']*100:.2f}%)")
    print(f"  召回率（Recall）: {summary['recall']:.4f} ({summary['recall']*100:.2f}%)")
    print(f"  F1分数: {summary['f1_score']:.4f}")
    print(f"  检测率（Detection Rate）: {summary['detection_rate']:.4f} ({summary['detection_rate']*100:.2f}%)")
    
    print(f"\n【检测成功率】")
    success_rate = summary['total_tp'] / summary['total_gt_polyps'] * 100 if summary['total_gt_polyps'] > 0 else 0
    print(f"  成功检测: {summary['total_tp']}/{summary['total_gt_polyps']} = {success_rate:.2f}%")
    
    # 统计有问题的图片
    perfect_images = sum(1 for s in image_stats if s['tp'] == s['gt_count'] and s['fp'] == 0)
    missed_images = sum(1 for s in image_stats if s['fn'] > 0)
    false_positive_images = sum(1 for s in image_stats if s['fp'] > 0)
    
    print(f"\n【图片级别统计】")
    print(f"  完美检测的图片（无漏检无误检）: {perfect_images}/{len(image_stats)}")
    print(f"  有漏检的图片: {missed_images}/{len(image_stats)}")
    print(f"  有误检的图片: {false_positive_images}/{len(image_stats)}")
    
    # 找出漏检最多的图片
    if missed_images > 0:
        print(f"\n【漏检最多的5张图片】")
        sorted_by_fn = sorted(image_stats, key=lambda x: x['fn'], reverse=True)
        for i, stat in enumerate(sorted_by_fn[:5]):
            if stat['fn'] > 0:
                print(f"  {i+1}. {stat['image']}: GT={stat['gt_count']}, 检测={stat['pred_count']}, 漏检={stat['fn']}, 召回率={stat['recall']:.2%}")
    
    # 找出误检最多的图片
    if false_positive_images > 0:
        print(f"\n【误检最多的5张图片】")
        sorted_by_fp = sorted(image_stats, key=lambda x: x['fp'], reverse=True)
        for i, stat in enumerate(sorted_by_fp[:5]):
            if stat['fp'] > 0:
                print(f"  {i+1}. {stat['image']}: GT={stat['gt_count']}, 检测={stat['pred_count']}, 误检={stat['fp']}, 精确率={stat['precision']:.2%}")
    
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="分析YOLO检测结果")
    parser.add_argument("--predictions", type=str,
                       default="/home/huangmanling/ultralytics/runs/detect/val41/predictions.json",
                       help="预测结果JSON文件")
    parser.add_argument("--test-labels", type=str,
                       default="Kvasir-SEG/test/labels",
                       help="测试标签目录")
    parser.add_argument("--test-images", type=str,
                       default="Kvasir-SEG/test/images",
                       help="测试图片目录")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU阈值")
    parser.add_argument("--output", type=str, default="detection_analysis.json",
                       help="输出JSON文件")
    
    args = parser.parse_args()
    
    # 分析结果
    results = analyze_detections(
        args.predictions,
        args.test_labels,
        args.test_images,
        args.iou_threshold
    )
    
    # 打印报告
    print_analysis_report(results)
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

