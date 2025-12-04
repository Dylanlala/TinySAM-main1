#!/usr/bin/env python3
"""
评估增强版YOLO+TinySAM（使用增强版推理脚本）
"""

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

from inference_yolo_tinysam_enhanced import EnhancedYOLOTinySAMInference


def calculate_metrics(pred_mask, gt_mask):
    """计算分割指标"""
    pred_flat = (pred_mask > 128).astype(np.uint8).flatten()
    gt_flat = (gt_mask > 128).astype(np.uint8).flatten()
    
    if gt_flat.sum() == 0 and pred_flat.sum() == 0:
        return {
            'dice': 1.0,
            'iou': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
        }
    
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
        'f1': float(f1),
    }


def evaluate_enhanced(
    inferencer,
    test_images_dir,
    test_masks_dir,
    output_dir="evaluation_results_enhanced",
):
    """评估增强版推理效果"""
    test_images_path = Path(test_images_dir)
    test_masks_path = Path(test_masks_dir)
    
    image_files = sorted(list(test_images_path.glob("*.jpg")))
    
    if len(image_files) == 0:
        print(f"未找到图片文件在: {test_images_dir}")
        return
    
    print(f"开始评估，共 {len(image_files)} 张图片...")
    
    all_metrics = []
    image_results = []
    
    for img_file in tqdm(image_files, desc="评估中"):
        image_name = img_file.stem
        
        # 读取GT mask
        gt_mask_path = test_masks_path / f"{image_name}.jpg"
        if not gt_mask_path.exists():
            gt_mask_path = test_masks_path / f"{image_name}.png"
        
        if not gt_mask_path.exists():
            continue
        
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        
        # 读取预测mask（从增强版推理结果）
        pred_mask_path = Path(output_dir) / "masks" / f"{image_name}_combined.png"
        if not pred_mask_path.exists():
            # 如果没有找到，尝试推理
            try:
                inferencer.predict(img_file, output_dir, enable_postprocess=True)
                pred_mask_path = Path(output_dir) / "masks" / f"{image_name}_combined.png"
            except Exception as e:
                print(f"处理图片 {img_file.name} 时出错: {e}")
                continue
        
        if not pred_mask_path.exists():
            continue
        
        pred_mask = cv2.imread(str(pred_mask_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            continue
        
        # 调整尺寸
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # 计算指标
        metrics = calculate_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        
        # 读取检测数量
        bbox_path = Path(output_dir) / "bboxes" / f"{image_name}.json"
        num_detections = 0
        if bbox_path.exists():
            with open(bbox_path, 'r') as f:
                bboxes = json.load(f)
                num_detections = len(bboxes)
        
        image_results.append({
            'image_name': image_name,
            'num_detections': num_detections,
            'metrics': metrics
        })
    
    if len(all_metrics) == 0:
        print("没有有效的评估结果")
        return
    
    # 计算平均指标
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
    }
    
    # 保存结果
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = {
        'summary': {
            'total_images': len(image_files),
            'valid_evaluations': len(all_metrics),
            'average_metrics': avg_metrics
        },
        'per_image_results': image_results
    }
    
    with open(Path(output_dir) / "evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印结果
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    print(f"测试图片数: {len(image_files)}")
    print(f"有效评估数: {len(all_metrics)}")
    print()
    print("平均指标:")
    print(f"  Dice (F1):     {avg_metrics['dice']:.4f} ({avg_metrics['dice']*100:.2f}%)")
    print(f"  IoU:           {avg_metrics['iou']:.4f} ({avg_metrics['iou']*100:.2f}%)")
    print(f"  Precision:     {avg_metrics['precision']:.4f} ({avg_metrics['precision']*100:.2f}%)")
    print(f"  Recall:        {avg_metrics['recall']:.4f} ({avg_metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:      {avg_metrics['f1']:.4f} ({avg_metrics['f1']*100:.2f}%)")
    print("="*70)
    print(f"\n详细评估报告已保存到: {Path(output_dir) / 'evaluation_report.json'}")


def main():
    parser = argparse.ArgumentParser(description="评估增强版YOLO+TinySAM")
    parser.add_argument("--yolo-weights", type=str, required=True,
                       help="YOLO模型权重路径")
    parser.add_argument("--sam-weights", type=str, required=True,
                       help="TinySAM模型权重路径")
    parser.add_argument("--test-images", type=str, required=True,
                       help="测试图片目录")
    parser.add_argument("--test-masks", type=str, required=True,
                       help="测试mask目录")
    parser.add_argument("--output-dir", type=str, default="evaluation_results_enhanced",
                       help="输出目录")
    parser.add_argument("--yolo-conf", type=float, default=0.15,
                       help="YOLO置信度阈值")
    parser.add_argument("--yolo-iou", type=float, default=0.4,
                       help="YOLO IoU阈值")
    parser.add_argument("--model-type", type=str, default="vit_t",
                       help="TinySAM模型类型")
    
    args = parser.parse_args()
    
    # 创建增强版推理器
    inferencer = EnhancedYOLOTinySAMInference(
        yolo_weights=args.yolo_weights,
        sam_weights=args.sam_weights,
        model_type=args.model_type,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        use_point_prompt=True
    )
    
    # 先运行推理（如果还没有运行）
    print("运行增强版推理...")
    image_paths = list(Path(args.test_images).glob("*.jpg"))
    for image_path in tqdm(image_paths, desc="推理中"):
        inferencer.predict(image_path, args.output_dir, enable_postprocess=True)
    
    # 评估
    evaluate_enhanced(
        inferencer,
        args.test_images,
        args.test_masks,
        args.output_dir
    )


if __name__ == "__main__":
    main()




