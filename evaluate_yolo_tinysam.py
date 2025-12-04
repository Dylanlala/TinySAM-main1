#!/usr/bin/env python3
"""
评估YOLO11n + TinySAM在测试集上的性能
计算分割指标：Dice, IoU, Precision, Recall等
"""

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

from inference_yolo_tinysam import YOLOTinySAMInference


def calculate_metrics(pred_mask, gt_mask):
    """计算分割指标"""
    pred_flat = (pred_mask > 128).astype(np.uint8).flatten()
    gt_flat = (gt_mask > 128).astype(np.uint8).flatten()
    
    if gt_flat.sum() == 0 and pred_flat.sum() == 0:
        # 两者都是空mask，认为是完美匹配
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


def evaluate_on_testset(
    inferencer,
    test_images_dir,
    test_masks_dir,
    output_dir="evaluation_results",
):
    """
    在测试集上评估性能
    
    Args:
        inferencer: YOLOTinySAMInference实例
        test_images_dir: 测试图片目录
        test_masks_dir: 测试mask目录
        output_dir: 输出目录
    """
    test_images_path = Path(test_images_dir)
    test_masks_path = Path(test_masks_dir)
    
    image_files = sorted(list(test_images_path.glob("*.jpg")))
    
    if len(image_files) == 0:
        print(f"未找到图片文件在: {test_images_dir}")
        return
    
    print(f"开始评估，共 {len(image_files)} 张图片...")
    
    # 存储每张图片的指标
    all_metrics = []
    image_results = []
    
    # 处理每张图片
    for img_file in tqdm(image_files, desc="评估中"):
        image_name = img_file.stem
        
        # 推理
        try:
            results = inferencer.predict(img_file)
        except Exception as e:
            print(f"处理图片 {img_file.name} 时出错: {e}")
            continue
        
        # 读取GT mask
        gt_mask_path = test_masks_path / f"{image_name}.jpg"
        if not gt_mask_path.exists():
            gt_mask_path = test_masks_path / f"{image_name}.png"
        
        if not gt_mask_path.exists():
            print(f"警告: 未找到GT mask: {gt_mask_path}")
            continue
        
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        
        # 调整pred mask尺寸以匹配GT
        pred_mask = results['combined_mask']
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        
        # 计算指标
        metrics = calculate_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        
        image_results.append({
            'image_name': image_name,
            'num_detections': len(results['detections']),
            'metrics': metrics,
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
    
    # 打印结果
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    print(f"测试图片数: {len(image_files)}")
    print(f"有效评估数: {len(all_metrics)}")
    print(f"\n平均指标:")
    print(f"  Dice (F1):     {avg_metrics['dice']:.4f} ({avg_metrics['dice']*100:.2f}%)")
    print(f"  IoU:           {avg_metrics['iou']:.4f} ({avg_metrics['iou']*100:.2f}%)")
    print(f"  Precision:     {avg_metrics['precision']:.4f} ({avg_metrics['precision']*100:.2f}%)")
    print(f"  Recall:        {avg_metrics['recall']:.4f} ({avg_metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:      {avg_metrics['f1']:.4f} ({avg_metrics['f1']*100:.2f}%)")
    print("="*70)
    
    # 保存详细结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    evaluation_report = {
        'summary': {
            'total_images': len(image_files),
            'valid_evaluations': len(all_metrics),
            'average_metrics': avg_metrics,
        },
        'per_image_results': image_results,
    }
    
    with open(output_path / "evaluation_report.json", 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print(f"\n详细评估报告已保存到: {output_path / 'evaluation_report.json'}")


def main():
    parser = argparse.ArgumentParser(description="评估YOLO11n + TinySAM性能")
    parser.add_argument("--yolo-weights", type=str, required=True,
                       help="YOLO模型权重路径")
    parser.add_argument("--sam-weights", type=str, required=True,
                       help="TinySAM模型权重路径")
    parser.add_argument("--sam-type", type=str, default="vit_t",
                       help="TinySAM模型类型")
    parser.add_argument("--test-images", type=str, required=True,
                       help="测试图片目录")
    parser.add_argument("--test-masks", type=str, required=True,
                       help="测试mask目录")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="输出目录")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                       help="YOLO置信度阈值")
    parser.add_argument("--yolo-iou", type=float, default=0.45,
                       help="YOLO IoU阈值")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备")
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = YOLOTinySAMInference(
        yolo_weights=args.yolo_weights,
        sam_weights=args.sam_weights,
        sam_model_type=args.sam_type,
        device=args.device,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
    )
    
    # 评估
    evaluate_on_testset(
        inferencer=inferencer,
        test_images_dir=args.test_images,
        test_masks_dir=args.test_masks,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

