#!/usr/bin/env python3
"""
评估YOLO11n-seg模型在测试集上的性能
对比YOLO11n-seg vs YOLO11n+TinySAM的效果
"""

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from ultralytics import YOLO


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


def evaluate_yolo_seg(
    model_weights,
    test_images_dir,
    test_masks_dir,
    output_dir="evaluation_results_yolo_seg",
    conf_threshold=0.5,
    iou_threshold=0.5,
):
    """
    评估YOLO11n-seg模型
    
    Args:
        model_weights: 模型权重路径
        test_images_dir: 测试图片目录
        test_masks_dir: 测试mask目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
    """
    print("="*70)
    print("评估YOLO11n-seg模型")
    print("="*70)
    print(f"模型权重: {model_weights}")
    print(f"测试图片: {test_images_dir}")
    print(f"测试mask: {test_masks_dir}")
    print("="*70)
    
    # 加载模型
    print(f"\n加载模型: {model_weights}")
    model = YOLO(model_weights)
    
    # 获取测试图片
    test_images_path = Path(test_images_dir)
    test_masks_path = Path(test_masks_dir)
    
    image_files = sorted(list(test_images_path.glob("*.jpg")))
    
    if len(image_files) == 0:
        print(f"未找到图片文件在: {test_images_dir}")
        return
    
    print(f"\n开始评估，共 {len(image_files)} 张图片...")
    
    # 存储结果
    all_metrics = []
    image_results = []
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    masks_output_dir = output_path / "masks"
    masks_output_dir.mkdir(exist_ok=True)
    
    # 处理每张图片
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
        
        original_h, original_w = gt_mask.shape
        
        # YOLO推理
        results = model.predict(
            str(img_file),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        
        result = results[0]
        
        # 合并所有检测的mask
        pred_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        num_detections = 0
        
        if result.masks is not None:
            # 获取所有mask
            masks = result.masks.data.cpu().numpy()  # [N, H, W]
            
            # 调整mask尺寸到原始图像尺寸
            for i, mask in enumerate(masks):
                # mask是模型输出尺寸，需要调整到原始尺寸
                mask_resized = cv2.resize(
                    (mask * 255).astype(np.uint8),
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST
                )
                # 合并mask（取最大值）
                pred_mask = np.maximum(pred_mask, mask_resized)
                num_detections += 1
        
        # 保存预测mask
        cv2.imwrite(str(masks_output_dir / f"{image_name}_pred.png"), pred_mask)
        
        # 计算指标
        metrics = calculate_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        
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
    report = {
        'model': str(model_weights),
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'summary': {
            'total_images': len(image_files),
            'valid_evaluations': len(all_metrics),
            'average_metrics': avg_metrics
        },
        'per_image_results': image_results
    }
    
    with open(output_path / "evaluation_report.json", 'w') as f:
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
    print(f"\n详细评估报告已保存到: {output_path / 'evaluation_report.json'}")


def main():
    parser = argparse.ArgumentParser(description="评估YOLO11n-seg模型")
    parser.add_argument("--model-weights", type=str, required=True,
                       help="模型权重路径")
    parser.add_argument("--test-images", type=str, required=True,
                       help="测试图片目录")
    parser.add_argument("--test-masks", type=str, required=True,
                       help="测试mask目录")
    parser.add_argument("--output-dir", type=str, default="evaluation_results_yolo_seg",
                       help="输出目录")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5,
                       help="IoU阈值")
    
    args = parser.parse_args()
    
    evaluate_yolo_seg(
        model_weights=args.model_weights,
        test_images_dir=args.test_images,
        test_masks_dir=args.test_masks,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )


if __name__ == "__main__":
    main()




