#!/usr/bin/env python3
"""
使用YOLO11n-seg在Kvasir-SEG测试集上进行分割推理
如果模型没有训练好，使用预训练模型进行推理
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


def inference_yolo_seg(
    model_weights,
    test_images_dir,
    test_masks_dir,
    output_dir="yolo_seg_results",
    conf_threshold=0.25,
    iou_threshold=0.45,
    save_visualizations=True,
):
    """
    使用YOLO11n-seg进行分割推理
    
    Args:
        model_weights: 模型权重路径（如果不存在，使用预训练模型）
        test_images_dir: 测试图片目录
        test_masks_dir: GT mask目录（用于评估）
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        save_visualizations: 是否保存可视化结果
    """
    print("=" * 70)
    print("YOLO11n-seg 分割推理")
    print("=" * 70)
    
    # 检查模型文件
    model_path = Path(model_weights)
    if not model_path.exists():
        print(f"警告: 模型文件不存在 {model_weights}")
        print("使用预训练模型 yolo11n-seg.pt")
        model_weights = "yolo11n-seg.pt"
    
    # 加载模型
    print(f"\n加载模型: {model_weights}")
    model = YOLO(model_weights)
    
    # 获取测试图片
    test_images_path = Path(test_images_dir)
    test_masks_path = Path(test_masks_dir)
    
    image_files = sorted(list(test_images_path.glob("*.jpg")))
    if not image_files:
        image_files = sorted(list(test_images_path.glob("*.png")))
    
    if not image_files:
        print(f"错误: 在 {test_images_dir} 中未找到图片文件")
        return
    
    print(f"\n找到 {len(image_files)} 张测试图片")
    print(f"置信度阈值: {conf_threshold}")
    print(f"IoU阈值: {iou_threshold}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    masks_output_dir = output_path / "masks"
    masks_output_dir.mkdir(exist_ok=True)
    if save_visualizations:
        vis_output_dir = output_path / "visualizations"
        vis_output_dir.mkdir(exist_ok=True)
    
    # 存储结果
    all_metrics = []
    image_results = []
    total_detections = 0
    images_with_detections = 0
    
    print("\n开始推理...")
    
    # 处理每张图片
    for img_file in tqdm(image_files, desc="处理中"):
        image_name = img_file.stem
        
        # 读取原图
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        original_h, original_w = image.shape[:2]
        
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
        
        if result.masks is not None and len(result.masks) > 0:
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
            
            images_with_detections += 1
            total_detections += num_detections
        
        # 保存预测mask
        cv2.imwrite(str(masks_output_dir / f"{image_name}_pred.png"), pred_mask)
        
        # 读取GT mask进行评估
        gt_mask_path = test_masks_path / f"{image_name}.jpg"
        if not gt_mask_path.exists():
            gt_mask_path = test_masks_path / f"{image_name}.png"
        
        metrics = None
        if gt_mask_path.exists():
            gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                # 调整GT mask尺寸
                if gt_mask.shape[:2] != (original_h, original_w):
                    gt_mask = cv2.resize(gt_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                
                # 计算指标
                metrics = calculate_metrics(pred_mask, gt_mask)
                all_metrics.append(metrics)
        
        # 保存可视化结果
        if save_visualizations and metrics is not None:
            # 创建对比图：原图 + 预测mask + GT mask
            vis_img = np.hstack([
                image,
                cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET),
                cv2.applyColorMap(gt_mask, cv2.COLORMAP_JET) if gt_mask_path.exists() else np.zeros_like(image)
            ])
            cv2.imwrite(str(vis_output_dir / f"{image_name}_vis.jpg"), vis_img)
        
        image_results.append({
            'image_name': image_name,
            'num_detections': num_detections,
            'metrics': metrics if metrics else {}
        })
    
    # 计算平均指标
    if len(all_metrics) > 0:
        avg_metrics = {
            'dice': np.mean([m['dice'] for m in all_metrics]),
            'iou': np.mean([m['iou'] for m in all_metrics]),
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1': np.mean([m['f1'] for m in all_metrics]),
        }
    else:
        avg_metrics = {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }
    
    # 保存结果
    report = {
        'model': str(model_weights),
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'summary': {
            'total_images': len(image_files),
            'valid_evaluations': len(all_metrics),
            'images_with_detections': images_with_detections,
            'total_detections': total_detections,
            'average_metrics': avg_metrics,
        },
        'per_image_results': image_results,
    }
    
    report_file = output_path / "evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("推理完成！")
    print("=" * 70)
    print(f"总图片数: {len(image_files)}")
    print(f"有检测结果的图片: {images_with_detections}")
    print(f"总检测数: {total_detections}")
    if len(all_metrics) > 0:
        print(f"\n平均指标:")
        print(f"  Dice (F1):     {avg_metrics['dice']:.4f} ({avg_metrics['dice']*100:.2f}%)")
        print(f"  IoU:           {avg_metrics['iou']:.4f} ({avg_metrics['iou']*100:.2f}%)")
        print(f"  Precision:     {avg_metrics['precision']:.4f} ({avg_metrics['precision']*100:.2f}%)")
        print(f"  Recall:        {avg_metrics['recall']:.4f} ({avg_metrics['recall']*100:.2f}%)")
        print(f"  F1 Score:      {avg_metrics['f1']:.4f} ({avg_metrics['f1']*100:.2f}%)")
    else:
        print("\n警告: 没有检测到任何结果，无法计算指标")
    print(f"\n结果保存在: {output_dir}")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用YOLO11n-seg进行分割推理")
    parser.add_argument(
        "--model-weights",
        type=str,
        default="runs/segment/yolo11n_seg_kvasir_final/weights/best.pt",
        help="模型权重路径"
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default="Kvasir-SEG/test/images",
        help="测试图片目录"
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
        default="yolo_seg_results",
        help="输出目录"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU阈值"
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="不保存可视化结果"
    )
    
    args = parser.parse_args()
    
    inference_yolo_seg(
        model_weights=args.model_weights,
        test_images_dir=args.test_images,
        test_masks_dir=args.test_masks,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_visualizations=not args.no_vis,
    )

