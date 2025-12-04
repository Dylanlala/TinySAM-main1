#!/usr/bin/env python3
"""
对比YOLO11n-seg vs YOLO11n+TinySAM的性能
"""

import json
from pathlib import Path
import argparse


def compare_models(
    yolo_seg_results,
    yolo_tinysam_results,
    output_file="model_comparison.json",
):
    """
    对比两个模型的结果
    
    Args:
        yolo_seg_results: YOLO11n-seg评估结果JSON文件
        yolo_tinysam_results: YOLO11n+TinySAM评估结果JSON文件
        output_file: 输出对比结果文件
    """
    print("="*80)
    print("模型性能对比：YOLO11n-seg vs YOLO11n+TinySAM")
    print("="*80)
    
    # 读取结果
    with open(yolo_seg_results, 'r') as f:
        yolo_seg = json.load(f)
    
    with open(yolo_tinysam_results, 'r') as f:
        yolo_tinysam = json.load(f)
    
    # 获取平均指标
    seg_metrics = yolo_seg['summary']['average_metrics']
    tinysam_metrics = yolo_tinysam['summary']['average_metrics']
    
    # 打印对比
    print()
    print(f"{'指标':<15} {'YOLO11n-seg':<20} {'YOLO11n+TinySAM':<20} {'差异':<15}")
    print("-"*80)
    
    metrics_names = ['dice', 'iou', 'precision', 'recall', 'f1']
    for metric in metrics_names:
        seg_val = seg_metrics.get(metric, 0) * 100
        tinysam_val = tinysam_metrics.get(metric, 0) * 100
        diff = seg_val - tinysam_val
        
        metric_name = metric.upper()
        seg_str = f"{seg_val:.2f}%"
        tinysam_str = f"{tinysam_val:.2f}%"
        diff_str = f"{diff:+.2f}%"
        
        # 标记更好的结果
        if diff > 0:
            diff_str = f"{diff_str} ⭐ (YOLO-seg更好)"
        elif diff < 0:
            diff_str = f"{diff_str} ⭐ (TinySAM更好)"
        
        print(f"{metric_name:<15} {seg_str:<20} {tinysam_str:<20} {diff_str:<15}")
    
    print()
    print("="*80)
    print("总结")
    print("="*80)
    
    # 分析哪个模型更好
    dice_diff = seg_metrics['dice'] - tinysam_metrics['dice']
    precision_diff = seg_metrics['precision'] - tinysam_metrics['precision']
    recall_diff = seg_metrics['recall'] - tinysam_metrics['recall']
    
    print(f"\nDice差异: {dice_diff*100:+.2f}%")
    print(f"Precision差异: {precision_diff*100:+.2f}%")
    print(f"Recall差异: {recall_diff*100:+.2f}%")
    
    if dice_diff > 0:
        print("\n✅ YOLO11n-seg在Dice上表现更好")
    else:
        print("\n✅ YOLO11n+TinySAM在Dice上表现更好")
    
    if precision_diff > 0:
        print("✅ YOLO11n-seg在Precision上表现更好（分的更准）")
    else:
        print("✅ YOLO11n+TinySAM在Precision上表现更好（分的更准）")
    
    if recall_diff > 0:
        print("✅ YOLO11n-seg在Recall上表现更好（检测更多）")
    else:
        print("✅ YOLO11n+TinySAM在Recall上表现更好（检测更多）")
    
    # 保存对比结果
    comparison = {
        'yolo_seg': {
            'model': yolo_seg.get('model', 'YOLO11n-seg'),
            'metrics': seg_metrics
        },
        'yolo_tinysam': {
            'model': 'YOLO11n+TinySAM',
            'metrics': tinysam_metrics
        },
        'differences': {
            'dice': dice_diff,
            'iou': seg_metrics['iou'] - tinysam_metrics['iou'],
            'precision': precision_diff,
            'recall': recall_diff,
            'f1': seg_metrics['f1'] - tinysam_metrics['f1'],
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n对比结果已保存到: {output_file}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="对比YOLO11n-seg和YOLO11n+TinySAM")
    parser.add_argument("--yolo-seg-results", type=str, required=True,
                       help="YOLO11n-seg评估结果JSON文件")
    parser.add_argument("--yolo-tinysam-results", type=str, required=True,
                       help="YOLO11n+TinySAM评估结果JSON文件")
    parser.add_argument("--output", type=str, default="model_comparison.json",
                       help="输出对比结果文件")
    
    args = parser.parse_args()
    
    compare_models(
        yolo_seg_results=args.yolo_seg_results,
        yolo_tinysam_results=args.yolo_tinysam_results,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()




