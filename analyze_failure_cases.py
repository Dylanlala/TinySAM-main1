#!/usr/bin/env python3
"""
分析失败案例，找出问题原因并提供优化建议
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


def analyze_failures(evaluation_report_path, inference_results_dir):
    """分析失败案例"""
    
    with open(evaluation_report_path, 'r') as f:
        report = json.load(f)
    
    results = report['per_image_results']
    inference_dir = Path(inference_results_dir)
    
    # 分类失败类型
    failure_types = {
        'zero_dice': [],  # 完全失败（Dice=0）
        'low_recall': [],  # 召回率低（漏检）
        'low_precision': [],  # 精确率低（误检）
        'low_iou': [],  # IoU低（分割不准确）
    }
    
    for item in results:
        metrics = item['metrics']
        dice = metrics['dice']
        recall = metrics['recall']
        precision = metrics['precision']
        iou = metrics['iou']
        
        if dice == 0:
            failure_types['zero_dice'].append(item)
        elif recall < 0.3:
            failure_types['low_recall'].append(item)
        elif precision < 0.5:
            failure_types['low_precision'].append(item)
        elif iou < 0.2:
            failure_types['low_iou'].append(item)
    
    print("\n" + "="*70)
    print("失败案例分析")
    print("="*70)
    
    # 1. 完全失败的案例
    print(f"\n【完全失败案例（Dice=0）】: {len(failure_types['zero_dice'])}张")
    if failure_types['zero_dice']:
        print("可能原因：")
        print("  1. YOLO未检测到息肉（检测数为0）")
        print("  2. YOLO检测到但TinySAM分割失败")
        print("  3. 检测位置偏差太大")
        
        print("\n典型案例：")
        for i, item in enumerate(failure_types['zero_dice'][:5]):
            print(f"  {i+1}. {item['image_name']}: 检测数={item['num_detections']}")
            if item['num_detections'] == 0:
                print("     → YOLO未检测到息肉（漏检）")
            else:
                print(f"     → YOLO检测到{item['num_detections']}个，但分割失败")
                # 检查bbox
                bbox_file = inference_dir / "bboxes" / f"{item['image_name']}.json"
                if bbox_file.exists():
                    with open(bbox_file, 'r') as f:
                        bbox_info = json.load(f)
                        for det in bbox_info['detections']:
                            print(f"       bbox: {det['bbox']}, conf={det['confidence']:.4f}")
    
    # 2. 低召回率案例（漏检）
    print(f"\n【低召回率案例（Recall<0.3）】: {len(failure_types['low_recall'])}张")
    if failure_types['low_recall']:
        print("可能原因：")
        print("  1. YOLO检测到的bbox不够准确")
        print("  2. TinySAM分割区域不完整")
        print("  3. 多个息肉只检测到部分")
        
        print("\n典型案例：")
        for i, item in enumerate(failure_types['low_recall'][:5]):
            metrics = item['metrics']
            print(f"  {i+1}. {item['image_name']}: Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f}")
    
    # 3. 低精确率案例（误检）
    print(f"\n【低精确率案例（Precision<0.5）】: {len(failure_types['low_precision'])}张")
    if failure_types['low_precision']:
        print("可能原因：")
        print("  1. YOLO误检（检测到非息肉区域）")
        print("  2. TinySAM分割超出息肉边界")
    
    # 4. 低IoU案例（分割不准确）
    print(f"\n【低IoU案例（IoU<0.2）】: {len(failure_types['low_iou'])}张")
    if failure_types['low_iou']:
        print("可能原因：")
        print("  1. 分割边界不准确")
        print("  2. 分割区域过大或过小")
    
    # 统计检测数量与性能的关系
    print(f"\n【检测数量与性能关系】")
    detection_count_stats = defaultdict(list)
    for item in results:
        detection_count_stats[item['num_detections']].append(item['metrics']['dice'])
    
    for count in sorted(detection_count_stats.keys()):
        dices = detection_count_stats[count]
        avg_dice = np.mean(dices)
        print(f"  检测数={count}: 平均Dice={avg_dice:.4f} (样本数={len(dices)})")
    
    # 优化建议
    print("\n" + "="*70)
    print("优化建议")
    print("="*70)
    
    if len(failure_types['zero_dice']) > 10:
        print("\n1. YOLO检测优化：")
        print("   - 降低置信度阈值（当前0.25，可尝试0.15-0.20）")
        print("   - 增加训练数据或数据增强")
        print("   - 检查YOLO训练是否充分")
    
    if len(failure_types['low_recall']) > 20:
        print("\n2. TinySAM分割优化：")
        print("   - 增加训练轮数（当前可能不够）")
        print("   - 调整学习率或使用学习率调度")
        print("   - 使用GT bbox训练（而不是YOLO检测结果）")
        print("   - 增加边界损失权重")
    
    if len(failure_types['low_iou']) > 20:
        print("\n3. 分割精度优化：")
        print("   - 使用多尺度训练")
        print("   - 增加边界损失（boundary loss）")
        print("   - 后处理：形态学操作优化mask")
    
    print("\n4. 整体优化策略：")
    print("   - 端到端微调：YOLO+TinySAM联合训练")
    print("   - 使用更强的数据增强")
    print("   - 集成多个模型的结果")
    
    return failure_types


def visualize_failure_cases(failure_types, inference_results_dir, output_dir="failure_analysis"):
    """可视化失败案例"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    inference_dir = Path(inference_results_dir)
    
    # 可视化完全失败的案例
    if failure_types['zero_dice']:
        vis_dir = output_path / "zero_dice_cases"
        vis_dir.mkdir(exist_ok=True)
        
        for item in failure_types['zero_dice'][:10]:  # 前10个
            img_name = item['image_name']
            
            # 读取原图、GT mask、预测mask
            img = cv2.imread(str(inference_dir / "images" / f"{img_name}.jpg"))
            pred_mask = cv2.imread(str(inference_dir / "masks" / f"{img_name}_combined.png"), cv2.IMREAD_GRAYSCALE)
            vis_img = cv2.imread(str(inference_dir / "visualizations" / f"{img_name}_vis.jpg"))
            
            if img is not None and vis_img is not None:
                # 保存可视化结果
                cv2.imwrite(str(vis_dir / f"{img_name}_failure.jpg"), vis_img)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="分析失败案例")
    parser.add_argument("--evaluation-report", type=str,
                       default="evaluation_results_kvasir/evaluation_report.json",
                       help="评估报告路径")
    parser.add_argument("--inference-results", type=str,
                       default="inference_results_kvasir",
                       help="推理结果目录")
    parser.add_argument("--output-dir", type=str,
                       default="failure_analysis",
                       help="分析结果输出目录")
    
    args = parser.parse_args()
    
    # 分析失败案例
    failure_types = analyze_failures(args.evaluation_report, args.inference_results)
    
    # 可视化失败案例
    visualize_failure_cases(failure_types, args.inference_results, args.output_dir)
    
    print(f"\n失败案例可视化已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

