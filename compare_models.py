#!/usr/bin/env python3
"""
对比不同TinySAM模型的性能
"""

import json
import argparse
from pathlib import Path
import subprocess
import sys


def run_evaluation(yolo_weights, sam_weights, test_images, test_masks, output_dir, yolo_conf=0.25):
    """运行评估并返回结果"""
    cmd = [
        sys.executable, "evaluate_yolo_tinysam.py",
        "--yolo-weights", yolo_weights,
        "--sam-weights", sam_weights,
        "--test-images", test_images,
        "--test-masks", test_masks,
        "--output-dir", output_dir,
        "--yolo-conf", str(yolo_conf),
        "--yolo-iou", "0.45",
    ]
    
    print(f"运行评估: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"评估失败: {result.stderr}")
        return None
    
    # 读取评估报告
    report_path = Path(output_dir) / "evaluation_report.json"
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
        return report['summary']['average_metrics']
    
    return None


def compare_models():
    """对比不同模型"""
    parser = argparse.ArgumentParser(description="对比不同TinySAM模型性能")
    parser.add_argument("--yolo-weights", type=str,
                       default="runs/detect/yolo11n_kvasir/weights/best.pt",
                       help="YOLO模型权重")
    parser.add_argument("--sam-weights-1", type=str,
                       default="results_tinysam_kvasir/best_model.pth",
                       help="第一个TinySAM模型（原始）")
    parser.add_argument("--sam-weights-2", type=str,
                       default="results_tinysam_kvasir_optimized/best_model.pth",
                       help="第二个TinySAM模型（优化）")
    parser.add_argument("--test-images", type=str,
                       default="Kvasir-SEG/test/images",
                       help="测试图片目录")
    parser.add_argument("--test-masks", type=str,
                       default="Kvasir-SEG/masks",
                       help="测试mask目录")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                       help="YOLO置信度阈值")
    
    args = parser.parse_args()
    
    print("="*70)
    print("模型性能对比")
    print("="*70)
    
    # 评估原始模型
    print("\n【评估原始模型】")
    print(f"模型: {args.sam_weights_1}")
    metrics1 = run_evaluation(
        args.yolo_weights,
        args.sam_weights_1,
        args.test_images,
        args.test_masks,
        "evaluation_original",
        args.yolo_conf
    )
    
    # 评估优化模型
    print("\n【评估优化模型】")
    print(f"模型: {args.sam_weights_2}")
    metrics2 = run_evaluation(
        args.yolo_weights,
        args.sam_weights_2,
        args.test_images,
        args.test_masks,
        "evaluation_optimized",
        args.yolo_conf
    )
    
    # 对比结果
    if metrics1 and metrics2:
        print("\n" + "="*70)
        print("性能对比结果")
        print("="*70)
        print(f"{'指标':<15} {'原始模型':<15} {'优化模型':<15} {'改进':<15}")
        print("-"*70)
        
        for key in ['dice', 'iou', 'precision', 'recall', 'f1']:
            val1 = metrics1.get(key, 0) * 100
            val2 = metrics2.get(key, 0) * 100
            improvement = val2 - val1
            sign = "+" if improvement > 0 else ""
            print(f"{key.capitalize():<15} {val1:>6.2f}%      {val2:>6.2f}%      {sign}{improvement:>6.2f}%")
        
        print("="*70)
        
        # 计算总体改进
        dice_improvement = (metrics2['dice'] - metrics1['dice']) / metrics1['dice'] * 100 if metrics1['dice'] > 0 else 0
        recall_improvement = (metrics2['recall'] - metrics1['recall']) / metrics1['recall'] * 100 if metrics1['recall'] > 0 else 0
        
        print(f"\n关键改进：")
        print(f"  Dice改进: {dice_improvement:+.2f}%")
        print(f"  召回率改进: {recall_improvement:+.2f}%")
    else:
        print("评估失败，无法对比")


if __name__ == "__main__":
    compare_models()





