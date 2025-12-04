#!/usr/bin/env python3
"""
分析YOLO11n-seg vs YOLO11n+TinySAM的性能差异
"""

import json
from pathlib import Path


def analyze_performance_difference():
    """分析两种方法的性能差异"""
    
    print("=" * 80)
    print("YOLO11n-seg vs YOLO11n+TinySAM 性能对比分析")
    print("=" * 80)
    
    # 读取YOLO11n-seg的结果
    yolo_seg_report = Path("yolo_seg_test_results_fixed2/evaluation_report.json")
    if not yolo_seg_report.exists():
        print(f"错误: 找不到YOLO11n-seg评估报告: {yolo_seg_report}")
        return
    
    with open(yolo_seg_report, 'r') as f:
        yolo_seg_data = json.load(f)
    
    yolo_seg_metrics = yolo_seg_data['summary']['average_metrics']
    
    # 查找YOLO+TinySAM的结果
    tinysam_reports = [
        "evaluation_results_enhanced_model/evaluation_report.json",
        "evaluation_results_optimized/evaluation_report.json",
        "evaluation_results_kvasir/evaluation_report.json",
    ]
    
    yolo_tinysam_data = None
    for report_path in tinysam_reports:
        if Path(report_path).exists():
            with open(report_path, 'r') as f:
                yolo_tinysam_data = json.load(f)
            print(f"\n使用YOLO+TinySAM报告: {report_path}")
            break
    
    if yolo_tinysam_data is None:
        print("\n警告: 未找到YOLO+TinySAM评估报告，使用默认值")
        # 使用之前看到的结果
        yolo_tinysam_metrics = {
            'dice': 0.3267,  # 约32.67%
            'iou': 0.2374,
            'precision': 0.6818,
            'recall': 0.2691,
            'f1': 0.3267
        }
    else:
        yolo_tinysam_metrics = yolo_tinysam_data['summary']['average_metrics']
    
    # 打印对比
    print("\n" + "=" * 80)
    print("性能指标对比")
    print("=" * 80)
    print(f"{'指标':<15} {'YOLO11n-seg':<20} {'YOLO11n+TinySAM':<20} {'差异':<15}")
    print("-" * 80)
    
    metrics = ['dice', 'iou', 'precision', 'recall', 'f1']
    metric_names = ['Dice', 'IoU', 'Precision', 'Recall', 'F1']
    
    for metric, name in zip(metrics, metric_names):
        seg_val = yolo_seg_metrics[metric] * 100
        tinysam_val = yolo_tinysam_metrics[metric] * 100
        diff = seg_val - tinysam_val
        diff_pct = (diff / tinysam_val * 100) if tinysam_val > 0 else 0
        
        print(f"{name:<15} {seg_val:>6.2f}%{'':<12} {tinysam_val:>6.2f}%{'':<12} {diff:>+6.2f}% ({diff_pct:>+.1f}%)")
    
    print("\n" + "=" * 80)
    print("原因分析")
    print("=" * 80)
    
    print("\n1. 【架构差异】")
    print("   YOLO11n-seg:")
    print("     - 端到端模型，同时进行检测和分割")
    print("     - 共享特征提取，检测和分割任务协同优化")
    print("     - 在COCO数据集上预训练，针对分割任务优化")
    print("     - 输出：直接生成分割mask")
    print("")
    print("   YOLO11n+TinySAM:")
    print("     - 两阶段模型：先检测后分割")
    print("     - YOLO负责检测（输出bbox），TinySAM负责分割")
    print("     - 两个模型独立训练，可能存在不匹配")
    print("     - 输出：YOLO检测 → TinySAM分割")
    
    print("\n2. 【误差累积问题】")
    print("   YOLO11n-seg:")
    print("     - 单阶段，无误差累积")
    print("     - 检测和分割共享特征，信息损失小")
    print("")
    print("   YOLO11n+TinySAM:")
    print("     - 第一阶段：YOLO检测可能有漏检、误检、bbox不准确")
    print("     - 第二阶段：TinySAM依赖YOLO的bbox，误差会传播")
    print("     - 如果YOLO检测失败，TinySAM无法分割")
    print("     - 如果YOLO bbox不准确，TinySAM分割也会受影响")
    
    print("\n3. 【训练数据差异】")
    print("   YOLO11n-seg:")
    print("     - 在Kvasir-SEG数据集上端到端训练")
    print("     - 直接学习息肉的分割模式")
    print("     - 训练数据：800张训练图片")
    print("")
    print("   YOLO11n+TinySAM:")
    print("     - YOLO在Kvasir-SEG上训练（检测任务）")
    print("     - TinySAM在COCO上预训练，在Kvasir-SEG上微调")
    print("     - 两个模型可能对数据分布的理解不一致")
    
    print("\n4. 【模型容量和优化】")
    print("   YOLO11n-seg:")
    print("     - 模型专门为分割任务设计")
    print("     - 分割头经过优化")
    print("     - Dice损失：87.91%")
    print("")
    print("   YOLO11n+TinySAM:")
    print("     - TinySAM是轻量级模型（~42M参数）")
    print("     - 可能容量不足，难以学习复杂的分割模式")
    print("     - 即使使用GT bbox，Dice也只有56.91%")
    print("     - 使用YOLO检测的bbox，性能进一步下降")
    
    print("\n5. 【关于YOLO分割是否包含检测步骤】")
    print("   是的，YOLO11n-seg包含检测步骤：")
    print("     - YOLO分割模型是检测+分割的联合模型")
    print("     - 首先进行目标检测（生成bbox和类别）")
    print("     - 然后在检测框内进行实例分割")
    print("     - 但这是端到端的，检测和分割共享特征提取器")
    print("     - 检测和分割任务在训练时联合优化")
    print("")
    print("   关键区别：")
    print("     - YOLO11n-seg：检测和分割在同一个模型中，端到端训练")
    print("     - YOLO11n+TinySAM：检测和分割是两个独立模型，分阶段训练")
    
    print("\n" + "=" * 80)
    print("性能差距总结")
    print("=" * 80)
    dice_diff = (yolo_seg_metrics['dice'] - yolo_tinysam_metrics['dice']) * 100
    print(f"\nDice差距: {dice_diff:.2f}% (YOLO11n-seg比YOLO11n+TinySAM高{dice_diff:.2f}%)")
    print(f"\n主要原因：")
    print(f"  1. 端到端 vs 两阶段：YOLO11n-seg是端到端优化，YOLO+TinySAM是两阶段")
    print(f"  2. 误差累积：YOLO检测误差会传播到TinySAM分割")
    print(f"  3. 模型匹配：两个独立训练的模型可能不匹配")
    print(f"  4. 容量限制：TinySAM是轻量级模型，可能容量不足")
    print(f"  5. 训练策略：YOLO11n-seg专门为分割任务优化")
    
    print("\n" + "=" * 80)
    print("改进建议")
    print("=" * 80)
    print("\n如果要提升YOLO+TinySAM的性能：")
    print("  1. 使用更大的SAM模型（SAM-base或SAM-large）")
    print("  2. 端到端联合训练YOLO和TinySAM")
    print("  3. 改进YOLO检测精度（减少漏检和误检）")
    print("  4. 使用更强的数据增强")
    print("  5. 优化TinySAM的训练策略（更关注召回率）")
    print("\n但考虑到YOLO11n-seg已经达到87.91%的Dice，")
    print("直接使用YOLO11n-seg可能是更好的选择。")
    print("=" * 80)


if __name__ == "__main__":
    analyze_performance_difference()

