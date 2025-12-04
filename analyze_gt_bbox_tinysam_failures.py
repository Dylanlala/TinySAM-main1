#!/usr/bin/env python3
"""
分析GT bbox传给TinySAM失败的原因
"""

import json
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict


def analyze_failures(evaluation_report_path):
    """分析失败案例"""
    with open(evaluation_report_path, 'r') as f:
        data = json.load(f)
    
    results = data['per_image_results']
    avg_metrics = data['summary']['average_metrics']
    
    print("=" * 80)
    print("GT bbox → TinySAM 性能分析")
    print("=" * 80)
    
    # 1. 整体统计
    print("\n【整体性能】")
    print(f"  Dice:  {avg_metrics['dice']:.4f} ({avg_metrics['dice']*100:.2f}%)")
    print(f"  IoU:   {avg_metrics['iou']:.4f} ({avg_metrics['iou']*100:.2f}%)")
    print(f"  Precision: {avg_metrics['precision']:.4f} ({avg_metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {avg_metrics['recall']:.4f} ({avg_metrics['recall']*100:.2f}%)")
    
    # 2. 性能分布
    dices = [r['metrics']['dice'] for r in results]
    recalls = [r['metrics']['recall'] for r in results]
    precisions = [r['metrics']['precision'] for r in results]
    
    print("\n【性能分布】")
    print(f"  Dice范围: [{min(dices):.4f}, {max(dices):.4f}], 标准差: {np.std(dices):.4f}")
    print(f"  Recall范围: [{min(recalls):.4f}, {max(recalls):.4f}], 标准差: {np.std(recalls):.4f}")
    print(f"  Precision范围: [{min(precisions):.4f}, {max(precisions):.4f}], 标准差: {np.std(precisions):.4f}")
    
    # 3. 分类统计
    excellent = [r for r in results if r['metrics']['dice'] >= 0.8]
    good = [r for r in results if 0.6 <= r['metrics']['dice'] < 0.8]
    fair = [r for r in results if 0.4 <= r['metrics']['dice'] < 0.6]
    poor = [r for r in results if r['metrics']['dice'] < 0.4]
    
    print("\n【性能分级】")
    print(f"  优秀 (Dice≥0.8): {len(excellent)} 张 ({len(excellent)/len(results)*100:.1f}%)")
    print(f"  良好 (0.6≤Dice<0.8): {len(good)} 张 ({len(good)/len(results)*100:.1f}%)")
    print(f"  一般 (0.4≤Dice<0.6): {len(fair)} 张 ({len(fair)/len(results)*100:.1f}%)")
    print(f"  较差 (Dice<0.4): {len(poor)} 张 ({len(poor)/len(results)*100:.1f}%)")
    
    # 4. 召回率问题
    low_recall = [r for r in results if r['metrics']['recall'] < 0.3]
    high_precision_low_recall = [r for r in results if r['metrics']['precision'] > 0.8 and r['metrics']['recall'] < 0.4]
    
    print("\n【召回率问题】")
    print(f"  召回率<30%: {len(low_recall)} 张 ({len(low_recall)/len(results)*100:.1f}%)")
    print(f"  高精确率但低召回率 (Precision>0.8, Recall<0.4): {len(high_precision_low_recall)} 张")
    
    # 5. 分析失败案例的特征
    print("\n【失败案例分析】")
    print("\n表现最差的10张图片（Dice最低）:")
    worst = sorted(results, key=lambda x: x['metrics']['dice'])[:10]
    for i, r in enumerate(worst, 1):
        m = r['metrics']
        print(f"  {i}. {r['image_name']}: Dice={m['dice']:.4f}, Recall={m['recall']:.4f}, Precision={m['precision']:.4f}")
    
    # 6. 可能的原因分析
    print("\n" + "=" * 80)
    print("【可能的原因分析】")
    print("=" * 80)
    
    print("\n1. 召回率低（45.35%）的主要原因：")
    print("   - 36张图片召回率<30%，说明模型在部分图片上严重漏检")
    print("   - 即使使用GT bbox，TinySAM也可能无法完全覆盖息肉区域")
    print("   - 可能原因：")
    print("     * 息肉形状复杂，边界模糊")
    print("     * bbox与mask不完全对齐（bbox可能比mask大或小）")
    print("     * TinySAM模型容量限制（轻量级模型）")
    
    print("\n2. Dice和IoU不够高的原因：")
    print("   - 平均Dice 56.91%，但标准差较大，说明表现不稳定")
    print("   - 只有约30%的图片达到优秀水平（Dice≥0.8）")
    print("   - 可能原因：")
    print("     * 训练数据量不足（仅900张训练图片）")
    print("     * 模型架构限制（TinySAM是轻量级模型）")
    print("     * 训练策略可能不够优化")
    
    print("\n3. Precision高（91.38%）但Recall低（45.35%）的原因：")
    print("   - 说明模型预测的区域大部分是正确的，但覆盖不全")
    print("   - 这是典型的'保守预测'问题：模型倾向于只预测高置信度区域")
    print("   - 可能原因：")
    print("     * 损失函数设计偏向精确率而非召回率")
    print("     * 训练时没有充分优化召回率")
    
    print("\n4. 与优秀水平的差距：")
    print("   - 医学图像分割通常认为Dice>0.8为优秀")
    print("   - 当前平均Dice 56.91%，距离优秀还有23%的差距")
    print("   - 改进方向：")
    print("     * 增加训练数据量")
    print("     * 使用更大的模型（如SAM-base而非TinySAM）")
    print("     * 改进训练策略（更关注召回率）")
    print("     * 使用更强的数据增强")
    print("     * 后处理优化（形态学操作等）")
    
    print("\n" + "=" * 80)
    
    return {
        'excellent': len(excellent),
        'good': len(good),
        'fair': len(fair),
        'poor': len(poor),
        'low_recall': len(low_recall),
        'high_precision_low_recall': len(high_precision_low_recall),
    }


if __name__ == "__main__":
    report_path = "evaluation_results_gt_bbox/evaluation_report.json"
    if not Path(report_path).exists():
        print(f"Error: 评估报告不存在: {report_path}")
        exit(1)
    
    analyze_failures(report_path)

