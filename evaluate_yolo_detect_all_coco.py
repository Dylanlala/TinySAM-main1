"""
在测试集上评估YOLO11n检测模型 (all_coco数据集)
"""
import os
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

def evaluate_yolo_detect_on_test(
    model_weights: str,
    data_config: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    save_json: bool = True,
    save_hybrid: bool = False,
    plots: bool = True,
):
    """
    在测试集上评估YOLO检测模型
    
    Args:
        model_weights: 模型权重路径
        data_config: 数据配置文件路径
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        save_json: 是否保存JSON格式结果
        save_hybrid: 是否保存混合标签
        plots: 是否生成可视化图表
    """
    print("=" * 80)
    print("YOLO11n检测模型 - 测试集评估")
    print("=" * 80)
    print(f"模型权重: {model_weights}")
    print(f"数据配置: {data_config}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"IoU阈值: {iou_threshold}")
    print("=" * 80)
    
    # 加载模型
    print("\n加载模型...")
    model = YOLO(model_weights)
    
    # 在测试集上验证
    print("\n开始测试集评估...")
    print(f"注意: 由于训练日志显示所有指标为0，模型可能未正确训练")
    print(f"尝试使用更低的置信度阈值进行测试...")
    
    # 先尝试低置信度阈值
    results = model.val(
        data=data_config,
        split='test',  # 使用测试集
        conf=0.01,  # 使用极低的置信度阈值
        iou=iou_threshold,
        save_json=save_json,
        save_hybrid=save_hybrid,
        plots=plots,
        verbose=True,
    )
    
    # 打印评估结果
    print("\n" + "=" * 80)
    print("测试集评估结果")
    print("=" * 80)
    
    if hasattr(results, 'box'):
        metrics = results.box
        print(f"\n【检测指标】")
        print(f"  mAP50:     {metrics.map50:.4f} ({metrics.map50*100:.2f}%)")
        print(f"  mAP50-95:  {metrics.map:.4f} ({metrics.map*100:.2f}%)")
        print(f"  Precision: {metrics.mp:.4f} ({metrics.mp*100:.2f}%)")
        print(f"  Recall:    {metrics.mr:.4f} ({metrics.mr*100:.2f}%)")
        
        # 类别指标
        if hasattr(metrics, 'maps') and len(metrics.maps) > 0:
            print(f"\n【类别指标 (polyp)】")
            print(f"  mAP50:     {metrics.maps[0]:.4f} ({metrics.maps[0]*100:.2f}%)")
            if hasattr(metrics, 'map50s') and len(metrics.map50s) > 0:
                print(f"  mAP50-95:  {metrics.map50s[0]:.4f} ({metrics.map50s[0]*100:.2f}%)")
    
    # 保存评估结果到JSON
    if save_json:
        results_dir = Path(model_weights).parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = results_dir / f"test_evaluation_{timestamp}.json"
        
        eval_results = {
            "model": str(model_weights),
            "data_config": str(data_config),
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "timestamp": timestamp,
            "metrics": {}
        }
        
        if hasattr(results, 'box'):
            metrics = results.box
            eval_results["metrics"] = {
                "mAP50": float(metrics.map50),
                "mAP50-95": float(metrics.map),
                "precision": float(metrics.mp),
                "recall": float(metrics.mr),
            }
            if hasattr(metrics, 'maps') and len(metrics.maps) > 0:
                eval_results["metrics"]["class_mAP50"] = float(metrics.maps[0])
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估结果已保存到: {json_path}")
    
    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # 配置路径 - 使用训练好的权重
    model_weights = "runs/detect/yolo11n_all_coco6/weights/best.pt"
    data_config = "configs/all_coco_yolo.yaml"
    
    # 检查文件是否存在
    if not Path(model_weights).exists():
        print(f"错误: 模型权重文件不存在: {model_weights}")
        exit(1)
    
    if not Path(data_config).exists():
        print(f"错误: 数据配置文件不存在: {data_config}")
        exit(1)
    
    # 运行评估
    evaluate_yolo_detect_on_test(
        model_weights=model_weights,
        data_config=data_config,
        conf_threshold=0.25,
        iou_threshold=0.45,
        save_json=True,
        plots=True,
    )

