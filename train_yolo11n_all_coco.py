#!/usr/bin/env python3
"""
训练YOLO11n模型用于all_coco息肉检测数据集
使用COCO格式的JSON标注文件
"""

import os
import sys
import argparse
from pathlib import Path

# 添加ultralytics路径
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRA = REPO_ROOT / "ultralyticss_new"
if LOCAL_ULTRA.exists() and str(LOCAL_ULTRA) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRA))

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error importing YOLO: {e}")
    print("Please ensure ultralytics is installed or ultralyticss_new directory exists")
    sys.exit(1)


def train_yolo11n_all_coco(
    data_yaml,
    model_name="yolo11n.pt",
    epochs=200,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/detect",
    name="yolo11n_all_coco",
    patience=50,
    save_period=10,
):
    """
    训练YOLO11n模型用于all_coco数据集
    
    Args:
        data_yaml: 数据配置文件路径
        model_name: 模型名称或权重路径
        epochs: 训练轮数
        imgsz: 图片尺寸
        batch: 批次大小
        device: 设备ID
        project: 项目目录
        name: 实验名称
        patience: 早停耐心值
        save_period: 保存周期
    """
    print("=" * 70)
    print("开始训练YOLO11n模型 - all_coco数据集")
    print("=" * 70)
    print(f"数据配置: {data_yaml}")
    print(f"模型: {model_name}")
    print(f"训练轮数: {epochs}")
    print(f"图片尺寸: {imgsz}")
    print(f"批次大小: {batch}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 检查数据配置文件
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        print(f"错误: 数据配置文件不存在: {data_yaml_path}")
        sys.exit(1)
    
    # 检查COCO JSON文件
    import yaml
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    train_json = base_path / config['train']
    val_json = base_path / config['val']
    
    if not train_json.exists():
        print(f"错误: 训练集JSON文件不存在: {train_json}")
        sys.exit(1)
    if not val_json.exists():
        print(f"错误: 验证集JSON文件不存在: {val_json}")
        sys.exit(1)
    
    print(f"\n✓ 数据文件检查通过")
    print(f"  训练集: {train_json}")
    print(f"  验证集: {val_json}")
    
    # 加载模型
    print(f"\n加载模型: {model_name}")
    model = YOLO(model_name)
    
    # 训练参数
    train_args = {
        'data': str(data_yaml_path.absolute()),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        'patience': patience,
        'save_period': save_period,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'verbose': True,
    }
    
    print("\n开始训练...")
    print("=" * 70)
    
    # 开始训练
    results = model.train(**train_args)
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    best_model = Path(project) / name / 'weights' / 'best.pt'
    last_model = Path(project) / name / 'weights' / 'last.pt'
    print(f"最佳模型: {best_model}")
    print(f"最终模型: {last_model}")
    print("=" * 70)
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description="训练YOLO11n用于all_coco息肉检测")
    parser.add_argument("--data-config", type=str, 
                       default="configs/all_coco_yolo.yaml",
                       help="数据配置文件路径")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                       help="模型名称或权重路径 (yolo11n.pt, yolo11s.pt, yolo11m.pt等)")
    parser.add_argument("--epochs", type=int, default=200,
                       help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="图片尺寸")
    parser.add_argument("--batch", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU设备ID")
    parser.add_argument("--project", type=str, default="runs/detect",
                       help="项目目录")
    parser.add_argument("--name", type=str, default="yolo11n_all_coco",
                       help="实验名称")
    parser.add_argument("--patience", type=int, default=50,
                       help="早停耐心值")
    
    args = parser.parse_args()
    
    # 开始训练
    train_yolo11n_all_coco(
        data_yaml=args.data_config,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()



