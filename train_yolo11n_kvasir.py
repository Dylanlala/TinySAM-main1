#!/usr/bin/env python3
"""
训练YOLO11n模型用于Kvasir-SEG息肉检测
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


def split_dataset(images_dir, labels_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    划分数据集为train/val/test
    
    Args:
        images_dir: 图片目录
        labels_dir: 标签目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    import random
    import shutil
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # 获取所有图片文件
    image_files = list(images_path.glob("*.jpg"))
    random.seed(42)  # 固定随机种子
    random.shuffle(image_files)
    
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")
    
    # 创建目录
    base_dir = images_path.parent
    for split in ['train', 'val', 'test']:
        (base_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        for img_file in files:
            # 复制图片
            shutil.copy2(img_file, base_dir / split_name / 'images' / img_file.name)
            # 复制标签
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, base_dir / split_name / 'labels' / f"{img_file.stem}.txt")
    
    return base_dir


def train_yolo11n(
    data_yaml,
    model_name="yolo11n.pt",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/detect",
    name="yolo11n_kvasir",
    patience=50,
    save_period=10,
):
    """
    训练YOLO11n模型
    
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
    print(f"开始训练YOLO11n...")
    print(f"数据配置: {data_yaml}")
    print(f"模型: {model_name}")
    print(f"训练轮数: {epochs}")
    print(f"图片尺寸: {imgsz}")
    print(f"批次大小: {batch}")
    print(f"设备: {device}")
    
    # 加载模型
    model = YOLO(model_name)
    
    # 训练参数
    train_args = {
        'data': data_yaml,
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
    }
    
    # 开始训练
    results = model.train(**train_args)
    
    print(f"\n训练完成！")
    print(f"最佳模型保存在: {Path(project) / name / 'weights' / 'best.pt'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="训练YOLO11n用于Kvasir-SEG息肉检测")
    parser.add_argument("--data", type=str, 
                       default="configs/kvasir_seg_yolo.yaml",
                       help="数据配置文件路径")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                       help="模型名称或权重路径")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="图片尺寸")
    parser.add_argument("--batch", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU设备ID")
    parser.add_argument("--project", type=str, default="runs/detect",
                       help="项目目录")
    parser.add_argument("--name", type=str, default="yolo11n_kvasir",
                       help="实验名称")
    parser.add_argument("--patience", type=int, default=50,
                       help="早停耐心值")
    parser.add_argument("--split", action="store_true",
                       help="是否划分数据集（如果数据集未划分）")
    
    args = parser.parse_args()
    
    # 如果需要划分数据集
    if args.split:
        base_dir = Path("/home/huangmanling/huangmanling/yolo_sam/TinySAM-main/Kvasir-SEG")
        images_dir = base_dir / "images"
        labels_dir = base_dir / "labels"
        
        if not labels_dir.exists():
            print("Error: labels directory not found. Please run convert_kvasir_to_yolo.py first!")
            return
        
        print("划分数据集...")
        split_dir = split_dataset(images_dir, labels_dir)
        
        # 更新数据配置文件
        data_yaml_path = Path(args.data)
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r') as f:
                content = f.read()
            content = content.replace('train: images', 'train: train/images')
            content = content.replace('val: images', 'val: val/images')
            with open(data_yaml_path, 'w') as f:
                f.write(content)
            print(f"已更新数据配置文件: {data_yaml_path}")
    
    # 开始训练
    train_yolo11n(
        data_yaml=args.data,
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


