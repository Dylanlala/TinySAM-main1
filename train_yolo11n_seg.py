#!/usr/bin/env python3
"""
训练YOLO11n-seg模型用于息肉分割
对比YOLO11n-seg vs YOLO11n+TinySAM的效果
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


def prepare_yolo_seg_config(data_dir, coco_json=None, output_config="configs/kvasir_seg_yolo_seg.yaml"):
    """
    准备YOLO分割的数据配置文件
    
    YOLO分割可以使用：
    1. COCO格式的JSON文件（推荐，包含segmentation多边形）
    2. 或者YOLO格式 + mask文件
    
    Args:
        data_dir: 数据集根目录
        coco_json: COCO格式的JSON文件路径（如果提供，将使用COCO格式）
        output_config: 输出配置文件路径
    """
    data_dir = Path(data_dir)
    
    # 如果提供了COCO JSON文件，使用COCO格式
    if coco_json and Path(coco_json).exists():
        # 检查是否有分离的train/val JSON文件
        coco_path = Path(coco_json)
        train_json = coco_path.parent / f"{coco_path.stem}-train.json"
        val_json = coco_path.parent / f"{coco_path.stem}-val.json"
        
        if train_json.exists() and val_json.exists():
            # 使用分离的train/val JSON文件
            # 使用绝对路径
            config = {
                'path': str(data_dir.absolute()),
                'train': str(train_json.absolute()),
                'val': str(val_json.absolute()),
                'nc': 1,
                'names': {
                    0: 'polyp'
                }
            }
            print(f"使用分离的COCO格式文件:")
            print(f"  训练集: {train_json.absolute()}")
            print(f"  验证集: {val_json.absolute()}")
        else:
            # 使用单个COCO JSON文件（YOLO会自动划分）
            coco_path = Path(coco_json)
            # 如果JSON文件在data_dir下，使用相对路径；否则使用绝对路径
            if coco_path.parent == data_dir:
                train_val_path = coco_path.name
            else:
                train_val_path = str(coco_path.absolute())
            config = {
                'path': str(data_dir.absolute()),
                'train': train_val_path,
                'val': train_val_path,
                'nc': 1,
                'names': {
                    0: 'polyp'
                }
            }
            print(f"使用COCO格式标注文件: {coco_json}")
            print("注意：train和val使用同一个文件，YOLO会自动划分")
    else:
        # 使用YOLO格式
        config = {
            'path': str(data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': {
                0: 'polyp'
            }
        }
    
    # 保存配置文件
    output_path = Path(output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"配置文件已保存到: {output_path}")
    return str(output_path)


def train_yolo11n_seg(
    data_config,
    model_name="yolo11n-seg.pt",
    epochs=200,
    batch=16,
    imgsz=640,
    device=0,
    conf=0.5,
    iou=0.5,
    project="runs/segment",
    name="yolo11n_seg_kvasir",
):
    """
    训练YOLO11n-seg模型
    
    Args:
        data_config: 数据配置文件路径
        model_name: 预训练模型名称
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 图像尺寸
        device: 设备ID
        conf: 置信度阈值
        iou: IoU阈值
        project: 项目目录
        name: 实验名称
    """
    print("="*70)
    print("开始训练YOLO11n-seg模型")
    print("="*70)
    print(f"数据配置: {data_config}")
    print(f"预训练模型: {model_name}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch}")
    print(f"图像尺寸: {imgsz}")
    print("="*70)
    
    # 加载模型
    print(f"\n加载模型: {model_name}")
    model = YOLO(model_name)
    
    # 训练
    print("\n开始训练...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        project=project,
        name=name,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"最佳模型保存在: {project}/{name}/weights/best.pt")
    print(f"最终模型保存在: {project}/{name}/weights/last.pt")
    print("="*70)
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description="训练YOLO11n-seg模型")
    parser.add_argument("--data-dir", type=str, default="Kvasir-SEG",
                       help="数据集根目录")
    parser.add_argument("--data-config", type=str, default="configs/kvasir_seg_yolo_seg.yaml",
                       help="数据配置文件路径")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt",
                       help="预训练模型（yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt）")
    parser.add_argument("--epochs", type=int, default=200,
                       help="训练轮数")
    parser.add_argument("--batch", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="图像尺寸")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU设备ID")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5,
                       help="IoU阈值")
    parser.add_argument("--project", type=str, default="runs/segment",
                       help="项目目录")
    parser.add_argument("--name", type=str, default="yolo11n_seg_kvasir",
                       help="实验名称")
    parser.add_argument("--prepare-config", action="store_true",
                       help="准备数据配置文件")
    parser.add_argument("--coco-json", type=str, default="Kvasir-SEG/kavsir-seg.json",
                       help="COCO格式的JSON标注文件路径")
    
    args = parser.parse_args()
    
    # 准备配置文件
    # 只有在明确指定--prepare-config或配置文件不存在时才重新生成
    if args.prepare_config:
        print("准备数据配置文件...")
        coco_json = args.coco_json if Path(args.coco_json).exists() else None
        data_config = prepare_yolo_seg_config(args.data_dir, coco_json, args.data_config)
    elif not Path(args.data_config).exists():
        print(f"配置文件不存在: {args.data_config}")
        print("准备数据配置文件...")
        coco_json = args.coco_json if Path(args.coco_json).exists() else None
        data_config = prepare_yolo_seg_config(args.data_dir, coco_json, args.data_config)
    else:
        # 配置文件已存在，直接使用
        data_config = args.data_config
        print(f"使用现有配置文件: {data_config}")
    
    # 训练
    model, results = train_yolo11n_seg(
        data_config=data_config,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
    )
    
    # 评估
    print("\n" + "="*70)
    print("在验证集上评估...")
    print("="*70)
    metrics = model.val()
    print(f"\n验证集指标:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    if hasattr(metrics, 'seg'):
        print(f"  Seg mAP50: {metrics.seg.map50:.4f}")
        print(f"  Seg mAP50-95: {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()

