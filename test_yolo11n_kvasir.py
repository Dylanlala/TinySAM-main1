#!/usr/bin/env python3
"""
在测试集上评估训练好的YOLO11n模型
"""

import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# 添加ultralytics路径
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRA = REPO_ROOT / "ultralyticss_new"
if LOCAL_ULTRA.exists() and str(LOCAL_ULTRA) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRA))

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error importing YOLO: {e}")
    sys.exit(1)


def test_yolo_model(
    weights_path,
    test_images_dir,
    test_labels_dir=None,
    conf_threshold=0.25,
    iou_threshold=0.45,
    imgsz=640,
    device=0,
    save_dir="test_results",
    save_txt=False,
    save_conf=True,
    save_crop=False,
    visualize=True,
):
    """
    在测试集上评估YOLO模型
    
    Args:
        weights_path: 模型权重路径
        test_images_dir: 测试图片目录
        test_labels_dir: 测试标签目录（可选，用于计算mAP）
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        imgsz: 图片尺寸
        device: 设备ID
        save_dir: 结果保存目录
        save_txt: 是否保存txt格式的检测结果
        save_conf: 是否在txt中保存置信度
        save_crop: 是否保存裁剪的检测框
        visualize: 是否保存可视化结果
    """
    print(f"开始测试YOLO11n模型...")
    print(f"模型权重: {weights_path}")
    print(f"测试图片目录: {test_images_dir}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"IoU阈值: {iou_threshold}")
    
    # 检查权重文件
    if not Path(weights_path).exists():
        print(f"Error: 权重文件不存在: {weights_path}")
        return None
    
    # 加载模型
    print(f"\n加载模型...")
    model = YOLO(weights_path)
    
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 如果有标签目录，进行验证（计算mAP等指标）
    if test_labels_dir and Path(test_labels_dir).exists():
        print(f"\n使用标签进行验证（计算mAP等指标）...")
        print(f"标签目录: {test_labels_dir}")
        
        # 使用YOLO的验证功能
        results = model.val(
            data=str(REPO_ROOT / "configs" / "kvasir_seg_yolo.yaml"),
            split='test',  # 使用test split
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            save_json=True,  # 保存JSON格式结果
            save_hybrid=False,
            plots=True,  # 生成图表
            verbose=True,
        )
        
        # 打印评估结果
        print("\n" + "="*60)
        print("测试集评估结果")
        print("="*60)
        if hasattr(results, 'box'):
            metrics = results.box
            print(f"mAP50: {metrics.map50:.4f}")
            print(f"mAP50-95: {metrics.map:.4f}")
            print(f"Precision: {metrics.mp:.4f}")
            print(f"Recall: {metrics.mr:.4f}")
        
        # 保存评估结果到JSON
        eval_results = {
            'model': weights_path,
            'test_images_dir': str(test_images_dir),
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'timestamp': datetime.now().isoformat(),
        }
        
        if hasattr(results, 'box'):
            eval_results['metrics'] = {
                'map50': float(results.box.map50),
                'map50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
            }
        
        with open(save_path / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"\n评估结果已保存到: {save_path / 'evaluation_results.json'}")
        
    else:
        print(f"\n未提供标签目录，仅进行推理（不计算mAP）...")
        # 仅推理模式
        results = model.predict(
            source=str(test_images_dir),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            save=True,  # 保存可视化结果
            save_txt=save_txt,  # 保存txt格式
            save_conf=save_conf,
            save_crop=save_crop,
            project=str(save_path),
            name='predictions',
        )
        
        print(f"\n推理完成！结果保存在: {save_path / 'predictions'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="在测试集上测试YOLO11n模型")
    parser.add_argument("--weights", type=str, 
                       default="runs/detect/yolo11n_kvasir/weights/best.pt",
                       help="模型权重路径")
    parser.add_argument("--test-images", type=str,
                       default="Kvasir-SEG/test/images",
                       help="测试图片目录")
    parser.add_argument("--test-labels", type=str,
                       default="Kvasir-SEG/test/labels",
                       help="测试标签目录（用于计算mAP）")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU阈值")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="图片尺寸")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU设备ID")
    parser.add_argument("--save-dir", type=str, default="test_results",
                       help="结果保存目录")
    parser.add_argument("--no-labels", action="store_true",
                       help="不使用标签（仅推理模式）")
    parser.add_argument("--save-txt", action="store_true",
                       help="保存txt格式的检测结果")
    parser.add_argument("--save-conf", action="store_true", default=True,
                       help="在txt中保存置信度")
    parser.add_argument("--save-crop", action="store_true",
                       help="保存裁剪的检测框")
    
    args = parser.parse_args()
    
    # 执行测试
    test_yolo_model(
        weights_path=args.weights,
        test_images_dir=args.test_images,
        test_labels_dir=None if args.no_labels else args.test_labels,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_dir=args.save_dir,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
    )


if __name__ == "__main__":
    main()


