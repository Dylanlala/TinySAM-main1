#!/usr/bin/env python3
"""
可视化YOLO11n-seg分割结果，显示预测mask和GT mask对比
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from ultralytics import YOLO


def visualize_yolo_seg_with_mask(
    model_weights,
    test_images_dir,
    test_masks_dir,
    output_dir="yolo_seg_visualizations",
    conf_threshold=0.25,
    iou_threshold=0.45,
    max_images=None,
):
    """
    可视化YOLO11n-seg分割结果，显示预测mask和GT mask对比
    
    Args:
        model_weights: 模型权重路径
        test_images_dir: 测试图片目录
        test_masks_dir: GT mask目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        max_images: 最大处理图片数（None表示处理所有）
    """
    print("=" * 70)
    print("YOLO11n-seg 分割结果可视化（带GT mask对比）")
    print("=" * 70)
    
    # 检查模型文件
    model_path = Path(model_weights)
    if not model_path.exists():
        print(f"错误: 模型文件不存在 {model_weights}")
        return
    
    # 加载模型
    print(f"\n加载模型: {model_weights}")
    model = YOLO(model_weights)
    
    # 获取测试图片
    test_images_path = Path(test_images_dir)
    test_masks_path = Path(test_masks_dir)
    
    image_files = sorted(list(test_images_path.glob("*.jpg")))
    if not image_files:
        image_files = sorted(list(test_images_path.glob("*.png")))
    
    if not image_files:
        print(f"错误: 在 {test_images_dir} 中未找到图片文件")
        return
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n找到 {len(image_files)} 张测试图片")
    print(f"输出目录: {output_dir}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"IoU阈值: {iou_threshold}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n开始处理...")
    
    # 统计信息
    total_detections = 0
    images_with_detections = 0
    
    # 处理每张图片
    for img_file in tqdm(image_files, desc="处理图片"):
        image_name = img_file.stem
        
        # 读取原图
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # YOLO分割推理
        results = model.predict(
            str(img_file),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        
        result = results[0]
        
        # 合并所有检测的mask
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        num_detections = 0
        confidences = []
        
        if result.masks is not None and len(result.masks) > 0:
            # 获取所有mask
            masks = result.masks.data.cpu().numpy()  # [N, H, W]
            
            # 调整mask尺寸到原始图像尺寸
            for i, mask in enumerate(masks):
                # mask是模型输出尺寸，需要调整到原始尺寸
                mask_resized = cv2.resize(
                    (mask * 255).astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
                # 合并mask（取最大值）
                pred_mask = np.maximum(pred_mask, mask_resized)
                num_detections += 1
                
                # 获取置信度
                if result.boxes is not None and i < len(result.boxes):
                    conf = float(result.boxes.conf[i])
                    confidences.append(conf)
            
            images_with_detections += 1
            total_detections += num_detections
        
        # 读取GT mask
        gt_mask_file = test_masks_path / f"{image_name}.jpg"
        if not gt_mask_file.exists():
            gt_mask_file = test_masks_path / f"{image_name}.png"
        
        gt_mask = None
        if gt_mask_file.exists():
            gt_mask = cv2.imread(str(gt_mask_file), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                # 调整GT mask尺寸以匹配原图
                if gt_mask.shape[:2] != (h, w):
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 创建可视化
        # 左侧：原图 + 预测mask叠加
        pred_overlay = image.copy()
        if num_detections > 0:
            # 将预测mask转换为彩色
            pred_mask_colored = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
            # 叠加到原图上（半透明）
            pred_overlay = cv2.addWeighted(image, 0.6, pred_mask_colored, 0.4, 0)
        
        # 右侧：原图 + GT mask叠加
        if gt_mask is not None:
            gt_mask_colored = cv2.applyColorMap(gt_mask, cv2.COLORMAP_JET)
            gt_overlay = cv2.addWeighted(image, 0.6, gt_mask_colored, 0.4, 0)
        else:
            gt_overlay = image.copy()
            cv2.putText(
                gt_overlay,
                "No GT Mask",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        # 确保两张图片高度一致
        h1, w1 = pred_overlay.shape[:2]
        h2, w2 = gt_overlay.shape[:2]
        
        # 调整到相同高度
        if h1 != h2:
            scale = h2 / h1
            new_w1 = int(w1 * scale)
            pred_overlay = cv2.resize(
                pred_overlay, (new_w1, h2), interpolation=cv2.INTER_LINEAR
            )
            h1 = h2
            w1 = new_w1
        
        # 水平拼接
        combined = np.hstack([pred_overlay, gt_overlay])
        
        # 添加标题和信息
        title_height = 60
        combined_with_title = np.ones(
            (h1 + title_height, w1 + w2, 3), dtype=np.uint8
        ) * 255
        
        # 放置图片
        combined_with_title[title_height:, :] = combined
        
        # 添加标题文本
        cv2.putText(
            combined_with_title,
            f"YOLO11n-seg Prediction (Conf>{conf_threshold})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        cv2.putText(
            combined_with_title,
            "Ground Truth Mask",
            (w1 + 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # 添加检测信息
        info_text = f"Detections: {num_detections}"
        if len(confidences) > 0:
            avg_conf = np.mean(confidences)
            info_text += f" | Avg Conf: {avg_conf:.2f}"
        
        cv2.putText(
            combined_with_title,
            info_text,
            (10, h1 + title_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )
        
        # 保存结果
        output_file = output_path / f"{image_name}_seg_mask.jpg"
        cv2.imwrite(str(output_file), combined_with_title)
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print(f"总图片数: {len(image_files)}")
    print(f"有检测结果的图片: {images_with_detections}")
    print(f"总检测数: {total_detections}")
    if images_with_detections > 0:
        print(f"平均每张图片检测数: {total_detections / images_with_detections:.2f}")
    print(f"\n可视化结果保存在: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="可视化YOLO11n-seg分割结果，显示预测mask和GT mask对比"
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="runs/segment/yolo11n_seg_kvasir_fixed2/weights/best.pt",
        help="模型权重路径"
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default="Kvasir-SEG/test/images",
        help="测试图片目录"
    )
    parser.add_argument(
        "--test-masks",
        type=str,
        default="Kvasir-SEG/masks",
        help="GT mask目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="yolo_seg_visualizations",
        help="输出目录"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU阈值"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="最大处理图片数（None表示处理所有）"
    )
    
    args = parser.parse_args()
    
    visualize_yolo_seg_with_mask(
        model_weights=args.model_weights,
        test_images_dir=args.test_images,
        test_masks_dir=args.test_masks,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_images=args.max_images,
    )

