#!/usr/bin/env python3
"""
可视化YOLO检测结果，显示检测框、置信度分数，以及对应的GT mask
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# 添加ultralytics路径
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRA = REPO_ROOT / "ultralyticss_new"
if LOCAL_ULTRA.exists() and str(LOCAL_ULTRA) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRA))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics not found")


def draw_detections(image, boxes, confidences, class_names=None):
    """
    在图片上绘制检测框和置信度
    
    Args:
        image: 输入图片 (BGR格式)
        boxes: 检测框列表 [[x1, y1, x2, y2], ...]
        confidences: 置信度列表
        class_names: 类别名称列表（可选）
    
    Returns:
        绘制后的图片
    """
    img = image.copy()
    
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制检测框
        color = (0, 255, 0)  # 绿色
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # 准备标签文本
        label = f"{conf:.2f}"
        if class_names and i < len(class_names):
            label = f"{class_names[i]}: {label}"
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # 绘制文本背景
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # 绘制文本
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return img


def load_mask(mask_path):
    """
    加载mask图片
    
    Args:
        mask_path: mask文件路径
    
    Returns:
        mask数组 (灰度图) 或 None
    """
    if not mask_path.exists():
        return None
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return mask


def visualize_detection_with_mask(
    yolo_weights,
    test_images_dir,
    test_masks_dir,
    output_dir,
    conf_threshold=0.25,
    iou_threshold=0.45,
    max_images=None,
):
    """
    可视化YOLO检测结果，显示检测框、置信度，以及对应的GT mask
    
    Args:
        yolo_weights: YOLO模型权重路径
        test_images_dir: 测试图片目录
        test_masks_dir: GT mask目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        max_images: 最大处理图片数（None表示处理所有）
    """
    print("=" * 60)
    print("YOLO检测结果可视化（带GT mask对比）")
    print("=" * 60)
    
    # 检查文件
    if YOLO is None:
        print("Error: ultralytics package not found")
        return
    
    if not Path(yolo_weights).exists():
        print(f"Error: 权重文件不存在: {yolo_weights}")
        return
    
    test_images_path = Path(test_images_dir)
    test_masks_path = Path(test_masks_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载YOLO模型
    print(f"\n加载YOLO模型: {yolo_weights}")
    model = YOLO(yolo_weights)
    
    # 获取所有测试图片
    image_files = sorted(list(test_images_path.glob("*.jpg")))
    if not image_files:
        image_files = sorted(list(test_images_path.glob("*.png")))
    
    if not image_files:
        print(f"Error: 在 {test_images_dir} 中未找到图片文件")
        return
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n找到 {len(image_files)} 张测试图片")
    print(f"输出目录: {output_dir}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"IoU阈值: {iou_threshold}")
    print("\n开始处理...")
    
    # 统计信息
    total_detections = 0
    images_with_detections = 0
    images_without_detections = 0
    
    # 处理每张图片
    for img_file in tqdm(image_files, desc="处理图片"):
        img_name = img_file.stem
        
        # 读取原图
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"Warning: 无法读取图片 {img_file}")
            continue
        
        h, w = image.shape[:2]
        
        # YOLO检测
        results = model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # 提取检测结果
        boxes = []
        confidences = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
        
        # 统计
        if len(boxes) > 0:
            images_with_detections += 1
            total_detections += len(boxes)
        else:
            images_without_detections += 1
        
        # 绘制检测结果
        img_with_detections = draw_detections(image, boxes, confidences)
        
        # 加载GT mask
        mask_file = test_masks_path / f"{img_name}.jpg"
        if not mask_file.exists():
            mask_file = test_masks_path / f"{img_name}.png"
        
        mask = None
        if mask_file.exists():
            mask = load_mask(mask_file)
            if mask is not None:
                # 调整mask尺寸以匹配原图
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                # 转换为彩色显示（绿色mask）
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                # 叠加到原图上（半透明）
                mask_overlay = cv2.addWeighted(image, 0.6, mask_colored, 0.4, 0)
            else:
                mask_overlay = image.copy()
        else:
            # 如果没有mask，显示原图
            mask_overlay = image.copy()
            cv2.putText(
                mask_overlay,
                "No GT Mask",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        # 创建对比图：左边是检测结果，右边是GT mask
        # 确保两张图片高度一致
        h1, w1 = img_with_detections.shape[:2]
        h2, w2 = mask_overlay.shape[:2]
        
        # 调整到相同高度
        if h1 != h2:
            scale = h2 / h1
            new_w1 = int(w1 * scale)
            img_with_detections = cv2.resize(
                img_with_detections, (new_w1, h2), interpolation=cv2.INTER_LINEAR
            )
            h1 = h2
            w1 = new_w1
        
        # 水平拼接
        combined = np.hstack([img_with_detections, mask_overlay])
        
        # 添加标题
        title_height = 40
        combined_with_title = np.ones(
            (h1 + title_height, w1 + w2, 3), dtype=np.uint8
        ) * 255
        
        # 放置图片
        combined_with_title[title_height:, :] = combined
        
        # 添加标题文本
        cv2.putText(
            combined_with_title,
            f"YOLO Detection (Conf>{conf_threshold})",
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
        
        # 添加检测数量信息
        info_text = f"Detections: {len(boxes)}"
        if len(boxes) > 0:
            avg_conf = np.mean(confidences)
            info_text += f" | Avg Conf: {avg_conf:.2f}"
        
        cv2.putText(
            combined_with_title,
            info_text,
            (10, h1 + title_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )
        
        # 保存结果
        output_file = output_path / f"{img_name}_detection_mask.jpg"
        cv2.imwrite(str(output_file), combined_with_title)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"总图片数: {len(image_files)}")
    print(f"有检测结果的图片: {images_with_detections}")
    print(f"无检测结果的图片: {images_without_detections}")
    print(f"总检测数: {total_detections}")
    if images_with_detections > 0:
        print(f"平均每张图片检测数: {total_detections / images_with_detections:.2f}")
    print(f"\n可视化结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="可视化YOLO检测结果，显示检测框、置信度，以及对应的GT mask"
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="runs/detect/yolo11n_kvasir/weights/best.pt",
        help="YOLO模型权重路径"
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
        default="yolo_detection_visualizations",
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
    
    visualize_detection_with_mask(
        yolo_weights=args.yolo_weights,
        test_images_dir=args.test_images,
        test_masks_dir=args.test_masks,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_images=args.max_images,
    )

