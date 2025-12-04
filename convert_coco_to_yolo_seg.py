#!/usr/bin/env python3
"""
将COCO格式的JSON转换为YOLO分割格式
YOLO分割需要：图片目录 + labels目录（包含对应的txt文件，格式为：class_id x1 y1 x2 y2 x3 y3 ...）
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_coco_to_yolo_seg(coco_json, images_dir, labels_output_dir):
    """
    将COCO格式转换为YOLO分割格式
    
    Args:
        coco_json: COCO格式的JSON文件路径
        images_dir: 图片目录
        labels_output_dir: 输出标签目录
    """
    print(f"读取COCO JSON: {coco_json}")
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # 创建输出目录
    labels_output_path = Path(labels_output_dir)
    labels_output_path.mkdir(parents=True, exist_ok=True)
    
    # 建立image_id到文件名的映射
    image_id_to_filename = {}
    for img_info in coco_data['images']:
        image_id_to_filename[img_info['id']] = img_info['file_name']
    
    # 按image_id分组annotations
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    print(f"处理 {len(image_annotations)} 张图片...")
    
    images_path = Path(images_dir)
    converted = 0
    
    for image_id, annotations in tqdm(image_annotations.items(), desc="转换中"):
        if image_id not in image_id_to_filename:
            continue
        
        filename = image_id_to_filename[image_id]
        image_path = images_path / filename
        
        if not image_path.exists():
            # 尝试不同的扩展名
            for ext in ['.jpg', '.png', '.jpeg']:
                alt_path = images_path / f"{Path(filename).stem}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break
            else:
                continue
        
        # 读取图片获取尺寸
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # 创建YOLO格式的标签文件
        label_file = labels_output_path / f"{Path(filename).stem}.txt"
        
        with open(label_file, 'w') as f:
            for ann in annotations:
                # COCO类别ID，如果已经是0开始则直接使用，否则减1
                category_id = ann['category_id']
                class_id = category_id if category_id == 0 else category_id - 1
                class_id = max(0, class_id)  # 确保类别ID非负
                
                segmentation = ann['segmentation']
                
                # COCO格式的segmentation是list of lists（多边形点）
                if isinstance(segmentation, list) and len(segmentation) > 0:
                    # 取第一个多边形（如果有多个，取最大的）
                    if len(segmentation) > 0 and isinstance(segmentation[0], list):
                        # 多个多边形，选择最大的
                        max_poly = max(segmentation, key=len)
                        polygon = max_poly
                    elif len(segmentation) > 0 and isinstance(segmentation[0], (int, float)):
                        # 单个多边形，已经是扁平化的列表
                        polygon = segmentation
                    else:
                        continue
                    
                    # 转换为归一化坐标 [x1, y1, x2, y2, ...]
                    if len(polygon) >= 6 and len(polygon) % 2 == 0:  # 至少3个点（6个坐标）
                        # 归一化坐标
                        normalized_poly = []
                        for i in range(0, len(polygon), 2):
                            x = max(0.0, min(1.0, polygon[i] / w))  # 限制在[0,1]
                            y = max(0.0, min(1.0, polygon[i + 1] / h))
                            normalized_poly.extend([x, y])
                        
                        # 写入YOLO格式：class_id x1 y1 x2 y2 ...
                        if len(normalized_poly) >= 6:  # 至少3个点
                            line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_poly])
                            f.write(line + "\n")
        
        converted += 1
    
    print(f"\n转换完成！共转换 {converted} 张图片的标签")
    return converted


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将COCO格式转换为YOLO分割格式")
    parser.add_argument("--coco-json", type=str, required=True, help="COCO格式的JSON文件")
    parser.add_argument("--images-dir", type=str, required=True, help="图片目录")
    parser.add_argument("--output-dir", type=str, required=True, help="输出标签目录")
    
    args = parser.parse_args()
    
    convert_coco_to_yolo_seg(
        coco_json=args.coco_json,
        images_dir=args.images_dir,
        labels_output_dir=args.output_dir
    )
