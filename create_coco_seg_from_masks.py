#!/usr/bin/env python3
"""
从mask文件创建COCO格式的分割标注文件
用于YOLO分割训练
"""

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
import argparse


def mask_to_polygon(mask):
    """
    将mask转换为COCO格式的多边形坐标
    
    Args:
        mask: 二值mask (H, W)
    
    Returns:
        polygon: [[x1, y1, x2, y2, ...]] 格式的列表
    """
    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return []
    
    # 选择最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 转换为多边形坐标 [x1, y1, x2, y2, ...]
    polygon = largest_contour.reshape(-1, 2).flatten().tolist()
    
    # COCO格式要求至少3个点（6个坐标）
    if len(polygon) < 6:
        return []
    
    return [polygon]


def create_coco_seg_dataset(
    data_dir="Kvasir-SEG",
    output_file="Kvasir-SEG/kavsir-seg.json",
    splits=['train', 'val'],
):
    """
    从mask文件创建COCO格式的分割标注
    
    Args:
        data_dir: 数据集根目录
        output_file: 输出JSON文件路径
        splits: 数据集划分
    """
    data_dir = Path(data_dir)
    
    # COCO格式结构
    coco_data = {
        "info": {
            "description": "Kvasir-SEG Polyp Segmentation Dataset",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [
            {
                "id": 0,
                "name": "polyp",
                "supercategory": "polyp"
            }
        ],
        "images": [],
        "annotations": []
    }
    
    image_id = 0
    annotation_id = 0
    
    # 处理每个split
    for split in splits:
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"
        masks_dir = data_dir / "masks"
        
        image_files = sorted(list(images_dir.glob("*.jpg")))
        
        print(f"\n处理 {split} 集，共 {len(image_files)} 张图片...")
        
        for img_file in tqdm(image_files, desc=f"处理{split}"):
            image_name = img_file.stem
            
            # 读取图像获取尺寸
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": f"{split}/images/{img_file.name}",
                "width": w,
                "height": h,
            })
            
            # 读取mask
            mask_file = masks_dir / f"{image_name}.jpg"
            if not mask_file.exists():
                mask_file = masks_dir / f"{image_name}.png"
            
            if not mask_file.exists():
                image_id += 1
                continue
            
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                image_id += 1
                continue
            
            # 调整mask尺寸（如果需要）
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 二值化
            mask_binary = (mask > 128).astype(np.uint8)
            
            # 连通域分析，为每个对象创建标注
            labeled, num_features = ndimage.label(mask_binary)
            
            # 读取YOLO标注获取bbox（如果有）
            label_file = labels_dir / f"{image_name}.txt"
            bboxes = []
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, cx, cy, w_norm, h_norm = map(float, parts)
                            # 转换为绝对坐标
                            x = (cx - w_norm / 2) * w
                            y = (cy - h_norm / 2) * h
                            bbox_w = w_norm * w
                            bbox_h = h_norm * h
                            bboxes.append([x, y, bbox_w, bbox_h])
            
            # 为每个连通域创建标注
            for obj_id in range(1, num_features + 1):
                # 提取单个对象的mask
                obj_mask = (labeled == obj_id).astype(np.uint8) * 255
                
                # 转换为多边形
                polygon = mask_to_polygon(obj_mask)
                
                if len(polygon) == 0:
                    continue
                
                # 计算bbox
                coords = np.array(polygon[0]).reshape(-1, 2)
                x_min = float(coords[:, 0].min())
                y_min = float(coords[:, 1].min())
                x_max = float(coords[:, 0].max())
                y_max = float(coords[:, 1].max())
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = float(cv2.contourArea(coords.reshape(-1, 1, 2)))
                
                # 添加标注
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "segmentation": polygon,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                })
                
                annotation_id += 1
            
            image_id += 1
    
    # 保存JSON文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✅ COCO格式标注文件已创建: {output_path}")
    print(f"   图像数量: {len(coco_data['images'])}")
    print(f"   标注数量: {len(coco_data['annotations'])}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="从mask文件创建COCO格式分割标注")
    parser.add_argument("--data-dir", type=str, default="Kvasir-SEG",
                       help="数据集根目录")
    parser.add_argument("--output-file", type=str, default="Kvasir-SEG/kavsir-seg.json",
                       help="输出JSON文件路径")
    parser.add_argument("--splits", type=str, nargs="+", default=['train', 'val'],
                       help="数据集划分")
    
    args = parser.parse_args()
    
    create_coco_seg_dataset(
        data_dir=args.data_dir,
        output_file=args.output_file,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()




