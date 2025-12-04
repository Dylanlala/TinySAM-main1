#!/usr/bin/env python3
"""
将Kvasir-SEG的bboxes.json转换为YOLO格式的txt标注文件
"""

import json
import os
from pathlib import Path
from PIL import Image

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """将像素坐标转换为YOLO归一化坐标 (cx, cy, w, h)"""
    # 计算中心点和宽高
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    # 归一化
    cx_norm = cx / img_width
    cy_norm = cy / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return cx_norm, cy_norm, w_norm, h_norm

def convert_kvasir_to_yolo(bboxes_json_path, images_dir, output_labels_dir):
    """
    转换Kvasir-SEG数据集为YOLO格式
    
    Args:
        bboxes_json_path: bboxes.json文件路径
        images_dir: 图片目录
        output_labels_dir: 输出标签目录
    """
    # 读取bboxes.json
    with open(bboxes_json_path, 'r') as f:
        bboxes_data = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # 类别ID（息肉类别为0）
    class_id = 0
    
    # 统计信息
    total_images = 0
    total_bboxes = 0
    skipped_images = 0
    
    # 遍历所有图片
    images_path = Path(images_dir)
    for img_file in images_path.glob("*.jpg"):
        img_name = img_file.stem
        
        # 检查是否有对应的bbox标注
        if img_name not in bboxes_data:
            print(f"Warning: No bbox found for {img_name}")
            skipped_images += 1
            continue
        
        # 读取图片尺寸
        try:
            img = Image.open(img_file)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading image {img_file}: {e}")
            skipped_images += 1
            continue
        
        # 获取bbox信息
        bbox_info = bboxes_data[img_name]
        img_width_from_json = bbox_info.get('width', img_width)
        img_height_from_json = bbox_info.get('height', img_height)
        
        # 使用JSON中的尺寸（如果可用），否则使用实际图片尺寸
        if img_width_from_json and img_height_from_json:
            img_width = img_width_from_json
            img_height = img_height_from_json
        
        # 获取bbox列表
        bboxes = bbox_info.get('bbox', [])
        
        if not bboxes:
            # 如果没有bbox，创建空文件（负样本）
            output_txt = output_labels_dir / f"{img_name}.txt"
            output_txt.write_text("")
            total_images += 1
            continue
        
        # 创建YOLO格式的txt文件
        output_txt = output_labels_dir / f"{img_name}.txt"
        with open(output_txt, 'w') as f:
            for bbox in bboxes:
                xmin = bbox['xmin']
                ymin = bbox['ymin']
                xmax = bbox['xmax']
                ymax = bbox['ymax']
                
                # 转换为YOLO格式
                cx, cy, w, h = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height)
                
                # 写入文件：class_id cx cy w h
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                total_bboxes += 1
        
        total_images += 1
    
    print(f"\n转换完成！")
    print(f"总图片数: {total_images}")
    print(f"总bbox数: {total_bboxes}")
    print(f"跳过图片数: {skipped_images}")
    print(f"标签文件保存在: {output_labels_dir}")

if __name__ == "__main__":
    # 设置路径
    base_dir = Path("/home/huangmanling/huangmanling/yolo_sam/TinySAM-main/Kvasir-SEG")
    bboxes_json = base_dir / "kavsir_bboxes.json"
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    
    # 检查路径是否存在
    if not bboxes_json.exists():
        print(f"Error: {bboxes_json} not found!")
        exit(1)
    
    if not images_dir.exists():
        print(f"Error: {images_dir} not found!")
        exit(1)
    
    # 执行转换
    convert_kvasir_to_yolo(bboxes_json, images_dir, labels_dir)


