#!/usr/bin/env python3
"""
将all_coco数据集的COCO格式转换为YOLO检测格式
COCO格式: JSON文件包含images, annotations, categories
YOLO格式: 每个图像对应一个txt文件，格式为: class_id center_x center_y width height (归一化)
"""

import json
from pathlib import Path
from tqdm import tqdm


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    将COCO格式的边界框转换为YOLO格式
    
    COCO格式: [x_min, y_min, width, height] (绝对像素坐标)
    YOLO格式: class_id center_x center_y width height (归一化到0-1)
    
    Args:
        bbox: COCO格式的边界框 [x_min, y_min, width, height]
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        (center_x, center_y, width, height) 归一化坐标
    """
    x_min, y_min, bbox_width, bbox_height = bbox
    
    # 计算中心点和尺寸（归一化）
    center_x = (x_min + bbox_width / 2) / img_width
    center_y = (y_min + bbox_height / 2) / img_height
    width = bbox_width / img_width
    height = bbox_height / img_height
    
    # 确保坐标在[0, 1]范围内
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height


def convert_coco_to_yolo_detect(coco_json, images_dir, labels_output_dir):
    """
    将COCO格式转换为YOLO检测格式
    
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
    
    # 建立image_id到图像信息的映射
    image_info_map = {}
    for img_info in coco_data['images']:
        image_info_map[img_info['id']] = img_info
    
    # 按image_id分组annotations
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    print(f"处理 {len(image_info_map)} 张图片，{len(image_annotations)} 张有标注...")
    
    images_path = Path(images_dir)
    converted = 0
    skipped = 0
    
    # 处理每张图像
    for image_id, img_info in tqdm(image_annotations.items(), desc="转换中"):
        if image_id not in image_info_map:
            skipped += 1
            continue
        
        img_data = image_info_map[image_id]
        filename = img_data['file_name']
        img_width = img_data['width']
        img_height = img_data['height']
        
        # 创建YOLO格式的标签文件
        label_file = labels_output_path / f"{Path(filename).stem}.txt"
        
        with open(label_file, 'w') as f:
            for ann in image_annotations[image_id]:
                # COCO类别ID，如果已经是0开始则直接使用，否则减1
                category_id = ann['category_id']
                class_id = category_id if category_id == 0 else category_id - 1
                class_id = max(0, class_id)  # 确保类别ID非负
                
                # 获取边界框
                if 'bbox' not in ann:
                    continue
                
                bbox = ann['bbox']
                if len(bbox) != 4:
                    continue
                
                # 转换为YOLO格式
                center_x, center_y, width, height = convert_bbox_to_yolo(
                    bbox, img_width, img_height
                )
                
                # 写入YOLO格式: class_id center_x center_y width height
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        converted += 1
    
    # 处理没有标注的图像（创建空文件）
    for image_id, img_info in tqdm(image_info_map.items(), desc="处理无标注图像"):
        if image_id not in image_annotations:
            filename = img_info['file_name']
            label_file = labels_output_path / f"{Path(filename).stem}.txt"
            # 创建空文件（表示没有标注）
            label_file.touch()
    
    print(f"\n转换完成!")
    print(f"  已转换: {converted} 张有标注图像")
    print(f"  无标注图像: {len(image_info_map) - len(image_annotations)} 张")
    print(f"  跳过: {skipped} 张")
    print(f"  标签文件保存在: {labels_output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="将all_coco的COCO格式转换为YOLO检测格式")
    parser.add_argument("--coco-dir", type=str, default="all_coco",
                       help="all_coco目录路径")
    parser.add_argument("--output-dir", type=str, default="all_coco_yolo",
                       help="输出YOLO格式目录")
    
    args = parser.parse_args()
    
    base_path = Path(args.coco_dir)
    output_path = Path(args.output_dir)
    
    # 转换train/val/test
    for split in ['train', 'val', 'test']:
        coco_json = base_path / "annotations" / f"{split}_coco_format.json"
        images_dir = base_path / split
        labels_output_dir = output_path / split / "labels"
        
        if not coco_json.exists():
            print(f"警告: {coco_json} 不存在，跳过")
            continue
        
        if not images_dir.exists():
            print(f"警告: {images_dir} 不存在，跳过")
            continue
        
        print(f"\n{'='*70}")
        print(f"转换 {split.upper()} 集")
        print(f"{'='*70}")
        
        convert_coco_to_yolo_detect(
            coco_json=coco_json,
            images_dir=images_dir,
            labels_output_dir=labels_output_dir
        )
        
        # 复制图像到输出目录
        images_output_dir = output_path / split / "images"
        images_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"复制图像到: {images_output_dir}")
        import shutil
        for img_file in tqdm(list(images_dir.glob("*")), desc="复制图像"):
            if img_file.is_file():
                shutil.copy2(img_file, images_output_dir / img_file.name)
    
    print(f"\n{'='*70}")
    print("所有转换完成!")
    print(f"YOLO格式数据集保存在: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()



