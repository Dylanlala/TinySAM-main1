#!/usr/bin/env python3
"""
准备YOLO分割数据集
将Kvasir-SEG的mask按对象分割，生成YOLO分割所需的格式
"""

import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage
import shutil
from tqdm import tqdm


def split_mask_by_objects(mask_path, output_dir, image_name):
    """
    将包含多个对象的mask分割成多个单独的mask文件
    
    Args:
        mask_path: 原始mask文件路径
        output_dir: 输出目录
        image_name: 图像名称（不含扩展名）
    
    Returns:
        分割后的mask文件列表
    """
    # 读取mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    # 二值化（阈值128）
    mask_binary = (mask > 128).astype(np.uint8)
    
    # 连通域分析
    labeled, num_features = ndimage.label(mask_binary)
    
    if num_features == 0:
        return []
    
    # 为每个连通域创建单独的mask文件
    mask_files = []
    for i in range(1, num_features + 1):
        # 提取单个对象
        single_mask = (labeled == i).astype(np.uint8) * 255
        
        # 保存mask文件
        mask_filename = f"{image_name}_obj_{i-1}.png"
        mask_filepath = output_dir / mask_filename
        cv2.imwrite(str(mask_filepath), single_mask)
        mask_files.append(mask_filename)
    
    return mask_files


def prepare_yolo_seg_dataset(
    data_dir="Kvasir-SEG",
    output_dir="Kvasir-SEG_yolo_seg",
    splits=['train', 'val'],
):
    """
    准备YOLO分割数据集
    
    YOLO分割需要：
    - images/: 图像文件
    - labels/: 检测标注（.txt格式，class_id cx cy w h）
    - masks/: 分割mask（每个对象一个mask文件）
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录结构
    for split in splits:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
    
    # 处理每个split
    for split in splits:
        print(f"\n处理 {split} 集...")
        
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"
        masks_dir = data_dir / "masks"
        
        image_files = sorted(list(images_dir.glob("*.jpg")))
        
        for img_file in tqdm(image_files, desc=f"处理{split}"):
            image_name = img_file.stem
            
            # 1. 复制图像
            shutil.copy(img_file, output_dir / split / "images" / img_file.name)
            
            # 2. 复制label文件
            label_file = labels_dir / f"{image_name}.txt"
            if label_file.exists():
                shutil.copy(label_file, output_dir / split / "labels" / f"{image_name}.txt")
            
            # 3. 处理mask文件
            mask_file = masks_dir / f"{image_name}.jpg"
            if not mask_file.exists():
                mask_file = masks_dir / f"{image_name}.png"
            
            if mask_file.exists():
                # 分割mask
                mask_files = split_mask_by_objects(
                    mask_file,
                    output_dir / split / "masks",
                    image_name
                )
                
                # 如果mask分割失败，使用原始mask
                if len(mask_files) == 0:
                    # 直接复制原始mask
                    shutil.copy(mask_file, output_dir / split / "masks" / f"{image_name}.png")
    
    print(f"\n数据集准备完成！")
    print(f"输出目录: {output_dir}")
    print(f"\n目录结构:")
    print(f"  {output_dir}/")
    print(f"    train/")
    print(f"      images/")
    print(f"      labels/")
    print(f"      masks/")
    print(f"    val/")
    print(f"      images/")
    print(f"      labels/")
    print(f"      masks/")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="准备YOLO分割数据集")
    parser.add_argument("--data-dir", type=str, default="Kvasir-SEG",
                       help="原始数据集目录")
    parser.add_argument("--output-dir", type=str, default="Kvasir-SEG_yolo_seg",
                       help="输出目录")
    parser.add_argument("--splits", type=str, nargs="+", default=['train', 'val'],
                       help="数据集划分")
    
    args = parser.parse_args()
    
    prepare_yolo_seg_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()




