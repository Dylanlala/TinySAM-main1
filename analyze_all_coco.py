#!/usr/bin/env python3
"""
分析all_coco目录的结构和内容
"""
import json
import os
from pathlib import Path
from collections import Counter, defaultdict

#我尝试一下

def analyze_all_coco(base_dir="all_coco"):
    base_path = Path(base_dir)
    
    print("=" * 80)
    print("all_coco 目录结构分析")
    print("=" * 80)
    
    # 1. 目录结构
    print("\n1. 目录结构:")
    for split in ['train', 'val', 'test']:
        split_dir = base_path / split
        if split_dir.exists():
            file_count = len(list(split_dir.glob("*")))
            total_size = sum(f.stat().st_size for f in split_dir.glob("*") if f.is_file())
            print(f"  {split}/: {file_count} 文件, {total_size / (1024**3):.2f} GB")
    
    # 2. 加载JSON文件
    print("\n2. COCO格式标注文件:")
    splits_data = {}
    for split in ['train', 'val', 'test']:
        json_file = base_path / "annotations" / f"{split}_coco_format.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            splits_data[split] = data
            print(f"  {split}_coco_format.json:")
            print(f"    - 图像数量: {len(data['images'])}")
            print(f"    - 标注数量: {len(data['annotations'])}")
            print(f"    - 类别: {data['categories']}")
            print(f"    - 文件大小: {json_file.stat().st_size / (1024**2):.2f} MB")
    
    # 3. 数据集组成分析
    print("\n3. 数据集组成分析 (基于文件名):")
    for split, data in splits_data.items():
        file_names = [img['file_name'] for img in data['images']]
        datasets = defaultdict(int)
        
        for fname in file_names:
            fname_lower = fname.lower()
            if 'kvasir' in fname_lower:
                key = 'Kvasir-SEG'
            elif 'cvc' in fname_lower:
                key = 'CVC-ClinicDB/ColonDB'
            elif 'case_m' in fname_lower:
                key = 'SUN-SEG'
            elif 'polygen' in fname_lower or 'polypgen' in fname_lower:
                key = 'PolypGen'
            elif 'image_' in fname_lower:
                key = 'Other (numbered)'
            else:
                key = 'Unknown'
            datasets[key] += 1
        
        print(f"\n  {split.upper()} 集:")
        for key, count in sorted(datasets.items(), key=lambda x: -x[1]):
            pct = count / len(file_names) * 100
            print(f"    {key}: {count} 张 ({pct:.1f}%)")
    
    # 4. 标注统计
    print("\n4. 标注统计:")
    for split, data in splits_data.items():
        images_with_anns = defaultdict(int)
        for ann in data['annotations']:
            images_with_anns[ann['image_id']] += 1
        
        multi_polyp = sum(1 for v in images_with_anns.values() if v > 1)
        no_ann = len(data['images']) - len(images_with_anns)
        
        print(f"\n  {split.upper()} 集:")
        print(f"    - 有标注的图像: {len(images_with_anns)}")
        print(f"    - 无标注的图像: {no_ann}")
        print(f"    - 多息肉图像: {multi_polyp} ({multi_polyp/len(images_with_anns)*100:.1f}%)" if images_with_anns else "")
        print(f"    - 单图像最大息肉数: {max(images_with_anns.values()) if images_with_anns else 0}")
    
    # 5. 图像尺寸分布
    print("\n5. 图像尺寸分布 (top 10):")
    for split, data in splits_data.items():
        sizes = Counter()
        for img in data['images']:
            size = f"{img['width']}x{img['height']}"
            sizes[size] += 1
        
        print(f"\n  {split.upper()} 集:")
        for size, count in sizes.most_common(10):
            pct = count / len(data['images']) * 100
            print(f"    {size}: {count} 张 ({pct:.1f}%)")
    
    # 6. 分割标注分析
    print("\n6. 分割标注分析:")
    for split, data in splits_data.items():
        anns_with_seg = [a for a in data['annotations'] if 'segmentation' in a and a['segmentation']]
        anns_with_bbox = [a for a in data['annotations'] if 'bbox' in a]
        
        print(f"\n  {split.upper()} 集:")
        print(f"    - 有分割标注: {len(anns_with_seg)}")
        print(f"    - 有边界框标注: {len(anns_with_bbox)}")
        
        if anns_with_seg:
            sample = anns_with_seg[0]
            seg = sample['segmentation']
            print(f"    - 分割格式: {type(seg).__name__}")
            if isinstance(seg, list) and len(seg) > 0:
                if isinstance(seg[0], list):
                    print(f"    - 分割点数量: {len(seg[0])} (第一个多边形)")
                    print(f"    - 多边形数量: {len(seg)} (第一个标注)")
                else:
                    print(f"    - 分割格式: RLE或其他")
    
    # 7. 边界框统计
    print("\n7. 边界框统计:")
    for split, data in splits_data.items():
        bbox_areas = []
        bbox_sizes = []
        for ann in data['annotations']:
            if 'bbox' in ann and 'area' in ann:
                bbox_areas.append(ann['area'])
                bbox = ann['bbox']
                if len(bbox) >= 4:
                    w, h = bbox[2], bbox[3]
                    bbox_sizes.append((w, h))
        
        if bbox_areas:
            print(f"\n  {split.upper()} 集:")
            print(f"    - 平均面积: {sum(bbox_areas)/len(bbox_areas):.1f} 像素²")
            print(f"    - 最小面积: {min(bbox_areas):.1f} 像素²")
            print(f"    - 最大面积: {max(bbox_areas):.1f} 像素²")
            if bbox_sizes:
                avg_w = sum(w for w, h in bbox_sizes) / len(bbox_sizes)
                avg_h = sum(h for w, h in bbox_sizes) / len(bbox_sizes)
                print(f"    - 平均边界框尺寸: {avg_w:.1f}x{avg_h:.1f}")
    
    # 8. 文件一致性检查
    print("\n8. 文件一致性检查:")
    for split, data in splits_data.items():
        split_dir = base_path / split
        if split_dir.exists():
            json_files = set(img['file_name'] for img in data['images'])
            dir_files = set(f.name for f in split_dir.glob("*") if f.is_file())
            
            match = len(json_files & dir_files)
            missing = len(json_files - dir_files)
            extra = len(dir_files - json_files)
            
            print(f"\n  {split.upper()} 集:")
            print(f"    - JSON中列出的文件: {len(json_files)}")
            print(f"    - 目录中的文件: {len(dir_files)}")
            print(f"    - 匹配: {match}")
            print(f"    - JSON中缺失: {missing}")
            print(f"    - 目录中多余: {extra}")
    
    # 9. 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    total_images = sum(len(data['images']) for data in splits_data.values())
    total_anns = sum(len(data['annotations']) for data in splits_data.values())
    print(f"总图像数: {total_images}")
    print(f"总标注数: {total_anns}")
    print(f"平均每张图像标注数: {total_anns/total_images:.2f}")
    print(f"数据集类别: {splits_data['train']['categories']}")
    print("=" * 80)

if __name__ == "__main__":
    analyze_all_coco()



