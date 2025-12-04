#!/usr/bin/env python3
"""
演示COCO格式到YOLO格式的转换过程
"""

import json
from pathlib import Path

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """将COCO格式转换为YOLO格式"""
    x_min, y_min, bbox_width, bbox_height = bbox
    
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

# 读取COCO JSON文件
coco_json = Path("all_coco/annotations/train_coco_format.json")
with open(coco_json, 'r') as f:
    coco_data = json.load(f)

# 获取第一个标注作为示例
ann = coco_data['annotations'][0]
img = next(i for i in coco_data['images'] if i['id'] == ann['image_id'])

print("=" * 70)
print("COCO格式到YOLO格式转换示例")
print("=" * 70)

print("\n【COCO格式 (原始数据)】")
print(f"图像文件名: {img['file_name']}")
print(f"图像尺寸: {img['width']} x {img['height']} 像素")
print(f"类别ID: {ann['category_id']}")
print(f"边界框(COCO格式): {ann['bbox']}")
print("  格式说明: [x_min, y_min, width, height] (绝对像素坐标)")

# 提取数据
x_min, y_min, w, h = ann['bbox']
img_w, img_h = img['width'], img['height']

print("\n【转换计算过程】")
print(f"1. COCO边界框: [{x_min}, {y_min}, {w}, {h}] (绝对像素)")

# 计算中心点
cx = (x_min + w/2) / img_w
cy = (y_min + h/2) / img_h
print(f"2. 计算中心点:")
print(f"   center_x = (x_min + width/2) / image_width")
print(f"            = ({x_min} + {w}/2) / {img_w}")
print(f"            = {x_min + w/2} / {img_w}")
print(f"            = {cx:.6f}")
print(f"   center_y = (y_min + height/2) / image_height")
print(f"            = ({y_min} + {h}/2) / {img_h}")
print(f"            = {y_min + h/2} / {img_h}")
print(f"            = {cy:.6f}")

# 归一化尺寸
nw = w / img_w
nh = h / img_h
print(f"3. 归一化尺寸:")
print(f"   normalized_width = width / image_width")
print(f"                    = {w} / {img_w}")
print(f"                    = {nw:.6f}")
print(f"   normalized_height = height / image_height")
print(f"                     = {h} / {img_h}")
print(f"                     = {nh:.6f}")

# 类别ID转换
class_id = ann['category_id'] - 1 if ann['category_id'] > 0 else 0
print(f"4. 类别ID转换:")
print(f"   COCO类别ID: {ann['category_id']} (从1开始)")
print(f"   YOLO类别ID: {class_id} (从0开始)")
print(f"   转换公式: YOLO_class_id = COCO_category_id - 1")

# 最终结果
print("\n【YOLO格式 (转换结果)】")
filename = Path(img['file_name']).stem
label_file = Path(f"all_coco_yolo/train/labels/{filename}.txt")

if label_file.exists():
    with open(label_file, 'r') as f:
        yolo_content = f.read().strip()
    print(f"标签文件: {label_file}")
    print(f"文件内容: {yolo_content}")
    print("\n格式说明: class_id center_x center_y width height (归一化坐标)")
    
    # 验证转换
    parts = yolo_content.split()
    if len(parts) == 5:
        yolo_class, yolo_cx, yolo_cy, yolo_w, yolo_h = map(float, parts)
        print(f"\n【验证】")
        print(f"转换结果: {class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        print(f"文件内容: {int(yolo_class)} {yolo_cx:.6f} {yolo_cy:.6f} {yolo_w:.6f} {yolo_h:.6f}")
        print(f"✓ 转换正确!" if abs(cx - yolo_cx) < 0.0001 else "✗ 转换有误")
else:
    print(f"标签文件不存在: {label_file}")
    print(f"预期内容: {class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

print("\n" + "=" * 70)
print("关键转换公式总结")
print("=" * 70)
print("""
COCO格式: [x_min, y_min, width, height] (绝对像素)
    ↓
YOLO格式: class_id center_x center_y width height (归一化0-1)

转换公式:
  center_x = (x_min + width/2) / image_width
  center_y = (y_min + height/2) / image_height
  normalized_width = width / image_width
  normalized_height = height / image_height
  class_id = COCO_category_id - 1
""")
print("=" * 70)



