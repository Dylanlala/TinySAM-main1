#!/usr/bin/env python3
"""
Kvasir-SEG数据集，使用YOLO检测的bbox作为TinySAM的prompt
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import sys

# 添加ultralytics路径
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRA = REPO_ROOT / "ultralyticss_new"
if LOCAL_ULTRA.exists() and str(LOCAL_ULTRA) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRA))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class KvasirYOLOTinySAMDataset(Dataset):
    """
    Kvasir-SEG数据集，使用YOLO检测的bbox作为prompt训练TinySAM
    """
    
    def __init__(
        self,
        images_dir,
        masks_dir,
        labels_dir=None,
        yolo_weights=None,
        img_size=1024,
        split='train',
        use_yolo_detection=True,
        conf_threshold=0.25,
        cache_detections=True,
    ):
        """
        Args:
            images_dir: 图片目录
            masks_dir: mask标注目录
            labels_dir: YOLO标签目录（可选，如果提供则使用GT bbox）
            yolo_weights: YOLO模型权重（如果use_yolo_detection=True）
            img_size: 图片尺寸
            split: 数据集划分（train/val/test）
            use_yolo_detection: 是否使用YOLO检测结果（True）或GT标签（False）
            conf_threshold: YOLO检测置信度阈值
            cache_detections: 是否缓存YOLO检测结果
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.img_size = img_size
        self.split = split
        self.use_yolo_detection = use_yolo_detection
        self.conf_threshold = conf_threshold
        
        # 加载YOLO模型（如果需要）
        self.yolo_model = None
        if use_yolo_detection and yolo_weights and YOLO:
            print(f"加载YOLO模型: {yolo_weights}")
            self.yolo_model = YOLO(yolo_weights)
        
        # 获取所有图片文件
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # 缓存检测结果
        self.detection_cache = {}
        cache_file = self.images_dir.parent / f"yolo_detections_{split}.json"
        
        if cache_detections and cache_file.exists():
            print(f"加载缓存的检测结果: {cache_file}")
            with open(cache_file, 'r') as f:
                self.detection_cache = json.load(f)
        elif use_yolo_detection and self.yolo_model:
            print("生成YOLO检测结果...")
            self._generate_detections()
            if cache_detections:
                with open(cache_file, 'w') as f:
                    json.dump(self.detection_cache, f, indent=2)
                print(f"检测结果已缓存到: {cache_file}")
    
    def _generate_detections(self):
        """使用YOLO模型生成检测结果"""
        for img_file in self.image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            results = self.yolo_model(img, conf=self.conf_threshold, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })
            
            self.detection_cache[img_file.name] = detections
    
    def _parse_yolo_label(self, txt_path, img_width, img_height):
        """解析YOLO格式标签，返回像素坐标的bbox列表"""
        boxes = []
        if not txt_path.exists():
            return boxes
        
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    # YOLO格式: class_id cx cy w h (归一化)
                    cx = float(parts[1]) * img_width
                    cy = float(parts[2]) * img_height
                    w = float(parts[3]) * img_width
                    h = float(parts[4]) * img_height
                    
                    # 转换为 xmin, ymin, xmax, ymax
                    xmin = max(0, int(cx - w / 2))
                    ymin = max(0, int(cy - h / 2))
                    xmax = min(img_width - 1, int(cx + w / 2))
                    ymax = min(img_height - 1, int(cy + h / 2))
                    
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
        
        return boxes
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_name = img_file.stem
        
        # 读取图片
        img = cv2.imread(str(img_file))
        if img is None:
            raise ValueError(f"Cannot read image: {img_file}")
        
        original_h, original_w = img.shape[:2]
        
        # 调整图片尺寸
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        
        # 读取mask
        mask_file = self.masks_dir / f"{img_name}.jpg"
        if not mask_file.exists():
            mask_file = self.masks_dir / f"{img_name}.png"
        
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask > 128).astype(np.float32)
        else:
            mask_binary = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # (1, H, W)
        
        # 获取bbox（prompt）
        boxes = []
        if self.use_yolo_detection:
            # 使用YOLO检测结果
            detections = self.detection_cache.get(img_file.name, [])
            for det in detections:
                if det['confidence'] >= self.conf_threshold:
                    # 将bbox缩放到resize后的尺寸
                    x1, y1, x2, y2 = det['bbox']
                    x1 = int(x1 * self.img_size / original_w)
                    y1 = int(y1 * self.img_size / original_h)
                    x2 = int(x2 * self.img_size / original_w)
                    y2 = int(y2 * self.img_size / original_h)
                    boxes.append([x1, y1, x2, y2])
        else:
            # 使用GT标签
            if self.labels_dir:
                label_file = self.labels_dir / f"{img_name}.txt"
                gt_boxes = self._parse_yolo_label(label_file, original_w, original_h)
                for x1, y1, x2, y2 in gt_boxes:
                    # 缩放到resize后的尺寸
                    x1 = int(x1 * self.img_size / original_w)
                    y1 = int(y1 * self.img_size / original_h)
                    x2 = int(x2 * self.img_size / original_w)
                    y2 = int(y2 * self.img_size / original_h)
                    boxes.append([x1, y1, x2, y2])
        
        # 如果没有检测到bbox，使用整个图片作为box（fallback）
        if len(boxes) == 0:
            boxes = [[0, 0, self.img_size, self.img_size]]
        
        # 转换为numpy数组
        boxes_array = np.array(boxes, dtype=np.float32)
        
        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'boxes': boxes_array,
            'image_name': img_name,
            'original_size': (original_h, original_w),
        }
    
    @staticmethod
    def collate_fn(batch):
        """自定义collate函数"""
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        boxes = [item['boxes'] for item in batch]
        image_names = [item['image_name'] for item in batch]
        original_sizes = [item['original_size'] for item in batch]
        
        return {
            'images': images,
            'masks': masks,
            'boxes': boxes,
            'image_names': image_names,
            'original_sizes': original_sizes,
        }


