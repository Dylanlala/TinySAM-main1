import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from typing import Tuple, Optional, List  # 添加必要的类型导入

def get_transform(split: str, img_size: int = 1024) -> A.Compose:
    """数据增强变换"""
    transforms = [
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=img_size, 
            min_width=img_size, 
            border_mode=cv2.BORDER_CONSTANT
        ),
    ]
    
    if split == "train":
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
        ])
    
    transforms.append(ToTensorV2())
    return A.Compose(transforms, additional_targets={'mask': 'mask'})

class KvasirDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 1024):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.transform = get_transform(split, img_size)
        
        # 加载COCO注解
        ann_path = os.path.join(root, "annotations.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        
        self.coco = COCO(ann_path)
        self.image_ids = list(self.coco.imgs.keys())
        
        # 检查图像文件是否存在
        self.valid_ids = []
        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(root, "images", img_info["file_name"])
            if os.path.exists(img_path):
                self.valid_ids.append(img_id)
        
        # 划分数据集 (80%训练，20%验证)
        random.seed(42)
        random.shuffle(self.valid_ids)
        split_idx = int(0.8 * len(self.valid_ids))
        
        if split == "train":
            self.image_ids = self.valid_ids[:split_idx]
        else:
            self.image_ids = self.valid_ids[split_idx:]
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # 修改这里
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, "images", img_info["file_name"])
        
        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载掩码
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        
        for ann in anns:
            try:
                # 跳过无效标注
                if not ann.get("segmentation"):
                    continue
                
                # 处理不同标注格式
                if ann.get("iscrowd", 0) == 1:
                    if isinstance(ann["segmentation"], dict):  # RLE格式
                        mask = np.maximum(mask, self.coco.annToMask(ann) * 255)
                else:
                    # 多边形格式
                    if isinstance(ann["segmentation"], list):
                        mask = np.maximum(mask, self.coco.annToMask(ann) * 255)
            except Exception as e:
                print(f"Warning: Failed to process annotation {ann['id']} in image {img_id}: {e}")
                continue
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # 归一化并调整掩码维度
        mask = mask.float() / 255.0
        mask = mask.unsqueeze(0)  # 添加通道维度 [1, H, W]
        
        image = image.float()  # 保证image为float32
        mask = mask.float()    # 保证mask为float32
        return image, mask

    @staticmethod
    def collate_fn(batch: List) -> Tuple:  # 这里也修改了类型注解
        """自定义collate函数处理可能的None值"""
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)

class CVCClinicDBDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 512):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.transform = get_transform(split, img_size)

        # 图像和掩码路径
        self.img_dir = os.path.join(root, "PNG", "Original")
        self.mask_dir = os.path.join(root, "PNG", "Ground Truth")
        self.img_names = sorted(os.listdir(self.img_dir))
        self.mask_names = sorted(os.listdir(self.mask_dir))

        # 只保留两边都存在的文件
        self.img_names = [f for f in self.img_names if f in self.mask_names]
        self.mask_names = [f for f in self.img_names]  # 保证顺序一致

        # 划分数据集 (80%训练，20%验证)
        random.seed(42)
        random.shuffle(self.img_names)
        split_idx = int(0.8 * len(self.img_names))
        if split == "train":
            self.img_names = self.img_names[:split_idx]
        else:
            self.img_names = self.img_names[split_idx:]
        self.mask_names = self.img_names  # 保证顺序一致

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        mask = mask.float() / 255.0
        mask = mask.unsqueeze(0)
        image = image.float()  # 保证image为float32
        mask = mask.float()    # 保证mask为float32
        return image, mask

    @staticmethod
    def collate_fn(batch: list):
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)
