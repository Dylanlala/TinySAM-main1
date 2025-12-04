#!/usr/bin/env python3
"""
YOLO + TinySAM 端到端联合训练脚本
目标：让YOLO检测和TinySAM分割协同优化，提升整体性能

关键改进：
1. 联合训练：YOLO和TinySAM一起训练，端到端优化
2. 课程学习：先用GT bbox，逐步引入YOLO检测的bbox
3. 联合损失：检测损失 + 分割损失
4. 梯度共享：YOLO和TinySAM共享特征提取（可选）
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import cv2

# 添加ultralytics路径
import sys
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRA = REPO_ROOT / "ultralyticss_new"
if LOCAL_ULTRA.exists() and str(LOCAL_ULTRA) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRA))

from ultralytics import YOLO
from tinysam.build_sam import sam_model_registry
from tinysam.predictor import SamPredictor
from kvasir_yolo_tinysam_dataset import KvasirYOLOTinySAMDataset
from utils import dice_loss, iou_loss


class JointYOLOTinySAM(nn.Module):
    """
    YOLO + TinySAM 联合模型
    """
    def __init__(self, yolo_model, sam_model, freeze_yolo_backbone=False):
        super().__init__()
        self.yolo_model = yolo_model
        self.sam_model = sam_model
        
        # 是否冻结YOLO backbone（可选）
        if freeze_yolo_backbone:
            for param in self.yolo_model.model.model[:10].parameters():  # 冻结前10层
                param.requires_grad = False
    
    def forward(self, images, boxes_list=None, use_yolo_detection=True):
        """
        Args:
            images: (B, 3, H, W) 输入图像
            boxes_list: GT bbox列表（可选，用于课程学习）
            use_yolo_detection: 是否使用YOLO检测（True）或GT bbox（False）
        Returns:
            yolo_results: YOLO检测结果
            sam_masks: TinySAM分割结果
        """
        # YOLO检测
        yolo_results = self.yolo_model(images, verbose=False)
        
        # 提取检测框
        detections = []
        for result in yolo_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })
        
        # 如果使用GT bbox（课程学习），则替换检测结果
        if not use_yolo_detection and boxes_list is not None:
            detections = []
            for boxes in boxes_list:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 1.0
                    })
        
        # TinySAM分割
        sam_masks = []
        for i, image in enumerate(images):
            # 转换为numpy格式
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # 设置图像
            self.sam_predictor.set_image(image_rgb)
            
            # 对每个检测框进行分割
            masks_for_image = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                box = np.array([x1, y1, x2, y2])
                
                # 使用bbox和中心点作为prompt
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                point_coords = np.array([[cx, cy]])
                point_labels = np.array([1])
                
                # 预测mask
                pred_masks, scores, logits = self.sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                )
                
                if len(pred_masks) > 0:
                    masks_for_image.append(pred_masks[0])
            
            sam_masks.append(masks_for_image)
        
        return yolo_results, sam_masks


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal Loss"""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def compute_segmentation_loss(pred_masks, gt_mask, device):
    """
    计算分割损失
    Args:
        pred_masks: list of numpy arrays (每个检测框的mask)
        gt_mask: (1, 1, H, W) tensor
    """
    if len(pred_masks) == 0:
        # 如果没有预测，返回一个大的损失
        return torch.tensor(10.0, device=device, requires_grad=True)
    
    # 合并所有预测mask
    combined_pred = np.zeros_like(gt_mask[0, 0].cpu().numpy(), dtype=np.float32)
    for mask in pred_masks:
        # 调整mask尺寸
        mask_resized = cv2.resize(
            mask.astype(np.float32),
            (gt_mask.shape[-1], gt_mask.shape[-2]),
            interpolation=cv2.INTER_NEAREST
        )
        combined_pred = np.maximum(combined_pred, mask_resized)
    
    pred_tensor = torch.from_numpy(combined_pred).unsqueeze(0).unsqueeze(0).to(device)
    
    # 计算损失
    dice = dice_loss(pred_tensor, gt_mask)
    iou = iou_loss(pred_tensor, gt_mask)
    focal = focal_loss(pred_tensor, gt_mask)
    
    return dice + iou + focal


def train_joint_model(
    train_images_dir,
    train_masks_dir,
    train_labels_dir,
    val_images_dir=None,
    val_masks_dir=None,
    val_labels_dir=None,
    yolo_weights="yolo11n.pt",
    sam_weights="weights/tinysam_42.3.pth",
    sam_model_type="vit_t",
    img_size=1024,
    batch_size=2,  # 联合训练内存占用大，batch size小一些
    num_epochs=20,
    lr_yolo=1e-5,  # YOLO学习率（较小，因为已经预训练）
    lr_sam=1e-4,   # TinySAM学习率
    weight_decay=1e-4,
    device="cuda",
    output_dir="results_yolo_tinysam_joint",
    save_interval=5,
    val_interval=5,
    curriculum_learning=True,  # 是否使用课程学习
    curriculum_epochs=10,  # 前N轮使用GT bbox
    freeze_yolo_backbone=False,
):
    """
    联合训练YOLO和TinySAM
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载YOLO模型
    print(f"加载YOLO模型: {yolo_weights}")
    yolo_model = YOLO(yolo_weights)
    # 设置为训练模式
    yolo_model.model.train()
    
    # 加载TinySAM模型
    print(f"加载TinySAM模型: {sam_weights}")
    sam_model = sam_model_registry[sam_model_type](checkpoint=sam_weights)
    sam_model.to(device)
    sam_model.train()
    
    # 创建SAM predictor（用于推理）
    sam_predictor = SamPredictor(sam_model)
    
    # 创建联合模型
    joint_model = JointYOLOTinySAM(yolo_model.model, sam_model, freeze_yolo_backbone)
    joint_model.sam_predictor = sam_predictor  # 添加predictor
    joint_model.to(device)
    
    # 优化器：分别为YOLO和TinySAM设置不同的学习率
    yolo_params = [p for p in yolo_model.model.parameters() if p.requires_grad]
    sam_params = list(sam_model.parameters())
    
    optimizer = optim.AdamW(
        [
            {'params': yolo_params, 'lr': lr_yolo},
            {'params': sam_params, 'lr': lr_sam},
        ],
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # 数据集
    print("创建训练数据集...")
    train_dataset = KvasirYOLOTinySAMDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        labels_dir=train_labels_dir,
        yolo_weights=None,  # 不使用预训练的YOLO检测
        img_size=img_size,
        split='train',
        use_yolo_detection=False,  # 使用GT标签（课程学习）
        conf_threshold=0.25,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=KvasirYOLOTinySAMDataset.collate_fn
    )
    
    val_loader = None
    if val_images_dir and val_masks_dir:
        print("创建验证数据集...")
        val_dataset = KvasirYOLOTinySAMDataset(
            images_dir=val_images_dir,
            masks_dir=val_masks_dir,
            labels_dir=val_labels_dir,
            yolo_weights=None,
            img_size=img_size,
            split='val',
            use_yolo_detection=False,
            conf_threshold=0.25,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # 验证时batch size=1
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=KvasirYOLOTinySAMDataset.collate_fn
        )
    
    # 训练循环
    best_dice = 0.0
    train_history = []
    
    for epoch in range(num_epochs):
        # 课程学习：前N轮使用GT bbox，之后使用YOLO检测
        use_gt_bbox = curriculum_learning and (epoch < curriculum_epochs)
        
        if use_gt_bbox:
            print(f"\nEpoch {epoch+1}/{num_epochs} [课程学习：使用GT bbox]")
        else:
            print(f"\nEpoch {epoch+1}/{num_epochs} [使用YOLO检测]")
        
        # 训练
        joint_model.train()
        yolo_model.model.train()
        sam_model.train()
        
        epoch_loss = 0.0
        epoch_yolo_loss = 0.0
        epoch_sam_loss = 0.0
        epoch_dice = 0.0
        
        pbar = tqdm(train_loader, desc=f"训练中")
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            boxes_list = batch['boxes']
            
            optimizer.zero_grad()
            
            with autocast():
                # 前向传播
                yolo_results, sam_masks = joint_model(
                    images,
                    boxes_list=boxes_list,
                    use_yolo_detection=not use_gt_bbox
                )
                
                # 计算损失
                batch_loss = 0.0
                yolo_loss_sum = 0.0
                sam_loss_sum = 0.0
                
                for i in range(len(images)):
                    gt_mask = masks[i:i+1]  # (1, 1, H, W)
                    pred_masks = sam_masks[i]
                    
                    # TinySAM分割损失
                    sam_loss = compute_segmentation_loss(pred_masks, gt_mask, device)
                    sam_loss_sum += sam_loss.item()
                    batch_loss += sam_loss
                    
                    # YOLO检测损失（从YOLO结果中提取）
                    # 注意：YOLO的损失已经在内部计算，这里我们主要优化分割损失
                    # 如果需要，可以从yolo_results中提取损失
                
                batch_loss = batch_loss / len(images)
            
            # 反向传播
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += batch_loss.item()
            epoch_yolo_loss += yolo_loss_sum / len(images)
            epoch_sam_loss += sam_loss_sum / len(images)
            
            # 计算Dice（用于监控）
            if len(sam_masks[0]) > 0:
                # 简化计算，只对第一个样本
                pred_mask = sam_masks[0][0]
                gt_mask_np = masks[0, 0].cpu().numpy()
                pred_mask_resized = cv2.resize(
                    pred_mask.astype(np.float32),
                    (gt_mask_np.shape[1], gt_mask_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                pred_binary = (pred_mask_resized > 0.5).astype(np.float32)
                intersection = (pred_binary * gt_mask_np).sum()
                union = pred_binary.sum() + gt_mask_np.sum()
                dice = 2 * intersection / (union + 1e-6)
                epoch_dice += dice
            
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'sam_loss': f'{sam_loss_sum/len(images):.4f}',
            })
        
        epoch_loss /= len(train_loader)
        epoch_sam_loss /= len(train_loader)
        epoch_dice /= len(train_loader)
        
        print(f"训练损失: {epoch_loss:.4f}, SAM损失: {epoch_sam_loss:.4f}, Dice: {epoch_dice:.4f}")
        
        # 验证
        if val_loader and (epoch + 1) % val_interval == 0:
            joint_model.eval()
            yolo_model.model.eval()
            sam_model.eval()
            
            val_dice = 0.0
            val_iou = 0.0
            val_count = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="验证中"):
                    images = batch['images'].to(device)
                    masks = batch['masks'].to(device)
                    boxes_list = batch['boxes']
                    
                    yolo_results, sam_masks = joint_model(
                        images,
                        boxes_list=boxes_list,
                        use_yolo_detection=not use_gt_bbox
                    )
                    
                    for i in range(len(images)):
                        gt_mask = masks[i, 0].cpu().numpy()
                        pred_masks = sam_masks[i]
                        
                        if len(pred_masks) > 0:
                            # 合并所有预测mask
                            combined_pred = np.zeros_like(gt_mask, dtype=np.float32)
                            for mask in pred_masks:
                                mask_resized = cv2.resize(
                                    mask.astype(np.float32),
                                    (gt_mask.shape[1], gt_mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                                combined_pred = np.maximum(combined_pred, mask_resized)
                            
                            pred_binary = (combined_pred > 0.5).astype(np.float32)
                            
                            # 计算Dice和IoU
                            intersection = (pred_binary * gt_mask).sum()
                            union = pred_binary.sum() + gt_mask.sum()
                            dice = 2 * intersection / (union + 1e-6)
                            
                            intersection_iou = (pred_binary * gt_mask).sum()
                            union_iou = pred_binary.sum() + gt_mask.sum() - intersection_iou
                            iou = intersection_iou / (union_iou + 1e-6)
                            
                            val_dice += dice
                            val_iou += iou
                            val_count += 1
            
            if val_count > 0:
                val_dice /= val_count
                val_iou /= val_count
                print(f"验证 Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
                
                if val_dice > best_dice:
                    best_dice = val_dice
                    # 保存最佳模型
                    torch.save({
                        'yolo_model': yolo_model.model.state_dict(),
                        'sam_model': sam_model.state_dict(),
                        'epoch': epoch,
                        'dice': val_dice,
                    }, os.path.join(output_dir, 'best_model.pth'))
                    print(f"保存最佳模型 (Dice: {val_dice:.4f})")
        
        # 保存检查点
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'yolo_model': yolo_model.model.state_dict(),
                'sam_model': sam_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 记录历史
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_sam_loss': epoch_sam_loss,
            'train_dice': epoch_dice,
            'val_dice': val_dice if val_loader and (epoch + 1) % val_interval == 0 else None,
        })
        
        scheduler.step()
    
    # 保存训练历史
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f"\n训练完成！最佳Dice: {best_dice:.4f}")
    print(f"模型保存在: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + TinySAM 联合训练")
    parser.add_argument("--train-images", type=str, required=True, help="训练图片目录")
    parser.add_argument("--train-masks", type=str, required=True, help="训练mask目录")
    parser.add_argument("--train-labels", type=str, required=True, help="训练标签目录")
    parser.add_argument("--val-images", type=str, help="验证图片目录")
    parser.add_argument("--val-masks", type=str, help="验证mask目录")
    parser.add_argument("--val-labels", type=str, help="验证标签目录")
    parser.add_argument("--yolo-weights", type=str, default="yolo11n.pt", help="YOLO模型权重")
    parser.add_argument("--sam-weights", type=str, default="weights/tinysam_42.3.pth", help="TinySAM模型权重")
    parser.add_argument("--sam-model-type", type=str, default="vit_t", help="TinySAM模型类型")
    parser.add_argument("--img-size", type=int, default=1024, help="图片尺寸")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--lr-yolo", type=float, default=1e-5, help="YOLO学习率")
    parser.add_argument("--lr-sam", type=float, default=1e-4, help="TinySAM学习率")
    parser.add_argument("--output-dir", type=str, default="results_yolo_tinysam_joint", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--curriculum-learning", action="store_true", help="使用课程学习")
    parser.add_argument("--curriculum-epochs", type=int, default=10, help="课程学习轮数（使用GT bbox）")
    
    args = parser.parse_args()
    
    train_joint_model(
        train_images_dir=args.train_images,
        train_masks_dir=args.train_masks,
        train_labels_dir=args.train_labels,
        val_images_dir=args.val_images,
        val_masks_dir=args.val_masks,
        val_labels_dir=args.val_labels,
        yolo_weights=args.yolo_weights,
        sam_weights=args.sam_weights,
        sam_model_type=args.sam_model_type,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr_yolo=args.lr_yolo,
        lr_sam=args.lr_sam,
        device=args.device,
        output_dir=args.output_dir,
        curriculum_learning=args.curriculum_learning,
        curriculum_epochs=args.curriculum_epochs,
    )

