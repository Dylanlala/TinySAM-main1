#!/usr/bin/env python3
"""
使用YOLO检测的bbox作为prompt训练TinySAM
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

from tinysam.build_sam import sam_model_registry
from tinysam.predictor import SamPredictor
from kvasir_yolo_tinysam_dataset import KvasirYOLOTinySAMDataset
from utils import dice_loss, iou_loss


def train_tinysam_with_yolo(
    train_images_dir,
    train_masks_dir,
    train_labels_dir=None,
    val_images_dir=None,
    val_masks_dir=None,
    val_labels_dir=None,
    yolo_weights=None,
    sam_weights="weights/tinysam_42.3.pth",
    model_type="vit_t",
    img_size=1024,
    batch_size=4,
    num_epochs=50,
    lr=1e-4,
    weight_decay=1e-4,
    device="cuda",
    output_dir="results_tinysam_kvasir",
    use_yolo_detection=True,
    conf_threshold=0.25,
    save_interval=10,
    val_interval=5,
):
    """
    训练TinySAM，使用YOLO检测的bbox作为prompt
    
    Args:
        train_images_dir: 训练图片目录
        train_masks_dir: 训练mask目录
        train_labels_dir: 训练标签目录（可选）
        val_images_dir: 验证图片目录
        val_masks_dir: 验证mask目录
        val_labels_dir: 验证标签目录（可选）
        yolo_weights: YOLO模型权重路径
        sam_weights: TinySAM预训练权重
        model_type: TinySAM模型类型
        img_size: 图片尺寸
        batch_size: 批次大小
        num_epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        device: 设备
        output_dir: 输出目录
        use_yolo_detection: 是否使用YOLO检测（True）或GT标签（False）
        conf_threshold: YOLO检测置信度阈值
        save_interval: 保存间隔
        val_interval: 验证间隔
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载TinySAM模型
    print(f"加载TinySAM模型: {sam_weights}")
    model = sam_model_registry[model_type](checkpoint=sam_weights)
    model.to(device)
    model.train()
    
    # 优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()
    
    # 数据集
    print("创建训练数据集...")
    train_dataset = KvasirYOLOTinySAMDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        labels_dir=train_labels_dir,
        yolo_weights=yolo_weights,
        img_size=img_size,
        split='train',
        use_yolo_detection=use_yolo_detection,
        conf_threshold=conf_threshold,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
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
            yolo_weights=yolo_weights,
            img_size=img_size,
            split='val',
            use_yolo_detection=use_yolo_detection,
            conf_threshold=conf_threshold,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=KvasirYOLOTinySAMDataset.collate_fn
        )
    
    # 训练循环
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            boxes_list = batch['boxes']
            
            optimizer.zero_grad()
            
            with autocast():
                # 对每张图片单独处理（因为bbox数量不同）
                batch_loss = 0.0
                for i in range(images.shape[0]):
                    img = images[i:i+1]  # (1, 3, H, W)
                    mask_gt = masks[i:i+1]  # (1, 1, H, W)
                    boxes = boxes_list[i]  # (N, 4)
                    
                    # 获取图像embedding
                    image_embeddings = model.image_encoder(img)
                    
                    # 对每个bbox进行预测
                    if len(boxes) > 0:
                        # 使用第一个bbox（或合并所有bbox）
                        box = boxes[0] if len(boxes) == 1 else boxes
                        if len(boxes) > 1:
                            # 合并多个bbox
                            x1 = min(box[:, 0])
                            y1 = min(box[:, 1])
                            x2 = max(box[:, 2])
                            y2 = max(box[:, 3])
                            box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                        else:
                            box = box.reshape(1, -1)
                        
                        # 使用box作为prompt
                        sparse_embeddings, dense_embeddings = model.prompt_encoder(
                            points=None,
                            boxes=torch.from_numpy(box).to(device),
                            masks=None,
                        )
                        
                        mask_output = model.mask_decoder(
                            image_embeddings=image_embeddings,
                            image_pe=model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                        )
                        
                        low_res_masks = mask_output[0] if isinstance(mask_output, tuple) else mask_output
                        
                        # 调整mask尺寸
                        masks_resized = F.interpolate(
                            mask_gt,
                            size=low_res_masks.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        if low_res_masks.size(1) != 1:
                            low_res_masks = low_res_masks[:, :1, :, :]
                        
                        # 计算损失
                        bce_loss = nn.BCEWithLogitsLoss()(low_res_masks, masks_resized)
                        
                        # 后处理
                        masks_pred = model.postprocess_masks(
                            low_res_masks,
                            (img_size, img_size),
                            (img_size, img_size),
                        )
                        masks_pred = masks_pred.sigmoid()
                        
                        # 确保masks_pred和mask_gt的通道数一致
                        if masks_pred.size(1) != mask_gt.size(1):
                            masks_pred = masks_pred[:, :1, :, :]
                        
                        dice = dice_loss(masks_pred, mask_gt)
                        iou = iou_loss(masks_pred, mask_gt)
                        
                        loss = bce_loss + dice + iou
                        batch_loss += loss
                    else:
                        # 如果没有bbox，使用默认prompt
                        sparse_embeddings, dense_embeddings = model.prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=None,
                        )
                        mask_output = model.mask_decoder(
                            image_embeddings=image_embeddings,
                            image_pe=model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                        )
                        # ... 类似处理
                        batch_loss += torch.tensor(0.0, device=device)
                
                loss = batch_loss / images.shape[0]
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        # 验证
        if val_loader and (epoch + 1) % val_interval == 0:
            val_dice = evaluate(model, val_loader, device, img_size)
            print(f"Validation Dice: {val_dice:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
                print(f"Saved best model with Dice: {best_dice:.4f}")
        
        # 保存checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{output_dir}/checkpoint_epoch_{epoch+1}.pth")
        
        scheduler.step()
    
    print(f"训练完成！最佳模型保存在: {output_dir}/best_model.pth")


def evaluate(model, dataloader, device, img_size):
    """评估模型"""
    model.eval()
    total_dice = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            boxes_list = batch['boxes']
            
            for i in range(images.shape[0]):
                img = images[i:i+1]
                mask_gt = masks[i:i+1]
                boxes = boxes_list[i]
                
                # 类似训练时的处理
                image_embeddings = model.image_encoder(img)
                
                if len(boxes) > 0:
                    box = boxes[0] if len(boxes) == 1 else boxes
                    if len(boxes) > 1:
                        x1 = min(box[:, 0])
                        y1 = min(box[:, 1])
                        x2 = max(box[:, 2])
                        y2 = max(box[:, 3])
                        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                    else:
                        box = box.reshape(1, -1)
                    
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=torch.from_numpy(box).to(device),
                        masks=None,
                    )
                    
                    mask_output = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                    )
                    
                    low_res_masks = mask_output[0] if isinstance(mask_output, tuple) else mask_output
                    
                    # 确保通道数为1（TinySAM可能输出多个mask）
                    if low_res_masks.size(1) != 1:
                        low_res_masks = low_res_masks[:, :1, :, :]
                    
                    masks_pred = model.postprocess_masks(
                        low_res_masks,
                        (img_size, img_size),
                        (img_size, img_size),
                    ).sigmoid()
                    
                    # 确保masks_pred和mask_gt的通道数一致
                    if masks_pred.size(1) != mask_gt.size(1):
                        masks_pred = masks_pred[:, :1, :, :]
                    
                    dice = 1.0 - dice_loss(masks_pred, mask_gt, reduction='mean')
                    total_dice += dice.item()
                    num_samples += 1
    
    return total_dice / num_samples if num_samples > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="使用YOLO检测结果训练TinySAM")
    parser.add_argument("--train-images", type=str, required=True,
                       help="训练图片目录")
    parser.add_argument("--train-masks", type=str, required=True,
                       help="训练mask目录")
    parser.add_argument("--train-labels", type=str, default=None,
                       help="训练标签目录（可选）")
    parser.add_argument("--val-images", type=str, default=None,
                       help="验证图片目录")
    parser.add_argument("--val-masks", type=str, default=None,
                       help="验证mask目录")
    parser.add_argument("--val-labels", type=str, default=None,
                       help="验证标签目录（可选）")
    parser.add_argument("--yolo-weights", type=str, required=True,
                       help="YOLO模型权重路径")
    parser.add_argument("--sam-weights", type=str, default="weights/tinysam_42.3.pth",
                       help="TinySAM预训练权重")
    parser.add_argument("--model-type", type=str, default="vit_t",
                       help="TinySAM模型类型")
    parser.add_argument("--img-size", type=int, default=1024,
                       help="图片尺寸")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--epochs", type=int, default=50,
                       help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="权重衰减")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备")
    parser.add_argument("--output-dir", type=str, default="results_tinysam_kvasir",
                       help="输出目录")
    parser.add_argument("--use-gt-labels", action="store_true",
                       help="使用GT标签而不是YOLO检测结果")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                       help="YOLO检测置信度阈值")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="保存间隔")
    parser.add_argument("--val-interval", type=int, default=5,
                       help="验证间隔")
    
    args = parser.parse_args()
    
    train_tinysam_with_yolo(
        train_images_dir=args.train_images,
        train_masks_dir=args.train_masks,
        train_labels_dir=args.train_labels,
        val_images_dir=args.val_images,
        val_masks_dir=args.val_masks,
        val_labels_dir=args.val_labels,
        yolo_weights=args.yolo_weights,
        sam_weights=args.sam_weights,
        model_type=args.model_type,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        output_dir=args.output_dir,
        use_yolo_detection=not args.use_gt_labels,
        conf_threshold=args.conf_threshold,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
    )


if __name__ == "__main__":
    main()

