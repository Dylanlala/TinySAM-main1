#!/usr/bin/env python3
"""
优化版TinySAM训练脚本
针对低召回率问题，使用GT bbox训练，增加训练轮数和边界损失
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


def boundary_loss_enhanced(pred, target):
    """增强的边界损失"""
    # 使用Sobel算子检测边界
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    
    pred_bound_x = F.conv2d(pred, sobel_x, padding=1)
    pred_bound_y = F.conv2d(pred, sobel_y, padding=1)
    pred_bound = torch.sqrt(pred_bound_x**2 + pred_bound_y**2 + 1e-6)
    
    target_bound_x = F.conv2d(target, sobel_x, padding=1)
    target_bound_y = F.conv2d(target, sobel_y, padding=1)
    target_bound = torch.sqrt(target_bound_x**2 + target_bound_y**2 + 1e-6)
    
    return F.l1_loss(pred_bound, target_bound)


def train_tinysam_optimized(
    train_images_dir,
    train_masks_dir,
    train_labels_dir,
    val_images_dir=None,
    val_masks_dir=None,
    val_labels_dir=None,
    sam_weights="weights/tinysam_42.3.pth",
    model_type="vit_t",
    img_size=1024,
    batch_size=4,
    num_epochs=100,  # 增加训练轮数
    lr=1e-4,
    weight_decay=1e-4,
    device="cuda",
    output_dir="results_tinysam_kvasir_optimized",
    save_interval=10,
    val_interval=5,
    use_boundary_loss=True,
    boundary_loss_weight=0.5,
):
    """
    优化版训练：使用GT bbox，增加训练轮数和边界损失
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
    # 使用warmup + cosine annealing
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs-10, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[10])
    
    scaler = GradScaler()
    
    # 数据集（使用GT标签而不是YOLO检测）
    print("创建训练数据集（使用GT bbox）...")
    train_dataset = KvasirYOLOTinySAMDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        labels_dir=train_labels_dir,
        yolo_weights=None,  # 不使用YOLO
        img_size=img_size,
        split='train',
        use_yolo_detection=False,  # 使用GT标签
        conf_threshold=0.25,
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
            yolo_weights=None,
            img_size=img_size,
            split='val',
            use_yolo_detection=False,
            conf_threshold=0.25,
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
                batch_loss = 0.0
                for i in range(images.shape[0]):
                    img = images[i:i+1]
                    mask_gt = masks[i:i+1]
                    boxes = boxes_list[i]
                    
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
                        
                        if low_res_masks.size(1) != 1:
                            low_res_masks = low_res_masks[:, :1, :, :]
                        
                        masks_resized = F.interpolate(
                            mask_gt,
                            size=low_res_masks.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        bce_loss = nn.BCEWithLogitsLoss()(low_res_masks, masks_resized)
                        
                        masks_pred = model.postprocess_masks(
                            low_res_masks,
                            (img_size, img_size),
                            (img_size, img_size),
                        )
                        masks_pred = masks_pred.sigmoid()
                        
                        if masks_pred.size(1) != mask_gt.size(1):
                            masks_pred = masks_pred[:, :1, :, :]
                        
                        dice = dice_loss(masks_pred, mask_gt)
                        iou = iou_loss(masks_pred, mask_gt)
                        
                        loss = bce_loss + dice + iou
                        
                        # 添加边界损失
                        if use_boundary_loss:
                            boundary = boundary_loss_enhanced(masks_pred, mask_gt)
                            loss = loss + boundary_loss_weight * boundary
                        
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
                        # 类似处理...
                        batch_loss += torch.tensor(0.0, device=device)
                
                loss = batch_loss / images.shape[0]
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")
        
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
                'val_dice': best_dice,
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
                    
                    if low_res_masks.size(1) != 1:
                        low_res_masks = low_res_masks[:, :1, :, :]
                    
                    masks_pred = model.postprocess_masks(
                        low_res_masks,
                        (img_size, img_size),
                        (img_size, img_size),
                    ).sigmoid()
                    
                    if masks_pred.size(1) != mask_gt.size(1):
                        masks_pred = masks_pred[:, :1, :, :]
                    
                    dice = 1.0 - dice_loss(masks_pred, mask_gt, reduction='mean')
                    total_dice += dice.item()
                    num_samples += 1
    
    return total_dice / num_samples if num_samples > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="优化版TinySAM训练")
    parser.add_argument("--train-images", type=str, required=True)
    parser.add_argument("--train-masks", type=str, required=True)
    parser.add_argument("--train-labels", type=str, required=True)
    parser.add_argument("--val-images", type=str, default=None)
    parser.add_argument("--val-masks", type=str, default=None)
    parser.add_argument("--val-labels", type=str, default=None)
    parser.add_argument("--sam-weights", type=str, default="weights/tinysam_42.3.pth")
    parser.add_argument("--model-type", type=str, default="vit_t")
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results_tinysam_kvasir_optimized")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--no-boundary-loss", action="store_true")
    
    args = parser.parse_args()
    
    train_tinysam_optimized(
        train_images_dir=args.train_images,
        train_masks_dir=args.train_masks,
        train_labels_dir=args.train_labels,
        val_images_dir=args.val_images,
        val_masks_dir=args.val_masks,
        val_labels_dir=args.val_labels,
        sam_weights=args.sam_weights,
        model_type=args.model_type,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        use_boundary_loss=not args.no_boundary_loss,
        boundary_loss_weight=args.boundary_weight,
    )


if __name__ == "__main__":
    main()

