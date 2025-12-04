#!/usr/bin/env python3
"""
增强版TinySAM训练脚本
改进点：
1. 使用中心点+bbox双重prompt（训练时）
2. 使用Focal Loss替代BCE（解决类别不平衡）
3. 增加召回率损失权重
4. 更强的数据增强
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


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss: 解决类别不平衡问题
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)  # pt = p if y=1, else 1-p
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def recall_loss(pred, target, eps=1e-6):
    """
    召回率损失：鼓励模型预测更多正样本
    recall = TP / (TP + FN)
    """
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > 0.5).float()
    
    tp = (pred_binary * target).sum()
    fn = ((1 - pred_binary) * target).sum()
    
    recall = tp / (tp + fn + eps)
    # 损失 = 1 - recall，鼓励提高召回率
    return 1.0 - recall


def boundary_loss_enhanced(pred, target):
    """增强的边界损失"""
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


def train_tinysam_enhanced(
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
    num_epochs=150,  # 增加训练轮数
    lr=1e-4,
    weight_decay=1e-4,
    device="cuda",
    output_dir="results_tinysam_enhanced",
    save_interval=10,
    val_interval=5,
    use_center_point=True,  # 使用中心点prompt
    use_focal_loss=True,  # 使用Focal Loss
    use_recall_loss=True,  # 使用召回率损失
    recall_loss_weight=0.3,  # 召回率损失权重
    use_boundary_loss=True,
    boundary_loss_weight=0.5,
):
    """
    增强版训练：中心点prompt + Focal Loss + 召回率损失
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
    
    # 数据集（使用GT标签）
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
        epoch_focal = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        epoch_recall = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            boxes_list = batch['boxes']
            
            optimizer.zero_grad()
            
            with autocast():
                batch_loss = 0.0
                
                for i in range(len(images)):
                    image = images[i:i+1]  # (1, 3, H, W)
                    mask_gt = masks[i:i+1]  # (1, 1, H, W)
                    boxes = boxes_list[i]  # (N, 4)
                    
                    # 图像编码
                    image_embeddings = model.image_encoder(image)
                    
                    if len(boxes) > 0:
                        # 对每个bbox进行训练
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            box_tensor = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32).to(device)
                            
                            # 准备prompt
                            if use_center_point:
                                cx = (x1 + x2) / 2
                                cy = (y1 + y2) / 2
                                point_coords = torch.tensor([[[cx, cy]]], dtype=torch.float32).to(device)
                                point_labels = torch.tensor([[1]], dtype=torch.int32).to(device)
                                points = (point_coords, point_labels)
                            else:
                                points = None
                            
                            # Prompt编码
                            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                                points=points,
                                boxes=box_tensor,
                                masks=None,
                            )
                            
                            # Mask解码
                            mask_output = model.mask_decoder(
                                image_embeddings=image_embeddings,
                                image_pe=model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                            )
                            
                            # 获取低分辨率mask
                            low_res_masks = mask_output[0] if isinstance(mask_output, tuple) else mask_output
                            if low_res_masks.size(1) != 1:
                                low_res_masks = low_res_masks[:, :1, :, :]
                            
                            # 调整GT mask尺寸
                            masks_resized = F.interpolate(
                                mask_gt,
                                size=low_res_masks.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            )
                            
                            # 计算损失
                            # 1. Focal Loss或BCE
                            if use_focal_loss:
                                focal = focal_loss(low_res_masks, masks_resized)
                                batch_loss += focal
                                epoch_focal += focal.item()
                            else:
                                bce = F.binary_cross_entropy_with_logits(low_res_masks, masks_resized)
                                batch_loss += bce
                            
                            # 后处理到原始尺寸
                            masks_pred = model.postprocess_masks(
                                low_res_masks,
                                (img_size, img_size),
                                (img_size, img_size),
                            )
                            masks_pred = masks_pred.sigmoid()
                            if masks_pred.size(1) != mask_gt.size(1):
                                masks_pred = masks_pred[:, :1, :, :]
                            
                            # 2. Dice Loss
                            dice = dice_loss(masks_pred, mask_gt)
                            batch_loss += dice
                            epoch_dice += dice.item()
                            
                            # 3. IoU Loss
                            iou = iou_loss(masks_pred, mask_gt)
                            batch_loss += iou
                            epoch_iou += iou.item()
                            
                            # 4. 召回率损失
                            if use_recall_loss:
                                recall = recall_loss(low_res_masks, masks_resized)
                                batch_loss += recall_loss_weight * recall
                                epoch_recall += recall.item()
                            
                            # 5. 边界损失
                            if use_boundary_loss:
                                boundary = boundary_loss_enhanced(masks_pred, mask_gt)
                                batch_loss += boundary_loss_weight * boundary
                    else:
                        # 如果没有bbox，跳过或使用默认prompt
                        continue
                
                batch_loss = batch_loss / max(len(images), 1)
            
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += batch_loss.item()
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        scheduler.step()
        
        # 打印epoch统计
        avg_loss = epoch_loss / len(train_loader)
        avg_focal = epoch_focal / len(train_loader) if epoch_focal > 0 else 0
        avg_dice = epoch_dice / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        avg_recall = epoch_recall / len(train_loader) if epoch_recall > 0 else 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, "
              f"Focal: {avg_focal:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, "
              f"Recall Loss: {avg_recall:.4f}")
        
        # 验证
        if val_loader and (epoch + 1) % val_interval == 0:
            val_dice = evaluate(model, val_loader, device, img_size)
            print(f"验证集 Dice: {val_dice:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                print(f"保存最佳模型 (Dice: {best_dice:.4f})")
        
        # 保存checkpoint
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    print(f"\n训练完成！最佳Dice: {best_dice:.4f}")


def evaluate(model, val_loader, device, img_size=1024):
    """评估模型"""
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            boxes_list = batch['boxes']
            
            for i in range(len(images)):
                image = images[i:i+1]
                mask_gt = masks[i:i+1]
                boxes = boxes_list[i]
                
                if len(boxes) == 0:
                    continue
                
                # 图像编码
                image_embeddings = model.image_encoder(image)
                
                # 合并所有bbox的预测
                combined_mask = torch.zeros_like(mask_gt)
                
                for box in boxes:
                    x1, y1, x2, y2 = box
                    box_tensor = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32).to(device)
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    point_coords = torch.tensor([[[cx, cy]]], dtype=torch.float32).to(device)
                    point_labels = torch.tensor([[1]], dtype=torch.int32).to(device)
                    points = (point_coords, point_labels)
                    
                    # Prompt编码
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=points,
                        boxes=box_tensor,
                        masks=None,
                    )
                    
                    # Mask解码
                    mask_output = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                    )
                    
                    low_res_masks = mask_output[0] if isinstance(mask_output, tuple) else mask_output
                    if low_res_masks.size(1) != 1:
                        low_res_masks = low_res_masks[:, :1, :, :]
                    
                    # 后处理
                    masks_pred = model.postprocess_masks(
                        low_res_masks,
                        (img_size, img_size),
                        (img_size, img_size),
                    )
                    masks_pred = masks_pred.sigmoid()
                    if masks_pred.size(1) != mask_gt.size(1):
                        masks_pred = masks_pred[:, :1, :, :]
                    
                    combined_mask = torch.maximum(combined_mask, masks_pred)
                
                # 计算Dice
                pred_binary = (combined_mask > 0.5).float()
                dice = dice_loss(pred_binary, mask_gt)
                dice_scores.append(dice.item())
    
    model.train()
    return np.mean(dice_scores) if dice_scores else 0.0


def main():
    parser = argparse.ArgumentParser(description="增强版TinySAM训练")
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
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results_tinysam_enhanced")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--no-center-point", action="store_true", help="不使用中心点prompt")
    parser.add_argument("--no-focal-loss", action="store_true", help="不使用Focal Loss")
    parser.add_argument("--no-recall-loss", action="store_true", help="不使用召回率损失")
    parser.add_argument("--recall-loss-weight", type=float, default=0.3)
    parser.add_argument("--no-boundary-loss", action="store_true", help="不使用边界损失")
    parser.add_argument("--boundary-loss-weight", type=float, default=0.5)
    
    args = parser.parse_args()
    
    train_tinysam_enhanced(
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
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        use_center_point=not args.no_center_point,
        use_focal_loss=not args.no_focal_loss,
        use_recall_loss=not args.no_recall_loss,
        recall_loss_weight=args.recall_loss_weight,
        use_boundary_loss=not args.no_boundary_loss,
        boundary_loss_weight=args.boundary_loss_weight,
    )


if __name__ == "__main__":
    main()

