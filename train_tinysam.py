import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

# 导入TinySAM模块
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from tinysam import sam_model_registry
from ldpolyvideo_dataset import LDPolyVideoDataset
from utils import dice_loss, iou_loss

# 配置参数
config = {
    "model_type": "vit_t",
    "checkpoint": "weights/tinysam_42.3.pth",
    "train_dir": "kvasir-seg/train",
    "test_dir": "Polys/Test",
    "batch_size": 8,
    "num_epochs":100,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "results_ldpolyvideo",
    "img_size": 1024,
    "save_interval": 10,
    "val_interval": 5,
    "test_after_train": True
}

def sobel_kernel(device=None, dtype=None):
    """Sobel算子用于边界检测"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float32
    
    kernel = torch.tensor([
        [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
        [[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]
    ], device=device, dtype=dtype)
    
    return kernel

def boundary_loss(pred, target):
    """边界损失 - 增强边缘分割能力"""
    # 获取输入的数据类型和设备
    dtype = pred.dtype
    device = pred.device
    
    # 生成与输入类型匹配的Sobel核
    kernel = sobel_kernel(device=device, dtype=dtype)
    
    # 调整卷积核维度以匹配输入通道数
    if pred.size(1) != kernel.size(0):
        kernel = kernel.repeat(pred.size(1), 1, 1, 1)
        # 重新归一化多通道核
        kernel = kernel / kernel.size(0)
    
    pred_bound = F.conv2d(pred, kernel, padding=1, groups=pred.size(1))
    target_bound = F.conv2d(target, kernel, padding=1, groups=target.size(1))
    return F.l1_loss(pred_bound, target_bound)

def generate_grid_points(image_size, grid_size, device):
    # image_size: int, grid_size: int
    # 返回 (1, N, 2), (1, N)
    step = image_size // grid_size
    coords = []
    for y in range(step // 2, image_size, step):
        for x in range(step // 2, image_size, step):
            coords.append([x, y])
    coords = torch.tensor([coords], dtype=torch.float, device=device)  # (1, N, 2)
    labels = torch.ones((1, len(coords[0])), dtype=torch.int, device=device)
    return coords, labels

def train():
    os.makedirs(config["output_dir"], exist_ok=True)
    
    model = sam_model_registry[config["model_type"]](checkpoint=config["checkpoint"])
    model.to(config["device"])
    model.train()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    scaler = GradScaler()
    
    train_dataset = LDPolyVideoDataset(root=config["train_dir"], split="train", img_size=config["img_size"])
    val_dataset = LDPolyVideoDataset(root=config["train_dir"], split="val", img_size=config["img_size"])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=LDPolyVideoDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=LDPolyVideoDataset.collate_fn)
    
    log_file = open(os.path.join(config["output_dir"], "training_log.csv"), "w")
    log_file.write("epoch,train_loss,val_dice,val_iou,val_f1,val_precision,val_recall,time\n")
    
    best_dice = 0.0
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for images, masks in pbar:
            images = images.to(config["device"], dtype=torch.float32)
            masks = masks.to(config["device"], dtype=torch.float32)
            
            optimizer.zero_grad()
            
            with autocast():
                image_embeddings = model.image_encoder(images)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
                print("dense_prompt_embeddings shape:", dense_embeddings.shape)
                print("image_embeddings shape:", image_embeddings.shape)
                mask_output = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                )
                low_res_masks = mask_output[0] if isinstance(mask_output, tuple) else mask_output
                
                # 调整mask维度
                masks_resized = F.interpolate(
                    masks, 
                    size=low_res_masks.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # 确保通道数一致
                if low_res_masks.size(1) != 1:
                    low_res_masks = low_res_masks[:, :1, :, :]
                
                bce_loss = nn.BCEWithLogitsLoss()(low_res_masks, masks_resized)
                
                # 后处理
                masks_pred = model.postprocess_masks(
                    low_res_masks,
                    (config["img_size"], config["img_size"]),
                    (config["img_size"], config["img_size"]),
                )
                masks_pred = masks_pred.sigmoid()
                
                dice = dice_loss(masks_pred, masks)
                iou = iou_loss(masks_pred, masks)
                loss = bce_loss + dice + iou
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if (epoch + 1) % config["val_interval"] == 0:
            val_metrics = evaluate(model, val_loader, config["device"], config["img_size"])
        else:
            val_metrics = {"dice": 0, "iou": 0, "f1": 0, "precision": 0, "recall": 0}
        
        epoch_time = time.time() - start_time
        
        log_line = f"{epoch+1},{epoch_loss/len(train_loader):.4f},{val_metrics['dice']:.4f},{val_metrics['iou']:.4f},{val_metrics['f1']:.4f},{val_metrics['precision']:.4f},{val_metrics['recall']:.4f},{epoch_time:.1f}"
        log_file.write(log_line + "\n")
        log_file.flush()
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Time: {epoch_time:.1f}s")
        
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), f"{config['output_dir']}/best_model.pth")
            print(f"Saved best model with Dice: {best_dice:.4f}")
        
        if (epoch + 1) % config["save_interval"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss/len(train_loader),
            }, f"{config['output_dir']}/checkpoint_epoch_{epoch+1}.pth")
        
        scheduler.step()
    
    log_file.close()
    
    if config["test_after_train"]:
        test_dataset = LDPolyVideoDataset(root=config["test_dir"], split="test", img_size=config["img_size"])
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                               num_workers=2, pin_memory=True, collate_fn=LDPolyVideoDataset.collate_fn)
        model.load_state_dict(torch.load(os.path.join(config["output_dir"], "best_model.pth")))
        model.eval()
        dices, ious, f1s, precisions, recalls, times = [], [], [], [], [], []
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(config["device"], dtype=torch.float32)
                masks = masks.to(config["device"], dtype=torch.float32)
                start = time.time()
                image_embeddings = model.image_encoder(images)
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
                infer_time = time.time() - start
                times.append(infer_time)
                masks_pred = model.postprocess_masks(
                    mask_output[0] if isinstance(mask_output, tuple) else mask_output,
                    (config["img_size"], config["img_size"]),
                    (config["img_size"], config["img_size"]),
                ).sigmoid()
                # 修复：保证pred和gt shape一致
                if masks_pred.ndim == 4:
                    masks_pred = masks_pred[:, 0]
                if masks.ndim == 4:
                    masks = masks[:, 0]
                pred = (masks_pred > 0.5).float().cpu().numpy().flatten()
                gt = (masks > 0.5).float().cpu().numpy().flatten()
                dices.append(f1_score(gt, pred, zero_division=0))
                ious.append(jaccard_score(gt, pred, zero_division=0))
                f1s.append(f1_score(gt, pred, zero_division=0))
                precisions.append(precision_score(gt, pred, zero_division=0))
                recalls.append(recall_score(gt, pred, zero_division=0))
        print("\n==== LDPolyVideo 测试集分割性能 ====")
        if times:
            print(f"FPS: {1/np.mean(times):.2f}")
        print(f"Dice: {np.mean(dices):.4f}")
        print(f"IoU: {np.mean(ious):.4f}")
        print(f"F1: {np.mean(f1s):.4f}")
        print(f"Precision: {np.mean(precisions):.4f}")
        print(f"Recall: {np.mean(recalls):.4f}")
        with open(os.path.join(config["output_dir"], "test_results.txt"), "w") as f:
            if times:
                f.write(f"FPS: {1/np.mean(times):.2f}\n")
            f.write(f"Dice: {np.mean(dices):.4f}\n")
            f.write(f"IoU: {np.mean(ious):.4f}\n")
            f.write(f"F1: {np.mean(f1s):.4f}\n")
            f.write(f"Precision: {np.mean(precisions):.4f}\n")
            f.write(f"Recall: {np.mean(recalls):.4f}\n")
        
        visualize_results(model, test_dataset, config["output_dir"], num_samples=5)

def evaluate(model, dataloader, device, img_size):
    """评估模型性能"""
    model.eval()
    total_dice, total_iou, total_f1, total_precision, total_recall = 0.0, 0.0, 0.0, 0.0, 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            
            image_embeddings = model.image_encoder(images)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            print("[VAL] dense_prompt_embeddings shape:", dense_embeddings.shape)
            print("[VAL] image_embeddings shape:", image_embeddings.shape)
            mask_output = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            low_res_masks = mask_output[0] if isinstance(mask_output, tuple) else mask_output
            
            # 调整mask维度
            masks_resized = F.interpolate(
                masks, 
                size=low_res_masks.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # 确保通道数一致
            if low_res_masks.size(1) != 1:
                low_res_masks = low_res_masks[:, :1, :, :]
            
            masks_pred = model.postprocess_masks(
                low_res_masks,
                (img_size, img_size),
                (img_size, img_size),
            )
            masks_pred = masks_pred.sigmoid()
            
            batch_dice = dice_loss(masks_pred, masks, reduction='mean')
            batch_iou = iou_loss(masks_pred, masks, reduction='mean')
            batch_f1, batch_precision, batch_recall = calculate_f1_precision_recall(masks_pred, masks)
            
            total_dice += batch_dice.item()
            total_iou += batch_iou.item()
            total_f1 += batch_f1
            total_precision += batch_precision
            total_recall += batch_recall
    
    avg_dice = 1 - total_dice / num_batches
    avg_iou = 1 - total_iou / num_batches
    avg_f1 = total_f1 / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    
    return {
        "dice": avg_dice,
        "iou": avg_iou,
        "f1": avg_f1,
        "precision": avg_precision,
        "recall": avg_recall
    }

def calculate_f1_precision_recall(pred, target):
    """计算F1分数、精确率和召回率"""
    pred = (pred > 0.5).float().cpu().numpy().flatten()
    gt = (target > 0.5).float().cpu().numpy().flatten()
    f1 = f1_score(gt, pred, zero_division=0)
    precision = precision_score(gt, pred, zero_division=0)
    recall = recall_score(gt, pred, zero_division=0)
    return f1, precision, recall

def visualize_results(model, dataset, output_dir, num_samples=5):
    """可视化分割结果"""
    import matplotlib.pyplot as plt
    import os
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    model.eval()
    import numpy as np
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for i, idx in enumerate(indices):
        image, true_mask = dataset[idx]
        image_tensor = image.unsqueeze(0).to(model.device, dtype=torch.float32)
        true_mask = true_mask.unsqueeze(0).to(model.device, dtype=torch.float32)
        with torch.no_grad():
            image_embeddings = model.image_encoder(image_tensor)
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
            pred_mask = model.postprocess_masks(
                mask_output[0] if isinstance(mask_output, tuple) else mask_output,
                (image_tensor.shape[-2], image_tensor.shape[-1]),
                (image_tensor.shape[-2], image_tensor.shape[-1]),
            ).sigmoid().squeeze().cpu().numpy()
        # 处理原图为uint8
        img = image.permute(1, 2, 0).cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        true_mask_np = true_mask.squeeze().cpu().numpy()
        # 修复：如果pred_mask为(3, H, W)，只取第一个通道
        if pred_mask.ndim == 3 and pred_mask.shape[0] == 3:
            pred_mask = pred_mask[0]
        # 可视化
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        ax[1].imshow(true_mask_np, cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[1].axis('off')
        ax[2].imshow(pred_mask > 0.5, cmap='gray')
        ax[2].set_title("Predicted Mask")
        ax[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "visualizations", f"result_{i}.png"))
        plt.close()

if __name__ == "__main__":
    train()
