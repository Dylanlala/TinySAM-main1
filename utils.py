import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-5, reduction='mean'):
    """Dice系数损失"""
    pred = pred.flatten(1)
    target = target.flatten(1)
    
    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def iou_loss(pred, target, smooth=1e-5, reduction='mean'):
    """IoU损失"""
    pred = pred.flatten(1)
    target = target.flatten(1)
    
    intersection = (pred * target).sum(1)
    total = (pred + target).sum(1)
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    loss = 1 - iou
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def boundary_loss(pred, target):
    """边界损失 - 增强边缘分割能力"""
    pred_bound = F.conv2d(pred, sobel_kernel(), padding=1)
    target_bound = F.conv2d(target, sobel_kernel(), padding=1)
    return F.l1_loss(pred_bound, target_bound)

def sobel_kernel():
    """Sobel算子用于边界检测"""
    return torch.tensor([
        [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], 
        dtype=torch.float32
    )
