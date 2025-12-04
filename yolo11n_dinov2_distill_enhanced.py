#!/usr/bin/env python3
"""
优化版DINOv2 -> YOLO11n特征蒸馏脚本
针对息肉检测任务进行优化
"""

import os
import copy
import sys
import math
import glob
import time
import yaml
import argparse
from typing import List, Tuple, Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Ultralytics
REPO_ROOT = os.path.dirname(__file__)
LOCAL_ULTRA = os.path.join(REPO_ROOT, "ultralyticss_new")
if os.path.isdir(LOCAL_ULTRA) and LOCAL_ULTRA not in sys.path:
    sys.path.insert(0, LOCAL_ULTRA)
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(f"Failed to import YOLO from local ultralyticss_new at {LOCAL_ULTRA}: {e}")


def find_latest_pth(directory: str) -> str:
    candidates = sorted(glob.glob(os.path.join(directory, "*.pth")), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else ""


class ImageFolderFromYAML(Dataset):
    """
    Minimal image-only dataset from a YOLO data.yaml for feature KD pretraining.
    - Uses the 'train' path in the YAML to gather images recursively.
    - No labels required; images are used to align student backbone features to teacher.
    """
    def __init__(self, data_yaml: str, split: str = "train", imgsz: int = 640):
        super().__init__()
        with open(data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        root = data_cfg.get("path", "")
        split_rel = data_cfg.get(split, split)
        split_path = split_rel if os.path.isabs(split_rel) else os.path.join(root, split_rel)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"Split path not found: {split_path}")
        self.img_paths: List[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"): 
            self.img_paths.extend(glob.glob(os.path.join(split_path, "**", ext), recursive=True))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found under: {split_path}")
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB").resize((self.imgsz, self.imgsz), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return img_tensor


def load_dinov2_teacher(teacher_ckpt: str, device: torch.device):
    """
    Try to build a DINOv2 ViT-S/14 teacher and load weights.
    Fallbacks: dinov2 hub -> timm vit_small_patch14_dinov2 -> plain timm vit_small_patch14_224.
    Returns a module exposing method 'extract_last_patch_feature(images)->B,C,h,w'.
    """
    class TeacherWrapper(nn.Module):
        def __init__(self, backbone: nn.Module, patch_h: int, patch_w: int, out_channels: int):
            super().__init__()
            self.backbone = backbone
            self.patch_h = patch_h
            self.patch_w = patch_w
            self.out_channels = out_channels

        @torch.no_grad()
        def extract_last_patch_feature(self, images: torch.Tensor) -> torch.Tensor:
            self.backbone.eval()
            # Ensure H and W are multiples of patch size (14 for ViT-S/14)
            _, _, H, W = images.shape
            patch = 14
            new_h = max(patch, (H // patch) * patch)
            new_w = max(patch, (W // patch) * patch)
            if new_h != H or new_w != W:
                images_in = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
            else:
                images_in = images
            # Attempt several common forward feature access patterns
            if hasattr(self.backbone, "get_intermediate_layers"):
                tokens = self.backbone.get_intermediate_layers(images_in, n=1)[0]  # (B,1+HW,C)
                patch = tokens[:, 1:]
            elif hasattr(self.backbone, "forward_features"):
                out = self.backbone.forward_features(images_in)
                if isinstance(out, dict) and "x_norm_patchtokens" in out:
                    patch = out["x_norm_patchtokens"]
                elif isinstance(out, (list, tuple)):
                    patch = out[-1]
                else:
                    # Last token set; assume (B, 1+HW, C)
                    patch = out[:, 1:] if out.dim() == 3 else out
            else:
                out = self.backbone(images_in)
                patch = out[:, 1:] if out.dim() == 3 else out

            B, HW, C = patch.shape
            # Derive a near-square grid from actual token count, pad if needed
            h_est = max(1, int(math.floor(math.sqrt(HW))))
            w_est = max(1, int(math.ceil(HW / h_est)))
            area = h_est * w_est
            if area > HW:
                pad_tokens = area - HW
                pad = patch.new_zeros((B, pad_tokens, C))
                patch = torch.cat([patch, pad], dim=1)
            elif area < HW:
                patch = patch[:, :area, :]
            feat = patch.transpose(1, 2).reshape(B, C, h_est, w_est)
            return feat

    backbone = None
    out_channels = 384  # ViT-S default
    try:
        import importlib
        thub = importlib.import_module('torch.hub')
        backbone = thub.load('facebookresearch/dinov2', 'dinov2_vits14')
        out_channels = 384
    except Exception:
        try:
            import timm
            backbone = timm.create_model('vit_small_patch14_dinov2', pretrained=True)
            out_channels = backbone.num_features if hasattr(backbone, 'num_features') else 384
        except Exception:
            import timm
            backbone = timm.create_model('vit_small_patch14_224', pretrained=True)
            out_channels = backbone.num_features if hasattr(backbone, 'num_features') else 384

    # Load custom checkpoint if provided
    if teacher_ckpt and os.path.isfile(teacher_ckpt):
        state = torch.load(teacher_ckpt, map_location='cpu')
        key = 'model' if isinstance(state, dict) and 'model' in state else None
        try:
            backbone.load_state_dict(state[key] if key else state, strict=False)
        except Exception:
            # Try nested 'state_dict'
            if isinstance(state, dict) and 'state_dict' in state:
                backbone.load_state_dict(state['state_dict'], strict=False)

    wrapper = TeacherWrapper(backbone, patch_h=16, patch_w=16, out_channels=out_channels).to(device)
    for p in wrapper.parameters():
        p.requires_grad = False
    wrapper.eval()
    return wrapper


class MultiScaleFeatureAdapter(nn.Module):
    """多尺度特征适配器，支持不同分辨率的特征对齐"""
    def __init__(self, in_channels: int, out_channels: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.adapters = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            for _ in range(num_scales)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果输入是单个特征图，为每个尺度使用相同的适配器
        if x.dim() == 4:
            return self.adapters[0](x)
        # 如果输入是多个特征图，为每个使用对应的适配器
        elif isinstance(x, (list, tuple)):
            return [self.adapters[i](feat) for i, feat in enumerate(x)]
        else:
            return self.adapters[0](x)


class EnhancedFeatureAdapter(nn.Module):
    """增强版特征适配器，包含更多非线性变换"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def cosine_kd_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """余弦相似度蒸馏损失"""
    s = F.normalize(student_feat, dim=1)
    t = F.normalize(teacher_feat, dim=1)
    return 1.0 - (s * t).sum(dim=1).mean()


def mse_kd_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """MSE蒸馏损失"""
    return F.mse_loss(student_feat, teacher_feat)


def combined_kd_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor, 
                    alpha: float = 0.7) -> torch.Tensor:
    """组合蒸馏损失：余弦相似度 + MSE"""
    cosine_loss = cosine_kd_loss(student_feat, teacher_feat)
    mse_loss = mse_kd_loss(student_feat, teacher_feat)
    return alpha * cosine_loss + (1 - alpha) * mse_loss


def run_enhanced_feature_distill_pretrain(
    data_yaml: str,
    student_ckpt: str,
    teacher_ckpt: str,
    imgsz: int,
    epochs: int,
    batch: int,
    lambda_kd: float,
    device_index: int,
    project: str,
    name: str,
    no_amp: bool = False,
    # 新增参数
    lr: float = 2e-4,  # 提高学习率
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,  # 预热轮数
    use_multi_scale: bool = True,  # 使用多尺度蒸馏
    use_enhanced_adapter: bool = True,  # 使用增强适配器
    loss_type: str = "combined",  # 损失类型：cosine, mse, combined
    alpha: float = 0.7,  # 组合损失权重
):
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    # Student: YOLO11n
    student = YOLO(student_ckpt if (student_ckpt and os.path.isfile(student_ckpt)) else "yolo11n.pt")
    student_model = student.model.to(device)
    student_model.train()

    # Enforce dataset-driven single-class metadata
    with open(data_yaml, "r") as f:
        data_cfg_for_names = yaml.safe_load(f)
    names_list = data_cfg_for_names.get("names", ["polyp"]) or ["polyp"]
    if isinstance(names_list, dict):
        names_list = [names_list[k] for k in sorted(names_list.keys())]
    nc_val = int(data_cfg_for_names.get("nc", len(names_list)))
    if hasattr(student_model, "nc"):
        student_model.nc = nc_val
    try:
        student_model.names = {i: n for i, n in enumerate(names_list)}
    except Exception:
        pass

    # Register hooks to capture multiple feature levels
    feats_s: Dict[str, torch.Tensor] = {}

    def hook_named(name: str):
        def _hook(module, inputs, output):
            feats_s[name] = output
        return _hook

    # 捕获多个特征层
    hook_handles = []
    hook_indices = [6, 8, 10] if use_multi_scale else [8]  # 多尺度特征
    for idx in hook_indices:
        if idx < len(student_model.model):
            try:
                handle = student_model.model[idx].register_forward_hook(hook_named(f"L{idx}"))
                hook_handles.append(handle)
            except Exception:
                pass

    # Teacher
    teacher = load_dinov2_teacher(teacher_ckpt, device)

    # Build adapters
    adapters = None
    adapters_added_to_optim = False

    # Dataset and loader
    dataset = ImageFolderFromYAML(data_yaml=data_yaml, split="train", imgsz=imgsz)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # One-batch probe to determine feature shapes
    try:
        sample_imgs = next(iter(loader)).to(device)
        probe_outputs: Dict[int, torch.Size] = {}

        def _probe_hook(idx: int):
            def _h(module, inputs, output):
                try:
                    probe_outputs[idx] = output.shape if hasattr(output, 'shape') else torch.Size([])
                except Exception:
                    probe_outputs[idx] = torch.Size([])
            return _h

        temp_handles = []
        for idx, layer in enumerate(student_model.model):
            try:
                temp_handles.append(layer.register_forward_hook(_probe_hook(idx)))
            except Exception:
                pass
        with torch.no_grad():
            _ = student_model(sample_imgs)
        for h in temp_handles:
            try:
                h.remove()
            except Exception:
                pass
        
        # Print feature shapes
        printable = sorted(probe_outputs.items(), key=lambda kv: kv[0])
        print("[Probe] Student feature shapes by layer index:")
        for idx, shp in printable:
            print(f"  L{idx}: {tuple(shp)}")
        
        # 建议的特征层
        candidate_idxs = [i for i, shp in printable if len(shp) == 4 and shp[-1] >= 4 and shp[-2] >= 4]
        if candidate_idxs:
            print(f"[Probe] Suggested hook candidates: {candidate_idxs[-3:]}")
    except Exception as e:
        print(f"[Probe] Skipped feature shape probing due to: {e}")

    # Optimizer with different learning rates for different parts
    backbone_params = []
    neck_params = []
    head_params = []
    
    for name, param in student_model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'neck' in name:
            neck_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': neck_params, 'lr': lr * 2, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr * 0.5, 'weight_decay': weight_decay},
    ])
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP setup
    use_amp = torch.cuda.is_available() and (not no_amp)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    save_dir = os.path.join(project, name, "weights")
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")
    last_ckpt = os.path.join(save_dir, "last.pt")
    best_ckpt = os.path.join(save_dir, "best.pt")

    print(f"开始增强版特征蒸馏训练...")
    print(f"学习率: {lr}, 权重衰减: {weight_decay}")
    print(f"预热轮数: {warmup_epochs}, 总轮数: {epochs}")
    print(f"多尺度蒸馏: {use_multi_scale}, 增强适配器: {use_enhanced_adapter}")
    print(f"损失类型: {loss_type}, 组合权重: {alpha}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()
        
        # Training dynamics stats
        feat_std_running = 0.0
        grad_norm_running = 0.0
        stat_batches = 0
        
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            feats_s.clear()

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                autocast_ctx = torch.amp.autocast('cuda')
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            
            with autocast_ctx:
                # Forward student to populate hooks
                _ = student_model(imgs)

                # Get student features
                student_features = []
                for idx in hook_indices:
                    key = f"L{idx}"
                    if key in feats_s:
                        student_features.append(feats_s[key])
                
                if not student_features:
                    raise RuntimeError("No student features captured. Adjust hook indices.")

                # Teacher feature
                t_feat = teacher.extract_last_patch_feature(imgs)

                # Initialize adapters
                if adapters is None:
                    if use_enhanced_adapter:
                        adapters = nn.ModuleDict({
                            f"adapter_{i}": EnhancedFeatureAdapter(
                                in_channels=t_feat.shape[1],
                                out_channels=feat.shape[1],
                            ).to(device)
                            for i, feat in enumerate(student_features)
                        })
                    else:
                        adapters = nn.ModuleDict({
                            f"adapter_{i}": MultiScaleFeatureAdapter(
                                in_channels=t_feat.shape[1],
                                out_channels=feat.shape[1],
                            ).to(device)
                            for i, feat in enumerate(student_features)
                        })
                
                # Ensure adapters in optimizer
                if not adapters_added_to_optim:
                    optimizer.add_param_group({
                        "params": adapters.parameters(),
                        "lr": lr,
                        "weight_decay": weight_decay,
                    })
                    adapters_added_to_optim = True

                # Multi-scale distillation
                total_kd_loss = 0.0
                for i, s_feat in enumerate(student_features):
                    adapter_key = f"adapter_{i}"
                    t_proj = adapters[adapter_key](t_feat)
                    
                    # Resize teacher feature to match student feature
                    if t_proj.shape[-2:] != s_feat.shape[-2:]:
                        t_proj = F.interpolate(t_proj, size=s_feat.shape[-2:], mode="bilinear", align_corners=False)
                    
                    # Calculate distillation loss
                    if loss_type == "cosine":
                        kd_loss = cosine_kd_loss(s_feat, t_proj)
                    elif loss_type == "mse":
                        kd_loss = mse_kd_loss(s_feat, t_proj)
                    elif loss_type == "combined":
                        kd_loss = combined_kd_loss(s_feat, t_proj, alpha)
                    else:
                        raise ValueError(f"Unknown loss type: {loss_type}")
                    
                    # Weight by feature scale (deeper features get higher weight)
                    weight = 1.0 / (i + 1) if use_multi_scale else 1.0
                    total_kd_loss += weight * kd_loss

                total_loss = lambda_kd * total_kd_loss

            # Stats
            try:
                feat_std = float(student_features[0].detach().float().std().cpu().item())
            except Exception:
                feat_std = 0.0

            # Backward pass
            if use_amp:
                scaler.scale(total_loss).backward()
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                grad_norm = 0.0
                for p in student_model.parameters():
                    if p.grad is not None:
                        grad_norm += float(p.grad.detach().data.norm(2).item())
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                grad_norm = 0.0
                for p in student_model.parameters():
                    if p.grad is not None:
                        grad_norm += float(p.grad.detach().data.norm(2).item())
                optimizer.step()

            epoch_loss += float(total_loss.detach().cpu().item())
            num_batches += 1
            feat_std_running += feat_std
            grad_norm_running += grad_norm
            stat_batches += 1

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_loss /= max(1, num_batches)
        dt = time.time() - t0
        mean_feat_std = (feat_std_running / max(1, stat_batches))
        mean_grad_norm = (grad_norm_running / max(1, stat_batches))
        
        print(f"[Enhanced KD] Epoch {epoch+1}/{epochs}  "
              f"kd_loss={epoch_loss:.4f}  "
              f"feat_std={mean_feat_std:.4f}  "
              f"grad_norm={mean_grad_norm:.4f}  "
              f"lr={current_lr:.6f}  "
              f"time={dt:.1f}s")

        # Save checkpoints
        try:
            model_cpu = copy.deepcopy(student_model).float().cpu()
            payload = {
                "model": model_cpu,
                "ema": None,
                "train_args": {},
                "epoch": epoch,
                "best_fitness": 0.0,
                "yaml": "yolo11n.yaml",
                "names": getattr(model_cpu, "names", None),
                "nc": getattr(model_cpu, "nc", None),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            torch.save(payload, last_ckpt)
        except Exception:
            torch.save(student_model.state_dict(), last_ckpt)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            try:
                model_cpu = copy.deepcopy(student_model).float().cpu()
                payload = {
                    "model": model_cpu,
                    "ema": None,
                    "train_args": {},
                    "epoch": epoch,
                    "best_fitness": 0.0,
                    "yaml": "yolo11n.yaml",
                    "names": getattr(model_cpu, "names", None),
                    "nc": getattr(model_cpu, "nc", None),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                torch.save(payload, best_ckpt)
            except Exception:
                torch.save(student_model.state_dict(), best_ckpt)

    # Cleanup hooks
    for h in hook_handles:
        try:
            h.remove()
        except Exception:
            pass

    print(f"Enhanced KD-pretrained weights saved: {best_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced DINOv2 -> YOLO11n feature distillation")
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--student", type=str, default="yolo11n.pt", help="Student ckpt (.pt) or model name")
    parser.add_argument("--teacher_ckpt", type=str, default="", help="Path to DINOv2 checkpoint .pth (optional)")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50, help="Increased epochs")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lambda_kd", type=float, default=1.0, help="Increased KD weight")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--project", type=str, default="/home/huangmanling/huangmanling/ultralytics/runs")
    parser.add_argument("--name", type=str, default="polyp_y11n_enhanced_kd_dinov2")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP and GradScaler")
    
    # 新增参数
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--use_multi_scale", action="store_true", default=True, help="Use multi-scale distillation")
    parser.add_argument("--use_enhanced_adapter", action="store_true", default=True, help="Use enhanced adapter")
    parser.add_argument("--loss_type", type=str, default="combined", choices=["cosine", "mse", "combined"], help="Loss type")
    parser.add_argument("--alpha", type=float, default=0.7, help="Combined loss weight")
    
    args = parser.parse_args()

    if (not args.teacher_ckpt) and os.path.isdir("/home/huangmanling/huangmanling/TinySAM-dinov2/multi_gpu_dinov2_output_fixed"):
        auto_pth = find_latest_pth("/home/huangmanling/huangmanling/TinySAM-dinov2/multi_gpu_dinov2_output_fixed")
        if auto_pth:
            print(f"Auto-selected teacher checkpoint: {auto_pth}")
            args.teacher_ckpt = auto_pth

    run_enhanced_feature_distill_pretrain(
        data_yaml=args.data,
        student_ckpt=args.student,
        teacher_ckpt=args.teacher_ckpt,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lambda_kd=args.lambda_kd,
        device_index=args.device,
        project=args.project,
        name=args.name,
        no_amp=args.no_amp,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        use_multi_scale=args.use_multi_scale,
        use_enhanced_adapter=args.use_enhanced_adapter,
        loss_type=args.loss_type,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()

