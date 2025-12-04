#!/usr/bin/env python3
"""
增强版推理脚本：降低YOLO阈值 + 后处理优化
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from ultralytics import YOLO
from tinysam.build_sam import sam_model_registry
from tinysam.predictor import SamPredictor
from scipy import ndimage
from scipy.ndimage import gaussian_filter


def postprocess_mask(mask, bbox=None):
    """
    后处理mask：形态学操作、连通域分析、边界平滑
    
    Args:
        mask: 二值mask (H, W)，应该是uint8类型，值0或255
        bbox: 可选，用于裁剪mask到bbox内
    
    Returns:
        处理后的mask
    """
    # 确保是uint8类型，值0或255
    if mask.dtype != np.uint8:
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.dtype in [np.float32, np.float64]:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    
    # 1. 形态学操作：闭运算填充小洞
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. 形态学操作：开运算去除小噪声
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # 3. 连通域分析：只保留最大的连通域
    labeled, num_features = ndimage.label(mask > 128)
    if num_features > 0:
        sizes = ndimage.sum(mask > 128, labeled, range(1, num_features + 1))
        if len(sizes) > 0:
            largest_label = np.argmax(sizes) + 1
            mask = (labeled == largest_label).astype(np.uint8) * 255
    
    # 4. 边界平滑：Gaussian滤波
    mask_float = mask.astype(float) / 255.0
    mask_smooth = gaussian_filter(mask_float, sigma=1.0)
    mask = (mask_smooth > 0.5).astype(np.uint8) * 255
    
    # 5. 如果提供了bbox，裁剪到bbox内（可选）
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        h, w = mask.shape
        # 创建bbox mask
        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        bbox_mask[int(y1):int(y2), int(x1):int(x2)] = 255
        # 只保留bbox内的区域
        mask = mask & bbox_mask
    
    return mask.astype(np.uint8)


class EnhancedYOLOTinySAMInference:
    """增强版YOLO+TinySAM推理"""
    
    def __init__(self, yolo_weights, sam_weights, model_type="vit_t", 
                 yolo_conf=0.15, yolo_iou=0.4, use_point_prompt=True):
        """
        Args:
            yolo_weights: YOLO模型权重
            sam_weights: TinySAM模型权重
            model_type: TinySAM模型类型
            yolo_conf: YOLO置信度阈值（降低以提高召回率）
            yolo_iou: YOLO IoU阈值（降低以减少NMS）
            use_point_prompt: 是否使用中心点作为prompt
        """
        # 加载YOLO
        print(f"加载YOLO模型: {yolo_weights}")
        self.yolo_model = YOLO(yolo_weights)
        
        # 加载TinySAM
        print(f"加载TinySAM模型: {sam_weights}")
        sam_model = sam_model_registry[model_type](checkpoint=sam_weights)
        self.sam_predictor = SamPredictor(sam_model)
        
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.use_point_prompt = use_point_prompt
    
    def predict(self, image_path, output_dir, enable_postprocess=True):
        """
        对单张图片进行推理
        
        Args:
            image_path: 图片路径
            output_dir: 输出目录
            enable_postprocess: 是否启用后处理
        """
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # YOLO检测（降低阈值）
        results = self.yolo_model.predict(
            image_rgb,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            verbose=False
        )
        
        detections = results[0].boxes
        
        if len(detections) == 0:
            print(f"未检测到息肉: {image_path.name}")
            # 仍然保存原图和空的bbox
            self._save_results(image_path, image_rgb, [], [], output_dir)
            return None
        
        # 设置SAM图像
        self.sam_predictor.set_image(image_rgb)
        
        # 对每个检测进行分割
        masks_list = []
        bboxes_list = []
        
        for i, box in enumerate(detections):
            # 获取bbox坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            # 计算中心点
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # 使用bbox + 中心点作为prompt
            if self.use_point_prompt:
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=np.array([[cx, cy]]),
                    point_labels=np.array([1]),
                    box=np.array([x1, y1, x2, y2]),
                )
            else:
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array([x1, y1, x2, y2]),
                )
            
            # 选择最好的mask（通常是第一个）
            # TinySAM返回的mask可能是bool类型，需要正确转换
            if masks[0].dtype == bool:
                mask = masks[0].astype(np.uint8) * 255
            elif masks[0].dtype == np.float32 or masks[0].dtype == np.float64:
                mask = (masks[0] * 255).astype(np.uint8)
            else:
                mask = masks[0].astype(np.uint8)
            
            # 后处理
            if enable_postprocess:
                mask = postprocess_mask(mask, bbox=[x1, y1, x2, y2])
            
            masks_list.append(mask)
            bboxes_list.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'mask_score': float(scores[0]) if len(scores) > 0 else 0.0
            })
        
        # 合并所有mask
        if masks_list:
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for mask in masks_list:
                combined_mask = np.maximum(combined_mask, mask)
        else:
            combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 保存结果
        self._save_results(image_path, image_rgb, bboxes_list, masks_list, 
                          combined_mask, output_dir)
        
        return {
            'num_detections': len(detections),
            'bboxes': bboxes_list,
            'masks': masks_list
        }
    
    def _save_results(self, image_path, image_rgb, bboxes_list, masks_list, 
                     combined_mask, output_dir):
        """保存推理结果"""
        # 创建输出目录
        images_dir = Path(output_dir) / "images"
        bboxes_dir = Path(output_dir) / "bboxes"
        masks_dir = Path(output_dir) / "masks"
        individual_masks_dir = masks_dir / "individual"
        vis_dir = Path(output_dir) / "visualizations"
        
        for d in [images_dir, bboxes_dir, masks_dir, individual_masks_dir, vis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        img_name = image_path.stem
        
        # 1. 保存原图
        cv2.imwrite(str(images_dir / f"{img_name}.jpg"), 
                   cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        
        # 2. 保存bbox信息
        with open(bboxes_dir / f"{img_name}.json", 'w') as f:
            json.dump(bboxes_list, f, indent=2)
        
        # 3. 保存mask
        cv2.imwrite(str(masks_dir / f"{img_name}_combined.png"), combined_mask)
        
        # 保存每个检测的独立mask
        for i, mask in enumerate(masks_list):
            cv2.imwrite(str(individual_masks_dir / f"{img_name}_mask_{i}.png"), mask)
        
        # 4. 保存可视化结果
        vis_image = image_rgb.copy()
        
        # 绘制bbox
        for bbox_info in bboxes_list:
            x1, y1, x2, y2 = bbox_info['bbox']
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (255, 0, 0), 2)
            cv2.putText(vis_image, f"{bbox_info['confidence']:.2f}", 
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 0, 0), 2)
        
        # 绘制mask（半透明）
        mask_colored = np.zeros_like(vis_image)
        mask_colored[combined_mask > 128] = [0, 255, 0]
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_colored, 0.3, 0)
        
        cv2.imwrite(str(vis_dir / f"{img_name}_vis.jpg"), 
                   cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="增强版YOLO+TinySAM推理")
    parser.add_argument("--yolo-weights", type=str, required=True,
                       help="YOLO模型权重路径")
    parser.add_argument("--sam-weights", type=str, required=True,
                       help="TinySAM模型权重路径")
    parser.add_argument("--test-images", type=str, required=True,
                       help="测试图片目录")
    parser.add_argument("--output-dir", type=str, default="inference_results_enhanced",
                       help="输出目录")
    parser.add_argument("--yolo-conf", type=float, default=0.15,
                       help="YOLO置信度阈值（降低以提高召回率）")
    parser.add_argument("--yolo-iou", type=float, default=0.4,
                       help="YOLO IoU阈值")
    parser.add_argument("--model-type", type=str, default="vit_t",
                       help="TinySAM模型类型")
    parser.add_argument("--no-postprocess", action="store_true",
                       help="禁用后处理")
    parser.add_argument("--no-point-prompt", action="store_true",
                       help="不使用中心点prompt")
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = EnhancedYOLOTinySAMInference(
        yolo_weights=args.yolo_weights,
        sam_weights=args.sam_weights,
        model_type=args.model_type,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        use_point_prompt=not args.no_point_prompt
    )
    
    # 处理所有图片
    image_paths = list(Path(args.test_images).glob("*.jpg"))
    print(f"找到 {len(image_paths)} 张图片")
    
    total_detections = 0
    images_with_detections = 0
    
    for image_path in image_paths:
        result = inferencer.predict(
            image_path,
            args.output_dir,
            enable_postprocess=not args.no_postprocess
        )
        
        if result:
            total_detections += result['num_detections']
            images_with_detections += 1
    
    print("\n" + "="*60)
    print("推理完成！")
    print("="*60)
    print(f"总图片数: {len(image_paths)}")
    print(f"检测到息肉的图片数: {images_with_detections}")
    print(f"总检测数: {total_detections}")
    print(f"平均每张图片检测数: {total_detections/len(image_paths):.2f}")
    print(f"\n结果保存在: {args.output_dir}")
    print("  - images/: 原图")
    print("  - bboxes/: bbox信息（JSON格式）")
    print("  - masks/: 分割mask（PNG格式）")
    print("  - visualizations/: 可视化结果")
    print("="*60)


if __name__ == "__main__":
    main()


