#!/usr/bin/env python3
"""
YOLO11n + TinySAM 推理脚本
在测试集上进行检测和分割，保存原图、bbox和mask
"""

import os
import sys
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# 添加路径
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRA = REPO_ROOT / "ultralyticss_new"
if LOCAL_ULTRA.exists() and str(LOCAL_ULTRA) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRA))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics not found")

from tinysam.build_sam import sam_model_registry
from tinysam.predictor import SamPredictor


class YOLOTinySAMInference:
    """YOLO检测 + TinySAM分割推理类"""
    
    def __init__(
        self,
        yolo_weights,
        sam_weights,
        sam_model_type="vit_t",
        device="cuda",
        yolo_conf=0.25,
        yolo_iou=0.45,
        use_point_prompt=True,
    ):
        """
        Args:
            yolo_weights: YOLO模型权重路径
            sam_weights: TinySAM模型权重路径
            sam_model_type: TinySAM模型类型
            device: 设备
            yolo_conf: YOLO置信度阈值
            yolo_iou: YOLO IoU阈值
            use_point_prompt: 是否使用中心点作为prompt
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.use_point_prompt = use_point_prompt
        
        # 加载YOLO模型
        if YOLO is None:
            raise ImportError("ultralytics package is required for YOLO detection")
        
        print(f"加载YOLO模型: {yolo_weights}")
        self.yolo_model = YOLO(yolo_weights)
        
        # 加载TinySAM模型
        print(f"加载TinySAM模型: {sam_weights}")
        self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_weights)
        self.sam_model.to(self.device)
        self.sam_predictor = SamPredictor(self.sam_model)
        
        print(f"模型加载完成，使用设备: {self.device}")
    
    @torch.no_grad()
    def predict(self, image_path):
        """
        对单张图片进行检测和分割
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: 包含原图、bbox列表、mask列表等
        """
        # 读取图片
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_bgr.shape[:2]
        
        # YOLO检测
        results = self.yolo_model(image_rgb, conf=self.yolo_conf, iou=self.yolo_iou, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id
                    })
        
        # TinySAM分割
        self.sam_predictor.set_image(image_rgb)
        
        masks = []
        mask_info = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # 使用bbox和中心点作为prompt
            if self.use_point_prompt:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                point_coords = np.array([[cx, cy]])
                point_labels = np.array([1])
            else:
                point_coords = None
                point_labels = None
            
            box = np.array([x1, y1, x2, y2])
            
            # 预测mask
            pred_masks, scores, logits = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
            )
            
            if len(pred_masks) > 0:
                mask = pred_masks[0].astype(np.uint8) * 255
                masks.append(mask)
                mask_info.append({
                    'bbox': det['bbox'],
                    'confidence': conf,
                    'mask_score': float(scores[0]) if len(scores) > 0 else 0.0,
                })
        
        # 合并所有mask（如果有多个检测）
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        if masks:
            combined_mask = np.maximum.reduce(masks)
        
        return {
            'image': image_bgr,
            'image_rgb': image_rgb,
            'detections': detections,
            'masks': masks,
            'combined_mask': combined_mask,
            'mask_info': mask_info,
            'image_size': (h, w),
        }
    
    def save_results(self, results, output_dir, image_name, save_individual_masks=True):
        """
        保存推理结果
        
        Args:
            results: predict()返回的结果字典
            output_dir: 输出目录
            image_name: 图片名称（不含扩展名）
            save_individual_masks: 是否保存每个检测的独立mask
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存原图
        original_dir = output_path / "images"
        original_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(original_dir / f"{image_name}.jpg"), results['image'])
        
        # 2. 保存bbox信息（JSON格式）
        bbox_dir = output_path / "bboxes"
        bbox_dir.mkdir(exist_ok=True)
        bbox_info = {
            'image_name': image_name,
            'image_size': results['image_size'],
            'num_detections': len(results['detections']),
            'detections': results['detections'],
        }
        with open(bbox_dir / f"{image_name}.json", 'w') as f:
            json.dump(bbox_info, f, indent=2)
        
        # 3. 保存mask
        mask_dir = output_path / "masks"
        mask_dir.mkdir(exist_ok=True)
        
        # 保存合并的mask
        cv2.imwrite(str(mask_dir / f"{image_name}_combined.png"), results['combined_mask'])
        
        # 保存每个检测的独立mask
        if save_individual_masks and len(results['masks']) > 0:
            individual_dir = mask_dir / "individual"
            individual_dir.mkdir(exist_ok=True)
            for i, mask in enumerate(results['masks']):
                cv2.imwrite(str(individual_dir / f"{image_name}_mask_{i}.png"), mask)
        
        # 4. 保存可视化结果（可选）
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        vis_image = results['image'].copy()
        
        # 绘制bbox
        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"{conf:.2f}", (x1, max(0, y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # 叠加mask
        if results['combined_mask'].sum() > 0:
            mask_colored = np.zeros_like(vis_image)
            mask_colored[:, :, 2] = results['combined_mask']  # 红色通道
            vis_image = cv2.addWeighted(vis_image, 1.0, mask_colored, 0.4, 0)
        
        cv2.imwrite(str(vis_dir / f"{image_name}_vis.jpg"), vis_image)
    
    def evaluate_on_dataset(
        self,
        test_images_dir,
        test_masks_dir=None,
        output_dir="inference_results",
        save_individual_masks=True,
    ):
        """
        在测试集上进行推理和评估
        
        Args:
            test_images_dir: 测试图片目录
            test_masks_dir: 测试mask目录（可选，用于计算指标）
            output_dir: 输出目录
            save_individual_masks: 是否保存每个检测的独立mask
        """
        test_images_path = Path(test_images_dir)
        image_files = sorted(list(test_images_path.glob("*.jpg")))
        
        if len(image_files) == 0:
            print(f"未找到图片文件在: {test_images_dir}")
            return
        
        print(f"开始推理，共 {len(image_files)} 张图片...")
        
        # 统计信息
        total_detections = 0
        total_images_with_detections = 0
        
        # 处理每张图片
        for img_file in tqdm(image_files, desc="推理中"):
            try:
                # 推理
                results = self.predict(img_file)
                
                # 保存结果
                image_name = img_file.stem
                self.save_results(results, output_dir, image_name, save_individual_masks)
                
                # 统计
                num_detections = len(results['detections'])
                total_detections += num_detections
                if num_detections > 0:
                    total_images_with_detections += 1
                    
            except Exception as e:
                print(f"处理图片 {img_file.name} 时出错: {e}")
                continue
        
        # 打印统计信息
        print("\n" + "="*60)
        print("推理完成！")
        print("="*60)
        print(f"总图片数: {len(image_files)}")
        print(f"检测到息肉的图片数: {total_images_with_detections}")
        print(f"总检测数: {total_detections}")
        print(f"平均每张图片检测数: {total_detections / len(image_files):.2f}")
        print(f"\n结果保存在: {output_dir}")
        print("  - images/: 原图")
        print("  - bboxes/: bbox信息（JSON格式）")
        print("  - masks/: 分割mask（PNG格式）")
        print("  - visualizations/: 可视化结果")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="YOLO11n + TinySAM 推理脚本")
    parser.add_argument("--yolo-weights", type=str, required=True,
                       help="YOLO模型权重路径")
    parser.add_argument("--sam-weights", type=str, required=True,
                       help="TinySAM模型权重路径")
    parser.add_argument("--sam-type", type=str, default="vit_t",
                       help="TinySAM模型类型")
    parser.add_argument("--test-images", type=str, required=True,
                       help="测试图片目录")
    parser.add_argument("--test-masks", type=str, default=None,
                       help="测试mask目录（可选，用于评估）")
    parser.add_argument("--output-dir", type=str, default="inference_results",
                       help="输出目录")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                       help="YOLO置信度阈值")
    parser.add_argument("--yolo-iou", type=float, default=0.45,
                       help="YOLO IoU阈值")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备")
    parser.add_argument("--no-point-prompt", action="store_true",
                       help="不使用中心点prompt，仅使用bbox")
    parser.add_argument("--no-individual-masks", action="store_true",
                       help="不保存每个检测的独立mask")
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = YOLOTinySAMInference(
        yolo_weights=args.yolo_weights,
        sam_weights=args.sam_weights,
        sam_model_type=args.sam_type,
        device=args.device,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        use_point_prompt=not args.no_point_prompt,
    )
    
    # 在测试集上推理
    inferencer.evaluate_on_dataset(
        test_images_dir=args.test_images,
        test_masks_dir=args.test_masks,
        output_dir=args.output_dir,
        save_individual_masks=not args.no_individual_masks,
    )


if __name__ == "__main__":
    main()

