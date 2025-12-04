#!/usr/bin/env python3
"""
分批处理Polys测试集验证 - 避免内存和时间问题
"""

import cv2
import torch
import numpy as np
import time
from pathlib import Path
from tinysam.build_sam import sam_model_registry
from tinysam.predictor import SamPredictor
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import json
import os
import gc

def load_tinysam_model():
    """加载TinySAM模型"""
    print("🚀 Loading TinySAM model...")
    
    # 使用results_ldpolyvideo目录下的权重文件
    tinysam_weights = "results_ldpolyvideo/checkpoint_epoch_100.pth"
    if not os.path.exists(tinysam_weights):
        print(f"❌ TinySAM weights not found at {tinysam_weights}")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    try:
        model = sam_model_registry["vit_t"](checkpoint=tinysam_weights).to(device)
        predictor = SamPredictor(model)
        print(f"✅ TinySAM model loaded from: {tinysam_weights}")
        return predictor
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def find_test_images(test_images_dir):
    """查找测试图像文件"""
    image_files = []
    for video_dir in Path(test_images_dir).iterdir():
        if video_dir.is_dir():
            images = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
            image_files.extend(images)
    return image_files

def find_mask_file(image_path, annotations_dir):
    """查找对应的掩码文件"""
    video_id = image_path.parent.name
    mask_path = Path(annotations_dir) / video_id / f"{image_path.stem}.png"
    return mask_path if mask_path.exists() else None

def test_single_image(predictor, image_path, mask_path=None):
    """测试单张图像"""
    try:
        # 加载图像
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 设置图像
        predictor.set_image(image_rgb)
        
        # 使用图像中心点进行分割
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        start_time = time.time()
        masks, _, _ = predictor.predict(
            point_coords=np.array([[center_x, center_y]]),
            point_labels=np.array([1])
        )
        inference_time = time.time() - start_time
        
        if len(masks) == 0:
            return None
        
        pred_mask = masks[0]
        
        # 计算指标
        metrics = {}
        if mask_path and mask_path.exists():
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask = (gt_mask > 128).astype(np.uint8)
                
                # 调整预测掩码大小
                pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]))
                
                # 计算指标
                pred_flat = pred_mask_resized.flatten()
                gt_flat = gt_mask.flatten()
                
                metrics = {
                    'dice': f1_score(gt_flat, pred_flat, zero_division=0),
                    'iou': jaccard_score(gt_flat, pred_flat, zero_division=0),
                    'precision': precision_score(gt_flat, pred_flat, zero_division=0),
                    'recall': recall_score(gt_flat, pred_flat, zero_division=0)
                }
        
        return {
            'inference_time': inference_time,
            'has_prediction': True,
            'metrics': metrics
        }
    except Exception as e:
        print(f"❌ Error processing {image_path.name}: {e}")
        return None

def process_batch(predictor, batch_images, annotations_dir, batch_num, total_batches):
    """处理一批图像"""
    print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch_images)} images)")
    
    batch_results = []
    batch_time = 0
    processed_count = 0
    
    for i, img_path in enumerate(batch_images):
        # 显示进度
        if (i + 1) % 10 == 0 or i == len(batch_images) - 1:
            print(f"  Progress: {i+1}/{len(batch_images)} - {img_path.name}")
        
        # 查找掩码文件
        mask_path = None
        if os.path.exists(annotations_dir):
            mask_path = find_mask_file(img_path, annotations_dir)
        
        # 测试图像
        result = test_single_image(predictor, img_path, mask_path)
        if result is not None:
            batch_results.append({
                'image': str(img_path),
                'mask': str(mask_path) if mask_path else None,
                'result': result
            })
            batch_time += result['inference_time']
            processed_count += 1
    
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return batch_results, batch_time, processed_count

def main():
    """主函数"""
    print("🧪 Batch Polys Test Set Validation with TinySAM")
    print("="*60)
    
    # 加载模型
    predictor = load_tinysam_model()
    if predictor is None:
        return
    
    # 设置路径
    test_images_dir = "Polys/Test/Images"
    test_annotations_dir = "Polys/Test/Annotations"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found: {test_images_dir}")
        return
    
    # 查找测试图像
    image_files = find_test_images(test_images_dir)
    print(f"📊 Found {len(image_files)} test images")
    
    if len(image_files) == 0:
        print("❌ No test images found")
        return
    
    # 分批处理参数
    batch_size = 100  # 每批处理100张图像
    max_total_images = 1000  # 最多处理1000张图像（可以修改）
    
    # 限制总测试数量
    if len(image_files) > max_total_images:
        image_files = image_files[:max_total_images]
        print(f"⚠️  Limiting to {max_total_images} images for testing")
    
    # 分批处理
    all_results = []
    total_time = 0
    total_processed = 0
    
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    
    print(f"🔬 Testing {len(image_files)} images in {num_batches} batches (batch size: {batch_size})")
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(image_files))
        batch_images = image_files[start_idx:end_idx]
        
        # 处理当前批次
        batch_results, batch_time, batch_processed = process_batch(
            predictor, batch_images, test_annotations_dir, batch_num + 1, num_batches
        )
        
        all_results.extend(batch_results)
        total_time += batch_time
        total_processed += batch_processed
        
        # 显示批次结果
        print(f"  ✅ Batch {batch_num + 1} completed: {batch_processed}/{len(batch_images)} images processed")
        print(f"  ⏱️  Batch time: {batch_time:.2f}s, Avg: {batch_time/len(batch_images):.3f}s/image")
        
        # 保存中间结果
        if (batch_num + 1) % 5 == 0 or batch_num == num_batches - 1:
            intermediate_report = {
                'summary': {
                    'total_batches': num_batches,
                    'completed_batches': batch_num + 1,
                    'processed_images': total_processed,
                    'total_time': total_time,
                    'avg_time_per_image': total_time / total_processed if total_processed > 0 else 0
                }
            }
            
            with open("intermediate_polys_test_results.json", 'w') as f:
                json.dump(intermediate_report, f, indent=2, default=str)
            
            print(f"  💾 Intermediate results saved")
    
    # 生成最终报告
    if total_processed > 0:
        # 计算平均指标
        metrics_list = [r['result']['metrics'] for r in all_results if r['result']['metrics']]
        
        if metrics_list:
            avg_dice = np.mean([m['dice'] for m in metrics_list])
            avg_iou = np.mean([m['iou'] for m in metrics_list])
            avg_precision = np.mean([m['precision'] for m in metrics_list])
            avg_recall = np.mean([m['recall'] for m in metrics_list])
        else:
            avg_dice = avg_iou = avg_precision = avg_recall = 0
        
        avg_time = total_time / total_processed
        
        # 打印最终结果
        print("\n" + "="*60)
        print("📊 FINAL TEST RESULTS")
        print("="*60)
        print(f"Total Images: {len(image_files)}")
        print(f"Processed Images: {total_processed}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Avg Time/Image: {avg_time:.3f}s")
        print(f"Avg Dice: {avg_dice:.3f}")
        print(f"Avg IoU: {avg_iou:.3f}")
        print(f"Avg Precision: {avg_precision:.3f}")
        print(f"Avg Recall: {avg_recall:.3f}")
        print("="*60)
        
        # 保存最终结果
        final_report = {
            'summary': {
                'total_images': len(image_files),
                'processed_images': total_processed,
                'total_time': total_time,
                'avg_time_per_image': avg_time,
                'avg_dice': avg_dice,
                'avg_iou': avg_iou,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall
            },
            'detailed_results': all_results
        }
        
        with open("final_polys_test_results.json", 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print("💾 Final results saved to: final_polys_test_results.json")
    else:
        print("❌ No images were successfully processed")

if __name__ == "__main__":
    main()
