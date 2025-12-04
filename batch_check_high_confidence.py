"""
批量检查高置信度图片，帮助识别假阳性
生成一个报告，列出所有高置信度的图片供人工审查
"""
from ultralytics import YOLO
import cv2
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

def batch_check_high_confidence(
    frames_dir: str,
    model_weights: str,
    conf_threshold: float = 0.7,
    min_confidence: float = 0.7,
):
    """
    批量检查高置信度图片
    
    Args:
        frames_dir: 帧目录路径
        model_weights: YOLO模型权重路径
        conf_threshold: 检测时使用的置信度阈值
        min_confidence: 要报告的最低置信度（只报告高于此值的图片）
    """
    print("=" * 80)
    print("批量检查高置信度图片")
    print("=" * 80)
    print(f"帧目录: {frames_dir}")
    print(f"模型权重: {model_weights}")
    print(f"检测阈值: {conf_threshold}")
    print(f"报告最低置信度: {min_confidence}")
    print("=" * 80)
    
    frames_path = Path(frames_dir)
    polyp_dir = frames_path / "有息肉"
    
    if not polyp_dir.exists():
        print(f"错误: 有息肉目录不存在: {polyp_dir}")
        return
    
    # 加载模型
    print("\n加载YOLO模型...")
    model = YOLO(model_weights)
    
    # 获取所有图片
    polyp_files = sorted(list(polyp_dir.glob("*.jpg")))
    print(f"\n检查 {len(polyp_files)} 张图片...")
    
    high_conf_files = []
    
    with tqdm(total=len(polyp_files), desc="检查图片") as pbar:
        for img_path in polyp_files:
            img = cv2.imread(str(img_path))
            if img is None:
                pbar.update(1)
                continue
            
            # 使用阈值检测
            results = model(img, conf=conf_threshold, verbose=False)
            
            max_conf = 0.0
            num_detections = 0
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    confidences = [float(box.conf) for box in boxes]
                    max_conf = max(confidences)
                    num_detections = len(boxes)
            
            # 只记录高置信度的图片
            if max_conf >= min_confidence:
                high_conf_files.append({
                    "file": img_path.name,
                    "confidence": max_conf,
                    "num_detections": num_detections,
                })
            
            pbar.update(1)
    
    # 按置信度排序
    high_conf_files.sort(key=lambda x: x["confidence"], reverse=True)
    
    # 生成报告
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_checked": len(polyp_files),
        "high_confidence_count": len(high_conf_files),
        "min_confidence_threshold": min_confidence,
        "files": high_conf_files,
    }
    
    # 保存报告
    report_path = frames_path / f"high_confidence_review_report_{min_confidence}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("检查结果")
    print("=" * 80)
    print(f"总检查数: {len(polyp_files)}")
    print(f"高置信度图片 (>= {min_confidence}): {len(high_conf_files)}")
    
    if high_conf_files:
        import numpy as np
        confidences = [f["confidence"] for f in high_conf_files]
        print(f"\n置信度统计:")
        print(f"  平均: {np.mean(confidences):.3f}")
        print(f"  中位数: {np.median(confidences):.3f}")
        print(f"  最小值: {np.min(confidences):.3f}")
        print(f"  最大值: {np.max(confidences):.3f}")
        
        print(f"\n最高置信度的10张图片（建议优先审查）:")
        for i, file_info in enumerate(high_conf_files[:10], 1):
            print(f"  {i}. {file_info['file']}: {file_info['confidence']:.3f}")
    
    print(f"\n详细报告已保存到: {report_path}")
    print("=" * 80)
    
    return report


if __name__ == "__main__":
    # 配置参数
    frames_dir = "钟木英230923_frames"
    model_weights = "runs/detect/yolo11n_all_coco6/weights/best.pt"
    conf_threshold = 0.6  # 检测时使用的阈值
    min_confidence = 0.7  # 只报告置信度 >= 0.7 的图片
    
    # 运行批量检查
    batch_check_high_confidence(
        frames_dir=frames_dir,
        model_weights=model_weights,
        conf_threshold=conf_threshold,
        min_confidence=min_confidence,
    )

