"""
使用更高置信度阈值严格重新分类，清理假阳性
"""
import cv2
from pathlib import Path
from ultralytics import YOLO
import shutil
from datetime import datetime
from tqdm import tqdm

def strict_reclassify(
    frames_dir: str,
    model_weights: str,
    conf_threshold: float = 0.75,
):
    """
    使用高置信度阈值严格重新分类
    
    Args:
        frames_dir: 帧目录路径
        model_weights: YOLO模型权重路径
        conf_threshold: 高置信度阈值（建议0.75或更高）
    """
    print("=" * 80)
    print("严格重新分类 - 清理假阳性")
    print("=" * 80)
    print(f"帧目录: {frames_dir}")
    print(f"模型权重: {model_weights}")
    print(f"置信度阈值: {conf_threshold} (高阈值，更严格)")
    print("=" * 80)
    
    frames_path = Path(frames_dir)
    polyp_dir = frames_path / "有息肉"
    no_polyp_dir = frames_path / "无息肉"
    
    if not polyp_dir.exists():
        print(f"错误: 有息肉目录不存在: {polyp_dir}")
        return
    
    if not no_polyp_dir.exists():
        no_polyp_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("\n加载YOLO模型...")
    model = YOLO(model_weights)
    
    # 获取所有图片
    polyp_files = sorted(list(polyp_dir.glob("*.jpg")))
    print(f"\n检查 {len(polyp_files)} 张图片...")
    
    stats = {
        "total": len(polyp_files),
        "confirmed": 0,  # 确认有息肉（通过高阈值）
        "moved": 0,  # 移动到无息肉目录
        "confidences": [],
    }
    
    files_to_move = []
    
    with tqdm(total=len(polyp_files), desc="检查图片") as pbar:
        for img_path in polyp_files:
            img = cv2.imread(str(img_path))
            if img is None:
                pbar.update(1)
                continue
            
            # 使用高置信度阈值检测
            results = model(img, conf=conf_threshold, verbose=False)
            
            has_polyp = False
            max_conf = 0.0
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    has_polyp = True
                    confidences = [float(box.conf) for box in boxes]
                    max_conf = max(confidences)
                    stats["confidences"].append(max_conf)
            
            if has_polyp:
                stats["confirmed"] += 1
            else:
                # 没有通过高阈值检测，移动到无息肉目录
                files_to_move.append(img_path)
                stats["moved"] += 1
            
            pbar.update(1)
    
    # 移动文件
    if files_to_move:
        print(f"\n移动 {len(files_to_move)} 张未通过高阈值检测的图片到无息肉目录...")
        for img_path in tqdm(files_to_move, desc="移动文件"):
            # 移除文件名中的 _det 部分
            new_filename = img_path.name.replace("_det", "")
            new_path = no_polyp_dir / new_filename
            shutil.move(str(img_path), str(new_path))
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("重新分类结果")
    print("=" * 80)
    print(f"检查总数: {stats['total']}")
    print(f"确认有息肉 (>= {conf_threshold}): {stats['confirmed']} ({stats['confirmed']/stats['total']*100:.2f}%)")
    print(f"移动到无息肉: {stats['moved']} ({stats['moved']/stats['total']*100:.2f}%)")
    
    if stats["confidences"]:
        import numpy as np
        confidences = np.array(stats["confidences"])
        print(f"\n确认有息肉图片的置信度统计:")
        print(f"  平均: {confidences.mean():.3f}")
        print(f"  中位数: {np.median(confidences):.3f}")
        print(f"  最小值: {confidences.min():.3f}")
        print(f"  最大值: {confidences.max():.3f}")
    
    print("=" * 80)
    
    # 更新统计文件
    stats_file = frames_path / "stats.txt"
    if stats_file.exists():
        with open(stats_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n严格重新分类记录 (置信度阈值={conf_threshold}):\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"确认有息肉: {stats['confirmed']}\n")
            f.write(f"移动到无息肉: {stats['moved']}\n")
    
    return stats


if __name__ == "__main__":
    # 配置参数
    frames_dir = "钟木英230923_frames"
    model_weights = "runs/detect/yolo11n_all_coco6/weights/best.pt"
    
    # 使用高置信度阈值（0.75或更高）
    # 可以根据实际情况调整：
    # - 0.75: 较严格，会过滤掉更多假阳性
    # - 0.80: 很严格，只保留高置信度检测
    # - 0.85: 非常严格，只保留极高置信度检测
    conf_threshold = 0.75
    
    print(f"警告: 这将使用 {conf_threshold} 的高置信度阈值重新分类")
    print(f"预计会移动更多图片到无息肉目录")
    print(f"如果确定要继续，请修改脚本中的 conf_threshold 参数")
    
    # 运行严格重新分类
    strict_reclassify(
        frames_dir=frames_dir,
        model_weights=model_weights,
        conf_threshold=conf_threshold,
    )

