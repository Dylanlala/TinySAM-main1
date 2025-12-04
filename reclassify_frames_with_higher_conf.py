"""
重新分类已提取的帧，使用更高的置信度阈值来过滤误检
"""
import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import shutil
from datetime import datetime

def reclassify_frames(
    frames_dir: str,
    model_weights: str,
    conf_threshold: float = 0.5,
    move_files: bool = True,
):
    """
    重新分类已提取的帧
    
    Args:
        frames_dir: 帧目录路径
        model_weights: YOLO模型权重路径
        conf_threshold: 新的置信度阈值
        move_files: 是否移动文件（True）或只报告（False）
    """
    print("=" * 80)
    print("重新分类帧 - 使用更高置信度阈值")
    print("=" * 80)
    print(f"帧目录: {frames_dir}")
    print(f"模型权重: {model_weights}")
    print(f"新置信度阈值: {conf_threshold}")
    print("=" * 80)
    
    frames_path = Path(frames_dir)
    polyp_dir = frames_path / "有息肉"
    no_polyp_dir = frames_path / "无息肉"
    
    if not polyp_dir.exists():
        print(f"错误: 有息肉目录不存在: {polyp_dir}")
        return
    
    if not no_polyp_dir.exists():
        print(f"错误: 无息肉目录不存在: {no_polyp_dir}")
        return
    
    # 加载模型
    print("\n加载YOLO模型...")
    model = YOLO(model_weights)
    
    # 检查有息肉目录中的所有图片
    polyp_files = list(polyp_dir.glob("*.jpg"))
    print(f"\n检查有息肉目录中的 {len(polyp_files)} 张图片...")
    
    stats = {
        "total_checked": len(polyp_files),
        "confirmed_polyp": 0,  # 确认有息肉
        "moved_to_no_polyp": 0,  # 移动到无息肉目录
        "low_confidence": 0,  # 低置信度检测
    }
    
    files_to_move = []
    
    with tqdm(total=len(polyp_files), desc="检查图片") as pbar:
        for img_path in polyp_files:
            img = cv2.imread(str(img_path))
            if img is None:
                pbar.update(1)
                continue
            
            # 使用新的置信度阈值进行检测
            results = model(img, conf=conf_threshold, verbose=False)
            
            has_polyp = False
            max_conf = 0.0
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    has_polyp = True
                    confidences = [float(box.conf) for box in boxes]
                    max_conf = max(confidences)
            
            if has_polyp:
                stats["confirmed_polyp"] += 1
                if max_conf < 0.5:
                    stats["low_confidence"] += 1
            else:
                # 没有检测到息肉，应该移动到无息肉目录
                files_to_move.append(img_path)
                stats["moved_to_no_polyp"] += 1
            
            pbar.update(1)
    
    # 移动文件
    if move_files and files_to_move:
        print(f"\n移动 {len(files_to_move)} 张误检图片到无息肉目录...")
        for img_path in tqdm(files_to_move, desc="移动文件"):
            new_path = no_polyp_dir / img_path.name.replace("_det", "")
            shutil.move(str(img_path), str(new_path))
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("重新分类结果")
    print("=" * 80)
    print(f"检查总数: {stats['total_checked']}")
    print(f"确认有息肉: {stats['confirmed_polyp']} ({stats['confirmed_polyp']/stats['total_checked']*100:.2f}%)")
    print(f"移动到无息肉: {stats['moved_to_no_polyp']} ({stats['moved_to_no_polyp']/stats['total_checked']*100:.2f}%)")
    if stats['low_confidence'] > 0:
        print(f"低置信度(<0.5): {stats['low_confidence']} ({stats['low_confidence']/stats['confirmed_polyp']*100:.2f}%)")
    print("=" * 80)
    
    # 更新统计文件
    stats_file = frames_path / "stats.txt"
    if stats_file.exists():
        with open(stats_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n重新分类记录 (置信度阈值={conf_threshold}):\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"确认有息肉: {stats['confirmed_polyp']}\n")
            f.write(f"移动到无息肉: {stats['moved_to_no_polyp']}\n")
    
    return stats


if __name__ == "__main__":
    # 配置参数
    frames_dir = "钟木英230923_frames"
    model_weights = "runs/detect/yolo11n_all_coco6/weights/best.pt"
    conf_threshold = 0.6  # 进一步提高置信度阈值到0.6，更严格过滤误检
    
    # 运行重新分类
    reclassify_frames(
        frames_dir=frames_dir,
        model_weights=model_weights,
        conf_threshold=conf_threshold,
        move_files=True,  # 设置为False可以先预览，不移动文件
    )

