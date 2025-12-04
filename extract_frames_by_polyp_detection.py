"""
从视频中提取帧，根据息肉检测结果分类保存
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime

def extract_frames_by_detection(
    video_path: str,
    model_weights: str,
    output_dir: str = None,
    conf_threshold: float = 0.4,
    frame_interval: int = 1,
    save_with_bbox: bool = False,
):
    """
    从视频中提取帧，根据检测结果分类保存
    
    Args:
        video_path: 视频文件路径
        model_weights: YOLO模型权重路径
        output_dir: 输出目录，如果为None则使用视频文件名
        conf_threshold: 检测置信度阈值
        frame_interval: 帧间隔（1表示每帧都处理，2表示每隔一帧处理）
        save_with_bbox: 是否在保存的帧上绘制检测框
    """
    print("=" * 80)
    print("视频帧提取 - 基于息肉检测")
    print("=" * 80)
    print(f"视频路径: {video_path}")
    print(f"模型权重: {model_weights}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"帧间隔: {frame_interval}")
    print("=" * 80)
    
    # 检查文件是否存在
    if not Path(video_path).exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    if not Path(model_weights).exists():
        print(f"错误: 模型权重文件不存在: {model_weights}")
        return
    
    # 创建输出目录
    if output_dir is None:
        video_name = Path(video_path).stem
        output_dir = f"{video_name}_frames"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 创建分类目录
    polyp_dir = output_path / "有息肉"
    no_polyp_dir = output_path / "无息肉"
    polyp_dir.mkdir(exist_ok=True)
    no_polyp_dir.mkdir(exist_ok=True)
    
    print(f"\n输出目录: {output_path}")
    print(f"  有息肉: {polyp_dir}")
    print(f"  无息肉: {no_polyp_dir}")
    
    # 加载模型
    print("\n加载YOLO模型...")
    model = YOLO(model_weights)
    
    # 打开视频
    print(f"\n打开视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"  总帧数: {total_frames}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  分辨率: {width}x{height}")
    print(f"  预计处理帧数: {total_frames // frame_interval}")
    
    # 统计信息
    stats = {
        "total_processed": 0,
        "with_polyp": 0,
        "without_polyp": 0,
    }
    
    # 处理帧
    print("\n开始处理视频帧...")
    frame_idx = 0
    processed_idx = 0
    
    with tqdm(total=total_frames // frame_interval, desc="处理帧") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 根据帧间隔决定是否处理
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            # 进行检测
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # 检查是否有检测结果
            has_polyp = False
            detections = []
            annotated_frame = frame.copy()  # 默认使用原始帧
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    has_polyp = True
                    detections = boxes
                    
                    # 如果需要在图像上绘制检测框
                    if save_with_bbox:
                        annotated_frame = results[0].plot()
                    # else: annotated_frame 已经是 frame.copy()
            
            # 保存帧
            if has_polyp:
                save_dir = polyp_dir
                stats["with_polyp"] += 1
            else:
                save_dir = no_polyp_dir
                stats["without_polyp"] += 1
            
            # 生成文件名（包含帧序号和检测信息）
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            if has_polyp:
                # 添加检测数量信息
                num_detections = len(detections)
                frame_filename = f"frame_{frame_idx:06d}_det{num_detections}.jpg"
            
            save_path = save_dir / frame_filename
            cv2.imwrite(str(save_path), annotated_frame)
            
            stats["total_processed"] += 1
            processed_idx += 1
            frame_idx += 1
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "有息肉": stats["with_polyp"],
                "无息肉": stats["without_polyp"]
            })
    
    cap.release()
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("处理完成!")
    print("=" * 80)
    print(f"总处理帧数: {stats['total_processed']}")
    print(f"有息肉帧数: {stats['with_polyp']} ({stats['with_polyp']/stats['total_processed']*100:.2f}%)")
    print(f"无息肉帧数: {stats['without_polyp']} ({stats['without_polyp']/stats['total_processed']*100:.2f}%)")
    print(f"\n输出目录: {output_path}")
    print(f"  有息肉: {polyp_dir} ({stats['with_polyp']} 帧)")
    print(f"  无息肉: {no_polyp_dir} ({stats['without_polyp']} 帧)")
    print("=" * 80)
    
    # 保存统计信息
    stats_file = output_path / "stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"视频处理统计信息\n")
        f.write(f"{'='*80}\n")
        f.write(f"视频路径: {video_path}\n")
        f.write(f"模型权重: {model_weights}\n")
        f.write(f"置信度阈值: {conf_threshold}\n")
        f.write(f"帧间隔: {frame_interval}\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n统计结果:\n")
        f.write(f"  总处理帧数: {stats['total_processed']}\n")
        f.write(f"  有息肉帧数: {stats['with_polyp']} ({stats['with_polyp']/stats['total_processed']*100:.2f}%)\n")
        f.write(f"  无息肉帧数: {stats['without_polyp']} ({stats['without_polyp']/stats['total_processed']*100:.2f}%)\n")
    
    print(f"\n统计信息已保存到: {stats_file}")


if __name__ == "__main__":
    # 配置参数
    video_path = "钟木英230923.TS"
    model_weights = "runs/detect/yolo11n_all_coco6/weights/best.pt"
    conf_threshold = 0.8  # 检测置信度阈值
    frame_interval = 1  # 处理每一帧（设为2则每隔一帧处理一次）
    save_with_bbox = True  # 是否在保存的帧上绘制检测框
    
    # 运行提取
    extract_frames_by_detection(
        video_path=video_path,
        model_weights=model_weights,
        conf_threshold=conf_threshold,
        frame_interval=frame_interval,
        save_with_bbox=save_with_bbox,
    )

