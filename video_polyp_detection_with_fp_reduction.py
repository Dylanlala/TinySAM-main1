"""
内窥镜视频息肉检测 - 带假阳性减少策略
使用时间连续性、多帧验证等策略减少假阳性
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import deque
from tqdm import tqdm
from datetime import datetime

class VideoPolypDetector:
    def __init__(
        self,
        model_weights: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        # 假阳性减少策略
        temporal_consistency: bool = True,  # 时间连续性检查
        min_consecutive_frames: int = 3,  # 最少连续帧数
        temporal_window: int = 5,  # 时间窗口大小
        min_detections_in_window: int = 3,  # 窗口内最少检测次数
        stability_check: bool = True,  # 稳定性检查
        min_stable_frames: int = 5,  # 稳定检测的最少帧数
        show_bbox: bool = False,  # 是否在保存的帧上绘制检测框
    ):
        """
        初始化视频息肉检测器
        
        Args:
            model_weights: YOLO模型权重路径
            conf_threshold: 基础置信度阈值
            iou_threshold: IoU阈值
            temporal_consistency: 是否启用时间连续性检查
            min_consecutive_frames: 最少连续帧数（连续检测到才认为是息肉）
            temporal_window: 时间窗口大小（滑动窗口）
            min_detections_in_window: 窗口内最少检测次数
            stability_check: 是否启用稳定性检查
            min_stable_frames: 稳定检测的最少帧数
            show_bbox: 是否在保存的帧上绘制检测框（默认False）
        """
        self.model = YOLO(model_weights)
        self.model_weights = model_weights  # 保存权重路径
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 假阳性减少策略参数
        self.temporal_consistency = temporal_consistency
        self.min_consecutive_frames = min_consecutive_frames
        self.temporal_window = temporal_window
        self.min_detections_in_window = min_detections_in_window
        self.stability_check = stability_check
        self.min_stable_frames = min_stable_frames
        
        # 显示控制
        self.show_bbox = show_bbox
        
        # 时间连续性跟踪
        self.detection_history = deque(maxlen=temporal_window)  # 检测历史
        self.consecutive_count = 0  # 连续检测计数
        self.stable_detections = []  # 稳定检测记录
        
    def detect_frame(self, frame):
        """
        检测单帧
        
        Returns:
            detections: 检测结果列表，每个元素为 (bbox, conf, class_id)
            has_polyp: 是否检测到息肉
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        has_polyp = False
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if len(boxes) > 0:
                has_polyp = True
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    cls = int(box.cls)
                    detections.append({
                        'bbox': xyxy,
                        'confidence': conf,
                        'class_id': cls,
                    })
        
        return detections, has_polyp
    
    def apply_temporal_filter(self, has_polyp, detections):
        """
        应用时间连续性过滤
        
        Args:
            has_polyp: 当前帧是否检测到息肉
            detections: 当前帧的检测结果
            
        Returns:
            filtered_has_polyp: 过滤后是否认为有息肉
            filtered_detections: 过滤后的检测结果
        """
        # 记录当前检测
        self.detection_history.append({
            'has_polyp': has_polyp,
            'detections': detections,
            'timestamp': len(self.detection_history)
        })
        
        if not self.temporal_consistency:
            return has_polyp, detections
        
        # 策略1: 连续帧检查
        if has_polyp:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0
        
        # 策略2: 滑动窗口检查
        window_detections = sum(1 for d in self.detection_history if d['has_polyp'])
        
        # 策略3: 稳定性检查
        if has_polyp and len(self.detection_history) >= self.min_stable_frames:
            recent_detections = list(self.detection_history)[-self.min_stable_frames:]
            stable_count = sum(1 for d in recent_detections if d['has_polyp'])
            is_stable = stable_count >= self.min_stable_frames * 0.6  # 至少60%的帧有检测
        else:
            is_stable = False
        
        # 综合判断
        filtered_has_polyp = False
        
        # 条件1: 连续检测
        if self.consecutive_count >= self.min_consecutive_frames:
            filtered_has_polyp = True
        
        # 条件2: 窗口内检测次数
        if window_detections >= self.min_detections_in_window:
            filtered_has_polyp = True
        
        # 条件3: 稳定性检查
        if self.stability_check and is_stable:
            filtered_has_polyp = True
        
        # 如果过滤后认为有息肉，返回检测结果
        if filtered_has_polyp:
            # 使用窗口内平均置信度最高的检测
            all_detections = []
            for d in self.detection_history:
                if d['has_polyp']:
                    all_detections.extend(d['detections'])
            
            if all_detections:
                # 选择置信度最高的检测
                best_detection = max(all_detections, key=lambda x: x['confidence'])
                filtered_detections = [best_detection]
            else:
                filtered_detections = detections if detections else []
        else:
            filtered_detections = []
        
        return filtered_has_polyp, filtered_detections
    
    def process_video(
        self,
        video_path: str,
        output_dir: str = None,
        save_frames: bool = True,
        save_with_bbox: bool = None,  # None表示使用类初始化时的设置
    ):
        """
        处理视频，提取有息肉的帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            save_frames: 是否保存帧
            save_with_bbox: 是否在保存的帧上绘制检测框
                          None=使用类初始化时的show_bbox设置
                          True/False=覆盖类初始化设置
        """
        print("=" * 80)
        print("内窥镜视频息肉检测 - 带假阳性减少")
        print("=" * 80)
        print(f"视频路径: {video_path}")
        print(f"模型权重: {self.model_weights}")
        print(f"置信度阈值: {self.conf_threshold}")
        print(f"\n假阳性减少策略:")
        print(f"  时间连续性: {self.temporal_consistency}")
        print(f"  最少连续帧: {self.min_consecutive_frames}")
        print(f"  时间窗口: {self.temporal_window}")
        print(f"  窗口内最少检测: {self.min_detections_in_window}")
        print(f"  稳定性检查: {self.stability_check}")
        print(f"  显示检测框: {save_with_bbox if save_with_bbox is not None else self.show_bbox}")
        print("=" * 80)
        
        # 确定是否显示检测框
        if save_with_bbox is None:
            save_with_bbox = self.show_bbox
        
        # 创建输出目录
        if output_dir is None:
            video_name = Path(video_path).stem
            output_dir = f"{video_name}_frames_filtered"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        polyp_dir = output_path / "有息肉"
        no_polyp_dir = output_path / "无息肉"
        polyp_dir.mkdir(exist_ok=True)
        no_polyp_dir.mkdir(exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n视频信息:")
        print(f"  总帧数: {total_frames}")
        print(f"  帧率: {fps:.2f} FPS")
        
        # 统计信息
        stats = {
            "total_processed": 0,
            "raw_detections": 0,  # 原始检测到的帧数
            "filtered_detections": 0,  # 过滤后的帧数
            "false_positives_filtered": 0,  # 被过滤的假阳性
        }
        
        # 重置状态（处理新视频时）
        self.consecutive_count = 0
        self.detection_history.clear()
        
        # 处理帧
        print(f"\n开始处理视频...")
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="处理帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测
                detections, has_polyp = self.detect_frame(frame)
                
                # 应用时间过滤
                filtered_has_polyp, filtered_detections = self.apply_temporal_filter(
                    has_polyp, detections
                )
                
                # 统计
                stats["total_processed"] += 1
                if has_polyp:
                    stats["raw_detections"] += 1
                if filtered_has_polyp:
                    stats["filtered_detections"] += 1
                if has_polyp and not filtered_has_polyp:
                    stats["false_positives_filtered"] += 1
                
                # 保存帧
                if save_frames:
                    if filtered_has_polyp:
                        save_dir = polyp_dir
                    else:
                        save_dir = no_polyp_dir
                    
                    # 绘制检测框
                    if save_with_bbox and filtered_detections:
                        annotated_frame = frame.copy()
                        for det in filtered_detections:
                            bbox = det['bbox'].astype(int)
                            conf = det['confidence']
                            cv2.rectangle(
                                annotated_frame,
                                (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]),
                                (0, 255, 0),
                                2
                            )
                            cv2.putText(
                                annotated_frame,
                                f"Polyp {conf:.2f}",
                                (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2
                            )
                        frame_to_save = annotated_frame
                    else:
                        frame_to_save = frame
                    
                    filename = f"frame_{frame_idx:06d}.jpg"
                    if filtered_has_polyp:
                        num_det = len(filtered_detections)
                        filename = f"frame_{frame_idx:06d}_det{num_det}.jpg"
                    
                    save_path = save_dir / filename
                    cv2.imwrite(str(save_path), frame_to_save)
                
                frame_idx += 1
                pbar.update(1)
                pbar.set_postfix({
                    "原始": stats["raw_detections"],
                    "过滤后": stats["filtered_detections"],
                    "过滤掉": stats["false_positives_filtered"]
                })
        
        cap.release()
        
        # 打印统计信息
        print("\n" + "=" * 80)
        print("处理完成!")
        print("=" * 80)
        print(f"总处理帧数: {stats['total_processed']}")
        print(f"原始检测帧数: {stats['raw_detections']} ({stats['raw_detections']/stats['total_processed']*100:.2f}%)")
        print(f"过滤后帧数: {stats['filtered_detections']} ({stats['filtered_detections']/stats['total_processed']*100:.2f}%)")
        print(f"过滤掉的假阳性: {stats['false_positives_filtered']} ({stats['false_positives_filtered']/stats['raw_detections']*100:.2f}% of detections)")
        print(f"\n假阳性减少率: {(stats['false_positives_filtered']/stats['raw_detections']*100):.2f}%")
        print(f"输出目录: {output_path}")
        print("=" * 80)
        
        # 保存统计信息
        stats_file = output_path / "detection_stats.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"视频息肉检测统计\n")
            f.write(f"{'='*80}\n")
            f.write(f"视频路径: {video_path}\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n检测参数:\n")
            f.write(f"  置信度阈值: {self.conf_threshold}\n")
            f.write(f"  时间连续性: {self.temporal_consistency}\n")
            f.write(f"  最少连续帧: {self.min_consecutive_frames}\n")
            f.write(f"  时间窗口: {self.temporal_window}\n")
            f.write(f"\n统计结果:\n")
            f.write(f"  总处理帧数: {stats['total_processed']}\n")
            f.write(f"  原始检测帧数: {stats['raw_detections']}\n")
            f.write(f"  过滤后帧数: {stats['filtered_detections']}\n")
            f.write(f"  过滤掉的假阳性: {stats['false_positives_filtered']}\n")
            f.write(f"  假阳性减少率: {(stats['false_positives_filtered']/stats['raw_detections']*100):.2f}%\n")
        
        return stats


if __name__ == "__main__":
    # 配置参数
    video_path = "钟木英230923.TS"
    model_weights = "runs/detect/yolo11n_all_coco6/weights/best.pt"
    
    # 创建检测器
    detector = VideoPolypDetector(
        model_weights=model_weights,
        conf_threshold=0.6,  # 基础置信度阈值
        iou_threshold=0.45,
        # 假阳性减少策略
        temporal_consistency=True,  # 启用时间连续性
        min_consecutive_frames=2,  # 需要连续2帧检测到
        temporal_window=5,  # 5帧滑动窗口
        min_detections_in_window=2,  # 窗口内至少2次检测
        stability_check=True,  # 启用稳定性检查
        min_stable_frames=5,  # 5帧稳定性检查
        show_bbox=True,  # 是否在保存的帧上绘制检测框 (True=显示, False=不显示)
    )
    
    # 处理视频
    detector.process_video(
        video_path=video_path,
        save_frames=True,
        # save_with_bbox 参数可以在这里覆盖类初始化时的设置
        # 如果为None，则使用类初始化时的 show_bbox 设置
        save_with_bbox=None,  # None=使用类初始化设置, True/False=覆盖设置
    )

