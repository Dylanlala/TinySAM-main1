"""
手动审查和清理假阳性（误检为息肉的图片）
支持交互式标记误检图片并移动到无息肉目录
"""
import cv2
from pathlib import Path
from ultralytics import YOLO
import shutil
from datetime import datetime
import json

def review_and_clean_false_positives(
    frames_dir: str,
    model_weights: str,
    conf_threshold: float = 0.7,
    auto_move_low_conf: bool = False,
    low_conf_threshold: float = 0.75,
):
    """
    审查和清理假阳性
    
    Args:
        frames_dir: 帧目录路径
        model_weights: YOLO模型权重路径
        conf_threshold: 检测置信度阈值
        auto_move_low_conf: 是否自动移动低置信度图片
        low_conf_threshold: 低置信度阈值（如果auto_move_low_conf为True）
    """
    print("=" * 80)
    print("手动审查和清理假阳性")
    print("=" * 80)
    print(f"帧目录: {frames_dir}")
    print(f"模型权重: {model_weights}")
    print(f"置信度阈值: {conf_threshold}")
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
    print(f"\n找到 {len(polyp_files)} 张图片需要审查")
    
    # 分析置信度分布
    print("\n分析置信度分布...")
    confidences = []
    file_conf_map = {}
    
    for img_path in polyp_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        results = model(img, conf=conf_threshold, verbose=False)
        max_conf = 0.0
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if len(boxes) > 0:
                confs = [float(box.conf) for box in boxes]
                max_conf = max(confs)
        
        confidences.append(max_conf)
        file_conf_map[img_path] = max_conf
    
    if confidences:
        import numpy as np
        confidences = np.array(confidences)
        print(f"  平均置信度: {confidences.mean():.3f}")
        print(f"  中位数: {np.median(confidences):.3f}")
        print(f"  最小值: {confidences.min():.3f}")
        print(f"  最大值: {confidences.max():.3f}")
        
        # 按置信度排序
        sorted_files = sorted(file_conf_map.items(), key=lambda x: x[1])
        
        # 如果启用自动移动低置信度图片
        if auto_move_low_conf:
            low_conf_files = [f for f, conf in sorted_files if conf < low_conf_threshold]
            if low_conf_files:
                print(f"\n自动移动 {len(low_conf_files)} 张低置信度(<{low_conf_threshold})图片...")
                for img_path in low_conf_files:
                    new_path = no_polyp_dir / img_path.name.replace("_det", "")
                    shutil.move(str(img_path), str(new_path))
                print(f"已移动到无息肉目录")
        
        # 生成审查报告
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(polyp_files),
            "conf_threshold": conf_threshold,
            "statistics": {
                "mean": float(confidences.mean()),
                "median": float(np.median(confidences)),
                "min": float(confidences.min()),
                "max": float(confidences.max()),
                "std": float(confidences.std()),
            },
            "low_confidence_files": [
                {
                    "file": str(f.name),
                    "confidence": float(conf)
                }
                for f, conf in sorted_files[:20]  # 前20个最低置信度
            ],
            "high_confidence_files": [
                {
                    "file": str(f.name),
                    "confidence": float(conf)
                }
                for f, conf in sorted_files[-20:]  # 后20个最高置信度
            ]
        }
        
        report_path = frames_path / "false_positive_review_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n审查报告已保存到: {report_path}")
        print(f"\n建议:")
        print(f"  1. 查看报告中的低置信度图片（前20个）")
        print(f"  2. 手动检查这些图片，确认是否为假阳性")
        print(f"  3. 如果确认是假阳性，可以手动移动到无息肉目录")
        print(f"  4. 或者使用更高的置信度阈值重新分类")
        
        # 显示最低置信度的图片列表
        print(f"\n最低置信度的10张图片（建议优先审查）:")
        for i, (img_path, conf) in enumerate(sorted_files[:10], 1):
            print(f"  {i}. {img_path.name}: {conf:.3f}")
    
    return report if 'report' in locals() else None


def create_manual_cleanup_script(frames_dir: str):
    """
    创建一个手动清理脚本，列出所有图片供用户审查
    """
    frames_path = Path(frames_dir)
    polyp_dir = frames_path / "有息肉"
    
    script_content = f"""#!/usr/bin/env python3
\"\"\"
手动清理假阳性图片
使用方法: 修改下面的文件列表，将误检的图片文件名添加到 false_positives 列表中
\"\"\"

import shutil
from pathlib import Path

# 配置
frames_dir = Path("{frames_dir}")
polyp_dir = frames_dir / "有息肉"
no_polyp_dir = frames_dir / "无息肉"

# 将误检的图片文件名添加到这个列表中（只写文件名，不包括路径）
false_positives = [
    # 示例:
    # "frame_000318_det1.jpg",
    # "frame_000319_det1.jpg",
]

# 移动文件
moved_count = 0
for filename in false_positives:
    src = polyp_dir / filename
    if src.exists():
        # 移除文件名中的 _det 部分
        new_filename = filename.replace("_det", "")
        dst = no_polyp_dir / new_filename
        shutil.move(str(src), str(dst))
        print(f"已移动: {{filename}} -> {{new_filename}}")
        moved_count += 1
    else:
        print(f"警告: 文件不存在: {{filename}}")

print(f"\\n总共移动了 {{moved_count}} 张图片")
"""
    
    script_path = frames_path / "manual_cleanup_false_positives.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 设置执行权限
    import os
    os.chmod(script_path, 0o755)
    
    print(f"\n手动清理脚本已创建: {script_path}")
    print("编辑该脚本，在 false_positives 列表中添加误检的图片文件名，然后运行它")


if __name__ == "__main__":
    # 配置参数
    frames_dir = "钟木英230923_frames"
    model_weights = "runs/detect/yolo11n_all_coco6/weights/best.pt"
    conf_threshold = 0.6  # 当前使用的阈值
    
    # 选项1: 自动移动低置信度图片（谨慎使用）
    # auto_move_low_conf = True
    # low_conf_threshold = 0.75  # 自动移动置信度低于0.75的图片
    
    # 选项2: 只生成报告，不自动移动
    auto_move_low_conf = False
    
    # 运行审查
    review_and_clean_false_positives(
        frames_dir=frames_dir,
        model_weights=model_weights,
        conf_threshold=conf_threshold,
        auto_move_low_conf=auto_move_low_conf,
        low_conf_threshold=0.75,
    )
    
    # 创建手动清理脚本
    create_manual_cleanup_script(frames_dir)

