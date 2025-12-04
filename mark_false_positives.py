"""
标记和清理假阳性图片
支持批量标记误检图片并移动到无息肉目录
"""
import shutil
from pathlib import Path
from datetime import datetime

def mark_and_clean_false_positives(
    frames_dir: str,
    false_positive_files: list,
    dry_run: bool = False,
):
    """
    标记假阳性图片并移动到无息肉目录
    
    Args:
        frames_dir: 帧目录路径
        false_positive_files: 假阳性图片文件名列表（只写文件名，不包括路径）
        dry_run: 如果为True，只显示将要移动的文件，不实际移动
    """
    frames_path = Path(frames_dir)
    polyp_dir = frames_path / "有息肉"
    no_polyp_dir = frames_path / "无息肉"
    
    if not polyp_dir.exists():
        print(f"错误: 有息肉目录不存在: {polyp_dir}")
        return
    
    if not no_polyp_dir.exists():
        no_polyp_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("标记和清理假阳性图片")
    print("=" * 80)
    print(f"假阳性图片数量: {len(false_positive_files)}")
    print(f"模式: {'预览模式（不实际移动）' if dry_run else '执行模式（将移动文件）'}")
    print("=" * 80)
    
    moved_count = 0
    not_found = []
    
    for filename in false_positive_files:
        src = polyp_dir / filename
        if src.exists():
            # 移除文件名中的 _det 部分（包括后面的数字）
            # 例如: frame_000323_det1.jpg -> frame_000323.jpg
            import re
            new_filename = re.sub(r'_det\d+', '', filename)
            dst = no_polyp_dir / new_filename
            
            if dry_run:
                print(f"[预览] 将移动: {filename} -> {new_filename}")
            else:
                shutil.move(str(src), str(dst))
                print(f"已移动: {filename} -> {new_filename}")
            moved_count += 1
        else:
            print(f"警告: 文件不存在: {filename}")
            not_found.append(filename)
    
    print("\n" + "=" * 80)
    if dry_run:
        print(f"预览完成: 将移动 {moved_count} 张图片")
        if not_found:
            print(f"未找到 {len(not_found)} 个文件")
    else:
        print(f"清理完成: 已移动 {moved_count} 张假阳性图片到无息肉目录")
        if not_found:
            print(f"未找到 {len(not_found)} 个文件")
    
    # 记录到日志文件
    if not dry_run and moved_count > 0:
        log_file = frames_path / "false_positives_removed.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"移除了 {moved_count} 张假阳性图片:\n")
            for filename in false_positive_files:
                if filename not in not_found:
                    f.write(f"  - {filename}\n")
        print(f"\n操作记录已保存到: {log_file}")
    
    print("=" * 80)
    
    return moved_count, not_found


if __name__ == "__main__":
    # 配置
    frames_dir = "钟木英230923_frames"
    
    # ============================================
    # 在这里添加假阳性图片的文件名
    # 只写文件名，不包括路径
    # ============================================
    false_positives = [
        "frame_000323_det1.jpg",  # 置信度0.89但实际不是息肉
        # 添加更多假阳性图片文件名：
        # "frame_000318_det1.jpg",
        # "frame_000319_det1.jpg",
    ]
    
    # 先预览（不实际移动文件）
    print("第一步：预览将要移动的文件...")
    mark_and_clean_false_positives(
        frames_dir=frames_dir,
        false_positive_files=false_positives,
        dry_run=True,  # 预览模式
    )
    
    # 如果确认无误，取消下面的注释来实际执行移动
    print("\n第二步：执行移动...")
    mark_and_clean_false_positives(
        frames_dir=frames_dir,
        false_positive_files=false_positives,
        dry_run=False,  # 执行模式
    )

