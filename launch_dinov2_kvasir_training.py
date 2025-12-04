#!/usr/bin/env python3
"""
启动DINOv2在Kvasir-SEG数据集上的预训练
基于现有配置文件和训练脚本的完整启动脚本
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import yaml
import shutil

def check_environment():
    """检查训练环境"""
    print("🔍 检查训练环境...")
    
    # 检查CUDA
    if not os.system("nvidia-smi > /dev/null 2>&1") == 0:
        print("❌ 未检测到CUDA环境")
        return False
    
    # 检查Python包
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    return True

def check_data_directory(data_path):
    """检查数据目录"""
    print(f"🔍 检查数据目录: {data_path}")
    
    if not Path(data_path).exists():
        print(f"❌ 数据目录不存在: {data_path}")
        return False
    
    # 统计图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_count = 0
    
    for ext in image_extensions:
        image_count += len(list(Path(data_path).rglob(f'*{ext}')))
        image_count += len(list(Path(data_path).rglob(f'*{ext.upper()}')))
    
    print(f"✅ 找到 {image_count} 张图像")
    
    if image_count == 0:
        print("❌ 数据目录中没有找到图像文件")
        return False
    
    return True

def prepare_config(args):
    """准备训练配置"""
    print("📝 准备训练配置...")
    
    # 基础配置
    config = {
        'MODEL': {'WEIGHTS': ''},
        'compute_precision': {'grad_scaler': True},
        
        # 学生和教师模型配置
        'student': {
            'arch': args.arch,
            'patch_size': args.patch_size,
            'drop_path_rate': 0.3,
            'layerscale': 1e-5,
            'drop_path_uniform': True,
            'pretrained_weights': '',
            'ffn_layer': 'mlp',
            'block_chunks': 0,
            'qkv_bias': True,
            'proj_bias': True,
            'ffn_bias': True,
            'num_register_tokens': 0,
            'interpolate_antialias': False,
            'interpolate_offset': 0.1
        },
        
        'teacher': {
            'momentum_teacher': 0.996,
            'final_momentum_teacher': 1.0,
            'warmup_teacher_temp': 0.04,
            'teacher_temp': 0.07,
            'warmup_teacher_temp_epochs': 30
        },
        
        # DINO和iBOT配置
        'dino': {
            'loss_weight': 1.0,
            'head_n_prototypes': 65536,
            'head_bottleneck_dim': 256,
            'head_nlayers': 3,
            'head_hidden_dim': 2048,
            'koleo_loss_weight': 0.1
        },
        
        'ibot': {
            'loss_weight': 1.0,
            'mask_sample_probability': 0.5,
            'mask_ratio_min_max': [0.1, 0.5],
            'separate_head': False,
            'head_n_prototypes': 65536,
            'head_bottleneck_dim': 256,
            'head_nlayers': 3,
            'head_hidden_dim': 2048
        },
        
        # 训练配置
        'train': {
            'batch_size_per_gpu': args.batch_size,
            'dataset_path': f'ImageNet:root={args.data_path}:split=TRAIN:extra=',
            'output_dir': args.output_dir,
            'saveckp_freq': 20,
            'seed': 0,
            'num_workers': args.num_workers,
            'OFFICIAL_EPOCH_LENGTH': args.epoch_length,
            'cache_dataset': True,
            'centering': 'centering'
        },
        
        # 优化器配置
        'optim': {
            'epochs': args.epochs,
            'weight_decay': 0.08,
            'weight_decay_end': 0.4,
            'base_lr': args.lr,
            'warmup_epochs': 20,
            'min_lr': 1e-6,
            'clip_grad': 1.0,
            'freeze_last_layer_epochs': 1,
            'scaling_rule': 'sqrt_wrt_1024',
            'patch_embed_lr_mult': 0.2,
            'layerwise_decay': 0.9,
            'adamw_beta1': 0.9,
            'adamw_beta2': 0.999,
            'batch_size_per_gpu': args.batch_size
        },
        
        # 裁剪配置
        'crops': {
            'global_crops_scale': [0.32, 1.0],
            'local_crops_number': 2,
            'local_crops_scale': [0.05, 0.32],
            'global_crops_size': args.img_size,
            'local_crops_size': args.img_size // 2
        },
        
        # 评估配置
        'evaluation': {
            'eval_period_iterations': 12500
        },
        
        # 模型配置
        'model': {
            'arch': args.arch
        }
    }
    
    # 保存配置文件
    config_path = Path(args.output_dir) / 'training_config.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ 配置文件已保存: {config_path}")
    return config_path

def launch_training(config_path, args):
    """启动分布式训练"""
    print("🚀 启动DINOv2训练...")
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd() / 'dinov2')
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    # 构建训练命令
    if args.distributed and args.num_gpus > 1:
        # 分布式训练
        cmd = [
            'torchrun',
            f'--nproc_per_node={args.num_gpus}',
            '--master_port=29500',
            'dinov2/dinov2/train/train.py',
            f'--config-file={config_path}',
            f'train.output_dir={args.output_dir}',
            f'optim.epochs={args.epochs}',
            f'optim.batch_size_per_gpu={args.batch_size}',
            f'model.arch={args.arch}',
            f'student.arch={args.arch}'
        ]
        
        if args.disable_xformers:
            cmd.append('--disable_xformers')
            
    else:
        # 单GPU训练
        cmd = [
            sys.executable,
            'dinov2/dinov2/train/train.py',
            f'--config-file={config_path}',
            f'train.output_dir={args.output_dir}',
            f'optim.epochs={args.epochs}',
            f'optim.batch_size_per_gpu={args.batch_size}',
            f'model.arch={args.arch}',
            f'student.arch={args.arch}'
        ]
        
        if args.disable_xformers:
            cmd.append('--disable_xformers')
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📊 训练配置:")
    print(f"   - 模型架构: {args.arch}")
    print(f"   - 训练轮数: {args.epochs}")
    print(f"   - 批大小: {args.batch_size}")
    print(f"   - 学习率: {args.lr}")
    print(f"   - GPU数量: {args.num_gpus}")
    print(f"   - 数据路径: {args.data_path}")
    
    try:
        # 启动训练
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出日志
        log_file = Path(args.output_dir) / 'training.log'
        with open(log_file, 'w') as f:
            for line in process.stdout:
                print(line.strip())
                f.write(line)
                f.flush()
        
        # 等待训练完成
        return_code = process.wait()
        
        if return_code == 0:
            print("🎉 训练成功完成！")
            print(f"📁 模型权重保存在: {args.output_dir}")
            return True
        else:
            print(f"❌ 训练失败，返回码: {return_code}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        if process:
            process.terminate()
        return False
    except Exception as e:
        print(f"❌ 训练启动失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='启动DINOv2在Kvasir-SEG数据集上的预训练')
    
    # 数据配置
    parser.add_argument('--data_path', type=str, 
                       default='/home/huangmanling/TinySAM-main/Kvasir-SEG/dummy',
                       help='Kvasir-SEG数据集路径')
    parser.add_argument('--output_dir', type=str,
                       default='./dinov2_kvasir_output',
                       help='输出目录')
    
    # 模型配置
    parser.add_argument('--arch', type=str, default='vit_small',
                       choices=['vit_small', 'vit_base', 'vit_large'],
                       help='模型架构')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch大小')
    parser.add_argument('--img_size', type=int, default=224,
                       help='输入图像尺寸')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='每GPU批大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--epoch_length', type=int, default=1250,
                       help='每个epoch的迭代次数')
    
    # 硬件配置
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='使用的GPU数量')
    parser.add_argument('--gpu_ids', type=str, default='1',
                       help='GPU ID，用逗号分隔')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='数据加载线程数')
    parser.add_argument('--distributed', action='store_true',
                       help='是否使用分布式训练')
    parser.add_argument('--disable_xformers', action='store_true',
                       help='禁用xformers优化')
    
    # 其他选项
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--dry_run', action='store_true',
                       help='只检查环境，不启动训练')
    
    args = parser.parse_args()
    
    print("🎯 DINOv2 Kvasir-SEG 预训练启动器")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败")
        return False
    
    # 检查数据
    if not check_data_directory(args.data_path):
        print("❌ 数据检查失败")
        return False
    
    # 如果是干运行，直接退出
    if args.dry_run:
        print("✅ 环境检查完成（干运行模式）")
        return True
    
    # 准备配置
    config_path = prepare_config(args)
    
    # 启动训练
    success = launch_training(config_path, args)
    
    if success:
        print("\n🎉 训练任务完成！")
        print(f"📁 检查输出目录: {args.output_dir}")
        print("💡 后续步骤:")
        print("   1. 验证生成的权重文件")
        print("   2. 运行评估脚本")
        print("   3. 将权重用于YOLO检测任务")
    else:
        print("\n❌ 训练任务失败")
        print("💡 故障排除:")
        print("   1. 检查GPU内存是否充足")
        print("   2. 检查数据路径是否正确")
        print("   3. 查看训练日志获取详细错误信息")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)