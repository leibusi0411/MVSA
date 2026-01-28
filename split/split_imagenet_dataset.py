"""
ImageNet数据集训练集划分工具

功能：
1. 从ImageNet训练集的每个类别中随机抽取50张图片
2. 将抽取的图片移动到新的验证集文件夹
3. 从原始训练集中删除这些图片

使用方法：
python split_imagenet_dataset.py --train_dir /path/to/project/DATA/imagenet/images/train  --val_dir /path/to/project/DATA/imagenet/images/val --samples_per_class 50

注意事项：
- 操作会直接移动文件，请确保有备份
- 建议先在小规模数据上测试
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def get_image_files(directory):
    """
    获取目录中的所有图片文件
    
    Args:
        directory (str): 目录路径
        
    Returns:
        List[str]: 图片文件路径列表
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return image_files


def split_imagenet_dataset(train_dir, val_dir, samples_per_class=50, seed=42, dry_run=False):
    """
    划分ImageNet数据集
    
    Args:
        train_dir (str): 原始训练集目录路径
        val_dir (str): 新验证集目录路径
        samples_per_class (int): 每个类别抽取的样本数
        seed (int): 随机种子
        dry_run (bool): 是否只是预览，不实际移动文件
        
    Returns:
        dict: 统计信息
    """
    # 设置随机种子
    random.seed(seed)
    
    # 检查训练集目录
    train_path = Path(train_dir)
    if not train_path.exists():
        raise ValueError(f"训练集目录不存在: {train_dir}")
    
    # 创建验证集目录
    val_path = Path(val_dir)
    if not dry_run:
        val_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'total_classes': 0,
        'processed_classes': 0,
        'total_moved_images': 0,
        'skipped_classes': [],
        'class_details': defaultdict(dict)
    }
    
    # 获取所有类别目录
    class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
    stats['total_classes'] = len(class_dirs)
    
    print(f"📊 发现 {len(class_dirs)} 个类别目录")
    print(f"🎯 每个类别抽取 {samples_per_class} 张图片")
    print(f"🌱 随机种子: {seed}")
    
    if dry_run:
        print("🔍 预览模式：不会实际移动文件")
    else:
        print("⚠️  实际执行模式：将会移动文件")
    
    # 处理每个类别
    for class_dir in tqdm(class_dirs, desc="处理类别"):
        class_name = class_dir.name
        
        # 获取该类别的所有图片
        image_files = get_image_files(class_dir)
        total_images = len(image_files)
        
        # 记录类别信息
        stats['class_details'][class_name]['total_images'] = total_images
        stats['class_details'][class_name]['moved_images'] = 0
        
        # 检查是否有足够的图片
        if total_images < samples_per_class:
            print(f"⚠️  类别 {class_name} 只有 {total_images} 张图片，少于要求的 {samples_per_class} 张")
            stats['skipped_classes'].append(class_name)
            continue
        
        # 随机抽取图片
        selected_images = random.sample(image_files, samples_per_class)
        
        # 创建验证集中的类别目录
        val_class_dir = val_path / class_name
        if not dry_run:
            val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动图片
        moved_count = 0
        for image_path in selected_images:
            image_file = Path(image_path)
            dest_path = val_class_dir / image_file.name
            
            if not dry_run:
                try:
                    # 移动文件（从训练集移动到验证集）
                    shutil.move(str(image_file), str(dest_path))
                    moved_count += 1
                except Exception as e:
                    print(f"❌ 移动文件失败 {image_file} -> {dest_path}: {e}")
            else:
                moved_count += 1
        
        # 更新统计信息
        stats['class_details'][class_name]['moved_images'] = moved_count
        stats['total_moved_images'] += moved_count
        stats['processed_classes'] += 1
    
    return stats


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "="*60)
    print("📊 数据集划分统计")
    print("="*60)
    
    print(f"总类别数: {stats['total_classes']}")
    print(f"成功处理的类别数: {stats['processed_classes']}")
    print(f"跳过的类别数: {len(stats['skipped_classes'])}")
    print(f"总共移动的图片数: {stats['total_moved_images']}")
    
    if stats['skipped_classes']:
        print(f"\n⚠️  跳过的类别 ({len(stats['skipped_classes'])} 个):")
        for class_name in stats['skipped_classes'][:10]:  # 只显示前10个
            total = stats['class_details'][class_name]['total_images']
            print(f"   - {class_name}: {total} 张图片")
        if len(stats['skipped_classes']) > 10:
            print(f"   ... 还有 {len(stats['skipped_classes']) - 10} 个类别")
    
    # 显示一些成功处理的类别示例
    processed_classes = [name for name in stats['class_details'] 
                        if stats['class_details'][name]['moved_images'] > 0]
    
    if processed_classes:
        print(f"\n✅ 成功处理的类别示例:")
        for class_name in processed_classes[:5]:
            details = stats['class_details'][class_name]
            print(f"   - {class_name}: {details['total_images']} -> 移动了 {details['moved_images']} 张")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ImageNet数据集训练集划分工具')
    
    parser.add_argument('--train_dir', type=str, required=True,
                       help='ImageNet训练集目录路径')
    parser.add_argument('--val_dir', type=str, required=True,
                       help='新验证集目录路径')
    parser.add_argument('--samples_per_class', type=int, default=50,
                       help='每个类别抽取的样本数 (默认: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--dry_run', action='store_true',
                       help='预览模式，不实际移动文件')
    
    args = parser.parse_args()
    
    print("🚀 ImageNet数据集划分工具")
    print(f"训练集目录: {args.train_dir}")
    print(f"验证集目录: {args.val_dir}")
    print(f"每类抽取: {args.samples_per_class} 张图片")
    
    # 确认操作
    if not args.dry_run:
        print("\n⚠️  警告：此操作将会移动文件，请确保已备份数据！")
        confirm = input("确认继续？(y/N): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return
    
    try:
        # 执行划分
        stats = split_imagenet_dataset(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            dry_run=args.dry_run
        )
        
        # 打印统计信息
        print_statistics(stats)
        
        if args.dry_run:
            print("\n🔍 预览完成！使用 --dry_run=False 执行实际操作")
        else:
            print(f"\n✅ 数据集划分完成！")
            print(f"   原训练集: {args.train_dir}")
            print(f"   新验证集: {args.val_dir}")
        
    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
