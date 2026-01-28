#!/usr/bin/env python3
"""
Places365数据集划分脚本

功能:
1. 自动从配置文件读取Places365数据集路径
2. 查找并读取places365_train_standard.txt文件
3. 按类别对所有图片进行分组
4. 从每个类别中随机抽取指定数量的图片作为验证集
5. 输出新的训练集和验证集划分文件

文件格式:
- 输入: places365_train_standard.txt (图片相对路径 类别索引)
- 输出: places365_train_standard.txt (更新后的训练集)
- 输出: places365_val.txt (新的验证集)

使用方法:
# 自动从配置文件查找
python split_places365_dataset.py --val_samples 100 --seed 42

# 手动指定文件路径
python split_places365_dataset.py --input /path/to/places365_train_standard.txt --val_samples 100 --seed 42

# 指定配置文件
python split_places365_dataset.py --config STN-Config/place365.yaml --val_samples 100 --seed 42
"""

import argparse
import random
import os
import yaml
from collections import defaultdict
from pathlib import Path


def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {config_path} - {e}")
        return None


def find_dataset_path_from_config(config_file=None, dataset_name="place365"):
    """
    从配置文件中查找数据集路径
    
    Args:
        config_file (str): 配置文件路径，如果为None则尝试自动查找
        dataset_name (str): 数据集名称
        
    Returns:
        str: 数据集路径，如果未找到返回None
    """
    # 可能的配置文件路径
    config_candidates = []
    
    if config_file:
        config_candidates.append(config_file)
    else:
        # 自动查找配置文件
        config_candidates.extend([
            f"STN-Config/{dataset_name}.yaml",
            f"cfgs/{dataset_name}.yaml",
            f"{dataset_name}.yaml"
        ])
    
    for config_path in config_candidates:
        if os.path.exists(config_path):
            print(f"📋 找到配置文件: {config_path}")
            config = load_config(config_path)
            
            if config and 'data_path' in config:
                data_path = config['data_path']
                print(f"📂 从配置文件读取数据集路径: {data_path}")
                return data_path
    
    print(f"⚠️ 未在配置文件中找到数据集路径")
    return None


def find_train_file(data_path, filename="places365_train_standard.txt"):
    """
    在数据集路径下查找训练文件
    
    Args:
        data_path (str): 数据集根路径
        filename (str): 训练文件名
        
    Returns:
        str: 训练文件完整路径，如果未找到返回None
    """
    if not data_path:
        return None
        
    # 可能的文件位置
    possible_paths = [
        os.path.join(data_path, filename),                    # 直接在数据集根目录
        os.path.join(data_path, "annotations", filename),     # 在annotations子目录
        os.path.join(data_path, "labels", filename),          # 在labels子目录
        os.path.join(data_path, "metadata", filename),        # 在metadata子目录
    ]
    
    for file_path in possible_paths:
        if os.path.exists(file_path):
            print(f"✅ 找到训练文件: {file_path}")
            return file_path
    
    print(f"❌ 在以下位置未找到 {filename}:")
    for path in possible_paths:
        print(f"     {path}")
    return None


def load_dataset_file(file_path):
    """
    加载Places365数据集文件
    
    Args:
        file_path (str): 数据集文件路径
        
    Returns:
        list: [(image_path, class_idx), ...] 格式的列表
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析格式: "图片相对路径 类别索引"
                parts = line.split()
                if len(parts) != 2:
                    print(f"⚠️ 警告: 第{line_num}行格式不正确: {line}")
                    continue
                    
                image_path = parts[0]
                class_idx = int(parts[1])
                data.append((image_path, class_idx))
                
            except ValueError as e:
                print(f"⚠️ 警告: 第{line_num}行解析失败: {line} - {e}")
                continue
    
    print(f"✅ 成功加载 {len(data)} 条数据")
    return data


def group_by_class(data):
    """
    按类别对数据进行分组
    
    Args:
        data (list): [(image_path, class_idx), ...] 格式的数据
        
    Returns:
        dict: {class_idx: [data_items], ...} 格式的字典
    """
    class_groups = defaultdict(list)
    
    for item in data:
        image_path, class_idx = item
        class_groups[class_idx].append(item)
    
    # 统计信息
    num_classes = len(class_groups)
    class_sizes = [len(items) for items in class_groups.values()]
    
    print(f"📊 数据统计:")
    print(f"   - 总类别数: {num_classes}")
    print(f"   - 每类样本数范围: {min(class_sizes)} - {max(class_sizes)}")
    print(f"   - 平均每类样本数: {sum(class_sizes) / num_classes:.1f}")
    
    return class_groups


def split_dataset(class_groups, val_samples_per_class, seed=42):
    """
    从每个类别中抽取验证集样本
    
    Args:
        class_groups (dict): 按类别分组的数据
        val_samples_per_class (int): 每个类别抽取的验证集样本数
        seed (int): 随机种子
        
    Returns:
        tuple: (train_data, val_data) 两个列表
    """
    random.seed(seed)
    print(f"🎲 使用随机种子: {seed}")
    
    train_data = []
    val_data = []
    
    insufficient_classes = []
    
    for class_idx, items in class_groups.items():
        num_items = len(items)
        
        if num_items < val_samples_per_class:
            print(f"⚠️ 类别 {class_idx} 只有 {num_items} 个样本，少于要求的 {val_samples_per_class} 个")
            insufficient_classes.append((class_idx, num_items))
            # 对于样本不足的类别，取一半作为验证集
            val_count = max(1, num_items // 2)
        else:
            val_count = val_samples_per_class
        
        # 随机打乱该类别的样本
        shuffled_items = items.copy()
        random.shuffle(shuffled_items)
        
        # 分割验证集和训练集
        val_items = shuffled_items[:val_count]
        train_items = shuffled_items[val_count:]
        
        val_data.extend(val_items)
        train_data.extend(train_items)
    
    print(f"📊 分割结果:")
    print(f"   - 训练集样本数: {len(train_data)}")
    print(f"   - 验证集样本数: {len(val_data)}")
    
    if insufficient_classes:
        print(f"⚠️ {len(insufficient_classes)} 个类别的样本数不足:")
        for class_idx, count in insufficient_classes[:5]:  # 只显示前5个
            print(f"     类别 {class_idx}: {count} 个样本")
        if len(insufficient_classes) > 5:
            print(f"     ... 还有 {len(insufficient_classes) - 5} 个类别")
    
    return train_data, val_data


def save_dataset_file(data, file_path):
    """
    保存数据集文件
    
    Args:
        data (list): [(image_path, class_idx), ...] 格式的数据
        file_path (str): 输出文件路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for image_path, class_idx in data:
            f.write(f"{image_path} {class_idx}\n")
    
    print(f"✅ 已保存 {len(data)} 条数据到: {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Places365数据集划分工具')
    parser.add_argument('--input', type=str, default=None,
                        help='输入的训练集文件路径 (如果未指定，将从配置文件中自动查找)')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (默认: 自动查找STN-Config/place365.yaml)')
    parser.add_argument('--dataset_name', type=str, default='place365',
                        help='数据集名称 (默认: place365)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出文件目录 (默认: 使用数据集路径)')
    parser.add_argument('--val_samples', type=int, default=100,
                        help='每个类别抽取的验证集样本数 (默认: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--backup', action='store_true',
                        help='是否备份原始文件')
    
    args = parser.parse_args()
    
    # 确定输入文件路径
    input_file = args.input
    
    if input_file is None:
        # 从配置文件中查找数据集路径
        print(f"🔍 从配置文件中查找数据集路径...")
        data_path = find_dataset_path_from_config(args.config, args.dataset_name)
        
        if data_path:
            # 在数据集路径下查找训练文件
            input_file = find_train_file(data_path)
            
            # 如果未指定输出目录，使用数据集路径
            if args.output_dir is None:
                args.output_dir = data_path
                print(f"📁 输出目录设置为数据集路径: {args.output_dir}")
        else:
            print(f"❌ 无法从配置文件中找到数据集路径")
            print(f"请使用 --input 参数手动指定训练文件路径")
            return
    
    # 检查输入文件
    if input_file is None or not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return
    
    # 如果仍然没有设置输出目录，使用当前目录
    if args.output_dir is None:
        args.output_dir = "."
    
    print(f"🚀 开始Places365数据集划分")
    print(f"   - 输入文件: {input_file}")
    print(f"   - 输出目录: {args.output_dir}")
    print(f"   - 每类验证样本数: {args.val_samples}")
    print(f"   - 随机种子: {args.seed}")
    
    # 备份原始文件
    if args.backup:
        backup_path = input_file + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(input_file, backup_path)
            print(f"📋 已备份原始文件到: {backup_path}")
    
    # 1. 加载数据
    print(f"\n📂 加载数据集文件...")
    data = load_dataset_file(input_file)
    
    if not data:
        print("❌ 错误: 没有成功加载任何数据")
        return
    
    # 2. 按类别分组
    print(f"\n🔄 按类别分组...")
    class_groups = group_by_class(data)
    
    # 3. 分割数据集
    print(f"\n✂️ 分割数据集...")
    train_data, val_data = split_dataset(class_groups, args.val_samples, args.seed)
    
    # 4. 保存结果
    print(f"\n💾 保存结果...")
    
    # 输出文件路径
    train_output = os.path.join(args.output_dir, 'places365_train_standard.txt')
    val_output = os.path.join(args.output_dir, 'places365_val.txt')
    
    save_dataset_file(train_data, train_output)
    save_dataset_file(val_data, val_output)
    
    # 验证结果
    print(f"\n🔍 验证结果:")
    original_count = len(data)
    new_train_count = len(train_data)
    new_val_count = len(val_data)
    
    print(f"   - 原始总数: {original_count}")
    print(f"   - 新训练集: {new_train_count}")
    print(f"   - 新验证集: {new_val_count}")
    print(f"   - 总计: {new_train_count + new_val_count}")
    print(f"   - 验证: {'✅ 通过' if original_count == new_train_count + new_val_count else '❌ 失败'}")
    
    print(f"\n🎉 数据集划分完成!")
    print(f"   - 训练集文件: {train_output}")
    print(f"   - 验证集文件: {val_output}")


if __name__ == '__main__':
    main()
