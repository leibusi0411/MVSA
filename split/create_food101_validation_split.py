#!/usr/bin/env python3
"""
Food-101数据集训练集分层划分脚本

功能：
1. 从Food-101训练集中按类别分层取20%作为验证集
2. 保持每个类别的比例一致
3. 生成新的train.json和val.json文件

使用方法：
python create_food101_validation_split.py

输出文件：
- train.json: 新的训练集文件（原训练集的80%）
- val.json: 新的验证集文件（原训练集的20%）

数据结构：
Food-101的JSON文件格式：
{
  "class_name1": ["image_path1", "image_path2", ...],
  "class_name2": ["image_path1", "image_path2", ...],
  ...
}
"""

import os
import json
import random
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import argparse


def load_food101_config():
    """加载Food-101数据集配置"""
    config_path = "cfgs/food101.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config['data_path']


def read_food101_train_json(data_root):
    """
    读取Food-101的train.json文件
    
    Returns:
        train_data: {class_name: [image_path, ...]}
    """
    print("📥 读取Food-101训练集文件...")
    
    # Food-101的meta目录路径
    meta_folder = os.path.join(data_root, 'meta')
    train_json_path = os.path.join(meta_folder, 'train.json')
    
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"训练集文件不存在: {train_json_path}")
    
    # 读取训练集JSON文件
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"   ✅ 读取到 {len(train_data)} 个食物类别")
    
    # 统计每个类别的图像数量
    total_images = 0
    class_counts = {}
    for class_name, image_paths in train_data.items():
        count = len(image_paths)
        class_counts[class_name] = count
        total_images += count
    
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    avg_count = total_images / len(class_counts)
    
    print(f"   ✅ 总训练图像数: {total_images}")
    print(f"   ✅ 每类图像数 - 最少: {min_count}, 最多: {max_count}, 平均: {avg_count:.1f}")
    
    return train_data, meta_folder


def stratified_split_food101(train_data, validation_ratio=0.2, seed=42):
    """
    分层划分Food-101验证集
    
    Args:
        train_data: {class_name: [image_path, ...]}
        validation_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        new_train_data: {class_name: [image_path, ...]} - 新训练集
        val_data: {class_name: [image_path, ...]} - 验证集
    """
    print(f"🎯 开始分层划分，验证集比例: {validation_ratio*100:.1f}%")
    
    random.seed(seed)
    
    new_train_data = {}
    val_data = {}
    
    split_stats = []
    
    for class_name, image_paths in train_data.items():
        # 随机打乱当前类别的图像
        shuffled_paths = image_paths.copy()
        random.shuffle(shuffled_paths)
        
        # 计算验证集数量（至少1个）
        total_count = len(image_paths)
        val_count = max(1, int(total_count * validation_ratio))
        train_count = total_count - val_count
        
        # 划分
        val_paths = shuffled_paths[:val_count]
        train_paths = shuffled_paths[val_count:]
        
        # 保存到结果字典
        new_train_data[class_name] = train_paths
        val_data[class_name] = val_paths
        
        # 记录统计信息
        split_stats.append({
            'class_name': class_name,
            'total': total_count,
            'train': train_count,
            'val': val_count,
            'val_ratio': val_count / total_count
        })
    
    # 输出统计信息
    total_new_train = sum(len(paths) for paths in new_train_data.values())
    total_val = sum(len(paths) for paths in val_data.values())
    
    print(f"   ✅ 新训练集图像数: {total_new_train}")
    print(f"   ✅ 验证集图像数: {total_val}")
    
    # 显示一些类别的划分详情
    print(f"   📊 各类别划分详情（前10个类别）:")
    for i, stat in enumerate(split_stats[:10]):
        print(f"      {stat['class_name']}: 总计{stat['total']:3d} → 训练{stat['train']:3d} + 验证{stat['val']:2d} (验证比例: {stat['val_ratio']:.1%})")
    
    if len(split_stats) > 10:
        print(f"      ... 还有 {len(split_stats) - 10} 个类别")
    
    # 整体统计
    total_original = sum(stat['total'] for stat in split_stats)
    actual_val_ratio = total_val / total_original
    
    print(f"   📈 整体统计:")
    print(f"      原训练集: {total_original} 张")
    print(f"      新训练集: {total_new_train} 张")
    print(f"      验证集: {total_val} 张")
    print(f"      实际验证比例: {actual_val_ratio:.1%}")
    
    return new_train_data, val_data


def save_json_files(meta_folder, new_train_data, val_data, output_dir=None):
    """
    保存新的JSON文件
    
    Args:
        meta_folder: Food-101的meta目录
        new_train_data: 新训练集数据
        val_data: 验证集数据
        output_dir: 输出目录，默认为meta_folder
    """
    print("💾 保存JSON文件...")
    
    if output_dir is None:
        output_dir = meta_folder
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存新的训练集文件
    train_json_path = os.path.join(output_dir, 'train.json')
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_train_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ 保存训练集文件: {train_json_path}")
    
    # 保存验证集文件
    val_json_path = os.path.join(output_dir, 'val.json')
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ 保存验证集文件: {val_json_path}")
    
    # 输出统计信息
    total_train_images = sum(len(paths) for paths in new_train_data.values())
    total_val_images = sum(len(paths) for paths in val_data.values())
    total_images = total_train_images + total_val_images
    
    print(f"   📊 保存统计:")
    print(f"      总类别数: {len(new_train_data)}")
    print(f"      总图像数: {total_images}")
    print(f"      新训练集: {total_train_images} ({total_train_images/total_images:.1%})")
    print(f"      验证集: {total_val_images} ({total_val_images/total_images:.1%})")
    
    return {
        'train_json': train_json_path,
        'val_json': val_json_path
    }


def verify_split_integrity(original_data, new_train_data, val_data):
    """验证划分的完整性"""
    print("🔍 验证划分完整性...")
    
    # 检查类别数量
    original_classes = set(original_data.keys())
    train_classes = set(new_train_data.keys())
    val_classes = set(val_data.keys())
    
    if original_classes != train_classes or original_classes != val_classes:
        print(f"   ❌ 类别不匹配")
        return False
    
    print(f"   ✅ 所有类别都正确保留")
    
    # 检查每个类别的图像分配
    missing_images = 0
    duplicate_images = 0
    
    for class_name in original_classes:
        original_paths = set(original_data[class_name])
        train_paths = set(new_train_data[class_name])
        val_paths = set(val_data[class_name])
        
        # 检查是否有重叠
        overlap = train_paths & val_paths
        if overlap:
            duplicate_images += len(overlap)
        
        # 检查是否有遗漏
        allocated_paths = train_paths | val_paths
        missing = original_paths - allocated_paths
        if missing:
            missing_images += len(missing)
    
    if missing_images > 0:
        print(f"   ❌ 有 {missing_images} 个图像未被分配")
        return False
    
    if duplicate_images > 0:
        print(f"   ❌ 有 {duplicate_images} 个图像被重复分配")
        return False
    
    print(f"   ✅ 所有图像都被正确分配，无重叠")
    print(f"   ✅ 划分完整性验证通过")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Food-101数据集训练集分层划分脚本')
    parser.add_argument('--validation_ratio', type=float, default=0.2, 
                       help='验证集比例 (默认: 0.2)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子 (默认: 42)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: 数据集meta目录)')
    
    args = parser.parse_args()
    
    print("🍕 Food-101数据集训练集分层划分")
    print(f"   验证集比例: {args.validation_ratio*100:.1f}%")
    print(f"   随机种子: {args.seed}")
    print("=" * 50)
    
    try:
        # 步骤1: 加载配置
        print("\n📥 步骤1: 加载数据集配置...")
        data_root = load_food101_config()
        print(f"✅ Food-101数据集路径: {data_root}")
        
        # 检查数据集是否存在
        if not os.path.exists(data_root):
            print(f"❌ Food-101数据集不存在: {data_root}")
            print("💡 请检查配置文件中的data_path是否正确")
            return
        
        # 步骤2: 读取训练集JSON文件
        original_data, meta_folder = read_food101_train_json(data_root)
        
        # 步骤3: 分层划分验证集
        new_train_data, val_data = stratified_split_food101(
            original_data, args.validation_ratio, args.seed
        )
        
        # 步骤4: 验证划分完整性
        if not verify_split_integrity(original_data, new_train_data, val_data):
            print("❌ 划分完整性验证失败")
            return
        
        # 步骤5: 保存JSON文件
        output_files = save_json_files(
            meta_folder, new_train_data, val_data, args.output_dir
        )
        
        print(f"\n🎉 分层划分完成！")
        print(f"📁 输出文件:")
        for name, path in output_files.items():
            print(f"   {name}: {path}")
        
        print(f"\n💡 使用说明:")
        print(f"   - 新的train.json包含每个类别80%的图像")
        print(f"   - 新的val.json包含每个类别20%的图像")
        print(f"   - 保持了原有的JSON格式和结构")
        print(f"   - 可直接用于Food101数据集类")
        
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
