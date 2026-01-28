#!/usr/bin/env python3
"""
CUB数据集训练集分层划分脚本

功能：
1. 从CUB训练集中按类别分层取20%作为验证集
2. 保持每个类别的比例一致
3. 生成新的训练/验证集划分文件

使用方法：
python create_cub_validation_split.py

输出文件：
- train_val_test_split.txt: 训练/验证/测试集划分文件 (1=训练, 2=验证, 0=测试)
"""

import os
import random
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import argparse


def load_cub_config():
    """加载CUB数据集配置"""
    config_path = "STN-Config/cub.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config['data_path']


def read_cub_files(data_root):
    """
    读取CUB数据集的相关文件
    
    Returns:
        images_dict: {image_id: filename}
        train_test_split: {image_id: is_train} (1=train, 0=test)
        class_labels: {image_id: class_id}
    """
    print("📥 读取CUB数据集文件...")
    
    # 读取图像文件名映射
    images_path = os.path.join(data_root, 'images.txt')
    images_dict = {}
    with open(images_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    image_id, filename = int(parts[0]), parts[1]
                    images_dict[image_id] = filename
    
    print(f"   ✅ 读取到 {len(images_dict)} 个图像文件")
    
    # 读取训练/测试集划分
    split_path = os.path.join(data_root, 'train_test_split.txt')
    train_test_split = {}
    with open(split_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ')
                if len(parts) == 2:
                    image_id, is_train = int(parts[0]), int(parts[1])
                    train_test_split[image_id] = is_train
    
    print(f"   ✅ 读取到 {len(train_test_split)} 个图像的训练/测试划分")
    
    # 读取类别标签
    labels_path = os.path.join(data_root, 'image_class_labels.txt')
    class_labels = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ')
                if len(parts) == 2:
                    image_id, class_id = int(parts[0]), int(parts[1])
                    class_labels[image_id] = class_id
    
    print(f"   ✅ 读取到 {len(class_labels)} 个图像的类别标签")
    
    return images_dict, train_test_split, class_labels


def build_training_data_structure(images_dict, train_test_split, class_labels):
    """
    构建训练数据的数据结构
    
    Returns:
        training_data: {class_id: [image_id, ...]} - 每个类别的训练图像ID列表
        training_samples: [(image_id, filename, class_id), ...] - 所有训练样本信息
    """
    print("🏗️ 构建训练数据结构...")
    
    training_data = defaultdict(list)
    training_samples = []
    
    # 筛选出训练集数据
    for image_id in images_dict:
        if train_test_split.get(image_id, 0) == 1:  # 训练集
            filename = images_dict[image_id]
            class_id = class_labels[image_id]
            
            training_data[class_id].append(image_id)
            training_samples.append((image_id, filename, class_id))
    
    # 统计信息
    total_train_images = len(training_samples)
    num_classes = len(training_data)
    
    print(f"   ✅ 训练集总图像数: {total_train_images}")
    print(f"   ✅ 类别数: {num_classes}")
    
    # 每个类别的图像数量统计
    class_counts = {class_id: len(image_ids) for class_id, image_ids in training_data.items()}
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    avg_count = sum(class_counts.values()) / len(class_counts)
    
    print(f"   ✅ 每类图像数 - 最少: {min_count}, 最多: {max_count}, 平均: {avg_count:.1f}")
    
    return training_data, training_samples


def stratified_split_validation(training_data, validation_ratio=0.2, seed=42):
    """
    分层划分验证集
    
    Args:
        training_data: {class_id: [image_id, ...]}
        validation_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        new_train_ids: set - 新训练集的图像ID
        validation_ids: set - 验证集的图像ID
    """
    print(f"🎯 开始分层划分，验证集比例: {validation_ratio*100:.1f}%")
    
    random.seed(seed)
    
    new_train_ids = set()
    validation_ids = set()
    
    split_stats = []
    
    for class_id, image_ids in training_data.items():
        # 随机打乱当前类别的图像
        shuffled_ids = image_ids.copy()
        random.shuffle(shuffled_ids)
        
        # 计算验证集数量（至少1个）
        total_count = len(image_ids)
        val_count = max(1, int(total_count * validation_ratio))
        train_count = total_count - val_count
        
        # 划分
        val_ids = shuffled_ids[:val_count]
        train_ids = shuffled_ids[val_count:]
        
        # 添加到结果集合
        validation_ids.update(val_ids)
        new_train_ids.update(train_ids)
        
        # 记录统计信息
        split_stats.append({
            'class_id': class_id,
            'total': total_count,
            'train': train_count,
            'val': val_count,
            'val_ratio': val_count / total_count
        })
    
    # 输出统计信息
    print(f"   ✅ 新训练集图像数: {len(new_train_ids)}")
    print(f"   ✅ 验证集图像数: {len(validation_ids)}")
    
    # 显示一些类别的划分详情
    print(f"   📊 各类别划分详情（前10个类别）:")
    for i, stat in enumerate(split_stats[:10]):
        print(f"      类别 {stat['class_id']:3d}: 总计{stat['total']:2d} → 训练{stat['train']:2d} + 验证{stat['val']:2d} (验证比例: {stat['val_ratio']:.1%})")
    
    if len(split_stats) > 10:
        print(f"      ... 还有 {len(split_stats) - 10} 个类别")
    
    # 整体统计
    total_original = sum(stat['total'] for stat in split_stats)
    total_new_train = sum(stat['train'] for stat in split_stats)
    total_val = sum(stat['val'] for stat in split_stats)
    actual_val_ratio = total_val / total_original
    
    print(f"   📈 整体统计:")
    print(f"      原训练集: {total_original} 张")
    print(f"      新训练集: {total_new_train} 张")
    print(f"      验证集: {total_val} 张")
    print(f"      实际验证比例: {actual_val_ratio:.1%}")
    
    return new_train_ids, validation_ids


def save_split_files(data_root, images_dict, train_test_split, new_train_ids, validation_ids, output_dir=None):
    """
    保存新的划分文件
    
    Args:
        data_root: CUB数据集根目录
        images_dict: {image_id: filename}
        train_test_split: 原始训练/测试划分
        new_train_ids: 新训练集ID
        validation_ids: 验证集ID
        output_dir: 输出目录，默认为data_root
    """
    print("💾 保存划分文件...")
    
    if output_dir is None:
        output_dir = data_root
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练/验证/测试集划分文件
    # 格式: image_id split_type (1=训练, 2=验证, 0=测试)
    train_val_test_path = os.path.join(output_dir, 'train_val_test_split.txt')
    with open(train_val_test_path, 'w', encoding='utf-8') as f:
        for image_id in sorted(images_dict.keys()):
            if image_id in new_train_ids:
                split_type = 1  # 新训练集
            elif image_id in validation_ids:
                split_type = 2  # 验证集
            else:
                split_type = 0  # 测试集（保持原样）
            
            f.write(f"{image_id} {split_type}\n")
    
    print(f"   ✅ 保存划分文件: {train_val_test_path}")
    
    # 输出统计信息
    total_images = len(images_dict)
    original_train = sum(1 for split in train_test_split.values() if split == 1)
    original_test = sum(1 for split in train_test_split.values() if split == 0)
    new_train_count = len(new_train_ids)
    val_count = len(validation_ids)
    test_count = total_images - new_train_count - val_count
    
    print(f"   📊 划分统计:")
    print(f"      总图像数: {total_images}")
    print(f"      新训练集: {new_train_count} ({new_train_count/total_images:.1%})")
    print(f"      验证集: {val_count} ({val_count/total_images:.1%})")
    print(f"      测试集: {test_count} ({test_count/total_images:.1%})")
    print(f"      验证集占原训练集比例: {val_count/original_train:.1%}")
    
    return train_val_test_path


def verify_split_integrity(images_dict, train_test_split, new_train_ids, validation_ids):
    """验证划分的完整性"""
    print("🔍 验证划分完整性...")
    
    # 检查是否有重叠
    overlap = new_train_ids & validation_ids
    if overlap:
        print(f"   ❌ 训练集和验证集有重叠: {len(overlap)} 个图像")
        return False
    else:
        print(f"   ✅ 训练集和验证集无重叠")
    
    # 检查是否所有原训练集图像都被分配
    original_train_ids = {img_id for img_id, is_train in train_test_split.items() if is_train == 1}
    allocated_ids = new_train_ids | validation_ids
    
    missing_ids = original_train_ids - allocated_ids
    extra_ids = allocated_ids - original_train_ids
    
    if missing_ids:
        print(f"   ❌ 有 {len(missing_ids)} 个原训练集图像未被分配")
        return False
    
    if extra_ids:
        print(f"   ❌ 有 {len(extra_ids)} 个非训练集图像被错误分配")
        return False
    
    print(f"   ✅ 所有原训练集图像都被正确分配")
    print(f"   ✅ 划分完整性验证通过")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CUB数据集训练集分层划分脚本')
    parser.add_argument('--validation_ratio', type=float, default=0.2, 
                       help='验证集比例 (默认: 0.2)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子 (默认: 42)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: 数据集根目录)')
    
    args = parser.parse_args()
    
    print("🎯 CUB数据集训练集分层划分")
    print(f"   验证集比例: {args.validation_ratio*100:.1f}%")
    print(f"   随机种子: {args.seed}")
    print("=" * 50)
    
    try:
        # 步骤1: 加载配置
        print("\n📥 步骤1: 加载数据集配置...")
        data_root = load_cub_config()
        print(f"✅ CUB数据集路径: {data_root}")
        
        # 检查数据集是否存在
        if not os.path.exists(data_root):
            print(f"❌ CUB数据集不存在: {data_root}")
            print("💡 请检查配置文件中的data_path是否正确")
            return
        
        # 步骤2: 读取数据集文件
        images_dict, train_test_split, class_labels = read_cub_files(data_root)
        
        # 步骤3: 构建训练数据结构
        training_data, training_samples = build_training_data_structure(
            images_dict, train_test_split, class_labels
        )
        
        # 步骤4: 分层划分验证集
        new_train_ids, validation_ids = stratified_split_validation(
            training_data, args.validation_ratio, args.seed
        )
        
        # 步骤5: 验证划分完整性
        if not verify_split_integrity(images_dict, train_test_split, new_train_ids, validation_ids):
            print("❌ 划分完整性验证失败")
            return
        
        # 步骤6: 保存划分文件
        output_file = save_split_files(
            data_root, images_dict, train_test_split, 
            new_train_ids, validation_ids, args.output_dir
        )
        
        print(f"\n🎉 分层划分完成！")
        print(f"📁 输出文件: {output_file}")
        
        print(f"\n💡 使用说明:")
        print(f"   - 文件格式: image_id split_type")
        print(f"   - 标志含义: 1=训练集, 2=验证集, 0=测试集")
        print(f"   - 验证集是从原训练集中按类别分层抽取20%得到")
        
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
