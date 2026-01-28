#!/usr/bin/env python3
"""
Oxford Pets数据集训练集分层划分脚本

功能：
1. 从Oxford Pets的trainval.txt中按类别分层取20%作为验证集
2. 保持每个宠物品种的比例一致
3. 生成新的train.txt和val.txt文件

使用方法：
python create_oxford_pets_validation_split.py

输出文件：
- train.txt: 新的训练集文件（原trainval的80%）
- val.txt: 新的验证集文件（原trainval的20%）

数据结构：
trainval.txt文件格式：
Abyssinian_1 1 1 1
图像ID 宠物品种标签(1-37) 物种标签(1=猫,2=狗) 品种内ID
"""

import os
import random
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import argparse


def load_oxford_pets_config():
    """加载Oxford Pets数据集配置"""
    config_path = "STN-Config/oxford_pets.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config['data_path']


def read_oxford_pets_trainval(data_root):
    """
    读取Oxford Pets的trainval.txt文件
    
    Returns:
        trainval_data: [(image_id, pet_label, species_label, breed_id), ...]
        class_to_samples: {pet_label: [(image_id, pet_label, species_label, breed_id), ...]}
    """
    print("📥 读取Oxford Pets训练验证集文件...")
    
    # Oxford Pets的annotations目录路径
    annotations_folder = os.path.join(data_root, 'oxford-iiit-pet', 'annotations')
    trainval_path = os.path.join(annotations_folder, 'trainval.txt')
    
    if not os.path.exists(trainval_path):
        raise FileNotFoundError(f"训练验证集文件不存在: {trainval_path}")
    
    # 读取trainval.txt文件
    trainval_data = []
    class_to_samples = defaultdict(list)
    
    with open(trainval_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    image_id = parts[0]
                    pet_label = int(parts[1])      # 宠物品种标签 (1-37)
                    species_label = int(parts[2])  # 物种标签 (1=猫, 2=狗)
                    breed_id = int(parts[3])       # 品种内ID
                    
                    sample = (image_id, pet_label, species_label, breed_id)
                    trainval_data.append(sample)
                    class_to_samples[pet_label].append(sample)
    
    print(f"   ✅ 读取到 {len(trainval_data)} 个训练验证样本")
    print(f"   ✅ 包含 {len(class_to_samples)} 个宠物品种")
    
    # 统计每个品种的样本数量
    class_counts = {pet_label: len(samples) for pet_label, samples in class_to_samples.items()}
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    avg_count = sum(class_counts.values()) / len(class_counts)
    
    print(f"   ✅ 每个品种样本数 - 最少: {min_count}, 最多: {max_count}, 平均: {avg_count:.1f}")
    
    return trainval_data, class_to_samples, annotations_folder


def stratified_split_oxford_pets(class_to_samples, validation_ratio=0.2, seed=42):
    """
    分层划分Oxford Pets验证集
    
    Args:
        class_to_samples: {pet_label: [sample, ...]}
        validation_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        train_samples: [sample, ...] - 新训练集
        val_samples: [sample, ...] - 验证集
    """
    print(f"🎯 开始分层划分，验证集比例: {validation_ratio*100:.1f}%")
    
    random.seed(seed)
    
    train_samples = []
    val_samples = []
    
    split_stats = []
    
    for pet_label, samples in class_to_samples.items():
        # 随机打乱当前品种的样本
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        # 计算验证集数量（至少1个）
        total_count = len(samples)
        val_count = max(1, int(total_count * validation_ratio))
        train_count = total_count - val_count
        
        # 划分
        val_breed_samples = shuffled_samples[:val_count]
        train_breed_samples = shuffled_samples[val_count:]
        
        # 添加到结果列表
        val_samples.extend(val_breed_samples)
        train_samples.extend(train_breed_samples)
        
        # 记录统计信息
        split_stats.append({
            'pet_label': pet_label,
            'total': total_count,
            'train': train_count,
            'val': val_count,
            'val_ratio': val_count / total_count
        })
    
    # 输出统计信息
    print(f"   ✅ 新训练集样本数: {len(train_samples)}")
    print(f"   ✅ 验证集样本数: {len(val_samples)}")
    
    # 显示一些品种的划分详情
    print(f"   📊 各品种划分详情（前10个品种）:")
    for i, stat in enumerate(split_stats[:10]):
        print(f"      品种 {stat['pet_label']:2d}: 总计{stat['total']:2d} → 训练{stat['train']:2d} + 验证{stat['val']:2d} (验证比例: {stat['val_ratio']:.1%})")
    
    if len(split_stats) > 10:
        print(f"      ... 还有 {len(split_stats) - 10} 个品种")
    
    # 整体统计
    total_original = sum(stat['total'] for stat in split_stats)
    actual_val_ratio = len(val_samples) / total_original
    
    print(f"   📈 整体统计:")
    print(f"      原训练验证集: {total_original} 张")
    print(f"      新训练集: {len(train_samples)} 张")
    print(f"      验证集: {len(val_samples)} 张")
    print(f"      实际验证比例: {actual_val_ratio:.1%}")
    
    return train_samples, val_samples


def save_txt_files(annotations_folder, train_samples, val_samples, output_dir=None):
    """
    保存新的txt文件
    
    Args:
        annotations_folder: Oxford Pets的annotations目录
        train_samples: 训练集样本
        val_samples: 验证集样本
        output_dir: 输出目录，默认为annotations_folder
    """
    print("💾 保存txt文件...")
    
    if output_dir is None:
        output_dir = annotations_folder
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存新的训练集文件
    train_txt_path = os.path.join(output_dir, 'train.txt')
    with open(train_txt_path, 'w', encoding='utf-8') as f:
        for image_id, pet_label, species_label, breed_id in train_samples:
            f.write(f"{image_id} {pet_label} {species_label} {breed_id}\n")
    
    print(f"   ✅ 保存训练集文件: {train_txt_path}")
    
    # 保存验证集文件
    val_txt_path = os.path.join(output_dir, 'val.txt')
    with open(val_txt_path, 'w', encoding='utf-8') as f:
        for image_id, pet_label, species_label, breed_id in val_samples:
            f.write(f"{image_id} {pet_label} {species_label} {breed_id}\n")
    
    print(f"   ✅ 保存验证集文件: {val_txt_path}")
    
    # 输出统计信息
    total_samples = len(train_samples) + len(val_samples)
    
    print(f"   📊 保存统计:")
    print(f"      总样本数: {total_samples}")
    print(f"      新训练集: {len(train_samples)} ({len(train_samples)/total_samples:.1%})")
    print(f"      验证集: {len(val_samples)} ({len(val_samples)/total_samples:.1%})")
    
    return {
        'train_txt': train_txt_path,
        'val_txt': val_txt_path
    }


def verify_split_integrity(original_samples, train_samples, val_samples):
    """验证划分的完整性"""
    print("🔍 验证划分完整性...")
    
    # 创建图像ID集合用于比较
    original_ids = {sample[0] for sample in original_samples}
    train_ids = {sample[0] for sample in train_samples}
    val_ids = {sample[0] for sample in val_samples}
    
    # 检查是否有重叠
    overlap = train_ids & val_ids
    if overlap:
        print(f"   ❌ 训练集和验证集有重叠: {len(overlap)} 个样本")
        return False
    else:
        print(f"   ✅ 训练集和验证集无重叠")
    
    # 检查是否所有原样本都被分配
    allocated_ids = train_ids | val_ids
    
    missing_ids = original_ids - allocated_ids
    extra_ids = allocated_ids - original_ids
    
    if missing_ids:
        print(f"   ❌ 有 {len(missing_ids)} 个原样本未被分配")
        return False
    
    if extra_ids:
        print(f"   ❌ 有 {len(extra_ids)} 个非原始样本被错误分配")
        return False
    
    print(f"   ✅ 所有原样本都被正确分配")
    print(f"   ✅ 划分完整性验证通过")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Oxford Pets数据集训练集分层划分脚本')
    parser.add_argument('--validation_ratio', type=float, default=0.2, 
                       help='验证集比例 (默认: 0.2)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子 (默认: 42)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: 数据集annotations目录)')
    
    args = parser.parse_args()
    
    print("🐱🐶 Oxford Pets数据集训练集分层划分")
    print(f"   验证集比例: {args.validation_ratio*100:.1f}%")
    print(f"   随机种子: {args.seed}")
    print("=" * 50)
    
    try:
        # 步骤1: 加载配置
        print("\n📥 步骤1: 加载数据集配置...")
        data_root = load_oxford_pets_config()
        print(f"✅ Oxford Pets数据集路径: {data_root}")
        
        # 检查数据集是否存在
        if not os.path.exists(data_root):
            print(f"❌ Oxford Pets数据集不存在: {data_root}")
            print("💡 请检查配置文件中的data_path是否正确")
            return
        
        # 步骤2: 读取trainval.txt文件
        original_samples, class_to_samples, annotations_folder = read_oxford_pets_trainval(data_root)
        
        # 步骤3: 分层划分验证集
        train_samples, val_samples = stratified_split_oxford_pets(
            class_to_samples, args.validation_ratio, args.seed
        )
        
        # 步骤4: 验证划分完整性
        if not verify_split_integrity(original_samples, train_samples, val_samples):
            print("❌ 划分完整性验证失败")
            return
        
        # 步骤5: 保存txt文件
        output_files = save_txt_files(
            annotations_folder, train_samples, val_samples, args.output_dir
        )
        
        print(f"\n🎉 分层划分完成！")
        print(f"📁 输出文件:")
        for name, path in output_files.items():
            print(f"   {name}: {path}")
        
        print(f"\n💡 使用说明:")
        print(f"   - 新的train.txt包含每个品种80%的样本")
        print(f"   - 新的val.txt包含每个品种20%的样本")
        print(f"   - 保持了原有的文件格式")
        print(f"   - 可直接用于Oxford Pets数据集类")
        
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
