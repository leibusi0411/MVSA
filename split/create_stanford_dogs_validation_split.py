#!/usr/bin/env python3
"""
Stanford Dogs Dataset - 创建验证集划分脚本

基于原始训练集进行分层抽样，每个类别选取20%作为验证集，80%作为新的训练集。
输入: train_list.mat (原始训练集)
输出: 
  - train_split.mat (新的训练集，80%)
  - val_split.mat (验证集，20%)
  - split_info.json (划分统计信息)
"""

import os
import scipy.io
import numpy as np
import json
from collections import defaultdict, Counter
import argparse


def get_stanford_dogs_data_root():
    """获取Stanford Dogs数据集的根路径"""
    # Stanford Dogs数据集的基础目录
    # 目录结构:
    # - lists/: 包含train_list.mat, test_list.mat
    # - images/: 包含120个品种子目录
    # - annotation/: 包含标注文件
    return '/mnt/e3319bd7-a0cc-41a8-9825-36b781a06ce8/xzy/project/DATA/stanford_dogs'


def read_train_mat_file(data_root):
    """读取原始的train_list.mat文件"""
    # .mat文件在lists子目录中
    train_mat_path = os.path.join(data_root, 'lists', 'train_list.mat')
    
    if not os.path.exists(train_mat_path):
        raise FileNotFoundError(f"找不到文件: {train_mat_path}")
    
    print(f"读取文件: {train_mat_path}")
    
    # 读取MATLAB文件
    data = scipy.io.loadmat(train_mat_path)
    
    # 提取annotation_list和labels
    annotation_list = data['annotation_list']
    labels = data['labels']
    
    print(f"原始数据形状: annotation_list={annotation_list.shape}, labels={labels.shape}")
    
    # 处理嵌套数组结构
    annotations = [item[0][0] for item in annotation_list]
    labels = [item[0] - 1 for item in labels]  # 1-based -> 0-based
    
    print(f"处理后数据长度: annotations={len(annotations)}, labels={len(labels)}")
    print(f"类别范围: {min(labels)} - {max(labels)} (共{max(labels)-min(labels)+1}个类别)")
    
    return annotations, labels


def analyze_class_distribution(annotations, labels):
    """分析类别分布"""
    class_counts = Counter(labels)
    
    print(f"\n📊 原始训练集类别分布分析:")
    print(f"总样本数: {len(annotations)}")
    print(f"类别数: {len(class_counts)}")
    print(f"每类样本数范围: {min(class_counts.values())} - {max(class_counts.values())}")
    print(f"平均每类样本数: {len(annotations) / len(class_counts):.1f}")
    
    # 显示前10个类别的分布
    print(f"\n前10个类别的样本数:")
    for class_id in sorted(class_counts.keys())[:10]:
        count = class_counts[class_id]
        # 从annotation中提取品种名称示例
        example_annotation = [ann for ann, label in zip(annotations, labels) if label == class_id][0]
        breed_name = example_annotation.split('-')[1].split('_')[0]
        print(f"  类别 {class_id:3d}: {count:3d} 样本 (例: {breed_name})")
    
    return class_counts


def create_stratified_split(annotations, labels, val_ratio=0.2, random_state=42):
    """创建分层抽样的训练/验证分割"""
    print(f"\n🔄 执行分层抽样 (验证集比例: {val_ratio:.1%}):")
    
    # 按类别分组
    class_groups = defaultdict(list)
    for i, (annotation, label) in enumerate(zip(annotations, labels)):
        class_groups[label].append((annotation, label, i))
    
    train_data = []
    val_data = []
    split_stats = {}
    
    print(f"每个类别的划分详情:")
    
    for class_id in sorted(class_groups.keys()):
        class_samples = class_groups[class_id]
        total_samples = len(class_samples)
        
        # 计算验证集样本数 (至少1个，最多不超过总数的val_ratio)
        val_size = max(1, int(total_samples * val_ratio))
        train_size = total_samples - val_size
        
        # 随机抽样
        np.random.seed(random_state + class_id)  # 确保每个类别的随机种子不同
        indices = np.random.permutation(total_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 分割数据
        class_train = [class_samples[i] for i in train_indices]
        class_val = [class_samples[i] for i in val_indices]
        
        train_data.extend(class_train)
        val_data.extend(class_val)
        
        # 记录统计信息
        split_stats[int(class_id)] = {
            'total': total_samples,
            'train': len(class_train),
            'val': len(class_val),
            'val_ratio': len(class_val) / total_samples
        }
        
        # 打印前20个类别的详情
        if class_id < 20:
            example_annotation = class_samples[0][0]
            breed_name = example_annotation.split('-')[1].split('_')[0]
            print(f"  类别 {class_id:3d} ({breed_name:15s}): {total_samples:3d} -> 训练 {train_size:3d}, 验证 {val_size:2d} ({val_size/total_samples:.1%})")
    
    print(f"\n📈 分割结果统计:")
    print(f"总样本数: {len(annotations)}")
    print(f"新训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    print(f"验证集比例: {len(val_data) / len(annotations):.1%}")
    
    return train_data, val_data, split_stats


def save_mat_files(train_data, val_data, output_dir):
    """保存新的.mat文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备训练集数据
    train_annotations = [item[0] for item in train_data]
    train_labels = [item[1] + 1 for item in train_data]  # 0-based -> 1-based
    
    # 准备验证集数据
    val_annotations = [item[0] for item in val_data]
    val_labels = [item[1] + 1 for item in val_data]  # 0-based -> 1-based
    
    # 基于原始文件真实结构的精确复制
    def format_for_matlab(annotations, labels):
        """
        根据验证发现的原始结构：
        
        annotation_list: (N, 1) object数组
        - annotation_list[i]: 包含一个numpy数组的object
        - annotation_list[i][0]: numpy数组包含字符串
        
        labels: (N, 1) uint8数组 - 不是object!
        - labels[i]: 包含一个uint8值的数组
        - labels[i][0]: 直接的uint8标量
        """
        
        n_samples = len(annotations)
        
        # 1. 创建annotation_list - (N, 1) object数组
        annotation_array = np.empty((n_samples, 1), dtype=object)
        for i, ann in enumerate(annotations):
            # 每个cell包含一个numpy字符串数组
            annotation_array[i, 0] = np.array([ann], dtype=np.str_)
        
        # 2. 创建labels - (N, 1) uint8数组，不是object!
        label_array = np.array([[label] for label in labels], dtype=np.uint8)
        
        return annotation_array, label_array
    
    # 格式化数据
    train_annotation_array, train_label_array = format_for_matlab(train_annotations, train_labels)
    val_annotation_array, val_label_array = format_for_matlab(val_annotations, val_labels)
    
    # 保存训练集
    train_output_path = os.path.join(output_dir, 'train_split.mat')
    scipy.io.savemat(train_output_path, {
        'annotation_list': train_annotation_array,
        'labels': train_label_array
    })
    print(f"✅ 保存新训练集: {train_output_path}")
    print(f"   包含 {len(train_annotations)} 个样本")
    
    # 保存验证集
    val_output_path = os.path.join(output_dir, 'val_split.mat')
    scipy.io.savemat(val_output_path, {
        'annotation_list': val_annotation_array,
        'labels': val_label_array
    })
    print(f"✅ 保存验证集: {val_output_path}")
    print(f"   包含 {len(val_annotations)} 个样本")
    
    return train_output_path, val_output_path


def save_split_info(split_stats, train_data, val_data, output_dir):
    """保存划分统计信息"""
    split_info = {
        'dataset': 'Stanford Dogs',
        'split_method': 'stratified_sampling',
        'validation_ratio': 0.2,
        'total_classes': len(split_stats),
        'total_samples': len(train_data) + len(val_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'class_distribution': split_stats,
        'summary': {
            'min_samples_per_class': min(stats['total'] for stats in split_stats.values()),
            'max_samples_per_class': max(stats['total'] for stats in split_stats.values()),
            'avg_samples_per_class': sum(stats['total'] for stats in split_stats.values()) / len(split_stats),
            'min_val_samples_per_class': min(stats['val'] for stats in split_stats.values()),
            'max_val_samples_per_class': max(stats['val'] for stats in split_stats.values()),
            'avg_val_samples_per_class': sum(stats['val'] for stats in split_stats.values()) / len(split_stats)
        }
    }
    
    info_path = os.path.join(output_dir, 'split_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 保存划分信息: {info_path}")
    return info_path


def verify_split_files(train_path, val_path):
    """验证生成的.mat文件"""
    print(f"\n🔍 验证生成的文件:")
    
    # 验证训练集文件
    train_data = scipy.io.loadmat(train_path)
    train_annotations = [str(item[0][0]) for item in train_data['annotation_list']]
    train_labels = [int(item[0]) for item in train_data['labels']]
    
    print(f"新训练集: {len(train_annotations)} 样本")
    print(f"  标签范围: {min(train_labels)} - {max(train_labels)}")
    print(f"  类别数: {len(set(train_labels))}")
    
    # 验证验证集文件
    val_data = scipy.io.loadmat(val_path)
    val_annotations = [str(item[0][0]) for item in val_data['annotation_list']]
    val_labels = [int(item[0]) for item in val_data['labels']]
    
    print(f"验证集: {len(val_annotations)} 样本")
    print(f"  标签范围: {min(val_labels)} - {max(val_labels)}")
    print(f"  类别数: {len(set(val_labels))}")
    
    # 检查是否有重叠 - 确保转换为字符串类型
    train_set = set(train_annotations)
    val_set = set(val_annotations)
    overlap = train_set & val_set
    
    if overlap:
        print(f"⚠️  警告: 发现 {len(overlap)} 个重叠样本")
    else:
        print(f"✅ 验证通过: 训练集和验证集无重叠")
    
    return len(overlap) == 0


def main():
    parser = argparse.ArgumentParser(description='创建Stanford Dogs数据集的验证集划分')
    parser.add_argument('--data_root', type=str, default=None, 
                        help='数据集根目录路径 (包含train_list.mat的目录)')
    parser.add_argument('--output_dir', type=str, default='/mnt/e3319bd7-a0cc-41a8-9825-36b781a06ce8/xzy/project/DATA/stanford_dogs/splits',
                        help='输出目录路径')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例 (默认: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 确定数据根目录
    if args.data_root is None:
        args.data_root = get_stanford_dogs_data_root()
    
    print("=" * 60)
    print("Stanford Dogs Dataset - 验证集划分工具")
    print("=" * 60)
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"验证集比例: {args.val_ratio:.1%}")
    print(f"随机种子: {args.random_seed}")
    
    try:
        # 1. 读取原始训练数据
        annotations, labels = read_train_mat_file(args.data_root)
        
        # 2. 分析类别分布
        class_counts = analyze_class_distribution(annotations, labels)
        
        # 3. 创建分层抽样分割
        train_data, val_data, split_stats = create_stratified_split(
            annotations, labels, 
            val_ratio=args.val_ratio, 
            random_state=args.random_seed
        )
        
        # 4. 保存.mat文件
        train_path, val_path = save_mat_files(train_data, val_data, args.output_dir)
        
        # 5. 保存统计信息
        info_path = save_split_info(split_stats, train_data, val_data, args.output_dir)
        
        # 6. 验证生成的文件
        is_valid = verify_split_files(train_path, val_path)
        
        print(f"\n{'='*60}")
        print("✅ 验证集划分完成!")
        print(f"{'='*60}")
        print(f"输出文件:")
        print(f"  📁 {train_path}")
        print(f"  📁 {val_path}")  
        print(f"  📁 {info_path}")
        print(f"\n使用方法:")
        print(f"  在stanford_dogs.py中修改load_split()方法，")
        print(f"  使用train_split.mat和val_split.mat替代原始文件")
        
        if is_valid:
            print(f"\n🎉 所有验证通过，可以安全使用生成的划分文件!")
        else:
            print(f"\n⚠️  请检查生成的文件，发现一些问题")
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
