"""
数据集工具函数模块

Dataset utilities for STN-CLIP project.

包含数据集相关的通用工具函数，避免模块间的循环导入问题。
"""

import json
from typing import Optional, Dict, List
from my_datasets import MyDataset
from utils import imagenet_classes


def wordify(string):
    """
    将下划线分隔的字符串转换为空格分隔的字符串
    
    Args:
        string (str): 输入字符串
        
    Returns:
        str: 转换后的字符串
    """
    word = string.replace("_", " ")
    return word



# === 文本描述处理相关函数 ===
#   加载数据集对应的文本描述
def load_text_prompts(dataset_name: str) -> Optional[Dict[str, List[str]]]:    #类型注解（Type Hints 
    """
     Args:
        dataset_name: 数据集名称
        
    Returns:
        返回字典，键为类别名称，值为文本描述列表，包含全部描述
        dict: 类别名称到文本描述列表的映射，如果加载失败返回None
    """
    try:
        # 处理数据集名称映射（某些数据集的文件夹名称可能不同）
        dataset_to_prompt_folder = {
            'oxford_pets': 'oxford_pet',  # oxford_pets -> oxford_pet
            # 可以在这里添加其他需要映射的数据集
        }
        #dict.get()，如果key存在，返回对应的value，否则返回默认值
        prompt_dataset_name = dataset_to_prompt_folder.get(dataset_name, dataset_name)
            
        with open(f'prompts/{prompt_dataset_name}/cupl.json', 'r') as f:
            prompts = json.load(f)  #返回一个字典    Dict[str, List[str]]   核心功能,加载全部描述
        
        # 显示加载统计信息
        print(f"📊 文本描述统计信息:")
        print(f"   - 加载的类别数量: {len(prompts)}")  #字典可以获取长度
        
        # 显示每个类别的描述数量统计
        desc_counts = [len(descriptions) for descriptions in prompts.values()]  #字典可以获取values
        if desc_counts:
            print(f"   - 平均每类描述数: {sum(desc_counts)/len(desc_counts):.1f}")
            print(f"   - 描述数量范围: {min(desc_counts)} - {max(desc_counts)}")
        
        # 显示前几个类别名称作为示例
        class_names = list(prompts.keys())[:3]  #字典可以获取keys
        print(f"   - 示例类别: {', '.join(class_names)}")
        
        return prompts
    except FileNotFoundError: #文件不存在
        print(f"⚠️ 警告: 未找到 prompts/{prompt_dataset_name}/cupl.json，使用默认模板")
        return None 
    except json.JSONDecodeError as e: #解析失败
        print(f"❌ 错误: 解析 prompts/{prompt_dataset_name}/cupl.json 失败: {e}")
        return None





def load_classes(dataset_name):
    """
    加载数据集的类别名称列表
    
    Args:
        dataset_name (str): 数据集名称
        
    Returns:
        List[str]: 处理后的类别名称列表，如果加载失败则返回None
    """
    # 对ImageNet数据集使用特殊处理（与helper.py保持一致）
    if dataset_name.startswith(MyDataset.ImageNet) or dataset_name == 'imagenet':
        classes = imagenet_classes
        print(f"✅ 使用预定义ImageNet类别: {len(classes)} 个类别")
        return classes
    
    # 对其他数据集使用JSON文件加载
    try:
        with open(f"features/{dataset_name}/{dataset_name}.json", "r") as f:
            classes = json.load(f)
        
        wordify_classes = []
        for c in classes:
            wordify_classes.append(wordify(c))
        
        print(f"✅ 成功加载类别名称: {len(wordify_classes)} 个类别")
        return wordify_classes
    
    except FileNotFoundError:
        print(f"❌ 未找到类别文件: features/{dataset_name}/{dataset_name}.json")
        return None
    except Exception as e:
        print(f"❌ 加载类别名称时发生错误: {e}")
        return None


