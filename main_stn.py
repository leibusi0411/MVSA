"""
Text-Guided STN 实验主脚本

文本指导的空间变换网络(STN)训练启动脚本

核心功能：
1. 训练STN模型：学习基于文本描述的智能图像裁剪
2. 评估STN性能：测试裁剪后的图像分类准确率
3. 可视化变换：生成原图与变换图的对比可视化

"""

import fire                          # 命令行接口库，用于将函数转换为CLI
import os                           # 操作系统接口，用于文件路径操作
import random                       # 随机数生成
import yaml                         # YAML配置文件解析

# === 第三方库导入 ===
import numpy as np                  # 数值计算库
import torch                         # PyTorch深度学习框架

# === 项目内部模块导入 ===
from clip import clip               # OpenAI CLIP模型
from data_preprocess import load_multi_view_dataset  # 多视角数据集加载
from stn.multi_view_stn import MultiViewSTNModel    # 多视角STN模型
from train_multi_view_stn import train_multi_view_stn_full  # 多视角STN训练


def set_seed(seed):
    """设置所有相关库的随机数种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保CuDNN的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_stn_config_path(dataset_name, config_file=None):
    """
    获取STN专用配置文件路径

    STN训练使用STN-Config目录下的专用配置文件，与原始WCA方法完全分离

    Args:
        dataset_name: 数据集名称
        config_file: 用户指定的STN配置文件（可选，必须在STN-Config目录下）

    Returns:
        str: STN配置文件路径
    """
    if config_file:
        # 用户指定了配置文件，确保在STN-Config目录下
        if config_file.startswith('STN-Config/'):
            return config_file
        elif config_file.endswith('.yaml'):
            return f"STN-Config/{config_file}"
        else:
            return f"STN-Config/{config_file}.yaml"

    # STN数据集特化配置映射（使用数据集名称作为文件名）
    stn_config_mapping = {
        'imagenet': 'STN-Config/imagenet.yaml',
        'cub': 'STN-Config/cub.yaml',
        'food101': 'STN-Config/food101.yaml',
        'fgvc-aircraft': 'STN-Config/fgvc-aircraft.yaml',
        'oxford_pets': 'STN-Config/oxford_pets.yaml',
        'dtd': 'STN-Config/dtd.yaml',
        'place365': 'STN-Config/place365.yaml'
    }

    # 返回STN特化配置，如果不存在则使用STN基础模板
    return stn_config_mapping.get(dataset_name, 'STN-Config/base_template.yaml')





def main(
    dataset_name: str = "imagenet",
    stn_config: str = None,  # STN专用配置文件名（在STN-Config目录下）
    num_workers: int = 4,
    seed: int = 42,
    device: str = "cuda"
):
    """
    STN训练主函数


    STN配置文件位置：
        STN-Config/{stn_config}.yaml 或自动选择的数据集特化配置

    输出文件：
        - 训练模型：models/stn_{dataset}_{model}.pth

    Main function for STN training with automatic evaluation after each epoch.
    """
    # === 步骤1：环境初始化 ===
    device = torch.device(device)
    print(f"Device: {device}")
    print(f"Mode: STN Training")

    set_seed(seed)  # 设置随机种子确保可重复性
    
    # === GPU缓存清理 ===
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU缓存已清理")

    # === 步骤2：加载STN专用配置文件 ===
    config_path = get_stn_config_path(dataset_name, stn_config)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 确保配置中包含正确的数据集信息
    config['dataset'] = dataset_name
    print(f"📋 加载配置: {config_path} (数据集: {dataset_name})")

    # === 步骤3：加载预训练CLIP模型 ===
    model_size = config['model_size']  # 如"ViT-B/32"
    print(f"Loading CLIP model: {model_size}")
    clip_model, preprocess = clip.load(model_size, device=device)
    
    # === 转换CLIP模型为float32精度 ===
    print("🔧 转换CLIP模型精度: float16 → float32")
    clip_model = clip_model.float()  # 转换为float32
    
    # 验证转换结果
    if hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'conv1'):
        clip_dtype = clip_model.visual.conv1.weight.dtype
        print(f"✅ CLIP模型精度转换完成: {clip_dtype}")
    else:
        print("✅ CLIP模型精度转换完成: float32")


    # === 步骤4：创建STN模型   
    print("Creating STN model...")
    


    # 选项3: 多视角STN模型
    num_views = config['stn_config'].get('num_views', 4)  # 从配置中读取，默认4个视角
    stn_model = MultiViewSTNModel(clip_model, config['stn_config'], num_views=num_views).to(device)
    
    # === 确保整个STN模型使用float32精度 ===
    stn_model = stn_model.float()  # 确保STN组件也是float32
    print(f"✅ 使用多视角STN模型 ({num_views}个视角)")
    print("🔧 整个模型精度统一为float32")




    # === 步骤5：准备数据集 ===
    print(f"📊 准备数据集: {dataset_name}")

 

    # 多视角STN模型训练集和验证集加载 - 使用新的统一接口
    print(f"\n🔥 加载训练集...")
    train_dataloader = load_multi_view_dataset(
        dataset_name=dataset_name,
        data_path=config['data_path'],
        split='train',  # 训练集
        batch_size=config['training']['batch_size'],
        num_workers=num_workers,
        target_size=448,  # 目标裁剪尺寸
        scale_short_edge=512,  # 短边缩放尺寸
        flip_prob=0.5,    # 训练时使用翻转
        center_crop=False  # 训练时使用随机裁剪
    )
    
    print(f"\n🔥 加载验证集...")
    val_dataloader = load_multi_view_dataset(
        dataset_name=dataset_name,
        data_path=config['data_path'],
        split='val',  # 验证集
        batch_size=config['training']['batch_size'],
        num_workers=num_workers,
        target_size=448,  # 目标裁剪尺寸
        scale_short_edge=512,   # 短边缩放尺寸
        flip_prob=0.0,    # 验证时不使用翻转
        center_crop=True  # 验证时使用中心裁剪
    )
    
    print(f"\n✅ 数据集加载完成 - 标准化优化方案:")
    print(f"    📊 总样本: 训练{len(train_dataloader.dataset)} + 验证{len(val_dataloader.dataset)}")
    print(f"    🎯 数据流: PIL图像 → 归一化+标准化 → STN变换 → CLIP")
    



    # === 步骤6：开始STN训练（每个epoch后在训练集上评估） ===
    print("=== 开始STN训练 ===")
  


    # === 多视角STN训练 (带验证集监控) ===
    # 执行多视角STN训练
    print("🚀 使用多视角STN训练流程 (带验证集监控)")
    print(f"   🎯 融合模式: {config['stn_config'].get('fusion_mode', 'simple')}")
    print(f"   👁️  视角数量: {config['stn_config'].get('num_views', 4)}")
    print(f"   📊 数据格式: 448x448单一输出 + GPU延迟CLIP处理")
    print(f"   📈 验证监控: 实时过拟合检测 + 基于验证损失的早停")
    
    model_save_path = train_multi_view_stn_full(
        stn_model, train_dataloader, val_dataloader, config, device, dataset_name, model_size)






    print(f"\n🎉 STN训练完成！")
    print(f"📁 模型保存路径: {model_save_path}")
    print(f"📊 训练过程中已包含每个epoch的自动评估")



# === 程序入口点 ===
if __name__ == "__main__":
    # 使用Fire库将main函数转换为命令行接口
    # 支持的调用方式：

    #python main_stn.py --dataset_name=imagenet --stn_config=imagenet --num_workers=4 --seed=42 --device=cuda
    #python main_stn.py --dataset_name=oxford_pets --stn_config=oxford_pets --num_workers=4 --seed=42 --device=cuda
    #python main_stn.py --dataset_name=cub --stn_config=cub --num_workers=8 --seed=42 --device=cuda:1
    fire.Fire(main)
