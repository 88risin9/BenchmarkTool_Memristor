"""
CNN模块 - 提供卷积神经网络相关功能
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional

class SimpleCNN(nn.Module):
    """简单的卷积神经网络模型"""
    
    def __init__(self, input_channels: int = 1, input_size: Tuple[int, int] = (28, 28)):
        """
        初始化简单CNN模型
        :param input_channels: 输入通道数，默认为1（灰度图像）
        :param input_size: 输入图像尺寸，默认为(28, 28)
        """
        super(SimpleCNN, self).__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        
        # 第一个卷积层，输入通道->16个输出通道，3x3卷积核
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层，16->32个通道，3x3卷积核
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # 计算全连接层的输入尺寸
        # 经过两次stride=1的卷积和两次2x2的池化后
        # 尺寸变为: input_size // 4
        fc_input_size = (input_size[0] // 4) * (input_size[1] // 4) * 32
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)  # 输出10个类别
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        :param x: 输入张量，形状为[batch_size, input_channels, height, width]
        :return: 输出张量，形状为[batch_size, 10]
        """
        # 第一个卷积块：卷积->ReLU->最大池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # 第二个卷积块：卷积->ReLU->最大池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # 展平操作，准备送入全连接层
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        计算模型的参数量和预估内存使用
        :return: 包含参数量和内存使用信息的字典
        """
        # 计算总参数量
        total_params = sum(p.numel() for p in self.parameters())
        
        # 估算参数内存占用 (假设每个参数是float32，4字节)
        param_memory_mb = total_params * 4 / (1024 * 1024)
        
        # 估算前向传播特征图内存
        # 假设batch_size为1
        batch_size = 1
        
        # 第一层卷积输出: [batch_size, 16, H, W]
        conv1_output = batch_size * 16 * self.input_size[0] * self.input_size[1]
        
        # 第一层池化输出: [batch_size, 16, H/2, W/2]
        pool1_output = batch_size * 16 * (self.input_size[0] // 2) * (self.input_size[1] // 2)
        
        # 第二层卷积输出: [batch_size, 32, H/2, W/2]
        conv2_output = batch_size * 32 * (self.input_size[0] // 2) * (self.input_size[1] // 2)
        
        # 第二层池化输出: [batch_size, 32, H/4, W/4]
        pool2_output = batch_size * 32 * (self.input_size[0] // 4) * (self.input_size[1] // 4)
        
        # FC1 输出: [batch_size, 128]
        fc1_output = batch_size * 128
        
        # FC2 输出: [batch_size, 10]
        fc2_output = batch_size * 10
        
        # 总特征图内存 (以浮点数计算，4字节)
        feature_memory = (conv1_output + pool1_output + conv2_output + 
                         pool2_output + fc1_output + fc2_output) * 4
        feature_memory_mb = feature_memory / (1024 * 1024)
        
        # 总内存
        total_memory_mb = param_memory_mb + feature_memory_mb
        
        return {
            "parameters": total_params,
            "parameter_memory_mb": param_memory_mb,
            "feature_memory_mb": feature_memory_mb,
            "total_memory_mb": total_memory_mb
        }

class MemristorCNN:
    """基于忆阻器实现的CNN模拟器类"""
    
    def __init__(self):
        """初始化忆阻器CNN模拟器"""
        self.model = None
        self.input_channels = 1
        self.input_size = (28, 28)
    
    def create_model(self, input_channels: int = 1, input_size: Tuple[int, int] = (28, 28)) -> None:
        """
        创建CNN模型
        :param input_channels: 输入通道数
        :param input_size: 输入尺寸(高度,宽度)
        """
        self.input_channels = input_channels
        self.input_size = input_size
        self.model = SimpleCNN(input_channels, input_size)
        print(f"Created CNN model with {input_channels} input channels and input size {input_size}")
    
    def generate_random_input(self, batch_size: int = 1) -> torch.Tensor:
        """
        生成随机输入数据用于测试
        :param batch_size: 批次大小
        :return: 随机生成的输入张量
        """
        return torch.rand(batch_size, self.input_channels, 
                          self.input_size[0], self.input_size[1])
    
    def simulate(self, input_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        模拟CNN的前向传播
        :param input_data: 输入数据，若为None则使用随机生成的数据
        :return: 模型输出
        """
        if self.model is None:
            raise ValueError("CNN model not initialized. Call create_model() first.")
        
        # 如果没有提供输入数据，则生成随机数据
        if input_data is None:
            input_data = self.generate_random_input()
        
        # 确保输入数据形状正确
        if input_data.dim() == 3:  # 如果缺少批次维度，添加它
            input_data = input_data.unsqueeze(0)
            
        # 检查输入形状
        expected_shape = (self.input_channels, self.input_size[0], self.input_size[1])
        if input_data.shape[1:] != expected_shape:
            raise ValueError(
                f"Input shape mismatch. Expected {expected_shape} but got {tuple(input_data.shape[1:])}"
            )
        
        # 执行前向传播
        with torch.no_grad():
            output = self.model(input_data)
        
        return output
    
    def extract_kernels(self) -> Dict[str, torch.Tensor]:
        """
        提取CNN中的卷积核参数
        :return: 包含各层卷积核的字典
        """
        if self.model is None:
            raise ValueError("CNN model not initialized")
            
        kernels = {
            "conv1": self.model.conv1.weight.data.clone(),
            "conv2": self.model.conv2.weight.data.clone()
        }
        
        return kernels
    
    def simulate_on_memristor_array(self) -> Dict[str, Any]:
        """
        模拟在忆阻器阵列上执行CNN（概念演示）
        :return: 模拟结果信息
        """
        # 这个方法展示了如何将CNN映射到忆阻器阵列上
        # 实际实现需要更详细的硬件模型
        
        if self.model is None:
            raise ValueError("CNN model not initialized")
        
        # 提取卷积层的参数
        kernels = self.extract_kernels()
        
        # 获取权重统计信息
        conv1_weights = kernels["conv1"].numpy().flatten()
        conv2_weights = kernels["conv2"].numpy().flatten()
        
        # 转化成电导值（假设简单的线性映射）
        # 假设权重范围在[-1,1]，映射到电导范围[0,1]
        conv1_conductance = (conv1_weights + 1) / 2
        conv2_conductance = (conv2_weights + 1) / 2
        
        # 计算理论上所需的忆阻器数量
        # 每个卷积核的每个权重需要一个忆阻器
        num_memristors_conv1 = self.model.conv1.weight.numel()
        num_memristors_conv2 = self.model.conv2.weight.numel()
        num_memristors_fc1 = self.model.fc1.weight.numel()
        num_memristors_fc2 = self.model.fc2.weight.numel()
        total_memristors = num_memristors_conv1 + num_memristors_conv2 + num_memristors_fc1 + num_memristors_fc2
        
        # 假设的性能指标
        delay_per_layer_ms = {
            "conv1": 0.2,
            "pool1": 0.1,
            "conv2": 0.4,
            "pool2": 0.1,
            "fc1": 0.3,
            "fc2": 0.1
        }
        total_delay_ms = sum(delay_per_layer_ms.values())
        
        # 假设的能耗指标
        energy_per_memristor_nJ = 0.1  # 每个忆阻器操作的能耗
        total_energy_uJ = total_memristors * energy_per_memristor_nJ / 1000
        
        return {
            "num_memristors": {
                "conv1": num_memristors_conv1,
                "conv2": num_memristors_conv2,
                "fc1": num_memristors_fc1,
                "fc2": num_memristors_fc2,
                "total": total_memristors
            },
            "performance": {
                "layer_delay_ms": delay_per_layer_ms,
                "total_delay_ms": total_delay_ms,
                "energy_consumption_uJ": total_energy_uJ
            },
            "mapping_info": {
                "conv1_conductance_range": [float(conv1_conductance.min()), float(conv1_conductance.max())],
                "conv2_conductance_range": [float(conv2_conductance.min()), float(conv2_conductance.max())]
            }
        }

def visualize_kernel(kernel: torch.Tensor) -> None:
    """
    可视化卷积核
    :param kernel: 卷积核张量
    """
    import matplotlib.pyplot as plt
    
    # 获取卷积核的维度
    out_channels, in_channels, kh, kw = kernel.shape
    
    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(out_channels)))
    
    # 创建子图
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # 展平轴数组以便索引
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # 绘制每个输出通道的第一个输入通道的卷积核
    for i in range(out_channels):
        if i < len(axes):
            # 显示第i个卷积核的第0个输入通道
            im = axes[i].imshow(kernel[i, 0].numpy(), cmap='viridis')
            axes[i].set_title(f'Filter {i+1}')
            axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(out_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.03, 0.8])
    plt.colorbar(im, cax=cax)
    plt.suptitle('Convolution Kernels Visualization', fontsize=16)
    plt.show()

def visualize_feature_maps(feature_maps: torch.Tensor) -> None:
    """
    可视化特征图
    :param feature_maps: 特征图张量 [C, H, W]
    """
    import matplotlib.pyplot as plt
    
    # 获取通道数
    channels = feature_maps.shape[0]
    
    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(channels)))
    
    # 创建子图
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # 展平轴数组以便索引
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # 绘制每个通道的特征图
    for i in range(channels):
        if i < len(axes):
            im = axes[i].imshow(feature_maps[i].numpy(), cmap='viridis')
            axes[i].set_title(f'Channel {i+1}')
            axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.03, 0.8])
    plt.colorbar(im, cax=cax)
    plt.suptitle('Feature Maps Visualization', fontsize=16)
    plt.show()