"""
神经网络模拟模块 - 使用PyTorch实现基于忆阻器的神经网络模拟
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

class SimpleFCNetwork(nn.Module):
    """简单的全连接神经网络模型，模拟忆阻器阵列的矩阵运算"""
    
    def __init__(self, input_size: int, output_size: int):
        """
        初始化神经网络
        :param input_size: 输入层大小
        :param output_size: 输出层大小
        """
        super(SimpleFCNetwork, self).__init__()
        # 定义无偏置的全连接层，权重矩阵模拟忆阻器的电导矩阵
        self.fc = nn.Linear(input_size, output_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        :param x: 输入张量
        :return: 输出张量
        """
        return self.fc(x)

class MemristorNN:
    """基于忆阻器的神经网络模拟器，将电导矩阵映射为神经网络权重"""
    
    def __init__(self):
        """初始化神经网络模拟器"""
        self.model = None
    
    def create_network(self, matrix: np.ndarray) -> None:
        """
        创建基于忆阻器电导矩阵的神经网络
        :param matrix: 电导矩阵，用作神经网络权重
        """
        input_size = matrix.shape[1]
        output_size = matrix.shape[0]
        
        # 创建模型实例
        self.model = SimpleFCNetwork(input_size, output_size)
        
        # 设置网络权重为电导矩阵值
        with torch.no_grad():
            self.model.fc.weight = nn.Parameter(torch.tensor(matrix, dtype=torch.float32))
    
    def simulate(self, input_vector: np.ndarray) -> np.ndarray:
        """
        模拟神经网络计算，执行前向传播
        :param input_vector: 输入电压向量
        :return: 输出电流向量
        """
        if self.model is None:
            raise ValueError("Network model not initialized")
            
        # 转换输入向量为PyTorch张量
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        
        # 前向传播计算输出
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 转换回NumPy数组返回
        return output.numpy()
    
    def batch_simulate(self, input_vectors: np.ndarray) -> np.ndarray:
        """
        批量模拟多个输入向量
        :param input_vectors: 多个输入向量组成的矩阵[批量大小, 输入维度]
        :return: 输出矩阵[批量大小, 输出维度]
        """
        if self.model is None:
            raise ValueError("Network model not initialized")
            
        # 转换输入为张量
        input_tensor = torch.tensor(input_vectors, dtype=torch.float32)
        
        # 批量前向传播
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output.numpy()