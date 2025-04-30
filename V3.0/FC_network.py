"""
神经网络模拟模块 - 使用PyTorch实现基于忆阻器的神经网络模拟
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from matrix_operations import split_weights_pos_neg, map_conductance_to_physical

class FCLayer:
    """全连接层类，用于保存层配置和权重矩阵"""
    
    def __init__(self, input_size: int, output_size: int, weight_matrix: Optional[np.ndarray] = None):
        """
        初始化全连接层
        :param input_size: 输入大小
        :param output_size: 输出大小
        :param weight_matrix: 权重矩阵(可选)，若未提供则随机初始化
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # 如果没有提供权重矩阵，则随机生成一个
        if weight_matrix is None:
            self.weight_matrix = np.random.uniform(0, 1, (output_size, input_size))
        else:
            # 检查矩阵维度是否匹配
            if weight_matrix.shape != (output_size, input_size):
                raise ValueError(f"Weight matrix shape {weight_matrix.shape} doesn't match expected shape ({output_size}, {input_size})")
            self.weight_matrix = weight_matrix

class MultilayerFCNetwork(nn.Module):
    """多层全连接神经网络模型，模拟忆阻器阵列的矩阵运算"""
    
    def __init__(self, layer_sizes: List[int]):
        """
        初始化多层神经网络
        :param layer_sizes: 各层大小列表，第一个是输入层，最后一个是输出层
        """
        super(MultilayerFCNetwork, self).__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output layers are required")
            
        self.layers = nn.ModuleList()
        
        # 创建各层
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        :param x: 输入张量
        :return: 输出张量
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def set_weights(self, weight_matrices: List[np.ndarray]) -> None:
        """
        设置网络各层权重
        :param weight_matrices: 权重矩阵列表
        """
        if len(weight_matrices) != len(self.layers):
            raise ValueError(f"Number of weight matrices {len(weight_matrices)} doesn't match number of layers {len(self.layers)}")
            
        with torch.no_grad():
            for i, matrix in enumerate(weight_matrices):
                self.layers[i].weight = nn.Parameter(torch.tensor(matrix, dtype=torch.float32))

class MemristorNN:
    """基于忆阻器的神经网络模拟器，将电导矩阵映射为神经网络权重"""
    
    def __init__(self):
        """初始化神经网络模拟器"""
        self.model = None
        self.fc_layers = []  # 保存层配置和权重矩阵
    
    def create_network(self, matrices: List[np.ndarray] = None, layer_sizes: List[int] = None) -> None:
        """
        创建多层神经网络
        :param matrices: 电导矩阵列表，用作各层网络权重
        :param layer_sizes: 各层大小列表，仅在未提供matrices时使用
        """
        # 检查参数
        if matrices is None and layer_sizes is None:
            raise ValueError("Either conductance matrix list or layer sizes list must be provided")
            
        # 如果提供了电导矩阵列表
        if matrices is not None:
            # 推断层大小
            layer_sizes = [matrices[0].shape[1]]  # 第一个矩阵的列数是输入大小
            for matrix in matrices:
                layer_sizes.append(matrix.shape[0])  # 每个矩阵的行数是该层输出大小
            
            # 创建模型
            self.model = MultilayerFCNetwork(layer_sizes)
            
            # 设置权重
            self.model.set_weights(matrices)
            
            # 保存层配置
            self.fc_layers = []
            for i, matrix in enumerate(matrices):
                input_size = matrix.shape[1]
                output_size = matrix.shape[0]
                self.fc_layers.append(FCLayer(input_size, output_size, matrix))
        
        # 如果只提供了层大小
        else:
            # 创建模型
            self.model = MultilayerFCNetwork(layer_sizes)
            
            # 保存层配置并生成随机权重矩阵
            self.fc_layers = []
            weight_matrices = []
            for i in range(len(layer_sizes) - 1):
                input_size = layer_sizes[i]
                output_size = layer_sizes[i+1]
                
                # 生成随机权重矩阵
                weight_matrix = np.random.uniform(0, 1, (output_size, input_size))
                
                # 保存层配置
                self.fc_layers.append(FCLayer(input_size, output_size, weight_matrix))
                weight_matrices.append(weight_matrix)
            
            # 设置权重
            self.model.set_weights(weight_matrices)
        
        print(f"Created {len(self.fc_layers)}-layer neural network: {layer_sizes}")
    
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
    
    def simulate_memristor_hardware(self, input_vector: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Simulate neural network forward pass using memristor hardware model
        模拟使用忆阻器硬件实现的神经网络前向传播
        :param input_vector: Input voltage vector (normalized 0-1)
        :param verbose: Whether to print detailed simulation info
        :return: Output current vector
        """
        if not self.fc_layers:
            raise ValueError("Network not initialized")
        
        # Physical voltage range: 0-0.2V
        physical_voltage = input_vector * 0.2
        
        if verbose:
            print("\nInput voltage vector (normalized 0-1):")
            print(input_vector)
            print("\nPhysical voltage vector (V):")
            print(physical_voltage)
        
        # Process through each layer
        current_vector = physical_voltage
        
        for i, layer in enumerate(self.fc_layers):
            weight_matrix = layer.weight_matrix
            
            # Split weight matrix into positive and negative components
            pos_matrix, neg_matrix = split_weights_pos_neg(weight_matrix)
            
            # Map to physical conductance (0-150μS)
            physical_pos_conductance = map_conductance_to_physical(pos_matrix)
            physical_neg_conductance = map_conductance_to_physical(neg_matrix)
            
            if verbose:
                print(f"\n=== Layer {i+1} Memristor Simulation ===")
                print(f"Normalized weight matrix (0-1 range):")
                print(weight_matrix)
                print(f"\nPositive weights (normalized 0-1 range):")
                print(pos_matrix)
                print(f"\nNegative weights (normalized 0-1 range):")
                print(neg_matrix)
                print(f"\nPositive physical conductance (S):")
                print(physical_pos_conductance)
                print(f"\nPositive physical conductance (μS):")
                print(physical_pos_conductance * 1e6)
                print(f"\nNegative physical conductance (S):")
                print(physical_neg_conductance)
                print(f"\nNegative physical conductance (μS):")
                print(physical_neg_conductance * 1e6)
            
            # Calculate output current from positive and negative arrays separately
            pos_current = np.dot(physical_pos_conductance, current_vector)
            neg_current = np.dot(physical_neg_conductance, current_vector)
            
            # Combine currents (positive contribution - negative contribution)
            current_vector = pos_current - neg_current
            
            if verbose:
                print(f"\nLayer {i+1} Output:")
                print(f"Positive array current contribution (A):")
                print(pos_current)
                print(f"Positive array current contribution (μA):")
                print(pos_current * 1e6)
                print(f"Negative array current contribution (A):")
                print(neg_current)
                print(f"Negative array current contribution (μA):")
                print(neg_current * 1e6)
                print(f"Combined output current (A):")
                print(current_vector)
                print(f"Combined output current (μA):")
                print(current_vector * 1e6)
                
            # Assuming a voltage conversion between layers
            # This would represent ADC -> DAC process between crossbar arrays
            if i < len(self.fc_layers) - 1:
                # Normalize back to 0-0.2V range for next layer
                # In real hardware, this would be ADC conversion followed by DAC
                max_current = max(abs(np.max(current_vector)), abs(np.min(current_vector)))
                if max_current > 0:
                    current_vector = current_vector / max_current * 0.2
                
                if verbose:
                    print(f"\nConverted to voltage for next layer:")
                    print(f"Voltage (0-0.2V range):")
                    print(current_vector)
        
        return current_vector
    
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
    
    def get_layer_matrices(self) -> List[np.ndarray]:
        """获取各层权重矩阵"""
        if not self.fc_layers:
            raise ValueError("Network not initialized")
        return [layer.weight_matrix for layer in self.fc_layers]
    
    def modify_layer_matrix(self, layer_idx: int, new_matrix: np.ndarray) -> None:
        """
        修改特定层的权重矩阵
        :param layer_idx: 层索引
        :param new_matrix: 新权重矩阵
        """
        if self.model is None or not self.fc_layers:
            raise ValueError("Network not initialized")
            
        if layer_idx < 0 or layer_idx >= len(self.fc_layers):
            raise ValueError(f"Invalid layer index: {layer_idx}")
            
        layer = self.fc_layers[layer_idx]
        
        # 检查矩阵形状
        if new_matrix.shape != (layer.output_size, layer.input_size):
            raise ValueError(f"Matrix shape mismatch: expected {(layer.output_size, layer.input_size)}, got {new_matrix.shape}")
        
        # 更新层矩阵
        layer.weight_matrix = new_matrix
        
        # 更新模型权重
        with torch.no_grad():
            self.model.layers[layer_idx].weight = nn.Parameter(torch.tensor(new_matrix, dtype=torch.float32))