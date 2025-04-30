"""
矩阵操作模块 - 包含所有与忆阻器电导矩阵相关的操作
"""
import numpy as np
import os
from typing import Tuple, Optional

def map_conductance_to_physical(conductance_normalized: np.ndarray) -> np.ndarray:
    """
    Map normalized conductance (0-1) to physical conductance range (0-150μS)
    将归一化电导值(0-1)映射到实际物理电导范围(0-150μS)
    """
    # Linear scaling: 0-1 -> 0-150μS
    return conductance_normalized * 150e-6  # Convert to Siemens (μS = 10^-6 S)

def map_physical_to_normalized(conductance_physical: np.ndarray) -> np.ndarray:
    """
    Map physical conductance (0-150μS) to normalized range (0-1)
    将实际物理电导值(0-150μS)映射回归一化范围(0-1)
    """
    # Linear scaling: 0-150μS -> 0-1
    return conductance_physical / 150e-6

def split_weights_pos_neg(weight_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split weight matrix into positive and negative components
    将权重矩阵分为正部分和负部分
    :param weight_matrix: Original weight matrix with positive and negative values
    :return: Tuple of (positive_weights, negative_weights) matrices
    """
    # Create positive matrix (keep positive values, set negative to 0)
    positive_weights = np.maximum(weight_matrix, 0)
    
    # Create negative matrix (keep absolute of negative values, set positive to 0)
    negative_weights = np.absolute(np.minimum(weight_matrix, 0))
    
    return positive_weights, negative_weights

class MemristorArray:
    """忆阻器阵列类，用于处理电导矩阵相关操作"""
    
    def __init__(self, conductance_matrix=None):
        """
        初始化忆阻器阵列
        :param conductance_matrix: 初始电导矩阵，默认为None
        """
        self.matrix = conductance_matrix
    
    def load_from_file(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从文件中加载电导矩阵和输入向量
        :param file_path: 数据文件路径
        :return: 电导矩阵和输入向量的元组，若加载失败则为(None, None)
        """
        try:
            with open(file_path, 'r') as file:
                # 读取矩阵维度
                rows, cols = map(int, file.readline().split())
                matrix = []
                for _ in range(rows):
                    row = list(map(float, file.readline().split()))
                    if len(row) != cols:
                        raise ValueError(f"Row length mismatch: expected {cols}, got {len(row)}")
                    matrix.append(row)
                
                # 读取向量维度和数据
                size = int(file.readline())
                vector = list(map(float, file.readline().split()))
                if len(vector) != size:
                    raise ValueError(f"Vector length mismatch: expected {size}, got {len(vector)}")
                
                self.matrix = np.array(matrix)
                return np.array(matrix), np.array(vector)
        except (IOError, ValueError) as e:
            print(f"File reading error: {e}")
            return None, None
    
    def calculate_current(self, voltage_vector: np.ndarray) -> np.ndarray:
        """
        Calculate output current using Ohm's law I = G * V with separate positive and negative memristor arrays
        使用欧姆定律计算输出电流 I = G * V，分别使用正负两个忆阻器阵列计算
        :param voltage_vector: Input voltage vector (normalized 0-1)
        :return: Output current vector
        """
        if self.matrix is None:
            raise ValueError("Conductance matrix not initialized")
        
        # Check dimension match
        if self.matrix.shape[1] != len(voltage_vector):
            raise ValueError(f"Dimension mismatch: matrix columns {self.matrix.shape[1]}, vector length {len(voltage_vector)}")
        
        # Split conductance matrix into positive and negative parts
        pos_matrix, neg_matrix = split_weights_pos_neg(self.matrix)
        
        # Map to physical units
        physical_pos_conductance = map_conductance_to_physical(pos_matrix)
        physical_neg_conductance = map_conductance_to_physical(neg_matrix)
        
        # Map voltage to 0-0.2V range
        physical_voltage = voltage_vector * 0.2  # Scale to 0-0.2V
        
        # Calculate current from positive and negative arrays separately
        pos_current = np.dot(physical_pos_conductance, physical_voltage)
        neg_current = np.dot(physical_neg_conductance, physical_voltage)
        
        # Combine currents: positive contribution - negative contribution
        physical_current = pos_current - neg_current
        
        # Display information about the separate calculations
        print("\nMemristor Array Calculation Details:")
        print("- Positive conductance array contribution")
        print("- Negative conductance array contribution")
        print(f"- Final result combines both: I_pos - I_neg")
        
        return physical_current  # Return physical current in Amperes
    
    def modify_weights(self) -> None:
        """
        用户交互式修改电导矩阵
        允许用户输入新的矩阵值
        """
        if self.matrix is None:
            print("Conductance matrix not initialized")
            return
            
        print("\nCurrent Conductance Matrix (G):")
        print(self.matrix)
        
        try:
            modify = input("\nDo you want to modify the conductance matrix? (yes/no): ").strip().lower()
            if modify == "yes":
                rows, cols = self.matrix.shape
                print(f"\nEnter new values for {rows}x{cols} conductance matrix:")
                new_matrix = []
                for i in range(rows):
                    while True:
                        try:
                            row_input = input(f"Row {i + 1} (space-separated values): ").strip().split()
                            row = list(map(float, row_input))
                            if len(row) != cols:
                                print(f"Error: Please enter {cols} values")
                                continue
                            new_matrix.append(row)
                            break
                        except ValueError:
                            print("Error: Please enter valid numbers")
                
                self.matrix = np.array(new_matrix)
                print("\nUpdated Conductance Matrix (G):")
                print(self.matrix)
        except Exception as e:
            print(f"Error modifying conductance matrix: {e}")
    
    def save_matrix(self, file_path: str) -> bool:
        """
        保存当前电导矩阵到文件
        :param file_path: 保存的文件路径
        :return: 保存是否成功
        """
        try:
            np.savetxt(file_path, self.matrix, fmt="%.6f")
            print(f"Conductance matrix saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving matrix: {e}")
            return False