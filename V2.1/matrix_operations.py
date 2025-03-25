"""
矩阵操作模块 - 包含所有与忆阻器电导矩阵相关的操作
"""
import numpy as np
import os
from typing import Tuple, Optional

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
        使用欧姆定律计算输出电流 I = G * V
        :param voltage_vector: 输入电压向量
        :return: 输出电流向量
        """
        if self.matrix is None:
            raise ValueError("Conductance matrix not initialized")
        
        # 检查维度匹配
        if self.matrix.shape[1] != len(voltage_vector):
            raise ValueError(f"Dimension mismatch: matrix columns {self.matrix.shape[1]}, vector length {len(voltage_vector)}")
            
        return np.dot(self.matrix, voltage_vector)
    
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