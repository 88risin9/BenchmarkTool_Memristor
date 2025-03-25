import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
from typing import Tuple, List, Dict, Any

class MemristorArray:
    """忆阻器阵列类，用于处理电导矩阵相关操作"""
    
    def __init__(self, conductance_matrix=None):
        self.matrix = conductance_matrix
        self.delay_memristor = 164.88e-6  # 忆阻器延时 (秒)
        self.energy_memristor = 59e-9  # 忆阻器能量消耗 (焦耳)
    
    def load_from_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """从文件中加载电导矩阵和输入向量"""
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
        """使用欧姆定律计算输出电流"""
        if self.matrix is None:
            raise ValueError("Conductance matrix not initialized")
        
        # 检查维度匹配
        if self.matrix.shape[1] != len(voltage_vector):
            raise ValueError(f"Dimension mismatch: matrix columns {self.matrix.shape[1]}, vector length {len(voltage_vector)}")
            
        return np.dot(self.matrix, voltage_vector)
    
    def modify_weights(self) -> None:
        """用户交互式修改电导矩阵"""
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
        """保存当前电导矩阵到文件"""
        try:
            np.savetxt(file_path, self.matrix, fmt="%.6f")
            print(f"Conductance matrix saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving matrix: {e}")
            return False
    
    def visualize_matrix(self) -> None:
        """将电导矩阵可视化为热力图"""
        if self.matrix is None:
            print("Conductance matrix not initialized")
            return
            
        plt.figure(figsize=(10, 8))
        plt.imshow(self.matrix, cmap='viridis')
        plt.colorbar(label='Conductance Value')
        plt.title('Memristor Conductance Matrix Visualization')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.show()

class PowerCalculator:
    """功率和延时计算器"""
    
    def __init__(self, node_size: int = 65):
        self.node_size = node_size  # 工艺节点 (nm)
        self.delay_memristor = 164.88e-6  # 忆阻器延时 (秒)
        self.energy_memristor = 59e-9  # 忆阻器能量消耗 (焦耳)
        self.energy_circuit = 3609e-9  # 外围电路能量消耗 (焦耳)
        self.power_memristor = self.energy_memristor / self.delay_memristor
        self.power_circuit = self.energy_circuit / self.delay_memristor
    
    def calculate_power_and_delay(self, matrix: np.ndarray) -> Tuple[float, float]:
        """计算功耗和延时，基于矩阵大小和当前功耗参数"""
        # 总功耗为忆阻器和电路功耗之和
        total_power = self.power_memristor + self.power_circuit
        # 总延时为忆阻器延时
        total_delay = self.delay_memristor
        return total_power, total_delay
    
    def modify_circuit_scaling(self) -> None:
        """修改工艺节点缩放，根据工艺节点平方关系调整功耗"""
        try:
            new_node = int(input(f"Enter new technology node size (current: {self.node_size}nm): ").strip())
            if new_node <= 0:
                print("Error: Technology node must be positive")
                return
                
            # 计算缩放后的功耗，使用平方关系缩放
            scaling_factor = (new_node / self.node_size) ** 2
            self.power_circuit *= scaling_factor
            
            print(f"\nCircuit power scaled from {self.node_size}nm to {new_node}nm.")
            print(f"New Circuit Power: {self.power_circuit:.6f} W")
            
            # 更新节点大小
            self.node_size = new_node
        except ValueError:
            print("Error: Please enter a valid integer")

class MemristorNN:
    """基于忆阻器的神经网络模拟器，使用PyTorch实现矩阵运算"""
    
    def __init__(self):
        self.model = None
    
    def create_network(self, matrix: np.ndarray) -> None:
        """创建基于忆阻器电导矩阵的神经网络，使用矩阵作为权重"""
        input_size = matrix.shape[1]
        output_size = matrix.shape[0]
        
        # 创建简单的全连接网络，不包含偏置
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size, bias=False)
        )
        
        # 设置权重为电导矩阵，使用无梯度模式防止优化器修改权重
        with torch.no_grad():
            self.model[0].weight = nn.Parameter(torch.tensor(matrix, dtype=torch.float32))
    
    def simulate(self, input_vector: np.ndarray) -> np.ndarray:
        """模拟神经网络计算，相当于矩阵向量乘法"""
        if self.model is None:
            raise ValueError("Network model not initialized")
            
        # 转换输入向量为张量
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        
        # 执行网络计算，无需梯度
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output.numpy()
    
    def batch_simulate(self, input_vectors: np.ndarray) -> np.ndarray:
        """批量模拟多个输入向量，提高计算效率"""
        if self.model is None:
            raise ValueError("Network model not initialized")
            
        # 转换输入向量为张量，批处理形式
        input_tensor = torch.tensor(input_vectors, dtype=torch.float32)
        
        # 执行网络计算，无需梯度
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output.numpy()

class MemristorSimulator:
    """忆阻器模拟器主类，整合所有功能模块"""
    
    def __init__(self):
        self.memristor_array = MemristorArray()
        self.power_calculator = PowerCalculator()
        self.neural_network = MemristorNN()
        self.voltage_vector = None
    
    def load_data(self, file_path: str) -> bool:
        """加载数据，包括电导矩阵和电压向量"""
        matrix, vector = self.memristor_array.load_from_file(file_path)
        if matrix is not None and vector is not None:
            self.voltage_vector = vector
            return True
        return False
    
    def display_menu(self) -> None:
        """显示主菜单，提供用户选择界面"""
        print("\n--- Memristor Array Simulator ---")
        print("1. Modify Conductance Matrix (G)")
        print("2. Modify Technology Node Scaling")
        print("3. Output Matrix Multiplication Result")
        print("4. Output Power and Delay")
        print("5. Simulate Neural Network (Fully Connected)")
        print("6. Save Conductance Matrix")
        print("7. Visualize Conductance Matrix")
        print("8. Exit")
    
    def run(self) -> None:
        """运行模拟器，主程序循环"""
        # 默认数据文件路径
        default_file_path = "data2.txt"
        
        # 尝试加载数据文件，提供默认选项 /Users/zhenzhou/Desktop/dissertation/app/data2.txt
        file_path = input(f"Enter data file path (default: {default_file_path}): ").strip()
        if not file_path:
            file_path = default_file_path
            
        if not self.load_data(file_path):
            print("Could not load data, using random matrix")
            # 创建一个随机的电导矩阵和电压向量作为备选
            self.memristor_array.matrix = np.random.rand(4, 3)
            self.voltage_vector = np.random.rand(3)
        
        while True:
            self.display_menu()
            try:
                choice = input("Enter your choice (1-8): ").strip()
                
                if choice == "1":
                    # 修改电导矩阵，允许用户自定义矩阵元素
                    self.memristor_array.modify_weights()
                
                elif choice == "2":
                    # 修改工艺节点缩放，调整功耗估算
                    self.power_calculator.modify_circuit_scaling()
                
                elif choice == "3":
                    # 输出矩阵运算结果，展示矩阵向量乘法
                    current = self.memristor_array.calculate_current(self.voltage_vector)
                    print("\nConductance Matrix (G):")
                    print(self.memristor_array.matrix)
                    print("Input Voltage Vector (V):")
                    print(self.voltage_vector)
                    print("Output Current Vector (I = G*V):")
                    print(current)
                
                elif choice == "4":
                    # 输出功耗和延时，基于当前设置计算
                    total_power, total_delay = self.power_calculator.calculate_power_and_delay(
                        self.memristor_array.matrix
                    )
                    print(f"\nTotal Power Consumption: {total_power:.6f} W")
                    print(f"Total Delay: {total_delay:.6f} s")
                
                elif choice == "5":
                    # 模拟神经网络，使用PyTorch进行计算
                    self.neural_network.create_network(self.memristor_array.matrix)
                    output = self.neural_network.simulate(self.voltage_vector)
                    print("\nUsing PyTorch Neural Network:")
                    print("Output Current Vector (I):")
                    print(output)
                
                elif choice == "6":
                    # 保存电导矩阵到文件
                    save_path = input("Enter save path: ").strip()
                    if not save_path:
                        save_path = "conductance_matrix.txt"
                    self.memristor_array.save_matrix(save_path)
                
                elif choice == "7":
                    # 可视化电导矩阵，生成热力图
                    self.memristor_array.visualize_matrix()
                
                elif choice == "8":
                    # 退出程序
                    print("Exiting...")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
            
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    simulator = MemristorSimulator()
    simulator.run()