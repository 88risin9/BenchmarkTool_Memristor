"""
主程序模块 - 集成所有功能模块并提供用户界面
"""
import numpy as np
import os
import sys
from typing import Dict, Any, Optional

import torch

# 导入其他模块
from matrix_operations import MemristorArray
from power_calculations import PowerCalculator
from FC_network import MemristorNN  
from visualization import visualize_matrix, visualize_current_distribution, visualize_power_comparison
from utils import generate_random_matrix, generate_random_vector, save_config, load_config, ensure_directory
from cnn_module import MemristorCNN, visualize_kernel, visualize_feature_maps

class MemristorSimulator:
    """忆阻器模拟器主类，整合所有功能模块"""
    
    def __init__(self):
        """初始化模拟器，创建所有必要的组件"""
        self.memristor_array = MemristorArray()
        self.power_calculator = PowerCalculator()
        self.FC_network = MemristorNN()
        self.cnn = MemristorCNN() 
        self.voltage_vector = None
        self.config_dir = "configs"
        self.results_dir = "results"
        
        # 确保目录存在
        ensure_directory(self.config_dir)
        ensure_directory(self.results_dir)
    
    def load_data(self, file_path: str) -> bool:
        """
        加载数据，包括电导矩阵和电压向量
        :param file_path: 数据文件路径
        :return: 加载是否成功
        """
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
        print("8. Visualize Current Distribution")
        print("9. CNN Operations")  # 添加新的CNN操作选项
        print("10. Save/Load Configuration")
        print("0. Exit")
    
    def run_cnn_operations(self) -> None:
        """运行CNN相关操作子菜单"""

        while True:
            print("\n--- CNN Operations ---")
            print("1. Create CNN Model")
            print("2. Simulate CNN Forward Pass")
            print("3. Visualize CNN Kernels")
            print("4. Estimate Memristor Requirements")
            print("5. Back to Main Menu")
            
            try:
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == "1":
                    # 创建CNN模型
                    try:
                        channels = int(input("Enter number of input channels (default: 1): ").strip() or "1")
                        height = int(input("Enter input height (default: 28): ").strip() or "28")
                        width = int(input("Enter input width (default: 28): ").strip() or "28")
                        
                        self.cnn.create_model(channels, (height, width))
                        print("CNN model created successfully")
                        
                        # 打印模型信息
                        memory_usage = self.cnn.model.get_memory_usage()
                        print(f"Model parameters: {memory_usage['parameters']:,}")
                        print(f"Parameter memory: {memory_usage['parameter_memory_mb']:.2f} MB")
                        print(f"Feature memory: {memory_usage['feature_memory_mb']:.2f} MB")
                        print(f"Total memory: {memory_usage['total_memory_mb']:.2f} MB")
                        
                    except ValueError as e:
                        print(f"Error creating CNN model: {e}")
                
                elif choice == "2":
                    # 模拟CNN前向传播
                    if self.cnn.model is None:
                        print("CNN model not created yet. Please create a model first.")
                        continue
                    
                    # 使用随机输入数据进行模拟
                    try:
                        batch_size = int(input("Enter batch size (default: 1): ").strip() or "1")
                        input_data = self.cnn.generate_random_input(batch_size)
                        
                        print(f"Input shape: {input_data.shape}")
                        output = self.cnn.simulate(input_data)
                        
                        print(f"Output shape: {output.shape}")
                        print("Output probabilities:")
                        
                        # 对于每个批次样本，打印输出概率（经过softmax）
                        import torch.nn.functional as F
                        probabilities = F.softmax(output, dim=1)
                        
                        for i in range(batch_size):
                            print(f"Sample {i+1}:")
                            for j in range(10):  # 假设有10个类别
                                print(f"  Class {j}: {probabilities[i, j].item():.4f}")
                        
                        # 可选：可视化特征图
                        if batch_size == 1 and input("Visualize feature maps? (yes/no): ").strip().lower() == "yes":
                            # 获取第一个卷积层输出的特征图
                            with torch.no_grad():
                                # 重新运行前向传播，这次保存中间结果
                                x = input_data
                                x = F.relu(self.cnn.model.conv1(x))
                                feature_maps = x[0]  # 获取第一个样本的特征图
                                
                                visualize_feature_maps(feature_maps)
                    
                    except Exception as e:
                        print(f"Error in CNN simulation: {e}")
                
                elif choice == "3":
                    # 可视化CNN卷积核
                    if self.cnn.model is None:
                        print("CNN model not created yet. Please create a model first.")
                        continue
                    
                    kernels = self.cnn.extract_kernels()
                    
                    print("\nAvailable kernels:")
                    for i, (name, kernel) in enumerate(kernels.items()):
                        print(f"{i+1}. {name}: shape={kernel.shape}")
                    
                    kernel_choice = input("Select kernel to visualize (1, 2, ...): ").strip()
                    try:
                        idx = int(kernel_choice) - 1
                        if idx < 0 or idx >= len(kernels):
                            print("Invalid kernel selection")
                            continue
                            
                        kernel_name = list(kernels.keys())[idx]
                        kernel = kernels[kernel_name]
                        
                        print(f"Visualizing {kernel_name} with shape {kernel.shape}")
                        visualize_kernel(kernel)
                        
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                
                elif choice == "4":
                    # 估算忆阻器需求
                    if self.cnn.model is None:
                        print("CNN model not created yet. Please create a model first.")
                        continue
                    
                    try:
                        memristor_info = self.cnn.simulate_on_memristor_array()
                        
                        print("\n--- Memristor Requirements for CNN ---")
                        print(f"Conv1 layer: {memristor_info['num_memristors']['conv1']:,} memristors")
                        print(f"Conv2 layer: {memristor_info['num_memristors']['conv2']:,} memristors")
                        print(f"FC1 layer: {memristor_info['num_memristors']['fc1']:,} memristors")
                        print(f"FC2 layer: {memristor_info['num_memristors']['fc2']:,} memristors")
                        print(f"Total memristors: {memristor_info['num_memristors']['total']:,}")
                        
                        print("\n--- Performance Estimates ---")
                        print(f"Total delay: {memristor_info['performance']['total_delay_ms']:.2f} ms")
                        print(f"Energy consumption: {memristor_info['performance']['energy_consumption_uJ']:.2f} µJ")
                        
                        # 绘制层级延迟柱状图
                        if input("Visualize layer delays? (yes/no): ").strip().lower() == "yes":
                            import matplotlib.pyplot as plt
                            
                            layers = list(memristor_info['performance']['layer_delay_ms'].keys())
                            delays = list(memristor_info['performance']['layer_delay_ms'].values())
                            
                            plt.figure(figsize=(10, 6))
                            plt.bar(layers, delays, color='teal')
                            plt.title('Layer-wise Delay in Memristor-based CNN')
                            plt.xlabel('Layer')
                            plt.ylabel('Delay (ms)')
                            plt.tight_layout()
                            plt.show()
                        
                    except Exception as e:
                        print(f"Error estimating memristor requirements: {e}")
                
                elif choice == "5":
                    # 返回主菜单
                    print("Returning to main menu...")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
            
            except Exception as e:
                print(f"Error: {e}")
    




    
    def save_current_config(self) -> None:
        """保存当前配置到文件"""
        if self.memristor_array.matrix is None:
            print("No matrix to save")
            return
            
        file_name = input("Enter configuration name to save: ").strip()
        if not file_name:
            file_name = "default_config"
            
        file_path = os.path.join(self.config_dir, f"{file_name}.json")
        
        # 准备配置数据
        config = {
            "matrix": self.memristor_array.matrix.tolist(),
            "vector": self.voltage_vector.tolist() if self.voltage_vector is not None else None,
            "node_size": self.power_calculator.node_size,
            "delay_memristor": self.power_calculator.delay_memristor,
            "power_memristor": self.power_calculator.power_memristor,
            "power_circuit": self.power_calculator.power_circuit
        }
        
        if save_config(config, file_path):
            print(f"Configuration saved to {file_path}")
    
    def load_saved_config(self) -> None:
        """从文件加载保存的配置"""
        # 获取可用配置列表
        configs = [f for f in os.listdir(self.config_dir) if f.endswith('.json')]
        
        if not configs:
            print("No saved configurations found")
            return
            
        print("\nAvailable configurations:")
        for i, config_file in enumerate(configs):
            print(f"{i+1}. {config_file[:-5]}")  # 去掉.json后缀
            
        try:
            choice = int(input("\nSelect configuration to load (0 to cancel): "))
            if choice == 0:
                return
                
            if 1 <= choice <= len(configs):
                file_path = os.path.join(self.config_dir, configs[choice-1])
                config = load_config(file_path)
                
                if config:
                    # 更新模拟器状态
                    self.memristor_array.matrix = np.array(config["matrix"])
                    if config["vector"]:
                        self.voltage_vector = np.array(config["vector"])
                    
                    self.power_calculator.node_size = config["node_size"]
                    self.power_calculator.delay_memristor = config["delay_memristor"]
                    self.power_calculator.power_memristor = config["power_memristor"]
                    self.power_calculator.power_circuit = config["power_circuit"]
                    
                    print(f"Configuration loaded from {file_path}")
                else:
                    print("Failed to load configuration")
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a valid number")
    
    def run(self) -> None:
        """运行模拟器，主程序循环"""
        # 默认数据文件路径
        default_file_path = "data2.txt"
        
        # 尝试加载数据文件，提供默认选项
        file_path = input(f"Enter data file path (default: {default_file_path}): ").strip()
        if not file_path:
            file_path = default_file_path
            
        if not self.load_data(file_path):
            print("Could not load data, using random matrix")
            # 创建一个随机的电导矩阵和电压向量作为备选
            self.memristor_array.matrix = generate_random_matrix(4, 3)
            self.voltage_vector = generate_random_vector(3)
        
        while True:
            self.display_menu()
            try:
                choice = input("Enter your choice (0-10): ").strip()
                
                if choice == "1":
                    # 修改电导矩阵
                    self.memristor_array.modify_weights()
                
                elif choice == "2":
                    # 修改工艺节点缩放
                    self.power_calculator.modify_circuit_scaling()
                
                elif choice == "3":
                    # 输出矩阵运算结果
                    current = self.memristor_array.calculate_current(self.voltage_vector)
                    print("\nConductance Matrix (G):")
                    print(self.memristor_array.matrix)
                    print("Input Voltage Vector (V):")
                    print(self.voltage_vector)
                    print("Output Current Vector (I = G*V):")
                    print(current)
                
                elif choice == "4":
                    # 输出功耗和延时
                    total_power, total_delay = self.power_calculator.calculate_power_and_delay(
                        self.memristor_array.matrix
                    )
                    print(f"\nTotal Power Consumption: {total_power:.6f} W")
                    print(f"Total Delay: {total_delay:.6f} s")
                    
                    # 显示功耗细分
                    powers = {
                        "Memristor": self.power_calculator.power_memristor,
                        "Circuit": self.power_calculator.power_circuit,
                        "Total": total_power
                    }
                    
                    # 询问是否显示功耗对比图
                    if input("Visualize power comparison? (yes/no): ").strip().lower() == "yes":
                        visualize_power_comparison(powers)
                
                elif choice == "5":
                    # 模拟神经网络
                    self.FC_network.create_network(self.memristor_array.matrix)
                    output = self.FC_network.simulate(self.voltage_vector)
                    print("\nUsing PyTorch Neural Network:")
                    print("Output Current Vector (I):")
                    print(output)
                
                elif choice == "6":
                    # 保存电导矩阵
                    save_path = input("Enter save path: ").strip()
                    if not save_path:
                        save_path = os.path.join(self.results_dir, "conductance_matrix.txt")
                    self.memristor_array.save_matrix(save_path)
                
                elif choice == "7":
                    # 可视化电导矩阵
                    visualize_matrix(self.memristor_array.matrix)
                
                elif choice == "8":
                    # 可视化电流分布
                    if self.memristor_array.matrix is not None and self.voltage_vector is not None:
                        current = self.memristor_array.calculate_current(self.voltage_vector)
                        visualize_current_distribution(current)
                    else:
                        print("Matrix or vector not initialized")
                
                elif choice == "9":
                    # CNN操作
                    self.run_cnn_operations()
                
                elif choice == "10":
                    # 保存/加载配置
                    sub_choice = input("1. Save current configuration\n2. Load configuration\nEnter choice: ").strip()
                    if sub_choice == "1":
                        self.save_current_config()
                    elif sub_choice == "2":
                        self.load_saved_config()
                    else:
                        print("Invalid choice")
                
                elif choice == "0":
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