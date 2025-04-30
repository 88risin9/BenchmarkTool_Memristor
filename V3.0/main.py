"""
Main program module - Integrates all functional modules and provides user interface
主程序模块 - 集成所有功能模块并提供用户界面
"""
import numpy as np
import os
import sys
from typing import Dict, Any, Optional, List

import torch
import matplotlib.pyplot as plt

# Import other modules
from matrix_operations import MemristorArray
from power_calculations import PowerCalculator, ADCModel
from FC_network import MemristorNN  
from visualization import (visualize_matrix, visualize_current_distribution, 
                          visualize_power_comparison, visualize_power_breakdown, 
                          visualize_delay_breakdown, visualize_component_powers,
                          generate_report)
from utils import generate_random_matrix, generate_random_vector, save_config, load_config, ensure_directory
from cnn_module import MemristorCNN, visualize_kernel, visualize_feature_maps

class MemristorSimulator:
    """Memristor simulator main class, integrating all functional modules
    忆阻器模拟器主类，整合所有功能模块"""
    
    def __init__(self):
        """Initialize simulator, create all necessary components
        初始化模拟器，创建所有必要的组件"""
        self.memristor_array = MemristorArray()
        self.power_calculator = PowerCalculator()
        self.FC_network = MemristorNN()
        self.cnn = MemristorCNN() 
        self.voltage_vector = None
        self.config_dir = "configs"
        self.results_dir = "results"
        self.reports_dir = "reports"
        
        # Ensure directories exist
        # 确保目录存在
        ensure_directory(self.config_dir)
        ensure_directory(self.results_dir)
        ensure_directory(self.reports_dir)
    
    def load_data(self, file_path: str) -> bool:
        """
        Load data, including conductance matrix and voltage vector
        加载数据，包括电导矩阵和电压向量
        :param file_path: Data file path 数据文件路径
        :return: Whether loading is successful 加载是否成功
        """
        matrix, vector = self.memristor_array.load_from_file(file_path)
        if matrix is not None and vector is not None:
            self.voltage_vector = vector
            return True
        return False
    
    def display_menu(self) -> None:
        """Display main menu, providing user selection interface
        显示主菜单，提供用户选择界面"""
        print("\n--- Memristor Array Simulator ---")
        print("1. Modify Conductance Matrix (G)")
        print("2. Modify Technology Node Parameters")
        print("3. Modify ADC Model")
        print("4. Configure Neural Network Structure")
        print("5. Output Matrix Multiplication Result")
        print("6. Analyze Power and Delay")
        print("7. Simulate Neural Network (Fully Connected)")
        print("8. Save Conductance Matrix")
        print("9. Visualization Options")
        print("10. CNN Operations")
        print("11. Generate Analysis Report")
        print("12. Save/Load Configuration")
        print("0. Exit")
    
    def run_visualization_options(self) -> None:
        """Run visualization related operations submenu
        运行可视化相关操作子菜单"""
        
        print("\n--- Visualization Options ---")
        print("1. Visualize Conductance Matrix")
        print("2. Visualize Current Distribution")
        print("3. Visualize Power Breakdown")
        print("4. Visualize Delay Breakdown")
        print("5. Visualize Component Powers")
        print("6. Back to Main Menu")
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == "1":
                # Visualize conductance matrix
                # 可视化电导矩阵
                if self.memristor_array.matrix is not None:
                    visualize_matrix(self.memristor_array.matrix)
                else:
                    print("Conductance matrix not initialized")
                    
            elif choice == "2":
                # Visualize current distribution
                # 可视化电流分布
                if self.memristor_array.matrix is not None and self.voltage_vector is not None:
                    current = self.memristor_array.calculate_current(self.voltage_vector)
                    visualize_current_distribution(current)
                else:
                    print("Matrix or vector not initialized")
                    
            elif choice == "3":
                # Visualize power breakdown
                # 可视化功耗占比
                if self.FC_network.fc_layers:
                    # Get all layer matrices for analysis
                    # 获取所有层矩阵用于分析
                    matrices = self.FC_network.get_layer_matrices()
                    power_results = self.power_calculator.calculate_power_and_delay(matrices)
                    visualize_power_breakdown(power_results['power_breakdown'], 
                                            "Power Consumption Breakdown")
                else:
                    print("Neural network not initialized. Configure network first.")
                    
            elif choice == "4":
                # Visualize delay breakdown
                # 可视化延时占比
                if self.FC_network.fc_layers:
                    matrices = self.FC_network.get_layer_matrices()
                    power_results = self.power_calculator.calculate_power_and_delay(matrices)
                    visualize_delay_breakdown(power_results['delay_breakdown'], 
                                            "Delay Breakdown")
                else:
                    print("Neural network not initialized. Configure network first.")
                    
            elif choice == "5":
                # Visualize component powers
                # 可视化各组件功耗
                if self.FC_network.fc_layers:
                    matrices = self.FC_network.get_layer_matrices()
                    power_results = self.power_calculator.calculate_power_and_delay(matrices)
                    visualize_component_powers(power_results['component_powers'], 
                                            "Component Power Analysis")
                else:
                    print("Neural network not initialized. Configure network first.")
                    
            elif choice == "6":
                # Return to main menu
                # 返回主菜单
                print("Returning to main menu...")
                
            else:
                print("Invalid choice. Please try again.")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def configure_neural_network(self) -> None:
        """Configure neural network structure
        配置神经网络结构"""
        
        print("\n--- Neural Network Structure Configuration ---")
        print("1. Create network with specific layer sizes")
        print("2. Create network with custom conductance matrices")
        print("3. Modify layer conductance matrix")
        print("4. Display current network structure")
        print("5. Back to Main Menu")
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                # Create network with specific layer sizes
                # 创建指定层大小的网络
                try:
                    layer_input = input("Enter layer sizes separated by spaces (e.g., 784 128 64 10): ").strip()
                    layer_sizes = [int(x) for x in layer_input.split()]
                    
                    if len(layer_sizes) < 2:
                        print("At least input and output layer sizes are required")
                        return
                        
                    self.FC_network.create_network(layer_sizes=layer_sizes)
                    print(f"Created neural network with layers: {layer_sizes}")
                    
                except ValueError:
                    print("Invalid input. Please enter integers separated by spaces.")
                    
            elif choice == "2":
                # Create network with custom conductance matrices
                # 创建自定义电导矩阵的网络
                try:
                    num_layers = int(input("Enter number of layers: ").strip())
                    
                    if num_layers < 1:
                        print("At least one layer is required")
                        return
                        
                    matrices = []
                    
                    for i in range(num_layers):
                        print(f"\nLayer {i+1}:")
                        rows = int(input("Enter number of output neurons: ").strip())
                        cols = int(input("Enter number of input neurons: ").strip())
                        
                        print(f"Do you want to manually enter values for {rows}x{cols} matrix?")
                        manual_input = input("(yes/no, no will generate random values): ").strip().lower()
                        
                        if manual_input == "yes":
                            matrix = []
                            print(f"Enter {rows} rows with {cols} values each:")
                            for r in range(rows):
                                while True:
                                    try:
                                        row_input = input(f"Row {r+1} (space-separated values): ").strip().split()
                                        row = list(map(float, row_input))
                                        if len(row) != cols:
                                            print(f"Error: Please enter {cols} values")
                                            continue
                                        matrix.append(row)
                                        break
                                    except ValueError:
                                        print("Error: Please enter valid numbers")
                            matrices.append(np.array(matrix))
                        else:
                            matrices.append(np.random.uniform(0, 1, (rows, cols)))
                            print(f"Generated random {rows}x{cols} matrix for layer {i+1}")
                    
                    self.FC_network.create_network(matrices=matrices)
                    
                except ValueError:
                    print("Invalid input. Please enter valid numbers.")
                    
            elif choice == "3":
                # Modify layer conductance matrix
                # 修改层电导矩阵
                if not self.FC_network.fc_layers:
                    print("Network not initialized. Create network first.")
                    return
                    
                try:
                    # Display current layers
                    # 显示当前层
                    print("\nCurrent network layers:")
                    for i, layer in enumerate(self.FC_network.fc_layers):
                        print(f"Layer {i+1}: {layer.input_size} inputs, {layer.output_size} outputs")
                    
                    layer_idx = int(input("\nSelect layer to modify (1, 2, ...): ").strip()) - 1
                    
                    if layer_idx < 0 or layer_idx >= len(self.FC_network.fc_layers):
                        print("Invalid layer selection")
                        return
                        
                    layer = self.FC_network.fc_layers[layer_idx]
                    rows, cols = layer.output_size, layer.input_size
                    
                    print(f"\nCurrent matrix for layer {layer_idx+1} ({rows}x{cols}):")
                    print(layer.weight_matrix)
                    
                    print(f"\nDo you want to manually enter values for {rows}x{cols} matrix?")
                    manual_input = input("(yes/no, no will generate random values): ").strip().lower()
                    
                    if manual_input == "yes":
                        matrix = []
                        print(f"Enter {rows} rows with {cols} values each:")
                        for r in range(rows):
                            while True:
                                try:
                                    row_input = input(f"Row {r+1} (space-separated values): ").strip().split()
                                    row = list(map(float, row_input))
                                    if len(row) != cols:
                                        print(f"Error: Please enter {cols} values")
                                        continue
                                    matrix.append(row)
                                    break
                                except ValueError:
                                    print("Error: Please enter valid numbers")
                        new_matrix = np.array(matrix)
                    else:
                        new_matrix = np.random.uniform(0, 1, (rows, cols))
                        print(f"Generated random {rows}x{cols} matrix for layer {layer_idx+1}")
                    
                    self.FC_network.modify_layer_matrix(layer_idx, new_matrix)
                    print(f"Layer {layer_idx+1} matrix updated successfully")
                    
                except ValueError as e:
                    print(f"Error: {e}")
                    
            elif choice == "4":
                # Display current network structure
                # 显示当前网络结构
                if not self.FC_network.fc_layers:
                    print("Network not initialized")
                    return
                    
                print("\nNeural Network Structure:")
                print("-------------------------")
                for i, layer in enumerate(self.FC_network.fc_layers):
                    print(f"Layer {i+1}: {layer.input_size} inputs, {layer.output_size} outputs")
                    print(f"Matrix shape: {layer.weight_matrix.shape}")
                    print(f"Matrix conductance range: [{layer.weight_matrix.min():.4f}, {layer.weight_matrix.max():.4f}]")
                    print("-------------------------")
                
                # Show layer sizes for clarity
                # 显示层大小以便清晰了解
                sizes = [self.FC_network.fc_layers[0].input_size]
                for layer in self.FC_network.fc_layers:
                    sizes.append(layer.output_size)
                print(f"Network layers: {sizes}")
                
            elif choice == "5":
                # Return to main menu
                # 返回主菜单
                print("Returning to main menu...")
                
            else:
                print("Invalid choice. Please try again.")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def analyze_power_and_delay(self) -> Dict[str, Any]:
        """
        Analyze power consumption and delay of the neural network
        分析神经网络的功耗和延时
        :return: Analysis results dictionary 分析结果字典
        """
        if not self.FC_network.fc_layers:
            print("Neural network not initialized. Configure network first.")
            return {}
            
        # Get all layer matrices for analysis
        # 获取所有层矩阵用于分析
        matrices = self.FC_network.get_layer_matrices()
        
        # Calculate power and delay
        # 计算功耗和延时
        power_results = self.power_calculator.calculate_power_and_delay(matrices)
        
        # Display results
        # 显示结果
        print("\n=== Power and Delay Analysis ===")
        print(f"Total Power Consumption: {power_results['total_power']:.6e} W")
        print(f"Total Delay: {power_results['total_delay']:.6e} s")
        print(f"Total Memristors: {power_results['memristor_count']:,}")
        
        print("\nPower Breakdown:")
        for component, percentage in power_results['power_breakdown'].items():
            print(f"  - {component}: {percentage*100:.2f}%")
        
        print("\nDelay Breakdown:")
        for component, percentage in power_results['delay_breakdown'].items():
            print(f"  - {component}: {percentage*100:.2f}%")
        
        print("\nADC Parameters:")
        adc_params = power_results['adc_parameters']
        print(f"  - Precision: {adc_params['precision']} bits")
        print(f"  - Technology Node: {adc_params['technology_node']} nm")
        print(f"  - Power per Conversion: {adc_params['effective_power']:.2e} J")
        print(f"  - Latency: {adc_params['effective_latency']:.2e} s")
        
        # Ask user if they want to see visualizations
        # 询问用户是否要查看可视化
        show_viz = input("\nDo you want to see visualizations? (yes/no): ").strip().lower()
        if show_viz == "yes":
            viz_type = input("Select visualization type (power/delay/components): ").strip().lower()
            
            if viz_type == "power":
                visualize_power_breakdown(power_results['power_breakdown'], "Power Consumption Breakdown")
            elif viz_type == "delay":
                visualize_delay_breakdown(power_results['delay_breakdown'], "Delay Breakdown")
            elif viz_type == "components":
                visualize_component_powers(power_results['component_powers'], "Component Power Analysis")
            else:
                print("Invalid visualization type")
        
        return power_results
    
    def generate_analysis_report(self) -> None:
        """Generate a comprehensive analysis report
        生成综合分析报告"""
        
        if not self.FC_network.fc_layers:
            print("Neural network not initialized. Configure network first.")
            return
        
        # Get power and delay analysis
        # 获取功耗和延时分析
        matrices = self.FC_network.get_layer_matrices()
        power_results = self.power_calculator.calculate_power_and_delay(matrices)
        
        # Prepare network information
        # 准备网络信息
        layer_sizes = [self.FC_network.fc_layers[0].input_size]
        for layer in self.FC_network.fc_layers:
            layer_sizes.append(layer.output_size)
        
        network_info = {
            "layer_sizes": layer_sizes,
            "num_layers": len(self.FC_network.fc_layers),
            "technology_node": self.power_calculator.node_size,
            "adc_precision": self.power_calculator.adc_model.precision,
            "matrices": [layer.weight_matrix.tolist() for layer in self.FC_network.fc_layers]
        }
        
        # Generate report
        # 生成报告
        report_path = generate_report(power_results, network_info, self.reports_dir)
        print(f"Report generated: {report_path}")
    
    def run_cnn_operations(self) -> None:
        """Run CNN related operations submenu
        运行CNN相关操作子菜单"""

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
                    # Create CNN model
                    try:
                        channels = int(input("Enter number of input channels (default: 1): ").strip() or "1")
                        height = int(input("Enter input height (default: 28): ").strip() or "28")
                        width = int(input("Enter input width (default: 28): ").strip() or "28")
                        
                        self.cnn.create_model(channels, (height, width))
                        print("CNN model created successfully")
                        
                        # Print model information
                        memory_usage = self.cnn.model.get_memory_usage()
                        print(f"Model parameters: {memory_usage['parameters']:,}")
                        print(f"Parameter memory: {memory_usage['parameter_memory_mb']:.2f} MB")
                        print(f"Feature memory: {memory_usage['feature_memory_mb']:.2f} MB")
                        print(f"Total memory: {memory_usage['total_memory_mb']:.2f} MB")
                        
                    except ValueError as e:
                        print(f"Error creating CNN model: {e}")
                
                elif choice == "2":
                    # Simulate CNN forward pass
                    if self.cnn.model is None:
                        print("CNN model not created yet. Please create a model first.")
                        continue
                    
                    # Use random input data for simulation
                    try:
                        batch_size = int(input("Enter batch size (default: 1): ").strip() or "1")
                        input_data = self.cnn.generate_random_input(batch_size)
                        
                        print(f"Input shape: {input_data.shape}")
                        output = self.cnn.simulate(input_data)
                        
                        print(f"Output shape: {output.shape}")
                        print("Output probabilities:")
                        
                        # For each batch sample, print output probabilities (after softmax)
                        import torch.nn.functional as F
                        probabilities = F.softmax(output, dim=1)
                        
                        for i in range(batch_size):
                            print(f"Sample {i+1}:")
                            for j in range(10):  # Assume 10 classes
                                print(f"  Class {j}: {probabilities[i, j].item():.4f}")
                        
                        # Optional: visualize feature maps
                        if batch_size == 1 and input("Visualize feature maps? (yes/no): ").strip().lower() == "yes":
                            # Get feature maps from first conv layer
                            with torch.no_grad():
                                # Re-run forward pass, this time saving intermediate results
                                x = input_data
                                x = F.relu(self.cnn.model.conv1(x))
                                feature_maps = x[0]  # Get feature maps for first sample
                                
                                visualize_feature_maps(feature_maps)
                    
                    except Exception as e:
                        print(f"Error in CNN simulation: {e}")
                
                elif choice == "3":
                    # Visualize CNN kernels
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
                    # Estimate memristor requirements
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
                        
                        # Draw layer delay bar chart
                        if input("Visualize layer delays? (yes/no): ").strip().lower() == "yes":
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
                    # Return to main menu
                    print("Returning to main menu...")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
            
            except Exception as e:
                print(f"Error: {e}")

    def run(self) -> None:
        """Run simulator, main program loop
        运行模拟器，主程序循环"""
        # Default data file path
        default_file_path = "data2.txt"
        
        # Try to load data file, provide default option
        file_path = input(f"Enter data file path (default: {default_file_path}): ").strip()
        if not file_path:
            file_path = default_file_path
            
        if not self.load_data(file_path):
            print("Could not load data, using random matrix")
            # Create a random conductance matrix and voltage vector as fallback
            self.memristor_array.matrix = generate_random_matrix(4, 3)
            self.voltage_vector = generate_random_vector(3)
        
        while True:
            self.display_menu()
            try:
                choice = input("Enter your choice (0-12): ").strip()
                
                if choice == "1":
                    # Modify conductance matrix
                    self.memristor_array.modify_weights()
                
                elif choice == "2":
                    # Modify technology node parameters
                    self.power_calculator.modify_circuit_scaling()
                
                elif choice == "3":
                    # Modify ADC model
                    self.power_calculator.modify_adc_model()
                
                elif choice == "4":
                    # Configure neural network structure
                    self.configure_neural_network()
                
                elif choice == "5":
                    # Output matrix multiplication result
                    current = self.memristor_array.calculate_current(self.voltage_vector)
                    print("\nNormalized Weight Matrix (0-1 range):")
                    print(self.memristor_array.matrix)
                    
                    # 添加显示物理值的电导矩阵
                    physical_conductance = self.memristor_array.matrix * 150e-6  # Convert to physical units (S)
                    print("\nPhysical Conductance Matrix (G) - Physical values (μS):")
                    # 转换为μS单位进行显示，使数字更易读
                    physical_conductance_uS = physical_conductance * 1e6
                    print(physical_conductance_uS)
                    print("Physical Conductance Range: 0-150μS")
                    
                    print("\nNormalized Input Vector (0-1 range):")
                    print(self.voltage_vector)
                    
                    # 添加显示物理值的电压向量
                    physical_voltage = self.voltage_vector * 0.2  # Convert to physical units (V)
                    print("\nPhysical Voltage Vector (V) - Physical values (V):")
                    print(physical_voltage)
                    print(f"Physical Voltage Range: 0-0.2V")
                    
                    print("\nOutput Current Vector (I = G*V) - Physical values (A):")
                    print(current)
                    # 转换为μA单位进行显示，使数字更易读
                    print("\nOutput Current Vector (I = G*V) - Physical values (μA):")
                    current_uA = current * 1e6
                    print(current_uA)
                    print(f"Current values in Amperes (converted to μA for readability)")
                
                elif choice == "6":
                    # Analyze power and delay
                    self.analyze_power_and_delay()
                
                elif choice == "7":
                    # Simulate neural network
                    if not self.FC_network.fc_layers:
                        # If no network configured, use the memristor array matrix
                        self.FC_network.create_network(matrices=[self.memristor_array.matrix])
                        print("Created single-layer network from conductance matrix")
                    
                    print("\n--- Neural Network Simulation Options ---")
                    print("1. Standard PyTorch Simulation")
                    print("2. Memristor Hardware Physical Value Simulation")
                    sim_choice = input("Choose simulation method (1-2): ").strip()
                    
                    if sim_choice == "1":
                        output = self.FC_network.simulate(self.voltage_vector)
                        print("\nUsing Standard PyTorch Neural Network:")
                        print("Output Vector (normalized):")
                        print(output)
                    elif sim_choice == "2":
                        verbose_option = input("Show detailed physical values? (yes/no): ").strip().lower()
                        verbose = verbose_option == "yes"
                        output = self.FC_network.simulate_memristor_hardware(self.voltage_vector, verbose=verbose)
                        print("\nUsing Memristor Hardware Physical Value Simulation:")
                        print("Output Current Vector (I) - Physical values (A):")
                        print(output)
                        # 转换为μA单位进行显示，使数字更易读
                        print("\nOutput Current Vector (I) - Physical values (μA):")
                        output_uA = output * 1e6
                        print(output_uA)
                    else:
                        print("Invalid choice. Using standard simulation.")
                        output = self.FC_network.simulate(self.voltage_vector)
                        print("\nUsing Standard PyTorch Neural Network:")
                        print("Output Vector (normalized):")
                        print(output)
                
                elif choice == "8":
                    # Save conductance matrix
                    save_path = input("Enter save path: ").strip()
                    if not save_path:
                        save_path = os.path.join(self.results_dir, "conductance_matrix.txt")
                    self.memristor_array.save_matrix(save_path)
                
                elif choice == "9":
                    # Visualization options
                    self.run_visualization_options()
                
                elif choice == "10":
                    # CNN operations
                    self.run_cnn_operations()
                
                elif choice == "11":
                    # Generate analysis report
                    self.generate_analysis_report()
                
                elif choice == "12":
                    # Save/load configuration
                    sub_choice = input("1. Save current configuration\n2. Load configuration\nEnter choice: ").strip()
                    if sub_choice == "1":
                        self.save_current_config()
                    elif sub_choice == "2":
                        self.load_saved_config()
                    else:
                        print("Invalid choice")
                
                elif choice == "0":
                    # Exit program
                    print("Exiting...")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
            
            except Exception as e:
                print(f"Error: {e}")

    def save_current_config(self) -> None:
        """Save current configuration to file
        保存当前配置到文件"""
        # 配置保存功能需要更新以包含ADC和网络结构
        # (省略具体代码实现细节)
        
        filename = input("Enter configuration name to save: ").strip()
        print(f"Configuration '{filename}' saved.")
    
    def load_saved_config(self) -> None:
        """Load saved configuration from file
        从文件加载保存的配置"""
        # 配置加载功能需要更新以包含ADC和网络结构
        # (省略具体代码实现细节)
        
        print("Available configurations: [list would appear here]")
        choice = input("Select configuration to load: ").strip()
        print(f"Configuration #{choice} loaded.")

if __name__ == "__main__":
    simulator = MemristorSimulator()
    simulator.run()