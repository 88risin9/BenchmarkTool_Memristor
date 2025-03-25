"""
可视化模块 - 提供矩阵和数据的可视化功能
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def visualize_matrix(matrix: np.ndarray, title: str = "Memristor Conductance Matrix") -> None:
    """
    将矩阵可视化为热力图
    :param matrix: 要可视化的矩阵
    :param title: 图表标题
    """
    if matrix is None:
        print("Matrix not initialized")
        return
        
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='Conductance Value')
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def visualize_current_distribution(current_vector: np.ndarray) -> None:
    """
    可视化电流分布直方图
    :param current_vector: 输出电流向量
    """
    plt.figure(figsize=(10, 6))
    plt.hist(current_vector, bins=20, alpha=0.7, color='blue')
    plt.title('Current Distribution')
    plt.xlabel('Current Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_power_comparison(powers: dict) -> None:
    """
    可视化不同组件的功率对比
    :param powers: 包含不同组件功率的字典
    """
    names = list(powers.keys())
    values = list(powers.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(names, values, color='teal')
    plt.title('Power Consumption Comparison')
    plt.xlabel('Component')
    plt.ylabel('Power (W)')
    plt.yscale('log')  # 使用对数刻度以便显示差异较大的值
    
    # 在柱状图上显示具体数值
    for i, v in enumerate(values):
        plt.text(i, v * 1.1, f"{v:.2e}W", ha='center')
    
    plt.tight_layout()
    plt.show()