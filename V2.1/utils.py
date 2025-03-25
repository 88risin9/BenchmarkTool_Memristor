"""
工具模块 - 提供各种辅助功能
"""
import numpy as np
import os
import json
from typing import Dict, Any, Optional

def generate_random_matrix(rows: int, cols: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """
    生成随机电导矩阵
    :param rows: 行数
    :param cols: 列数
    :param low: 最小值
    :param high: 最大值
    :return: 随机矩阵
    """
    return np.random.uniform(low=low, high=high, size=(rows, cols))

def generate_random_vector(size: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """
    生成随机电压向量
    :param size: 向量大小
    :param low: 最小值
    :param high: 最大值
    :return: 随机向量
    """
    return np.random.uniform(low=low, high=high, size=size)

def save_config(config: Dict[str, Any], file_path: str) -> bool:
    """
    保存配置到JSON文件
    :param config: 配置字典
    :param file_path: 文件路径
    :return: 保存是否成功
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def load_config(file_path: str) -> Optional[Dict[str, Any]]:
    """
    从JSON文件加载配置
    :param file_path: 文件路径
    :return: 配置字典，加载失败则返回None
    """
    try:
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def ensure_directory(directory: str) -> bool:
    """
    确保目录存在，如不存在则创建
    :param directory: 目录路径
    :return: 操作是否成功
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except Exception as e:
        print(f"Error creating directory: {e}")
        return False