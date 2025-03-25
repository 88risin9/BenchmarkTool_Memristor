"""
功率和延时计算模块 - 实现忆阻器系统的功耗和延时估算
"""
import numpy as np
from typing import Tuple

class PowerCalculator:
    """功率和延时计算器，考虑工艺节点和能耗参数"""
    
    def __init__(self, node_size: int = 65):
        """
        初始化功率计算器
        :param node_size: 初始工艺节点大小(nm)，默认为65nm
        """
        self.node_size = node_size  # 工艺节点 (nm)
        self.delay_memristor = 164.88e-6  # 忆阻器延时 (秒)
        self.energy_memristor = 59e-9  # 忆阻器能量消耗 (焦耳)
        self.energy_circuit = 3609e-9  # 外围电路能量消耗 (焦耳)
        
        # 计算功率值
        self.power_memristor = self.energy_memristor / self.delay_memristor
        self.power_circuit = self.energy_circuit / self.delay_memristor
    
    def calculate_power_and_delay(self, matrix: np.ndarray) -> Tuple[float, float]:
        """
        计算功耗和延时
        :param matrix: 电导矩阵，用于计算矩阵大小相关的功耗
        :return: (总功耗, 总延时)元组
        """
        # 总功耗为忆阻器和电路功耗之和
        # 在更复杂的模型中，功耗可能会与矩阵大小和运算量相关
        total_power = self.power_memristor + self.power_circuit
        
        # 总延时为忆阻器延时
        # 在更复杂的系统中，延时可能会考虑并行度和其他因素
        total_delay = self.delay_memristor
        
        return total_power, total_delay
    
    def modify_circuit_scaling(self) -> None:
        """
        修改工艺节点缩放
        根据工艺节点的平方关系调整功耗
        """
        try:
            new_node = int(input(f"Enter new technology node size (current: {self.node_size}nm): ").strip())
            if new_node <= 0:
                print("Error: Technology node must be positive")
                return
                
            # 计算缩放后的功耗
            # 工艺节点按平方关系影响功耗：P_new = P_old * (node_new/node_old)^2
            scaling_factor = (new_node / self.node_size) ** 2
            self.power_circuit *= scaling_factor
            
            print(f"\nCircuit power scaled from {self.node_size}nm to {new_node}nm.")
            print(f"New Circuit Power: {self.power_circuit:.6f} W")
            
            # 更新节点大小
            self.node_size = new_node
        except ValueError:
            print("Error: Please enter a valid integer")