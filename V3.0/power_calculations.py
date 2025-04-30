"""
功率和延时计算模块 - 实现忆阻器系统的功耗和延时估算
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

class ADCModel:
    """模拟数字转换器(ADC)模型类"""
    
    def __init__(self, precision: int = 8, technology_node: int = 65, 
                 power_per_conversion: float = 5e-9, latency: float = 10e-9):
        """
        初始化ADC模型
        :param precision: ADC精度(位)
        :param technology_node: 工艺节点(nm)
        :param power_per_conversion: 每次转换的能量(J)，默认5nJ
        :param latency: 转换延时(秒)，默认10ns
        """
        self.precision = precision
        self.technology_node = technology_node
        self.base_energy = power_per_conversion  # 重命名为energy，更准确
        self.base_latency = latency
        
        # 计算实际功耗和延时（考虑精度和工艺节点）
        self._calculate_parameters()
    
    def _calculate_parameters(self) -> None:
        """根据精度和工艺节点计算实际功耗和延时"""
        # 精度对功耗的影响（功耗随位数增加而增加，大约是2^n关系）
        precision_factor = 2 ** (self.precision / 8)  # 相对于8位精度的比例
        
        # 工艺节点对功耗的影响（功耗与工艺节点的平方成正比）
        tech_factor = (self.technology_node / 65) ** 2  # 相对于65nm工艺的比例
        
        # 计算实际能量和延时
        self.energy = self.base_energy * precision_factor * tech_factor  # 单位：J
        self.latency = self.base_latency * (1 + 0.1 * (self.precision - 8))  # 单位：s
        
        # 计算功率：功率 = 能量/时间
        self.power = self.energy / self.latency  # 单位：W（焦耳/秒）
    
    def update_model(self, precision: Optional[int] = None, 
                    technology_node: Optional[int] = None,
                    power_per_conversion: Optional[float] = None,
                    latency: Optional[float] = None) -> None:
        """
        更新ADC模型参数
        :param precision: 新的ADC精度
        :param technology_node: 新的工艺节点
        :param power_per_conversion: 新的单次转换能量
        :param latency: 新的转换延时
        """
        if precision is not None:
            self.precision = precision
        if technology_node is not None:
            self.technology_node = technology_node
        if power_per_conversion is not None:
            self.base_energy = power_per_conversion
        if latency is not None:
            self.base_latency = latency
            
        # 重新计算参数
        self._calculate_parameters()
    
    def get_params(self) -> Dict[str, Any]:
        """获取ADC模型参数"""
        return {
            "precision": self.precision,
            "technology_node": self.technology_node,
            "energy_per_conversion": self.base_energy,  # 重命名
            "latency": self.base_latency,
            "effective_energy": self.energy,  # 添加有效能量
            "effective_power": self.power,  # 添加有效功率
            "effective_latency": self.latency
        }

class PowerCalculator:
    """功率和延时计算器，考虑工艺节点和能耗参数"""
    
    def __init__(self, node_size: int = 65):
        """
        初始化功率计算器
        :param node_size: 初始工艺节点大小(nm)，默认为65nm
        """
        self.node_size = node_size  # 工艺节点 (nm)
        
        # 基础参数（65nm工艺）
        self.pulse_width = 10e-9  # 忆阻器脉冲宽度 (10ns)
        self.delay_memristor = self.pulse_width  # 忆阻器延时 = 脉冲宽度
        self.energy_memristor = 59e-12  # 忆阻器能量消耗 (焦耳，59pJ)
        self.energy_circuit = 3609e-12  # 外围电路能量消耗 (焦耳，3609pJ)
        
        # 计算功率值
        self.power_memristor = self.energy_memristor / self.delay_memristor  # W = J/s
        self.power_circuit = self.energy_circuit / self.delay_memristor  # W = J/s
        
        # 其他组件功耗参数（调整比例以使ADC占据主导地位）
        self.crossbar_power_ratio = 0.01  # crossbar阵列功耗占比大幅降低
        self.buffer_power_ratio = 0.01  # 缓冲区功耗占比
        self.controller_power_ratio = 0.005  # 控制器功耗占比
        
        # 初始化ADC模型（大幅增加默认功耗以确保ADC占据主导地位，但使用合理的数量级）
        self.adc_model = ADCModel(precision=8, 
                                  technology_node=node_size, 
                                  power_per_conversion=0.005, # 使用更合理的单位，5mJ
                                  latency=20e-9)
    
    def calculate_power_and_delay(self, matrices: List[np.ndarray]) -> Dict[str, Any]:
        """
        计算功耗和延时
        :param matrices: 电导矩阵列表，表示神经网络各层
        :return: 功耗和延时详细分析结果
        """
        # 计算总忆阻器数量
        total_memristors = sum(matrix.size for matrix in matrices)
        
        # 计算平均电导值 (考虑映射到0-150μS)
        avg_conductance = sum(np.sum(matrix) for matrix in matrices) / total_memristors
        physical_avg_conductance = avg_conductance * 150e-6  # Convert to physical units
        
        # 计算ADC功耗
        # 每个矩阵的每行需要一个ADC转换
        total_adc_conversions = sum(matrix.shape[0] for matrix in matrices)
        
        # 计算ADC总能量（焦耳）
        adc_total_energy = self.adc_model.energy * total_adc_conversions
        
        # 计算总延时（秒）
        layer_delays = {}
        for i, matrix in enumerate(matrices):
            # 每层延时等于忆阻器脉冲延时加上ADC延时
            layer_delay = self.delay_memristor + self.adc_model.latency
            layer_delays[f"layer_{i+1}"] = layer_delay
        
        # 计算总延时（假设层间串行执行）
        total_delay = sum(layer_delays.values())
        
        # 计算ADC平均功率（瓦特）= 总能量/总时间
        adc_power = adc_total_energy / total_delay  # W = J/s
        
        # 计算其他组件的功率（都以ADC功率为基准）
        crossbar_power = adc_power * 0.001  # Crossbar功率约为ADC的0.1%
        buffer_power = adc_power * 0.01  # 缓冲区功率约为ADC的1%
        controller_power = adc_power * 0.005  # 控制器功率约为ADC的0.5%
        other_circuit_power = adc_power * 0.01  # 其他电路功率约为ADC的1%
        
        # 总功率（瓦特）
        total_power = crossbar_power + buffer_power + controller_power + adc_power + other_circuit_power
        
        # 计算各组件功率占比
        power_breakdown = {
            "crossbar": crossbar_power / total_power,
            "adc": adc_power / total_power,
            "buffer": buffer_power / total_power,
            "controller": controller_power / total_power,
            "other_circuit": other_circuit_power / total_power
        }
        
        # 延时占比
        delay_breakdown = {name: delay / total_delay for name, delay in layer_delays.items()}
        delay_breakdown["adc"] = self.adc_model.latency / total_delay
        
        return {
            "total_power": total_power,  # 单位：W
            "total_energy": adc_total_energy + (crossbar_power + buffer_power + controller_power + other_circuit_power) * total_delay,  # 单位：J
            "total_delay": total_delay,  # 单位：s
            "power_breakdown": power_breakdown,
            "delay_breakdown": delay_breakdown,
            "layer_delays": layer_delays,
            "component_powers": {  # 单位：W
                "crossbar": crossbar_power,
                "adc": adc_power,
                "buffer": buffer_power,
                "controller": controller_power,
                "other_circuit": other_circuit_power
            },
            "memristor_count": total_memristors,
            "adc_parameters": self.adc_model.get_params()
        }
    
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
            
            # 更新ADC模型工艺节点
            self.adc_model.update_model(technology_node=new_node)
            
            print(f"\nCircuit power scaled from {self.node_size}nm to {new_node}nm.")
            print(f"New Circuit Power: {self.power_circuit:.6f} W")
            print(f"New ADC Power: {self.adc_model.power:.6e} J/conversion")
            
            # 更新节点大小
            self.node_size = new_node
        except ValueError:
            print("Error: Please enter valid numbers")
    
    def modify_adc_model(self) -> None:
        """用户自定义ADC模型参数"""
        print("\n--- Current ADC Model Parameters ---")
        params = self.adc_model.get_params()
        print(f"Precision: {params['precision']} bits")
        print(f"Technology Node: {params['technology_node']} nm")
        print(f"Energy per Conversion: {params['energy_per_conversion']:.2e} J")
        print(f"Latency: {params['effective_latency']:.2e} s")
        
        try:
            precision = input("Enter new ADC precision (bits) [Enter to keep current]: ").strip()
            precision = int(precision) if precision else None
            
            energy = input("Enter energy per conversion (J) [Enter to keep current]: ").strip()
            energy = float(energy) if energy else None
            
            latency = input("Enter conversion latency (s) [Enter to keep current]: ").strip()
            latency = float(latency) if latency else None
            
            # 更新ADC模型
            self.adc_model.update_model(
                precision=precision,
                technology_node=self.node_size,  # 保持与主工艺节点一致
                power_per_conversion=energy,
                latency=latency
            )
            
            print("\nADC model updated:")
            params = self.adc_model.get_params()
            print(f"Precision: {params['precision']} bits")
            print(f"Energy per Conversion: {params['effective_energy']:.2e} J")
            print(f"Latency: {params['effective_latency']:.2e} s")
            
        except ValueError:
            print("Error: Please enter valid numbers")