import numpy as np
import torch
import torch.nn as nn

def menu():
    """
    初始菜单，提供用户选择的功能
    """
    print("\n--- Menu ---")
    print("1. Modify Conductance Matrix (G)")
    print("2. Modify Circuit Scaling")
    print("3. Output Matrix Multiplication Result")
    print("4. Output Power and Delay")
    print("5. Simulate Neural Network(fully connected)")
    print("6. Exit")
    choice = input("Enter your choice (1-5): ").strip()
    return choice

def scale_power_for_new_node(current_power, old_node, new_node):
    """
    根据工艺节点缩放外围电路功耗
    :param current_power: 当前外围电路功耗 (W)
    :param old_node: 当前工艺节点 (nm)
    :param new_node: 目标工艺节点 (nm)
    :return: 缩放后的功耗 (W)
    """
    scaling_factor = (new_node / old_node) ** 2  # 计算缩放因子
    scaled_power = current_power * scaling_factor  # 缩放功耗
    return scaled_power



def modify_circuit_scaling(power_circuit):
    """
    修改外围电路的工艺节点 scaling
    """
    old_node = 65  # 当前工艺节点 (nm)
    new_node = int(input(f"Enter the new node size (current: {old_node}nm): ").strip())
    scaled_power = scale_power_for_new_node(power_circuit, old_node, new_node)
    print(f"\nCircuit power scaled from {old_node}nm to {new_node}nm.")
    print(f"New Circuit Power: {scaled_power:.6f} W")
    return scaled_power

def simulate_neural_network(matrix, vector):
    """
    使用 PyTorch 模拟神经网络计算
    """
    input_size = matrix.shape[1]
    output_size = matrix.shape[0]
    model = SimpleFCNetwork(input_size, output_size)

    # 将电导矩阵 G 设置为全连接层的权重
    with torch.no_grad():
        model.fc.weight = nn.Parameter(torch.tensor(matrix, dtype=torch.float32))

    # 将输入电压向量转换为 PyTorch 张量
    input_vector = torch.tensor(vector, dtype=torch.float32)

    # 前向传播，计算输出电流向量
    output_current = model(input_vector)

    print("\nUsing PyTorch Neural Network:")
    print("Output Current Vector (I):")
    print(output_current.detach().numpy())  # 转换为 NumPy 数组以便打印





def get_matrix_input_from_file(file):
    """从文件中读取电导矩阵 G"""
    rows, cols = map(int, file.readline().split())
    matrix = []
    for _ in range(rows):
        row = list(map(float, file.readline().split()))
        matrix.append(row)
    return np.array(matrix)

def get_vector_input_from_file(file):
    """从文件中读取输入电压向量 V"""
    size = int(file.readline())
    vector = list(map(float, file.readline().split()))
    return np.array(vector)

def calculate_current(matrix, vector):
    """
    使用欧姆定律和基尔霍夫定律计算输出电流向量 I
    I = G * V
    """
    # 欧姆定律：矩阵乘法 G * V
    current = np.dot(matrix, vector)
    return current

def calculate_power_and_delay(matrix, vector):
    """
    计算功耗和延时
    假设每个忆阻器的功耗和延时是已知的常数
    """
    delay_memristor = 164.88e-6  # 忆阻器延时 (秒)
    energy_memristor = 59e-9  # 忆阻器能量消耗 (焦耳)
   
    # energy_1t1r = 59e-9  # 1T1R能量消耗 (焦耳)
    # energy_BL = 17.1e-9  # BL Driver能量消耗 (焦耳)
    # energy_WL = 17.1e-9  # WL Driver能量消耗 (焦耳)
    # energy_SH = 5.6e-9  # S&H能量消耗 (焦耳)
    # energy_mux = 82e-9  # MUX能量消耗 (焦耳)
    # energy_ADC = 2785.3e-9  # ADC能量消耗 (焦耳)
    # energy_ShiftandAdd = 700.2e-9  # Shift-and-Add能量消耗 (焦耳)
    # energy_ReLu = 1.1e-9  # ReLu能量消耗 (焦耳)
    # energy_maxpooling = 1.6e-9  # MaxPooling能量消耗 (焦耳)

    energy_circuit = 3609e-9  # 外围电路能量消耗 (焦耳)
    # energy_circuit =  energy_1t1r + energy_BL + energy_WL + energy_SH + energy_mux + energy_ADC + energy_ShiftandAdd + energy_ReLu + energy_maxpooling

    power_memristor = energy_memristor / delay_memristor  # 忆阻器功率 (瓦特)
    power_circuit = energy_circuit / delay_memristor  # 电路功率 (瓦特)

    # 假设总功耗为忆阻器和电路功耗之和
    total_power = power_memristor + power_circuit
    # 假设总延时为忆阻器延时
    total_delay = delay_memristor

    return total_power, total_delay

class SimpleFCNetwork(nn.Module):
    """
    一个简单的全连接神经网络，模拟忆阻器 crossbar 的计算
    """
    def __init__(self, input_size, output_size):
        super(SimpleFCNetwork, self).__init__()
        # 定义一个全连接层，权重矩阵模拟忆阻器的电导矩阵
        self.fc = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        # 前向传播，模拟 G * V 的计算
        return self.fc(x)

def modify_weights(matrix):
    """
    允许用户修改电导矩阵的权重
    """
    print("\nCurrent Conductance Matrix (G):")
    print(matrix)
    modify = input("\nDo you want to modify the conductance matrix? (yes/no): ").strip().lower()
    if modify == "yes":
        rows, cols = matrix.shape
        print(f"\nEnter the new values for the {rows}x{cols} conductance matrix:")
        new_matrix = []
        for i in range(rows):
            row = input(f"Row {i + 1} (space-separated values): ").strip().split()
            new_matrix.append(list(map(float, row)))
        matrix = np.array(new_matrix)
        print("\nUpdated Conductance Matrix (G):")
        print(matrix)
    return matrix

def main():
    file_path = "/Users/zhenzhou/Desktop/dissertation/app/data2.txt"
    with open(file_path, 'r') as file:
        # 从文件中读取电导矩阵 G 和输入电压向量 V
        matrix = get_matrix_input_from_file(file)  # 电导矩阵 G
        vector = get_vector_input_from_file(file)  # 输入电压向量 V
    
    # 允许用户修改电导矩阵
    # matrix = modify_weights(matrix)
    
    # 使用欧姆定律计算输出电流向量 I
    result = calculate_current(matrix, vector)
    
    # 计算功耗和延时
    total_power, total_delay = calculate_power_and_delay(matrix, vector)

    # 初始化外围电路功耗
    delay_memristor = 164.88e-6  # 忆阻器延时 (秒)
    energy_circuit = 3609e-9  # 外围电路能量消耗 (焦耳)
    power_circuit = energy_circuit / delay_memristor  # 外围电路功率 (瓦特)


    while True:
        choice = menu()
        if choice == "1":
            # 修改电导矩阵
            matrix = modify_weights(matrix)
        elif choice == "2":
            # 修改外围电路 scaling
            power_circuit = modify_circuit_scaling(power_circuit)
        elif choice == "3":
            # 输出矩阵运算结果
            result = calculate_current(matrix, vector)
            print("\nMatrix (matrix of memristor conductance values):")  # 忆阻器电导值矩阵
            print(matrix)
            print("Vector (Input voltage vector):")  # 输入电压向量
            print(vector)
            print("Result of multiplication (Output current vector):")  # 输出电流向量
            print(result)
        elif choice == "4":
            # 输出功耗和延时
            total_power, total_delay = calculate_power_and_delay(matrix, vector)  # 总功耗 总延时
            
            print("\nTotal Power Consumption (W):")  # 总功耗
            print(total_power)
            print("Total Delay (s):")  # 总延时
            print(total_delay)
        elif choice == "5":
            # 模拟神经网络
            simulate_neural_network(matrix, vector)
        elif choice == "6":
            # 退出程序
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")



    # # 使用 PyTorch 模拟神经网络计算
    # input_size = matrix.shape[1]
    # output_size = matrix.shape[0]
    # model = SimpleFCNetwork(input_size, output_size)

    # # 将电导矩阵 G 设置为全连接层的权重
    # with torch.no_grad():
    #     model.fc.weight = nn.Parameter(torch.tensor(matrix, dtype=torch.float32))

    # # 将输入电压向量转换为 PyTorch 张量
    # input_vector = torch.tensor(vector, dtype=torch.float32)

    # # 前向传播，计算输出电流向量
    # output_current = model(input_vector)

    # print("\nUsing PyTorch Neural Network:")
    # print("Output Current Vector (I):")
    # print(output_current.detach().numpy())  # 转换为 NumPy 数组以便打印

if __name__ == "__main__":
    main()





