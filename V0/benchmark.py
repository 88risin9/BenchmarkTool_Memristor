# import numpy as np

# def get_matrix_input(rows, cols):
#     matrix = []
#     print(f"Enter the values for a {rows}x{cols} matrix:")
#     for i in range(rows):
#         row = list(map(float, input(f"Row {i+1}: ").split()))
#         matrix.append(row)
#     return np.array(matrix)

# def get_vector_input(size):
#     print(f"Enter the values for a vector of size {size}:")
#     vector = list(map(float, input("Vector: ").split()))
#     return np.array(vector)

# def main():
#     rows = int(input("Enter the number of rows for the matrix: "))
#     cols = int(input("Enter the number of columns for the matrix: "))
#     matrix = get_matrix_input(rows, cols)
    
#     vector_size = int(input("Enter the size of the vector: "))
#     vector = get_vector_input(vector_size)
    
#     if cols != vector_size:
#         print("Error: The number of columns in the matrix must be equal to the size of the vector.")
#         return
    
#     result = np.dot(matrix, vector)
#     print("The result of the matrix-vector multiplication is:")
#     print(result)

# if __name__ == "__main__":
#     main()






# input from file
# import numpy as np

# def get_matrix_input_from_file(file):
#     rows, cols = map(int, file.readline().split())
#     matrix = []
#     for _ in range(rows):
#         row = list(map(float, file.readline().split()))
#         matrix.append(row)
#     return np.array(matrix)

# def get_vector_input_from_file(file):
#     size = int(file.readline())
#     vector = list(map(float, file.readline().split()))
#     return np.array(vector)

# def main():
#     #file_path = input("Enter the path to the data file: ")
#     file_path = "/Users/zhenzhou/Desktop/dissertation/app/data.txt"
#     with open(file_path, 'r') as file:
#         matrix = get_matrix_input_from_file(file)
#         vector = get_vector_input_from_file(file)
    
#     # 矩阵和向量相乘
#     result = np.dot(matrix, vector)
    
#     print("Matrix(matrix of memristor conductance values):")#忆阻器电导值矩阵
#     print(matrix)
#     print("Vector(Input voltage vector):")#输入电压向量
#     print(vector)
#     print("Result of multiplication(Output current vector):")#电流向量
#     print(result)

# if __name__ == "__main__":
#     main()





import numpy as np

def get_matrix_input_from_file(file):
    rows, cols = map(int, file.readline().split())
    matrix = []
    for _ in range(rows):
        row = list(map(float, file.readline().split()))
        matrix.append(row)
    return np.array(matrix)

def get_vector_input_from_file(file):
    size = int(file.readline())
    vector = list(map(float, file.readline().split()))
    return np.array(vector)

def calculate_power_and_delay(matrix, vector):
    # 假设每个忆阻器的功耗和延时是已知的常数

    delay_memristor = 164.88e-6  #latency unit: s
    
    energy_memristor = 59e-9  # energy unit: J
    energy_circuit = 3609e-9  # energy unit: J

    power_memristor = energy_memristor/delay_memristor  # energy/latency unit: W
    # power_memristor = 59e-9/164.88e-6  # energy/latency unit: W
    
    
    power_circuit = energy_circuit/delay_memristor # energy/latency unit: W
    # power_circuit = 3609e-9/164.88e-6 # energy/latency unit: W
    
    
    #num_memristors = matrix.size
    total_power =  power_memristor + power_circuit
    total_delay =  delay_memristor
    
    return total_power, total_delay

def main():
    file_path = "/Users/zhenzhou/Desktop/dissertation/app/data2.txt"
    with open(file_path, 'r') as file:
        matrix = get_matrix_input_from_file(file)
        vector = get_vector_input_from_file(file)
    
    # 矩阵和向量相乘
    result = np.dot(matrix, vector)
    
    # 计算功耗和延时
    total_power, total_delay = calculate_power_and_delay(matrix, vector)
    
    print("Matrix (matrix of memristor conductance values):")  # 忆阻器电导值矩阵
    print(matrix)
    print("Vector (Input voltage vector):")  # 输入电压向量
    print(vector)
    print("Result of multiplication (Output current vector):")  # 电流向量
    print(result)
    print("Total Power Consumption (W):")  # 总功耗
    print(total_power)
    print("Total Delay (s):")  # 总延时
    print(total_delay)

if __name__ == "__main__":
    main()