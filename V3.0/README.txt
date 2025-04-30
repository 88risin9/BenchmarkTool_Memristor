Read me 

这个benchmark tool V3.0 在V2.1的基础之上

更新：

更新readme 使用说明 

新增 
输出报告功能 直观体现输出
工艺节点缩放 改进



📘 Overview
This project is runs and tests on environment Python 3.11.4 64bit
This system simulates the behavior of memristor crossbar arrays in executing neural network operations. It supports both fully connected (FC) and convolutional neural networks (CNNs), and provides detailed power/delay estimation, visualization, and automated reports.

📂 Project Structure

memristor_simulator/
├── main.py
├── matrix_operations.py
├── power_calculations.py
├── visualization.py
├── FC_network.py
├── cnn_module.py
├── utils.py
├── configs/
├── reports/
└── README.md


✨ Features
✅ Define custom or imported neural network structures
✅ Simulate memristor-based matrix multiplication
✅ Estimate power & latency under different technology nodes
✅ Visualize current distribution, conductance, and power breakdown
✅ Support CNN kernel/feature map visualization
✅ Generate HTML reports automatically


🚀 Getting Started
1. Installation

 manually:

pip install numpy matplotlib torch seaborn

2. Run the System
python main.py
🧩 Key Functionalities
1. Define Network Structure
Manual input of layer sizes (e.g., 784 128 64 10)
Import PyTorch model and map to memristor weights

2. Matrix Simulation
Load conductance matrix → Input voltage → Compute I = G × V

3. Power & Delay Estimation
Configure ADC (precision, power)
Select technology node (e.g., 180nm → 3nm)
View total and component-level power/delay


4. Visualization
Heatmap – Conductance matrix
Histogram – Current distribution
Pie Chart – Power breakdown
Bar Chart – Comparison across components


5. CNN Support
Simulate 2 conv + 2 FC CNN
Visualize kernels and feature maps
📁 Configuration Example
{
  "adc_bits": 8,
  "tech_node_nm": 45,
  "power_model": "scaled",
  "crossbar_size": [128, 128]
}

📄 HTML Report
After simulation, an HTML report is generated containing:

Network structure
Power/delay results
Matrix snapshots
Current and component charts



🔄 Roadmap
🛠 GUI version
🛠 Multi-layer CNN and dataset support
🛠 Hardware noise modeling


