"""
可视化模块 - 提供矩阵和数据的可视化功能
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import os
from datetime import datetime

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

def visualize_power_breakdown(power_breakdown: Dict[str, float], title: str = "Power Consumption Breakdown") -> None:
    """
    将功耗占比可视化为饼图
    :param power_breakdown: Power breakdown dictionary
    :param title: Chart title
    """
    labels = list(power_breakdown.keys())
    values = list(power_breakdown.values())
    
    # Component name mapping for better display
    label_map = {
        "crossbar": "Crossbar Array", 
        "adc": "ADC Converters",
        "buffer": "Buffers",
        "controller": "Controller",
        "other_circuit": "Other Circuits"
    }
    
    # Apply mapping
    labels = [label_map.get(label, label) for label in labels]
    
    # Highlight the largest portion
    explode = [0.1 if v == max(values) else 0 for v in values]
    
    plt.figure(figsize=(10, 8))
    plt.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    plt.title(title)
    plt.tight_layout()
    plt.show()

def visualize_delay_breakdown(delay_breakdown: Dict[str, float], title: str = "Delay Breakdown") -> None:
    """
    将延时占比可视化为饼图
    :param delay_breakdown: Delay breakdown dictionary
    :param title: Chart title
    """
    labels = list(delay_breakdown.keys())
    values = list(delay_breakdown.values())
    
    # Replace layer_ with Layer in labels
    labels = [label.replace("layer_", "Layer ") if "layer_" in label else label for label in labels]
    
    # Translate labels
    if "adc" in labels:
        idx = labels.index("adc")
        labels[idx] = "ADC Conversion"
    
    # Highlight the largest portion
    explode = [0.1 if v == max(values) else 0 for v in values]
    
    plt.figure(figsize=(10, 8))
    plt.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def visualize_component_powers(component_powers: Dict[str, float], title: str = "Component Power Analysis") -> None:
    """
    将各组件功耗可视化为条形图
    :param component_powers: Component power dictionary
    :param title: Chart title
    """
    # Component name mapping
    label_map = {
        "crossbar": "Crossbar Array", 
        "adc": "ADC Converters",
        "buffer": "Buffers",
        "controller": "Controller",
        "other_circuit": "Other Circuits"
    }
    
    # Apply mapping and sort
    names = [label_map.get(key, key) for key in component_powers.keys()]
    values = list(component_powers.values())
    
    # Sort by power from high to low
    sorted_indices = np.argsort(values)[::-1]
    names = [names[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values, color='teal')
    
    plt.title(title)
    plt.xlabel('Component')
    plt.ylabel('Power (W)')
    plt.yscale('log')  # Use log scale
    
    # Display values on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height*1.05,
                f'{height:.2e}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def generate_report(power_results: Dict[str, Any], network_info: Dict[str, Any], 
                    output_dir: str = "reports") -> str:
    """
    生成分析报告并保存为HTML文件
    :param power_results: Power analysis results
    :param network_info: Network configuration information
    :param output_dir: Output directory
    :return: Report file path
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"memristor_analysis_report_{timestamp}.html"
    filepath = os.path.join(output_dir, filename)
    
    # Generate HTML report
    with open(filepath, 'w', encoding='utf-8') as f:
        # Report header
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Memristor Neural Network Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .highlight {{ color: #e74c3c; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Memristor Neural Network Analysis Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>Performance Summary</h2>
                <p>Total Power Consumption: <span class="highlight">{power_results['total_power']:.4e} W</span></p>
                <p>Total Delay: <span class="highlight">{power_results['total_delay']:.4e} s</span></p>
                <p>Total Memristors: <span class="highlight">{power_results['memristor_count']:,}</span></p>
            </div>
            
            <h2>Network Configuration</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Network Structure</td>
                    <td>{network_info['layer_sizes']}</td>
                </tr>
                <tr>
                    <td>Number of Layers</td>
                    <td>{network_info['num_layers']}</td>
                </tr>
                <tr>
                    <td>Technology Node</td>
                    <td>{network_info['technology_node']} nm</td>
                </tr>
                <tr>
                    <td>ADC Precision</td>
                    <td>{network_info['adc_precision']} bits</td>
                </tr>
            </table>
            
            <h2>Power Analysis</h2>
            <h3>Power Breakdown</h3>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Power (W)</th>
                    <th>Percentage</th>
                </tr>
        """)
        
        # Power breakdown table
        for component, percentage in power_results['power_breakdown'].items():
            component_power = power_results['component_powers'][component]
            f.write(f"""
                <tr>
                    <td>{component}</td>
                    <td>{component_power:.4e}</td>
                    <td>{percentage*100:.2f}%</td>
                </tr>
            """)
        
        f.write("""
            </table>
            
            <h2>Delay Analysis</h2>
            <h3>Delay Breakdown</h3>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Delay (s)</th>
                    <th>Percentage</th>
                </tr>
        """)
        
        # Delay breakdown table
        total_delay = power_results['total_delay']
        for component, percentage in power_results['delay_breakdown'].items():
            if component == "adc":
                component_delay = power_results['adc_parameters']['effective_latency']
                component_name = "ADC Conversion"
            else:
                component_delay = power_results['layer_delays'].get(component, 0)
                component_name = component.replace("layer_", "Layer ")
            
            f.write(f"""
                <tr>
                    <td>{component_name}</td>
                    <td>{component_delay:.4e}</td>
                    <td>{percentage*100:.2f}%</td>
                </tr>
            """)
        
        f.write("""
            </table>
            
            <h2>ADC Parameters</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
        """)
        
        # ADC parameters table
        adc_params = power_results['adc_parameters']
        f.write(f"""
                <tr>
                    <td>Precision</td>
                    <td>{adc_params['precision']} bits</td>
                </tr>
                <tr>
                    <td>Technology Node</td>
                    <td>{adc_params['technology_node']} nm</td>
                </tr>
                <tr>
                    <td>Power per Conversion</td>
                    <td>{adc_params['effective_power']:.4e} J</td>
                </tr>
                <tr>
                    <td>Latency</td>
                    <td>{adc_params['effective_latency']:.4e} s</td>
                </tr>
        """)
        
        # Report footer
        f.write("""
            </table>
            
            <h2>Recommendations</h2>
            <ul>
                <li>Consider optimizing the component with the highest power consumption.</li>
                <li>Evaluate trade-offs between ADC precision and power consumption.</li>
                <li>Investigate opportunities to parallelize operations to reduce total delay.</li>
            </ul>
            
            <p><em>This report was automatically generated by the Memristor Array Simulator.</em></p>
        </body>
        </html>
        """)
    
    return filepath