"""
DNDA (Drivable Network Dynamic Assessment) - 自动驾驶风险评估系统

主要功能:
1. 计算车辆可行驶区域 (Drivable Area, DA) - 衡量可用驾驶空间大小
2. 计算风险等级 (Risk Level, RL) - 衡量当前驾驶场景的危险程度

核心算法流程:
1. 读取输入数据 (道路信息、自车状态、周围车辆状态、参考路径)
2. 坐标系转换 (笛卡尔坐标系 → Frenet坐标系)
3. 生成自车轨迹簇 (多条候选轨迹)
4. 预测周围车辆未来轨迹
5. 碰撞检测 (使用分离轴定理)
6. 计算DA和RL指标
7. 输出评估结果

应用场景:
- 自动驾驶安全评估
- 轨迹规划决策支持
- 驾驶场景复杂度量化
- 数据集自动标注

作者: DNDA团队
版本: Python实现版本 (对应C++原始版本)
"""

import numpy as np
import os
import pandas as pd

# 导入dnda_functions模块中的核心函数和类
from dnda_functions import (
    Way,                                    # 路径类，存储基准线和轨迹信息
    surVehicle,                             # 周围车辆类
    Cartesian_trans_Frenet,                 # 坐标系转换函数
    create_test_data_folder,                # 创建测试数据文件夹
    set_scenario_name,                      # 设置场景名称（用于指定输出路径）
    DrivableArea_RiskLevel_Calculation,     # DA和RL计算主函数
    Generate_Ego_TraCluster,                # 生成自车轨迹簇
    Generate_surVehicle_Traj,               # 生成周围车辆轨迹
    Collison_Detection,                     # 碰撞检测
    Calculate_Drivable_Area,                # 计算可行驶区域
    Calculate_RiskLevel                     # 计算风险等级
)

# ==================== 数据结构定义 ====================

class Road:
    """
    道路类 - 存储道路相关信息
    
    属性:
        cross_centerline_: 是否允许跨越中心线
        lane_num_: 车道总数量
        lane_egodirect_: 自车所在方向的车道数量
        lane_width_: 车道宽度(米)
        maxAbsoluteAcc_: 车辆最大绝对加速度(m/s²)，用于计算曲率约束
    """
    def __init__(self, cross_centerline=False, lane_num=1, lane_egodirect=1, lane_width=4.0, maxAbsoluteAcc=9.8):
        self.cross_centerline_ = cross_centerline  # 是否允许跨越中心线
        self.lane_num_ = lane_num             # 车道数量
        self.lane_egodirect_ = lane_egodirect       # 自车所在车道方向
        self.lane_width_ = lane_width         # 车道宽度(米)
        self.maxAbsoluteAcc_ = maxAbsoluteAcc     # 最大加速度(m/s²)

class Vehicle:
    """
    车辆类 - 存储车辆状态信息
    
    属性:
        x_, y_: 车辆位置(笛卡尔坐标系，米)
        length_, width_: 车辆尺寸(米)
        speed_: 速度大小(m/s)
        speed_x_, speed_y_: 速度分量(m/s)
        acc_: 加速度(m/s²)
        init_q_: 初始横向偏移(相对于参考路径)
        lane_posi_: 车道位置(从左往右数第几条车道)
        absolute_theta_: 绝对方向角(相对于绝对坐标系y轴)
        relative_theta_: 相对方向角(相对于参考路径)
    """
    def __init__(self, x=0.0, y=0.0, length=5.0, width=2.0, speed=0.0, speed_x=0.0, speed_y=0.0, acc=0.0, init_q=0.0, lane_posi=1.0, absolute_theta=0.0, relative_theta=0.0):
        self.x_ = x                  # x坐标(米)
        self.y_ = y                  # y坐标(米)
        self.length_ = length             # 车长(米)
        self.width_ = width              # 车宽(米)
        self.speed_ = speed              # 速度大小(m/s)
        self.speed_x_ = speed_x            # x方向速度(m/s)
        self.speed_y_ = speed_y            # y方向速度(m/s)
        self.acc_ = acc                # 加速度(m/s²)
        self.init_q_ = init_q             # 初始横向偏移(米)
        self.lane_posi_ = lane_posi          # 车道位置(编号)
        self.absolute_theta_ = absolute_theta     # 绝对方向角(弧度)
        self.relative_theta_ = relative_theta     # 相对方向角(弧度)

# ==================== 数据读取函数 ====================

def read_input_data(filepath):
    """
    读取输入数据文件
    
    从CSV格式的输入文件中读取DNDA算法所需的所有参数，包括:
    - 道路参数 (road.*)
    - 自车状态 (egoVeh.*)
    - 参考路径 (baseline_data)
    - 周围车辆状态 (surVhe_input_data)
    - C++输出结果 (用于对比验证)
    
    参数:
        filepath: 输入文件路径
        
    返回:
        dict: 包含所有参数的字典
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取文件 {filepath} 时出错: {e}")
        return {}
    
    data = {}
    baseline_data = []
    surVhe_data = []
    
    reading_baseline = False
    reading_surVhe = False
    reading_output = False
    
    print(f"开始解析文件: {filepath}")
    
    # 先检查文件是否包含关键参数
    has_surVhe_length = False
    has_baseline_data = False
    for line in lines:
        if 'surVhe_input_length' in line:
            has_surVhe_length = True
        if 'baseline_data' in line:
            has_baseline_data = True
    
    if not has_surVhe_length:
        print("警告: 文件中未找到 'surVhe_input_length' 参数")
    
    if not has_baseline_data:
        print("警告: 文件中未找到 'baseline_data' 参数")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 直接处理 surVhe_input_length
        if 'surVhe_input_length' in line:
            try:
                key, value = line.split(',', 1)
                data[key] = int(value)
                print(f"读取到 surVhe_input_length: {value}")
                # 如果在读取 baseline 数据，这里结束它
                if reading_baseline:
                    reading_baseline = False
            except Exception as e:
                print(f"解析 surVhe_input_length 时出错: {e}")
                data['surVhe_input_length'] = 1  # 设置默认值
            continue
        
        if reading_baseline:
            # 在这里不中断 baseline 数据读取，而是在函数中检测到特定关键字时处理
            values = line.split(',')
            if len(values) == 2:
                try:
                    baseline_data.append(float(values[0]))
                    baseline_data.append(float(values[1]))
                except ValueError:
                    # 如果这行不是有效的 baseline 数据，可能是新的键值对
                    if ',' in line:
                        try:
                            key, value = line.split(',', 1)
                            if key == 'surVhe_input_length':
                                reading_baseline = False
                                data[key] = int(value)
                                print(f"在 baseline 数据中读取到 surVhe_input_length: {value}")
                            else:
                                try:
                                    data[key] = float(value)
                                except ValueError:
                                    data[key] = value
                        except Exception:
                            pass
        elif reading_surVhe:
            if 'frame' in line:
                reading_surVhe = False
                # 处理 frame 行
                if ',' in line:
                    key, value = line.split(',', 1)
                    try:
                        data[key] = int(value)
                    except ValueError:
                        data[key] = value
            else:
                values = line.split(',')
                if len(values) == 9:
                    for val in values:
                        try:
                            surVhe_data.append(float(val))
                        except ValueError:
                            surVhe_data.append(0.0)  # 默认值
        elif 'baseline_data' in line:
            reading_baseline = True
        elif 'surVhe_input_data' in line:
            reading_surVhe = True
        elif 'output_result' in line:
            reading_output = True
        elif reading_output:
            if line.startswith('DA,'):
                try:
                    data['DA'] = float(line.split(',')[1])
                except (IndexError, ValueError):
                    data['DA'] = 0.0
            elif line.startswith('RL,'):
                try:
                    data['RL'] = float(line.split(',')[1])
                except (IndexError, ValueError):
                    data['RL'] = 0.0
        else:
            # 处理一般的键值对
            if ',' in line:
                try:
                    key, value = line.split(',', 1)
                    try:
                        # 尝试转换为数值
                        data[key] = float(value)
                    except ValueError:
                        data[key] = value
                except Exception as e:
                    print(f"解析行 '{line}' 时出错: {e}")
    
    # 确保必要的数据存在
    data['baseline'] = baseline_data
    data['surVhe_input'] = surVhe_data
    
    # 如果仍然没有 surVhe_input_length，计算它
    if 'surVhe_input_length' not in data:
        # 计算车辆数量 (每个车辆有9个参数)
        if surVhe_data:
            data['surVhe_input_length'] = len(surVhe_data) // 9
            print(f"计算得到 surVhe_input_length: {data['surVhe_input_length']}")
        else:
            data['surVhe_input_length'] = 0
            print("警告: 没有找到周围车辆数据，设置 surVhe_input_length 为 0")
    
    # print(f"文件解析完成，数据键: {list(data.keys())}")
    return data


def test_DNDA(input_file):
    """
    测试DNDA算法与C++输出结果的一致性
    
    该函数从输入文件中读取数据，调用DNDA算法计算DA和RL值，
    并与C++版本的输出结果进行对比，验证Python实现的正确性。
    
    参数:
        input_file: 输入数据文件路径
        
    返回:
        tuple: (Python结果[DA, RL], C++结果[DA, RL])
    """
    # 读取输入数据
    data = read_input_data(input_file)
    
    # 创建道路对象
    road = Road()
    road.cross_centerline_ = bool(data.get('road.cross_centerline_', False))
    road.lane_num_ = int(data.get('road.lane_num_', 1))
    road.lane_egodirect_ = int(data.get('road.lane_egodirect_', 1))
    road.lane_width_ = data.get('road.lane_width_', 4.0)
    road.maxAbsoluteAcc_ = data.get('road.maxAbsoluteAcc_', 9.8)
    
    # 创建自车对象
    egoVeh = Vehicle()
    egoVeh.x_ = data.get('egoVeh.x_', 0.0)
    egoVeh.y_ = data.get('egoVeh.y_', 0.0)
    egoVeh.length_ = data.get('egoVeh.length_', 5.0)
    egoVeh.width_ = data.get('egoVeh.width_', 2.0)
    egoVeh.speed_ = data.get('egoVeh.speed_', 0.0)
    egoVeh.speed_x_ = data.get('egoVeh.speed_x_', 0.0)
    egoVeh.speed_y_ = data.get('egoVeh.speed_y_', 0.0)
    egoVeh.acc_ = data.get('egoVeh.acc_', 0.0)
    egoVeh.init_q_ = data.get('egoVeh.init_q_', 0.0)
    egoVeh.lane_posi_ = data.get('egoVeh.lane_posi_', 1.0)
    egoVeh.absolute_theta_ = data.get('egoVeh.absolute_theta_', 0.0)
    egoVeh.relative_theta_ = data.get('egoVeh.relative_theta_', 0.0)
    
    # 获取其他参数
    timeHorizon = data.get('timeHorizon', 5.0)
    timeStep = data.get('timeStep', 0.1)
    baseline = data['baseline']
    Mrow_b = int(data.get('baseline_length', len(baseline) // 2))
    surVhe_input = data['surVhe_input']
    Mrow_s = int(data.get('surVhe_input_length', len(surVhe_input) // 9))
    frame = int(data.get('frame', 1))
    log_flag = bool(data.get('log_flag', False))
    basepoint_num = int(data.get('basepoint_num', 100))
    
    # 从全局配置中获取场景名称
    from dnda_functions import get_scenario_name
    scenario_name = get_scenario_name()
    
    # 设置输出路径到当前文件夹下的 output/<场景名称> 目录
    output_base = os.path.join('output', scenario_name)
    polygon_folder = os.path.join(output_base, 'polygon')
    surRect_folder = os.path.join(output_base, 'surRect')
    
    # 确保目录存在
    os.makedirs(polygon_folder, exist_ok=True)
    os.makedirs(surRect_folder, exist_ok=True)
    
    # 调用DNDA算法
    result = DrivableArea_RiskLevel_Calculation(timeHorizon, timeStep, road, egoVeh, baseline, Mrow_b, 
                                              surVhe_input, Mrow_s, frame, log_flag, basepoint_num, 
                                              polygon_folder, surRect_folder)
    
    # 比较结果
    print(f"Frame: {frame}")
    if 'DA' in data and 'RL' in data:
        print(f"Python DA: {result[0]}, C++ DA: {data['DA']}")
        print(f"Python RL: {result[1]}, C++ RL: {data['RL']}")
        print(f"DA差异: {abs(result[0] - data['DA'])}")
        print(f"RL差异: {abs(result[1] - data['RL'])}")
        return result, [data.get('DA', 0), data.get('RL', 0)]
    else:
        print(f"错误: {frame}帧无对比值")
        return result, [0, 0]


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    """
    主程序 - 批量测试DNDA算法
    
    功能:
    1. 读取指定目录下的所有输入数据文件
    2. 对每个文件运行DNDA算法
    3. 比较Python实现与C++版本的输出结果
    4. 统计并输出差异信息
    """
    # 设置输入数据路径 (包含C++版本生成的测试数据)
    # 使用相对路径，相对于 DrivableArea_to_python 文件夹
    input_dir = "./build/data/output/51_nihe2_1.csv/input_data"
    
    # 从 input_dir 中提取场景名称（如 "39_nihe2.csv"）
    # 路径格式: "./build/data/output/场景名称/input_data"
    scenario_name = os.path.basename(os.path.dirname(input_dir))
    print(f"当前场景: {scenario_name}")
    
    # 设置场景名称，影响测试数据的保存路径
    # 测试数据将保存到: output/<场景名称>/frame_<帧号>/
    set_scenario_name(scenario_name)
    
    # 列出所有输入文件
    all_files = [f for f in os.listdir(input_dir) if f.endswith('_input.csv')]
    all_files.sort()  # 按文件名排序，确保按时间顺序处理
    print(f"找到{len(all_files)}个输入文件")
    
    # 测试单个文件
    if all_files:
        test_file = os.path.join(input_dir, all_files[0])
        print(f"\n测试单个文件: {all_files[0]}")
        test_DNDA(test_file)
    
    # 测试多个文件
    print("\n测试多个文件:")
    results = []
    
    # 限制只测试前5个文件，以免输出过多
    # test_files = all_files[10:15] if len(all_files) > 5 else all_files
    test_files = all_files
    
    for file in test_files:  
        file_path = os.path.join(input_dir, file)
        print(f"\n处理文件: {file}")
        python_result, cpp_result = test_DNDA(file_path)
        results.append({
            'file': file,
            'python_DA': python_result[0],
            'cpp_DA': cpp_result[0],
            'python_RL': python_result[1],
            'cpp_RL': cpp_result[1],
            'DA_diff': abs(python_result[0] - cpp_result[0]),
            'RL_diff': abs(python_result[1] - cpp_result[1])
        })
        print("-" * 50)
    
    # 输出结果统计
    if results:
        df = pd.DataFrame(results)
        print("\n结果统计:")
        print(f"平均DA差异: {df['DA_diff'].mean()}")
        print(f"最大DA差异: {df['DA_diff'].max()}")
        print(f"平均RL差异: {df['RL_diff'].mean()}")
        print(f"最大RL差异: {df['RL_diff'].max()}") 