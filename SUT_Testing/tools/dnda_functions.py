"""
DNDA核心算法函数库
========================

本文件包含DNDA (Drivable Network Dynamic Assessment) 算法的核心实现函数。

主要功能模块:
1. 坐标系转换 (Cartesian ↔ Frenet)
2. 样条插值 (三次样条)
3. 自车轨迹簇生成
4. 周围车辆轨迹预测
5. 碰撞检测 (分离轴定理)
6. 可行驶区域计算
7. 风险等级评估

核心算法说明:
- Frenet坐标系: 沿道路中心线的自然坐标系，便于轨迹规划
- 分离轴定理(SAT): 高效的矩形碰撞检测算法
- 高斯加权: 考虑轨迹曲率对舒适性的影响

作者: DNDA团队
版本: Python实现 (对应C++原始版本)
"""

import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy import interpolate

# ==================== 数据结构定义 ====================

class Way:
    """
    路径类 - 存储参考路径或轨迹的几何信息
    
    在Frenet坐标系中表示路径，包含坐标、导数、曲率等信息。
    用于：
    1. 存储参考路径(道路中心线)
    2. 存储生成的自车轨迹
    
    属性:
        baseline_x, baseline_y: 路径点的坐标数组
        d1x, d1y: 一阶导数 (dx/ds, dy/ds)，s为弧长
        d2x, d2y: 二阶导数
        kb: 曲率数组，曲率 = (dx*d2y - dy*d2x) / (dx²+dy²)^(3/2)
        theta: 切线角数组，表示路径在各点的方向
    """
    def __init__(self):
        # 基准线x,y坐标
        self.baseline_x = None  # 基准线x坐标数组
        self.baseline_y = None  # 基准线y坐标数组
        # 一阶导数 (关于弧长s的导数)
        self.d1x = None  # dx/ds
        self.d1y = None  # dy/ds
        # 二阶导数
        self.d2x = None  # d²x/ds²
        self.d2y = None  # d²y/ds²
        # 曲率和切线角
        self.kb = None   # 曲率数组
        self.theta = None  # 切线角数组

class surVehicle:
    """
    周围车辆类 - 存储环境车辆的状态信息
    
    用于表示自车周围的其他车辆，存储其位置、速度、加速度等信息，
    用于预测其未来轨迹和碰撞检测。
    
    属性:
        x_, y_: 车辆位置(米)
        length_, width_: 车辆尺寸(米)
        speed_x_, speed_y_: 速度分量(m/s)
        acc_x_, acc_y_: 加速度分量(m/s²)
        heading_: 车辆朝向角(弧度)
    """
    def __init__(self, x=0.0, y=0.0, length=0.0, width=0.0, speed_x=0.0, speed_y=0.0, acc_x=0.0, acc_y=0.0, heading=0.0):
        self.x_ = x              # x坐标(米)
        self.y_ = y              # y坐标(米)
        self.length_ = length    # 车长(米)
        self.width_ = width      # 车宽(米)
        self.speed_x_ = speed_x  # x方向速度(m/s)
        self.speed_y_ = speed_y  # y方向速度(m/s)
        self.acc_x_ = acc_x      # x方向加速度(m/s²)
        self.acc_y_ = acc_y      # y方向加速度(m/s²)
        self.heading_ = heading  # 朝向角(弧度)

# ==================== 样条插值函数 ====================
# 三次样条插值用于路径平滑和曲率计算

def spline(n, end1, end2, slope1, slope2, x, y, b, c, d):
    """
    实现三次样条插值的基础函数，对应C++中的spline函数
    
    参数:
    n: 已知点的数量
    end1, end2: 边界条件标志
    slope1, slope2: 边界斜率
    x, y: 已知点的坐标数组
    b, c, d: 用于存储插值系数的数组
    
    返回:
    iflag: 0表示成功，其他值表示错误
    """
    iflag = 0
    
    # 检查输入点数是否足够
    if n < 2:
        return 1  # 不可能进行插值
    
    # 检查x值是否单调递增
    ascend = all(x[i] > x[i-1] for i in range(1, n))
    if not ascend:
        return 2  # x值不是单调递增的
    
    # 如果点数大于等于3，计算插值系数
    if n >= 3:
        nm1 = n - 1
        d[0] = x[1] - x[0]
        c[1] = (y[1] - y[0]) / d[0]
        
        for i in range(1, nm1):
            d[i] = x[i+1] - x[i]
            b[i] = 2.0 * (d[i-1] + d[i])
            c[i+1] = (y[i+1] - y[i]) / d[i]
            c[i] = c[i+1] - c[i]
        
        # 默认边界条件
        b[0] = -d[0]
        b[nm1] = -d[n-2]
        c[0] = 0.0
        c[nm1] = 0.0
        
        if n != 3:
            c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
            c[nm1] = c[n-2] / (x[nm1] - x[n-3]) - c[n-3] / (x[n-2] - x[n-4])
            c[0] = c[0] * d[0] * d[0] / (x[3] - x[0])
            c[nm1] = -c[nm1] * d[n-2] * d[n-2] / (x[nm1] - x[n-4])
        
        # 替代边界条件 - 已知斜率
        if end1 == 1:
            b[0] = 2.0 * (x[1] - x[0])
            c[0] = (y[1] - y[0]) / (x[1] - x[0]) - slope1
        
        if end2 == 1:
            b[nm1] = 2.0 * (x[nm1] - x[n-2])
            c[nm1] = slope2 - (y[nm1] - y[n-2]) / (x[nm1] - x[n-2])
        
        # 前向消元
        for i in range(1, n):
            t = d[i-1] / b[i-1]
            b[i] = b[i] - t * d[i-1]
            c[i] = c[i] - t * c[i-1]
        
        # 回代
        c[nm1] = c[nm1] / b[nm1]
        for i in range(n-2, -1, -1):
            c[i] = (c[i] - d[i] * c[i+1]) / b[i]
        
        # 计算其他系数
        b[nm1] = (y[nm1] - y[n-2]) / d[n-2] + d[n-2] * (c[n-2] + 2.0 * c[nm1])
        for i in range(nm1):
            b[i] = (y[i+1] - y[i]) / d[i] - d[i] * (c[i+1] + 2.0 * c[i])
            d[i] = (c[i+1] - c[i]) / d[i]
            c[i] = 3.0 * c[i]
        
        c[nm1] = 3.0 * c[nm1]
        d[nm1] = d[n-2]
    else:
        # 点数等于2的特殊情况
        b[0] = (y[1] - y[0]) / (x[1] - x[0])
        c[0] = 0.0
        d[0] = 0.0
        b[1] = b[0]
        c[1] = 0.0
        d[1] = 0.0
    
    return iflag

# 实现seval子函数 - 对应C++中的seval函数
def seval(ni, u, n, x, y, b, c, d, last):
    """
    计算插值点的值，对应C++中的seval函数
    
    参数:
    ni: 插值点的数量
    u: 待插值的x坐标
    n: 已知点的数量
    x, y: 已知点的坐标数组
    b, c, d: 插值系数数组
    last: 上次查找到的区间索引
    
    返回:
    w: 插值点的y坐标
    kpath: 插值点的曲率
    """
    if last >= n - 1:
        last = 0
    if last < 0:
        last = 0
    
    # 如果u不在当前区间，二分查找合适的区间
    if x[last] > u or x[last+1] < u:
        i, j = 0, n
        while j > i + 1:
            k = (i + j) // 2
            if u < x[k]:
                j = k
            if u >= x[k]:
                i = k
        last = i
    
    # 计算插值
    w = u - x[last]
    d1y = 3 * d[last] * w**2 + 2 * c[last] * w + b[last]
    d2y = 6 * d[last] * w + 2 * c[last]
    kpath = abs(d2y) / ((1 + d1y**2)**(1.5))
    w = y[last] + w * (b[last] + w * (c[last] + w * d[last]))
    
    return w, kpath, last

# 实现SPL函数 - 对应C++中的SPL函数
def SPL(n, x, y, ni, xi, yi, b, c, d, curvature):
    """
    样条插值函数，对应C++中的SPL函数
    
    参数:
    n: 已知点的数量
    x, y: 已知点的坐标数组
    ni: 插值点的数量
    xi, yi: 插值点的坐标数组(xi为输入，yi为输出)
    b, c, d: 用于存储插值系数的数组
    curvature: 用于存储曲率的数组
    """
    if d is None:
        print("没有足够的内存用于b,c,d")
        return
    
    # 调用spline函数计算插值系数
    iflag = spline(n, 0, 0, 0, 0, x, y, b, c, d)
    
    if iflag == 0:
        # 计算成功
        last = 0
        for i in range(ni):
            yi[i], curv, last = seval(ni, xi[i], n, x, y, b, c, d, last)
            curvature.append(curv)
        # print(curvature)
    else:
        print("x不是单调递增的或者存在其他错误")
        # print(x)

# ==================== 坐标系转换函数 ====================

def Cartesian_trans_Frenet(baseline, Mrow_b, m_equal, 
                          unitArcL, arcL, trans_x, trans_y, a0_x, 
                          a1_x, a2_x, a0_y, a1_y, a2_y, cdnt_arc):
    """
    笛卡尔坐标系到Frenet坐标系的转换
    
    Frenet坐标系是沿道路中心线的自然坐标系，其中：
    - s: 沿路径的弧长坐标
    - d: 垂直于路径的横向偏移
    
    该函数的主要步骤：
    1. 旋转基准线使其与x轴对齐
    2. 使用样条插值平滑路径
    3. 将路径等分为m_equal段
    4. 计算各段的弧长和样条系数
    5. 旋转回原坐标系
    
    参数:
        baseline: 基准线坐标数组 [x1, y1, x2, y2, ...]
        Mrow_b: 基准线点数
        m_equal: 等分段数
        unitArcL, arcL: 单位弧长和总弧长(输出)
        trans_x, trans_y: 转换后的坐标(输出)
        a0_x, a1_x, a2_x: x方向样条系数(输出)
        a0_y, a1_y, a2_y: y方向样条系数(输出)
        cdnt_arc: 累计弧长数组(输出)
        
    返回:
        unitArcL: 单位弧长
        arcL: 总弧长
    """
    # 将baseline转换为矩阵格式
    BaseLine = np.array(baseline).reshape(2, Mrow_b)
    
    # 计算旋转角度使基准线与x轴对齐
    colomn = BaseLine.shape[1]
    projV = np.array([BaseLine[0, -1] - BaseLine[0, 0], BaseLine[1, -1] - BaseLine[1, 0]])
    projectVec = projV.reshape(1, 2)
    xVect = np.array([1, 0]).reshape(1, 2)
    
    # 严格按照C++代码计算旋转角度
    dot_product = np.dot(projectVec, xVect.T)[0, 0]
    norm_product = np.linalg.norm(projectVec) * np.linalg.norm(xVect)
    rotateDegree = np.arccos(dot_product / norm_product)
    
    # 检查Y分量是否为负，调整旋转角度
    if projV[1] < 0:
        rotateDegree = -rotateDegree
    
    # 构建旋转矩阵 - 精确匹配C++代码
    Value_rom = np.array([
        np.cos(-rotateDegree), -np.sin(-rotateDegree), 
        np.sin(-rotateDegree), np.cos(-rotateDegree)
    ]).reshape(2, 2)
    rotateMatrix = Value_rom
    
    # 应用旋转矩阵
    newBaseLine = np.dot(rotateMatrix, BaseLine)
    
    # 按照C++代码提取baseline_x和baseline_y
    baseline_x = newBaseLine[0, :]
    baseline_y = newBaseLine[1, :]
    
    # 创建等间距数组u - 这是关键点
    u = np.zeros(m_equal)
    s = np.zeros(m_equal)
    
    for i in range(m_equal):
        # 严格按照C++代码创建u数组，确保单调递增
        u[i] = baseline_x[0] + i * (baseline_x[-1] - baseline_x[0]) / (m_equal - 1)
    
    # 实现等效于SPL函数的功能
    # 初始化辅助数组
    a = np.zeros(colomn)
    b = np.zeros(colomn)
    c = np.zeros(colomn)
    useless = []
    
    # 使用自定义SPL函数进行插值，确保与C++版本行为一致
    SPL(colomn, baseline_x, baseline_y, m_equal, u, s, a, b, c, useless)
    
    # 计算弧长
    li = np.zeros(m_equal - 1)
    for j in range(1, m_equal):
        microVec = np.array([u[j] - u[j-1], s[j] - s[j-1]])
        li[j-1] = np.linalg.norm(microVec)
    
    # 计算总弧长
    arcL = 0
    for k in range(m_equal - 1):
        arcL += li[k]
    
    # 计算单位弧长
    unitarcL = arcL / m_equal
    
    # 坐标变换部分
    count = 0
    knot_interval = np.zeros((m_equal, 3))
    
    while count < m_equal:
        for z in range(m_equal - 2):
            L_temp1 = np.sum(li[0:z+1])
            L_temp2 = np.sum(li[0:z+2])
            ans = (count + 1) * unitarcL
            
            if ans <= li[0]:
                knot_interval[count, 0] = 0
                knot_interval[count, 1] = 1
                knot_interval[count, 2] = (count + 1) * unitarcL
                count += 1
                break
                
            if (ans > L_temp1) and (ans <= L_temp2 + 0.001):
                knot_interval[count, 0] = z + 1
                knot_interval[count, 1] = z + 2
                knot_interval[count, 2] = (count + 1) * unitarcL - L_temp1
                count += 1
                break
    
    # 生成新的基准线
    new2BaseLine = np.zeros((2, m_equal))
    
    for i in range(m_equal):
        x_basepoint1 = u[int(knot_interval[i, 0])]
        y_basepoint1 = s[int(knot_interval[i, 0])]
        x_basepoint2 = u[int(knot_interval[i, 1])]
        y_basepoint2 = s[int(knot_interval[i, 1])]
        
        if x_basepoint1 != x_basepoint2:
            SIN = (y_basepoint2 - y_basepoint1) / li[int(knot_interval[i, 1] - 1)]
            COS = (x_basepoint2 - x_basepoint1) / li[int(knot_interval[i, 1] - 1)]
            new2BaseLine[0, i] = x_basepoint1 + knot_interval[i, 2] * COS
            new2BaseLine[1, i] = y_basepoint1 + knot_interval[i, 2] * SIN
        else:
            new2BaseLine[0, i] = x_basepoint1
            if y_basepoint2 >= y_basepoint1:
                new2BaseLine[1, i] = y_basepoint1 + knot_interval[i, 2]
            else:
                new2BaseLine[1, i] = y_basepoint1 - knot_interval[i, 2]
    
    # 添加初始点
    init = np.array([[u[0]], [s[0]]])
    new3BaseLine = np.zeros((2, m_equal + 1))
    new3BaseLine[:, 0] = init.flatten()
    new3BaseLine[:, 1:] = new2BaseLine
    
    # 旋转回原始坐标系
    Value_rom2 = np.array([
        np.cos(rotateDegree), -np.sin(rotateDegree), 
        np.sin(rotateDegree), np.cos(rotateDegree)
    ]).reshape(2, 2)
    rotate2Matrix = Value_rom2
    new3BaseLine = np.dot(rotate2Matrix, new3BaseLine)
    
    # 提取结果坐标
    for i in range(m_equal + 1):
        trans_x[i] = new3BaseLine[0, i]
        trans_y[i] = new3BaseLine[1, i]
    
    # 生成弧长数组
    for i in range(m_equal + 1):
        cdnt_arc[i] = i * unitarcL
    
    # 最终插值，获取样条系数
    x_splin = new3BaseLine[0, :]
    y_splin = np.zeros(m_equal + 1)
    
    # 对x和y坐标分别进行插值，获取系数
    # 使用自定义SPL函数对x坐标进行插值
    a2_x_temp = np.zeros(m_equal + 1)
    a1_x_temp = np.zeros(m_equal + 1)
    a0_x_temp = np.zeros(m_equal + 1)
    curvature_x = []
    SPL(m_equal + 1, cdnt_arc, trans_x, m_equal + 1, x_splin, y_splin, a2_x_temp, a1_x_temp, a0_x_temp, curvature_x)
    
    # 复制系数
    for i in range(m_equal + 1):
        if i < len(a2_x_temp):
            a2_x[i] = a2_x_temp[i]
            a1_x[i] = a1_x_temp[i]
            a0_x[i] = a0_x_temp[i]
    
    # 使用自定义SPL函数对y坐标进行插值
    a2_y_temp = np.zeros(m_equal + 1)
    a1_y_temp = np.zeros(m_equal + 1)
    a0_y_temp = np.zeros(m_equal + 1)
    curvature_y = []
    SPL(m_equal + 1, cdnt_arc, trans_y, m_equal + 1, x_splin, y_splin, a2_y_temp, a1_y_temp, a0_y_temp, curvature_y)
    
    # 复制系数
    for i in range(m_equal + 1):
        if i < len(a2_y_temp):
            a2_y[i] = a2_y_temp[i]
            a1_y[i] = a1_y_temp[i]
            a0_y[i] = a0_y_temp[i]
    
    return unitarcL, arcL

# 全局变量，用于存储当前场景名称（由上层调用代码设置）
_current_scenario_name = "test_data"

def set_scenario_name(scenario_name):
    """
    设置当前场景名称 - 由上层调用代码设置
    
    该函数允许上层代码指定场景名称，从而改变测试数据的保存路径。
    例如: set_scenario_name("39_nihe2.csv") 会将测试数据保存到 output/39_nihe2.csv/ 目录下
    
    参数:
        scenario_name: 场景名称（如 "39_nihe2.csv"）
    """
    global _current_scenario_name
    _current_scenario_name = scenario_name

def get_scenario_name():
    """
    获取当前场景名称
    
    返回:
        当前场景名称字符串
    """
    global _current_scenario_name
    return _current_scenario_name

# 生成测试数据目录
def create_test_data_folder(frame):
    """
    创建测试数据文件夹 - 用于保存中间调试数据
    
    在 DrivableArea_to_python/output/<场景名称>/ 目录下创建测试数据文件夹，
    用于保存各帧的中间计算结果，便于调试和验证。
    场景名称通过 set_scenario_name() 函数设置。
    
    参数:
        frame: 当前帧号
    
    返回:
        frame_folder: 帧数据文件夹路径
    """
    # 获取当前脚本所在目录（DrivableArea_to_python文件夹）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 使用全局变量中的场景名称，如果未设置则使用默认值 "test_data"
    scenario_name = get_scenario_name()
    
    # 设置输出目录到当前文件夹下的 output/<场景名称>
    test_data_folder = os.path.join(current_dir, "output", scenario_name)
    
    # 打印路径以便调试
    print(f"创建测试数据文件夹: {test_data_folder}")
    
    # 创建目录
    os.makedirs(test_data_folder, exist_ok=True)
    
    # 创建帧号子文件夹
    frame_folder = os.path.join(test_data_folder, f"frame_{frame}")
    os.makedirs(frame_folder, exist_ok=True)
    
    return frame_folder


# ==================== 主算法函数 ====================

def DrivableArea_RiskLevel_Calculation(timeHorizon, timeStep, road, egoVeh, baseline, Mrow_b, surVhe_input, Mrow_s, 
                                            frame, log_flag, test_data_log, basepoint_num, polygon_folder, surRect_folder):
    """
    DNDA主算法 - 计算可行驶区域(DA)和风险等级(RL)
    
    这是DNDA算法的核心函数，完成以下步骤：
    1. 坐标系转换 (将道路坐标转换到车辆局部坐标系)
    2. 计算曲率约束 (基于车辆动力学限制)
    3. 生成自车轨迹簇 (多条候选轨迹)
    4. 预测周围车辆未来轨迹
    5. 碰撞检测 (每条轨迹与周围车辆)
    6. 计算DA值 (可行驶区域面积)
    7. 计算RL值 (风险等级)
    
    参数:
        timeHorizon: 预测时间范围(秒)
        timeStep: 时间步长(秒)
        road: Road对象，包含道路信息
        egoVeh: Vehicle对象，自车状态
        baseline: 参考路径坐标数组 [x1,y1,x2,y2,...]
        Mrow_b: 参考路径点数
        surVhe_input: 周围车辆数据 [x,y,len,wid,vx,vy,ax,ay,heading]×N
        Mrow_s: 周围车辆数量
        frame: 当前帧号
        log_flag: 是否保存中间结果
        basepoint_num: 使用的基准点数量
        polygon_folder: 多边形保存路径
        surRect_folder: 周围车辆矩形保存路径
    
    返回:
        [DA, RL]: 可行驶区域面积(m²)和风险等级(0-1)
        
    算法说明:
        - DA值越大表示可行驶空间越大，越安全
        - RL值越大表示风险越高
        - 使用Frenet坐标系简化轨迹规划
        - 使用分离轴定理进行碰撞检测
    """
    
    if test_data_log == True:
        # 创建测试数据文件夹
        test_data_folder = create_test_data_folder(frame)
    
    # 处理自车数据
    # 计算自车朝向，将自车速度向量转换为绝对坐标系下的角度
    if egoVeh.speed_x_**2 + egoVeh.speed_y_**2 != 0:
        # 自车朝向向量
        directVec = np.array([egoVeh.speed_x_, egoVeh.speed_y_])
        # 绝对x轴和y轴单位向量
        xVec = np.array([1.0, 0.0])
        yVec = np.array([0.0, 1.0])
        
        # 计算与x轴和y轴的夹角
        xDegree = np.arccos(np.dot(directVec, xVec) / np.linalg.norm(directVec))
        yDegree = np.arccos(np.dot(directVec, yVec) / np.linalg.norm(directVec))
        
        # 确定自车绝对方向角
        if xDegree < np.pi / 2:
            egoVeh.absolute_theta_ = -yDegree
        else:
            egoVeh.absolute_theta_ = yDegree
    
    if test_data_log == True:
        # 记录自车角度
        with open(os.path.join(test_data_folder, f"frame_{frame}_ego_theta.csv"), 'w') as f:
            f.write(f"absolute_theta,{egoVeh.absolute_theta_}\n")
            f.write(f"relative_theta,{egoVeh.relative_theta_}\n")
    
    # 计算路径相对于绝对y轴的夹角，用于后续坐标转换
    path_Y_Degree = egoVeh.absolute_theta_ - egoVeh.relative_theta_  # 路径相对于绝对y轴的夹角，逆时针为正
    
    if test_data_log == True:
        # 记录路径夹角
        with open(os.path.join(test_data_folder, f"frame_{frame}_path_degree.csv"), 'w') as f:
            f.write(f"path_Y_Degree,{path_Y_Degree}\n")
    
    # 构建旋转矩阵，用于坐标系转换
    cos_val = np.cos(-path_Y_Degree)
    sin_val = np.sin(-path_Y_Degree)
    rotateMatrix = np.array([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0],
        [0, 0, 1]
    ])
    
    # 记录旋转矩阵
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_rotate_matrix.csv"), 'w') as f:
            np.savetxt(f, rotateMatrix, delimiter=',')
    
    # 处理道路数据
    # 计算最大横向加速度，基于最大绝对加速度和当前纵向加速度
    maxLatAcc = np.sqrt(road.maxAbsoluteAcc_**2 - egoVeh.acc_**2)
    # 计算由最大横向加速度决定的最大曲率
    Kmax_LatAcc = maxLatAcc / egoVeh.speed_**2 if egoVeh.speed_ != 0 else float('inf')  # 曲率由最大绝对加速度决定
    # 方向盘最大转角对应的曲率
    kmax_Delta = 1.0 / 6.0  # 曲率由最大方向盘转角决定
    # 取两者的较小值作为最大曲率约束
    k_max = min(kmax_Delta, Kmax_LatAcc)
    
    # 记录曲率约束
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_k_max.csv"), 'w') as f:
            f.write(f"maxLatAcc,{maxLatAcc}\n")
            f.write(f"Kmax_LatAcc,{Kmax_LatAcc}\n")
            f.write(f"kmax_Delta,{kmax_Delta}\n")
            f.write(f"k_max,{k_max}\n")
    
    # 处理基准线数据
    # 将基准线数据转换为numpy数组格式，使用Fortran顺序（先填充行再向下填充列）
    BaseLine1 = np.array(baseline).reshape(2, Mrow_b, order='F')
    
    # 记录原始基准线数据
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_baseline1.csv"), 'w') as f:
            for i in range(Mrow_b):
                f.write(f"{BaseLine1[0, i]},{BaseLine1[1, i]}\n")
    
    # 添加齐次坐标第三维
    BaseLine2 = np.vstack((BaseLine1, np.ones(Mrow_b)))
    
    # 记录添加齐次坐标后的基准线
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_baseline2.csv"), 'w') as f:
            for i in range(Mrow_b):
                f.write(f"{BaseLine2[0, i]},{BaseLine2[1, i]},{BaseLine2[2, i]}\n")
    
    # 应用旋转矩阵，将基准线转换到局部坐标系
    BaseLine3 = np.dot(rotateMatrix, BaseLine2)
    
    # 记录旋转后的基准线
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_baseline3.csv"), 'w') as f:
            for i in range(Mrow_b):
                f.write(f"{BaseLine3[0, i]},{BaseLine3[1, i]},{BaseLine3[2, i]}\n")
    
    # 将自车坐标也转换到同一坐标系
    egoXY = np.array([egoVeh.x_, egoVeh.y_, 1])
    egoXY = np.dot(rotateMatrix, egoXY)
    
    # 记录旋转后的自车坐标
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_ego_xy.csv"), 'w') as f:
            f.write(f"{egoXY[0]},{egoXY[1]},{egoXY[2]}\n")
    
    # 寻找基准线上距离自车最近的点
    SquareDis = []
    for i in range(BaseLine3.shape[1]):
        SquareDis.append((BaseLine3[0, i] - egoXY[0])**2 + (BaseLine3[1, i] - egoXY[1])**2)
    
    # 找出最小距离点的索引
    minIndex = np.argmin(SquareDis)
    
    # 记录最小距离索引
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_min_index.csv"), 'w') as f:
            f.write(f"min_index,{minIndex}\n")
    
    # 从最近点开始，截取一段基准线作为新的参考路径
    newBaseline = np.zeros((3, basepoint_num))
    
    for i in range(minIndex, minIndex + basepoint_num):
        if minIndex + basepoint_num > BaseLine3.shape[1]:
            print(f"[Error: No enough baseline point OR the calculation is finished]")
            print(minIndex + basepoint_num)
            # 在Python中，我们应该处理这种情况，但为了与C++代码保持一致，这里保留错误信息
            break
        
        newBaseline[0, i - minIndex] = BaseLine3[0, i]
        newBaseline[1, i - minIndex] = BaseLine3[1, i]
        newBaseline[2, i - minIndex] = 1
    
    # 记录截取的基准线
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_newbaseline.csv"), 'w') as f:
            for i in range(basepoint_num):
                f.write(f"{newBaseline[0, i]},{newBaseline[1, i]},{newBaseline[2, i]}\n")
    
    # 构建平移矩阵，将坐标系原点移到基准线起点
    transMatrix = np.array([
        [1, 0, -newBaseline[0, 0]],
        [0, 1, -newBaseline[1, 0]],
        [0, 0, 1]
    ])
    
    # 记录平移矩阵
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_trans_matrix.csv"), 'w') as f:
            np.savetxt(f, transMatrix, delimiter=',')
    
    # 应用平移矩阵
    newBaseline = np.dot(transMatrix, newBaseline)
    
    # 记录应用平移后的基准线
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_translated_baseline.csv"), 'w') as f:
            for i in range(basepoint_num):
                f.write(f"{newBaseline[0, i]},{newBaseline[1, i]},{newBaseline[2, i]}\n")
    
    # 将矩阵转回列表格式，用于后续处理
    baseline1 = newBaseline[:2, :].flatten()  # 只取前两行(x,y坐标)并按列优先顺序展平
    newBaseline_num = newBaseline.shape[1]
    
    # 记录转成列表格式的基准线
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_baseline_ptr.csv"), 'w') as f:
            for i in range(2 * newBaseline_num):
                f.write(f"{baseline1[i]}\n")
    
    # 处理周围车辆数据
    # 将周围车辆数据转换为numpy数组格式
    inputData = np.array(surVhe_input).reshape(Mrow_s, 9)
    
    # 记录周围车辆输入数据
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_input_data.csv"), 'w') as f:
            for i in range(Mrow_s):
                for j in range(8):
                    f.write(f"{inputData[i, j]},")
                f.write(f"{inputData[i, 8]}\n")
    
    surVeh_num = inputData.shape[0]  # 获取周围车辆数量
    surVeh = []
    
    # 逐个处理每辆周围车辆
    for i in range(surVeh_num):
        # 提取车辆数据并应用坐标变换
        surM = np.zeros((3, 3))
        surM[0, 0] = inputData[i, 0]  # x 旋转平移
        surM[1, 0] = inputData[i, 1]  # y 旋转平移
        surM[0, 1] = inputData[i, 4]  # vx 旋转
        surM[1, 1] = inputData[i, 5]  # vy 旋转
        surM[0, 2] = inputData[i, 6]  # ax 旋转
        surM[1, 2] = inputData[i, 7]  # ay 旋转
        surM[2, 0] = 1
        surM[2, 1] = 1
        surM[2, 2] = 1
        
        # 应用旋转变换
        newInput = np.dot(rotateMatrix, surM)  # 旋转
        
        # 应用平移变换
        newInput[0, 0] = newInput[0, 0] + transMatrix[0, 2]  # 坐标x,y平移
        newInput[1, 0] = newInput[1, 0] + transMatrix[1, 2]
        
        # 创建周围车辆对象并存储
        temp_veh = surVehicle(
            newInput[0, 0], newInput[1, 0],
            inputData[i, 2], inputData[i, 3],
            newInput[0, 1], newInput[1, 1],
            newInput[0, 2], newInput[1, 2],
            inputData[i, 8] - path_Y_Degree
        )
        surVeh.append(temp_veh)
    
    # 记录处理后的周围车辆数据
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_sur_veh.csv"), 'w') as f:
            for veh in surVeh:
                f.write(f"{veh.x_},{veh.y_},{veh.length_},{veh.width_},{veh.speed_x_},{veh.speed_y_},{veh.acc_x_},{veh.acc_y_},{veh.heading_}\n")
    
    # 坐标转换：笛卡尔坐标系到Frenet坐标系
    # 初始设置等分段数
    m_equal = 50
    # 初始化数组
    trans_x = np.zeros(m_equal + 1)
    trans_y = np.zeros(m_equal + 1)
    unitArcL = 0
    ArcL = 0
    # 多项式拟合参数
    a0_x = np.zeros(m_equal + 1)
    a1_x = np.zeros(m_equal + 1)
    a2_x = np.zeros(m_equal + 1)
    a0_y = np.zeros(m_equal + 1)
    a1_y = np.zeros(m_equal + 1)
    a2_y = np.zeros(m_equal + 1)
    cdnt_arc = np.zeros(m_equal + 1)
    
    # 第一次坐标转换，获取基准线长度
    unitArcL, ArcL = Cartesian_trans_Frenet(baseline1, newBaseline_num, m_equal, unitArcL, ArcL, 
                                          trans_x, trans_y, a0_x, a1_x, a2_x, a0_y, a1_y, a2_y, cdnt_arc)
    
    # 记录第一次坐标转换结果
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_first_transform.csv"), 'w') as f:
            f.write(f"unitArcL,{unitArcL}\n")
            f.write(f"ArcL,{ArcL}\n")
            for i in range(m_equal + 1):
                f.write(f"trans_x[{i}],{trans_x[i]}\n")
                f.write(f"trans_y[{i}],{trans_y[i]}\n")
                f.write(f"a0_x[{i}],{a0_x[i]}\n")
                f.write(f"a1_x[{i}],{a1_x[i]}\n")
                f.write(f"a2_x[{i}],{a2_x[i]}\n")
                f.write(f"a0_y[{i}],{a0_y[i]}\n")
                f.write(f"a1_y[{i}],{a1_y[i]}\n")
                f.write(f"a2_y[{i}],{a2_y[i]}\n")
                f.write(f"cdnt_arc[{i}],{cdnt_arc[i]}\n")
    
    # 基于实际长度重新设置等分段数
    m_equal = round(ArcL / 1)
    # 初始化新的数组
    trans_x2 = np.zeros(m_equal + 1)
    trans_y2 = np.zeros(m_equal + 1)
    unitArcL2 = 0
    ArcL2 = 0
    a0_x2 = np.zeros(m_equal + 1)
    a1_x2 = np.zeros(m_equal + 1)
    a2_x2 = np.zeros(m_equal + 1)
    a0_y2 = np.zeros(m_equal + 1)
    a1_y2 = np.zeros(m_equal + 1)
    a2_y2 = np.zeros(m_equal + 1)
    cdnt_arc2 = np.zeros(m_equal + 1)
    
    # 第二次坐标转换，更精确的分段
    unitArcL2, ArcL2 = Cartesian_trans_Frenet(baseline1, newBaseline_num, m_equal, unitArcL2, ArcL2, 
                                           trans_x2, trans_y2, a0_x2, a1_x2, a2_x2, a0_y2, a1_y2, a2_y2, cdnt_arc2)
    
    # 记录第二次坐标转换结果
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_second_transform.csv"), 'w') as f:
            f.write(f"m_equal,{m_equal}\n")
            f.write(f"unitArcL2,{unitArcL2}\n")
            f.write(f"ArcL2,{ArcL2}\n")
            for i in range(m_equal + 1):
                f.write(f"{trans_x2[i]},{trans_y2[i]},{a0_x2[i]},{a1_x2[i]},{a2_x2[i]},{a0_y2[i]},{a1_y2[i]},{a2_y2[i]},{cdnt_arc2[i]}\n")
    
    # 创建Way对象，存储基准线信息
    way = Way()
    way.baseline_x = trans_x2
    way.baseline_y = trans_y2
    way.d1x = a2_x2
    way.d2x = a1_x2
    way.d1y = a2_y2
    way.d2y = a1_y2
    
    # 计算曲率和切线角
    # 为避免除零错误，添加一个小值
    epsilon = 1e-10
    d1x_squared = way.d1x**2
    d1y_squared = way.d1y**2
    denominator = ((d1x_squared + d1y_squared)**(3/2)) + epsilon
    way.kb = (way.d1x * way.d2y - way.d2x * way.d1y) / denominator
    # 计算切线角，处理除零情况
    way.theta = np.zeros_like(way.d1x)
    for i in range(len(way.d1x)):
        if abs(way.d1x[i]) > epsilon:
            way.theta[i] = np.arctan(way.d1y[i] / way.d1x[i])
        else:
            way.theta[i] = np.pi/2 if way.d1y[i] > 0 else -np.pi/2
    
    # 记录way对象数据
    if test_data_log == True:
        with open(os.path.join(test_data_folder, f"frame_{frame}_way.csv"), 'w') as f:
            f.write("// way.baseline_x, way.baseline_y, way.d1x, way.d2x, way.d1y, way.d2y, way.kb, way.theta\n")
            for i in range(m_equal + 1):
                f.write(f"{way.baseline_x[i]},{way.baseline_y[i]},{way.d1x[i]},{way.d2x[i]},{way.d1y[i]},{way.d2y[i]},{way.kb[i]},{way.theta[i]}\n")
    
    # 生成自车轨迹簇
    traj_num = 0
    egoRect = []
    UseablePath_num = 0
    AvailablePath = []
    unitArcPath = []
    K_max = []
    Path = []
    
    # 调用生成自车轨迹簇函数
    traj_num, egoRect, UseablePath_num, AvailablePath, unitArcPath, K_max, Path = Generate_Ego_TraCluster(
        egoVeh, road, k_max, unitArcL2, way, cdnt_arc2, m_equal, timeHorizon, timeStep,
        traj_num, egoRect, UseablePath_num, AvailablePath, unitArcPath, K_max, Path
    )
    
    # 生成周围车辆轨迹
    surRect = []
    Generate_surVehicle_Traj(surVeh, surRect, timeHorizon, timeStep, frame, log_flag, surRect_folder)
    
    # 碰撞检测
    CollisonIndex = [0] * UseablePath_num
    Collison_Detection(surVeh, egoVeh, egoRect, surRect, timeHorizon, timeStep, AvailablePath, 
                      UseablePath_num, unitArcPath, CollisonIndex)
    
    # 计算可行驶区域
    DA = Calculate_Drivable_Area(UseablePath_num, egoRect, CollisonIndex, frame, log_flag, polygon_folder)
    
    # 计算风险等级
    RL = Calculate_RiskLevel(UseablePath_num, AvailablePath, K_max, CollisonIndex, unitArcPath,
                           egoVeh, timeHorizon, timeStep, Path)
    
    print(f" 第 {frame} 帧计算结果: DA = {DA}, RL = {RL}")
    
    # 返回计算结果
    return [DA, RL]

# ==================== 轨迹生成函数 ====================

def generate_path(ego, way, path_planning_segment, traj_num, qf, cdnt_arc,
                path_q, path_theta, path_x, path_y, path_L):
    """
    生成自车轨迹簇 - 五次多项式轨迹规划
    
    使用五次多项式在Frenet坐标系中生成平滑轨迹:
    q(s) = a*s³ + b*s² + c*s + d
    
    其中:
    - q: 横向偏移量
    - s: 弧长坐标
    - 边界条件: q(0)=init_q, q'(0)=tan(θ), q(sf)=qf, q'(sf)=0
    
    参数:
        ego: Vehicle对象，自车当前状态
        way: Way对象，参考路径信息
        path_planning_segment: 各轨迹的规划长度(点数)
        traj_num: 轨迹数量
        qf: 各轨迹终点的横向偏移量
        cdnt_arc: 参考路径累计弧长数组
    
    输出 (通过引用修改):
        path_q: 各轨迹的横向偏移量序列
        path_theta: 各轨迹的斜率序列
        path_x, path_y: 各轨迹在笛卡尔坐标系中的坐标
        path_L: 各轨迹的长度
    """
    import numpy as np
    
    # 计算样条曲线的a, b, c, d值
    d = ego.init_q_  # 初始横向偏移
    c = np.tan(ego.relative_theta_)  # 初始角度对应的斜率
    
    a = []
    b = []
    
    # 依次生成轨迹簇中的每一条轨迹
    for i in range(traj_num):
        cdn_len = cdnt_arc[path_planning_segment[i]] - cdnt_arc[0]
        # 构建矩阵A
        A = np.array([
            [pow(cdn_len, 3), pow(cdn_len, 2)],
            [3 * pow(cdn_len, 2), 2 * cdn_len]
        ])
        # 构建矩阵B
        B = np.array([
            [qf[i] - ego.init_q_ - c * cdn_len],
            [-c]
        ])
        # 求解线性方程组
        Result = np.linalg.inv(A) @ B
        a_val = Result[0, 0]
        if np.isnan(a_val):
            a_val = 0
        a.append(a_val)
        
        b_val = Result[1, 0]
        if np.isnan(b_val):
            b_val = 0
        b.append(b_val)
    
    # 生成各轨迹的点
    for j in range(traj_num):
        temp_q = []
        temp_theta = []
        temp_x = []
        temp_y = []
        sum_length = 0
        
        for i in range(path_planning_segment[j] + 1):
            delta_s = cdnt_arc[i] - cdnt_arc[0]
            # 计算横向偏移
            q_value = a[j] * pow(delta_s, 3) + b[j] * pow(delta_s, 2) + c * delta_s + d
            temp_q.append(q_value)
            
            # 计算斜率
            dqds = 3 * a[j] * pow(delta_s, 2) + 2 * b[j] * delta_s + c
            temp_theta.append(np.arctan(dqds))
            
            # 计算x, y坐标
            x_value = way.baseline_x[i] - q_value * way.d1y[i]
            y_value = way.baseline_y[i] + q_value * way.d1x[i]
            temp_x.append(x_value)
            temp_y.append(y_value)
            
            # 计算路径长度
            if i > 0:
                sum_length += np.sqrt(pow((temp_x[i] - temp_x[i-1]), 2) + pow((temp_y[i] - temp_y[i-1]), 2))
        
        # 添加到结果列表
        path_q.append(temp_q)
        path_theta.append(temp_theta)
        path_x.append(temp_x)
        path_y.append(temp_y)
        path_L.append(sum_length)
        
        # 清空临时列表
        temp_x = []
        temp_y = []
        temp_q = []
        temp_theta = []

# 补齐Path函数 - 对应C++中的addPath函数
def addPath(path_x, path_y, qf, way, AvailablePath, UseablePath_num, Path_Planning_Segments, realPath_segments, AddFlag):
    """
    补齐generate_path生成的轨迹簇
    
    参数:
    path_x - 各轨迹x坐标
    path_y - 各轨迹y坐标
    qf - 各轨迹相对参考路径的横向偏移量
    way - Way类，参考路径信息
    AvailablePath - 满足曲率要求的轨迹编号列表
    UseablePath_num - 满足曲率要求的轨迹数量
    Path_Planning_Segments - 各轨迹规划长度
    realPath_segments - 需补齐的长度
    AddFlag - 各轨迹是否需要补齐的标志
    """
    for i in range(UseablePath_num):
        if AddFlag[i] == 1:
            for j in range(Path_Planning_Segments[AvailablePath[i]] + 1, realPath_segments):
                path_x[AvailablePath[i]].append(way.baseline_x[j] - qf[AvailablePath[i]] * way.d1y[j])
                path_y[AvailablePath[i]].append(way.baseline_y[j] + qf[AvailablePath[i]] * way.d1x[j])

# 生成自车轨迹簇函数 - 对应C++中的Generate_Ego_TraCluster函数
def Generate_Ego_TraCluster(ego, road, k_max, unitArcL, way, cdnt_arc, m_equal, timeHorizon, timeStep,
                           traj_num, egoRect, UseablePath_num, AvailablePath, unitArcPath, K_max, Path):
    """
    生成自车轨迹簇，并沿轨迹簇生成自车的占用矩形
    
    参数:
    ego - Vehicle类，存储自车信息
    road - Road类，存储道路信息
    k_max - 最大曲率约束
    unitArcL - 参考路径单位弧长
    way - Way类，参考路径信息
    cdnt_arc - 参考路径累计弧长
    m_equal - 参考路径点数量
    timeHorizon - 计算时长
    timeStep - 计算步长
    
    输出:
    traj_num - 轨迹数量
    egoRect - 自车占用矩形列表
    UseablePath_num - 满足曲率要求的轨迹数量
    AvailablePath - 满足曲率要求的轨迹编号列表
    unitArcPath - 各轨迹单位弧长列表
    K_max - 各轨迹最大曲率列表
    Path - 各轨迹Way信息列表
    """
    import numpy as np
    import math
    
    # 计算实际可行驶车道数
    actual_lane = 0
    if road.cross_centerline_:
        actual_lane = road.lane_num_
    else:
        actual_lane = road.lane_egodirect_
    
    # 计算轨迹簇中的轨迹数
    traj_num = math.ceil((actual_lane * road.lane_width_ / ego.width_ - 1) * 2)
    
    # 针对单车道进行优化
    if actual_lane == 1:
        traj_num = 3
    
    # 计算左右偏移范围
    deviationLeft = (actual_lane + 1 - ego.lane_posi_) * road.lane_width_ - ego.width_ / 2
    deviationRight = (ego.lane_posi_ - 1) * road.lane_width_ - ego.width_ / 2
    
    # 计算最小转弯半径
    Rmin = pow(k_max, -1)
    
    # 计算横向间隔
    unitDeviance = (actual_lane * road.lane_width_ - ego.width_) / (traj_num - 1)
    
    # 初始化各轨迹规划长度和横向偏移量
    path_planning_segment = []
    qf = []
    
    # 计算每一条轨迹的初始规划长度
    for i in range(traj_num):
        temp_deviance = abs(deviationRight - unitDeviance * i)
        temp_seg = math.ceil((np.sqrt(4 * Rmin * temp_deviance - pow(temp_deviance, 2)) + Rmin * np.sin(-ego.relative_theta_)) / unitArcL)
        
        if temp_seg < 10:
            temp_seg = 10
        if temp_seg > m_equal:
            temp_seg = m_equal
            
        path_planning_segment.append(temp_seg)
        temp_qf = -deviationRight + unitDeviance * i
        qf.append(temp_qf)
    
    # 初始化轨迹数据列表
    path_q = []
    path_theta = []
    path_x = []
    path_y = []
    path_L = []
    
    # 轨迹生成迭代标志
    EndFlag_all = 0
    EndFlag = [0] * traj_num
    OverPredictFlag = [0] * traj_num
    
    # 不断迭代直到生成符合曲率要求的轨迹
    while EndFlag_all != 1:
        # 生成轨迹
        generate_path(ego, way, path_planning_segment, traj_num, qf, cdnt_arc,
                     path_q, path_theta, path_x, path_y, path_L)
        
        # 计算各轨迹的最大曲率
        for i in range(traj_num):
            # 轨迹投影向量
            proVector = [path_x[i][path_planning_segment[i]] - path_x[i][0], 
                        path_y[i][path_planning_segment[i]] - path_y[i][0]]
            projVec = np.array(proVector).reshape(1, 2)
            xVec = np.array([1, 0]).reshape(1, 2)
            # print(f"轨迹 {i} 投影向量: {proVector}")
            
            # 计算旋转角度
            dot_product = np.dot(projVec, xVec.T)[0, 0]
            norm_product = np.linalg.norm(projVec) * np.linalg.norm(xVec)
            rotateDegree = np.arccos(dot_product / norm_product)
            # print(f"轨迹 {i} 旋转角度: {np.rad2deg(rotateDegree)}")

            if projVec[0, 1] < 0:  # 如果y分量为负，调整旋转角度为负
                rotateDegree = -rotateDegree
            
            # 构建旋转矩阵
            Value_rom = np.array([
                np.cos(-rotateDegree), -np.sin(-rotateDegree),
                np.sin(-rotateDegree), np.cos(-rotateDegree)
            ]).reshape(2, 2)
            rotateMatrix = Value_rom
            
            # 准备轨迹点数据
            path_xy = path_x[i] + path_y[i]  # 合并x,y坐标
            path_xy_arr = np.array(path_xy).reshape(2, path_planning_segment[i] + 1)
            # print(f"轨迹 {i} 原始坐标: {path_xy_arr}")
            
            # 旋转轨迹点
            newXY = np.dot(rotateMatrix, path_xy_arr)
            
            # 插值点数量
            point_num = max(100, path_planning_segment[i])
            
            # 提取旋转后的坐标
            cur_x = newXY[0, :]
            cur_y = newXY[1, :]
            # print(f"轨迹 {i} 旋转后X坐标: {cur_x}")
            
            # 准备插值数据
            trans_x = np.zeros(point_num)
            trans_y = np.zeros(point_num)
            # b, c, d 数组大小应该基于原始点数，不是插值点数
            b = np.zeros(len(cur_x))
            c = np.zeros(len(cur_x))
            d = np.zeros(len(cur_x))
            curvature = []
            
            # 生成等间距插值点
            for j in range(point_num):
                trans_x[j] = cur_x[0] + j * (cur_x[-1] - cur_x[0]) / (point_num - 1)
            
            # 样条插值
            SPL(len(cur_x), cur_x, cur_y, point_num, trans_x, trans_y, b, c, d, curvature)
            # print(cur_x)
            
            # 记录最大曲率
            K_max.append(max(curvature))
        
        # 检查各轨迹是否满足曲率要求
        for i in range(traj_num):
            if EndFlag[i] == 1:
                continue
                
            if K_max[i] > k_max:
                # 如果超过最大曲率，增加规划长度
                path_planning_segment[i] += 2
                if path_planning_segment[i] >= m_equal - 1:
                    path_planning_segment[i] -= 2
                    OverPredictFlag[i] = 1
                    EndFlag[i] = 1
            else:
                EndFlag[i] = 1
        
        # 检查是否所有轨迹都已完成
        EndFlag_all = 1
        for flag in EndFlag:
            if flag == 0:
                EndFlag_all = 0
                break
        
        # 如果未完成，清空数据重新迭代
        if EndFlag_all == 0:
            K_max.clear()
            path_q.clear()
            path_theta.clear()
            path_x.clear()
            path_y.clear()
            path_L.clear()
    
    # 记录满足曲率要求的轨迹
    UseablePath_num = 0
    for i in range(traj_num):
        if OverPredictFlag[i] == 0:
            UseablePath_num += 1
            AvailablePath.append(i)
    
    # 计算实际规划长度
    realPath_segments = int(math.ceil((ego.speed_ * timeHorizon) / unitArcL)) + 10
    
    # 标记需要补齐的轨迹
    AddFlag = []
    for i in range(UseablePath_num):
        if path_planning_segment[AvailablePath[i]] < realPath_segments:
            AddFlag.append(1)
        else:
            AddFlag.append(0)
    
    # 补齐轨迹
    addPath(path_x, path_y, qf, way, AvailablePath, UseablePath_num, path_planning_segment, realPath_segments, AddFlag)
    
    # 构建自车基本矩形
    BasRec = np.array([
        ego.length_ / 2, -ego.length_ / 2, -ego.length_ / 2, ego.length_ / 2,
        -ego.width_ / 2, -ego.width_ / 2, ego.width_ / 2, ego.width_ / 2
    ]).reshape(2, 4)  # 使用列优先(Fortran风格)存储
    BasicRect = BasRec
    
    # 为每条轨迹生成占用矩形
    for i in range(UseablePath_num):
        # 准备轨迹数据
        path_XY = []
        if len(path_x[AvailablePath[i]]) <= realPath_segments:
            path_XY = path_x[AvailablePath[i]] + path_y[AvailablePath[i]]
        else:
            path_XY = path_x[AvailablePath[i]][:realPath_segments] + path_y[AvailablePath[i]][:realPath_segments]
        
        # 转换为数组格式
        X_Y = np.array(path_XY).reshape(2, -1)
        
        # 插值点数量
        point_num = 100
        
        # 初始化插值数组
        spline_path_x = np.zeros(point_num + 1)
        spline_path_y = np.zeros(point_num + 1)
        unitPathL = 0
        PathL = 0
        a0_x = np.zeros(point_num + 1)
        a1_x = np.zeros(point_num + 1)
        a2_x = np.zeros(point_num + 1)
        a0_y = np.zeros(point_num + 1)
        a1_y = np.zeros(point_num + 1)
        a2_y = np.zeros(point_num + 1)
        cdnt_arc_local = np.zeros(point_num + 1)
        
        # 坐标转换
        unitPathL, PathL = Cartesian_trans_Frenet(X_Y.flatten(), X_Y.shape[1], point_num, 
                                                unitPathL, PathL, spline_path_x, spline_path_y,
                                                a0_x, a1_x, a2_x, a0_y, a1_y, a2_y, cdnt_arc_local)
        
        # 记录轨迹单位弧长
        unitArcPath.append(unitPathL)
        
        # 创建Way对象存储轨迹信息
        path = Way()
        path.baseline_x = spline_path_x
        path.baseline_y = spline_path_y
        path.d1x = a2_x
        path.d2x = a1_x
        path.d1y = a2_y
        path.d2y = a1_y
        
        # 计算曲率和切线角
        epsilon = 1e-10
        path.kb = np.zeros_like(path.d1x)
        for j in range(len(path.d1x)):
            denominator = ((path.d1x[j]**2 + path.d1y[j]**2)**(3/2)) + epsilon
            path.kb[j] = abs((path.d1x[j] * path.d2y[j] - path.d2x[j] * path.d1y[j]) / denominator)
        
        path.theta = np.zeros_like(path.d1x)
        for j in range(len(path.d1x)):
            if abs(path.d1x[j]) > epsilon:
                path.theta[j] = np.arctan(path.d1y[j] / path.d1x[j])
            else:
                path.theta[j] = np.pi/2 if path.d1y[j] > 0 else -np.pi/2
        
        # 添加轨迹信息
        Path.append(path)
        
        # 生成自车占用矩形
        temp_RecVec = []
        for j in range(point_num + 1):
            # 计算旋转角度
            rotateDeg = path.theta[j]
            if path.theta[j] < 0:
                rotateDeg = path.theta[j] + np.pi
                
            # 构建旋转矩阵
            rotaM = np.array([
                np.cos(rotateDeg), -np.sin(rotateDeg),
                np.sin(rotateDeg), np.cos(rotateDeg)
            ]).reshape(2, 2)
            RotateM = rotaM
            
            # 旋转基本矩形
            temp_Rect = np.dot(RotateM, BasicRect)
            
            # 平移到轨迹点位置
            tran = np.array([
                path.baseline_x[j], path.baseline_x[j], path.baseline_x[j], path.baseline_x[j],
                path.baseline_y[j], path.baseline_y[j], path.baseline_y[j], path.baseline_y[j]
            ]).reshape(2, 4)
            temp_Rect = temp_Rect + tran
            
            # 添加到矩形列表
            temp_RecVec.append(temp_Rect)
        
        # 添加到全局矩形列表
        egoRect.append(temp_RecVec)
    
    return traj_num, egoRect, UseablePath_num, AvailablePath, unitArcPath, K_max, Path

# ==================== 周围车辆轨迹预测 ====================

def Generate_surVehicle_Traj(surVeh, surRect, timeHorizon, timeStep, frame, log_flag, surRect_folder):
    """
    生成周围车辆未来轨迹 - 匀变速运动模型
    
    使用简单的匀变速运动模型预测周围车辆未来轨迹:
    x(t) = x₀ + vₓ*t + 0.5*aₓ*t²
    y(t) = y₀ + vᵧ*t + 0.5*aᵧ*t²
    
    为每个时间步生成车辆占用矩形，用于后续碰撞检测。
    
    注意：
    - 当速度降为零后，车辆停止运动
    - 车辆朝向根据速度方向动态计算
    
    参数:
        surVeh: 周围车辆列表(surVehicle对象)
        surRect: 输出的占用矩形列表
        timeHorizon: 预测时间范围(秒)
        timeStep: 时间步长(秒)
        frame: 当前帧号
        log_flag: 是否保存到文件
        surRect_folder: 保存路径
    
    输出:
        surRect: 每辆车在各时间步的占用矩形 [车辆数][时间步][2×4矩阵]
    """
    import numpy as np
    import os
    
    surVeh_num = len(surVeh)
    
    for i in range(surVeh_num):
        # 生成当前车辆的占用矩形列表
        temp_surRect = []
        
        # 创建基本矩形，表示车辆形状
        BasRec = np.array([
            surVeh[i].length_ / 2, -surVeh[i].length_ / 2, -surVeh[i].length_ / 2, surVeh[i].length_ / 2,
            -surVeh[i].width_ / 2, -surVeh[i].width_ / 2, surVeh[i].width_ / 2, surVeh[i].width_ / 2
        ]).reshape(2, 4)
        BasicRect = BasRec
        
        # 防止车辆速度为负时倒车的标志
        stopFlag = 0
        surveh_theta = 0
        T = 0
        
        # 对每个时间步长生成占用矩形
        for j in range(int(timeHorizon / timeStep) + 1):
            t = j * timeStep
            
            # 检查速度是否变为负值
            if j > 0:
                preVelVec = np.array([
                    surVeh[i].speed_x_ + surVeh[i].acc_x_ * (t - timeStep), 
                    surVeh[i].speed_y_ + surVeh[i].acc_y_ * (t - timeStep)
                ])
                newVelVec = np.array([
                    surVeh[i].speed_x_ + surVeh[i].acc_x_ * t,
                    surVeh[i].speed_y_ + surVeh[i].acc_y_ * t
                ])
                if np.dot(preVelVec, newVelVec) <= 0:
                    stopFlag = 1
            
            # 计算车辆朝向
            speed_squared = (surVeh[i].speed_x_ + surVeh[i].acc_x_ * t)**2 + (surVeh[i].speed_y_ + surVeh[i].acc_y_ * t)**2
            if speed_squared == 0:
                if t == 0:
                    surveh_theta = surVeh[i].heading_ + 1.57
                else:
                    stopFlag = 1
            elif surVeh[i].speed_x_ + surVeh[i].acc_x_ * t == 0:
                surveh_theta = 1.5708  # 约等于π/2
            else:
                if stopFlag == 0:
                    surveh_theta = np.arctan((surVeh[i].speed_y_ + surVeh[i].acc_y_ * t) / (surVeh[i].speed_x_ + surVeh[i].acc_x_ * t))
            
            # 更新时间，用于位置计算
            if stopFlag == 0:
                T = t
            
            # 计算车辆位置
            surveh_x = surVeh[i].x_ + surVeh[i].speed_x_ * T + 0.5 * surVeh[i].acc_x_ * T**2
            surveh_y = surVeh[i].y_ + surVeh[i].speed_y_ * T + 0.5 * surVeh[i].acc_y_ * T**2
            
            # 构建旋转矩阵
            rotaM = np.array([
                np.cos(surveh_theta), -np.sin(surveh_theta),
                np.sin(surveh_theta), np.cos(surveh_theta)
            ]).reshape(2, 2)
            RotateM = rotaM
            
            # 旋转基本矩形
            temp_Rect = np.dot(RotateM, BasicRect)
            
            # 构建平移矩阵
            tran = np.array([
                surveh_x, surveh_x, surveh_x, surveh_x,
                surveh_y, surveh_y, surveh_y, surveh_y
            ]).reshape(2, 4)
            
            # 平移旋转后的矩形
            temp_Rect = temp_Rect + tran
            
            # 添加到结果列表
            temp_surRect.append(temp_Rect)
        
        # 将当前车辆的所有占用矩形添加到总结果中
        surRect.append(temp_surRect)
        
        # 如果需要记录日志，将结果保存到文件
        if log_flag:
            os.makedirs(surRect_folder, exist_ok=True)
            
            with open(os.path.join(surRect_folder, f"surVeh_{frame}_{i}.csv"), 'w') as file_surVeh:
                for k in range(int(timeHorizon / timeStep)):
                    file_surVeh.write(f"{surRect[i][k][0, 0]},{surRect[i][k][0, 1]},{surRect[i][k][0, 2]},{surRect[i][k][0, 3]}\n")
                    file_surVeh.write(f"{surRect[i][k][1, 0]},{surRect[i][k][1, 1]},{surRect[i][k][1, 2]},{surRect[i][k][1, 3]}\n")

# ==================== 碰撞检测函数 ====================

def Collison_Detection(surVeh, egoVeh, egoRect, surRect, timeHorizon, timeStep, AvailablePath, 
                      UseablePath_num, unitArcPath, CollisonIndex):
    """
    碰撞检测 - 检测自车轨迹与周围车辆的碰撞
    
    对每条候选轨迹，检测其在未来timeHorizon时间内是否与周围车辆发生碰撞。
    使用分离轴定理(SAT)进行高效的矩形碰撞检测。
    
    检测策略:
    1. 对每条轨迹，遍历所有时间步
    2. 对每个时间步，检测自车与所有周围车辆
    3. 一旦检测到碰撞，记录碰撞点并停止该轨迹的检测
    
    参数:
        surVeh: 周围车辆列表
        egoVeh: 自车对象
        egoRect: 自车在各轨迹上的占用矩形 [轨迹数][点数][2×4]
        surRect: 周围车辆占用矩形 [车辆数][时间步][2×4]
        timeHorizon: 预测时间范围(秒)
        timeStep: 时间步长(秒)
        AvailablePath: 可用轨迹编号列表
        UseablePath_num: 可用轨迹数量
        unitArcPath: 各轨迹单位弧长
    
    输出:
        CollisonIndex: 各轨迹首次碰撞点的索引，-1表示无碰撞
    """
    for i in range(UseablePath_num):  # 对每条潜在轨迹
        detectonOverFlag = 0
        
        # 初始化碰撞索引为轨迹最大长度
        CollisonIndex[i] = int(np.ceil((egoVeh.speed_ * timeHorizon) / unitArcPath[i]))
        
        stopTime = 0
        for j in range(int(timeHorizon / timeStep)):  # 对每个时间步长
            # 计算检测起始和结束索引
            dectSrart_index = int(np.ceil((egoVeh.speed_ * j * timeStep) / unitArcPath[i]))
            dectEnd_index = int(np.ceil((egoVeh.speed_ * (j + 1) * timeStep) / unitArcPath[i]))
            
            for check_index in range(dectSrart_index, dectEnd_index):
                for k in range(len(surVeh)):
                    # 准备检测矩形
                    underdetectRect = []
                    underdetectRect.append(egoRect[i][check_index])
                    underdetectRect.append(surRect[k][j])
                    
                    # 进行碰撞检测
                    collisonFlag = collison_detection(underdetectRect)
                    underdetectRect.clear()
                    
                    if collisonFlag == 1:
                        # 发生碰撞，记录碰撞点前一个点的索引
                        CollisonIndex[i] = check_index - 1
                        detectonOverFlag = 1
                        break
                
                if detectonOverFlag == 1:
                    break
            
            if detectonOverFlag == 1:
                break

def collison_detection(underdetectRect):
    """
    分离轴定理(SAT)碰撞检测 - 高效的矩形碰撞检测算法
    
    分离轴定理原理:
    两个凸多边形不相交，当且仅当存在一条轴，使得两个多边形在该轴上的投影不重叠。
    
    对于矩形，只需检查4条轴(每个矩形的两条边的法向量):
    1. 矩形1的两条边的法向量
    2. 矩形2的两条边的法向量
    
    如果在所有4条轴上的投影都重叠，则两矩形碰撞；
    否则，存在分离轴，两矩形不碰撞。
    
    参数:
        underdetectRect: 包含两个矩形的列表
                        每个矩形为2×4矩阵，4列为4个顶点坐标
    
    返回:
        CollisionFlag: 1表示碰撞，0表示未碰撞
    """
    import numpy as np
    
    # 基于分离轴定律的矩形重叠检测算法
    CollisionFlag = 1  # 默认发生碰撞
    checkVec = []
    unitVec = []
    
    # 计算第一轴
    temp1 = np.array([
        underdetectRect[0][0, 1] - underdetectRect[0][0, 0], 
        underdetectRect[0][1, 1] - underdetectRect[0][1, 0]
    ])
    temp1u = temp1 / np.linalg.norm(temp1)
    checkVec.append(temp1)
    unitVec.append(temp1u)
    
    # 计算第二轴
    temp1 = np.array([
        underdetectRect[0][0, 2] - underdetectRect[0][0, 1], 
        underdetectRect[0][1, 2] - underdetectRect[0][1, 1]
    ])
    temp1u = temp1 / np.linalg.norm(temp1)
    checkVec.append(temp1)
    unitVec.append(temp1u)
    
    # 计算第三轴
    temp1 = np.array([
        underdetectRect[1][0, 1] - underdetectRect[1][0, 0], 
        underdetectRect[1][1, 1] - underdetectRect[1][1, 0]
    ])
    temp1u = temp1 / np.linalg.norm(temp1)
    checkVec.append(temp1)
    unitVec.append(temp1u)
    
    # 计算第四轴
    temp1 = np.array([
        underdetectRect[1][0, 2] - underdetectRect[1][0, 1], 
        underdetectRect[1][1, 2] - underdetectRect[1][1, 1]
    ])
    temp1u = temp1 / np.linalg.norm(temp1)
    checkVec.append(temp1)
    unitVec.append(temp1u)
    
    # 计算矩形中心点
    center_Ego_x = (underdetectRect[0][0, 1] + underdetectRect[0][0, 3]) / 2
    center_Ego_y = (underdetectRect[0][1, 1] + underdetectRect[0][1, 3]) / 2
    center_Sur_x = (underdetectRect[1][0, 1] + underdetectRect[1][0, 3]) / 2
    center_Sur_y = (underdetectRect[1][1, 1] + underdetectRect[1][1, 3]) / 2
    
    centerVec = np.array([center_Sur_x - center_Ego_x, center_Sur_y - center_Ego_y])
    
    # 检查四个轴上的投影是否有分离
    PassFlag = 0
    for i in range(4):
        radiusProj_sum = 0
        for j in range(4):
            halfcheckVec = 0.5 * checkVec[j]
            radiusProj_sum += abs(np.dot(unitVec[i], halfcheckVec))
        
        centerLine = abs(np.dot(centerVec, unitVec[i]))
        
        if centerLine > radiusProj_sum:
            CollisionFlag = 0  # 在某一轴上分离，没有碰撞
            PassFlag = 1
        else:
            if PassFlag == 1:
                continue
            CollisionFlag = 1
    
    return CollisionFlag

# ==================== 可行驶区域计算 ====================

def Calculate_Drivable_Area(UseablePath_num, egoRect, CollisonIndex, frame, log_flag, polygon_folder):
    """
    计算可行驶区域(DA) - 将无碰撞轨迹端点连接成多边形并计算面积
    
    算法步骤:
    1. 对每条轨迹，找到碰撞点(或最远点)
    2. 按特定规则连接各轨迹的端点，形成封闭多边形
    3. 使用鞋带公式计算多边形面积
    
    多边形构建规则:
    - 第一条轨迹: 左边界 → 端点 → 右边界
    - 中间轨迹: 前一轨迹端点 → 当前轨迹端点 → 右边界
    - 最后一条: 前一轨迹端点 → 当前轨迹端点 → 右边界 → 起点
    
    参数:
        UseablePath_num: 可用轨迹数量
        egoRect: 自车占用矩形 [轨迹数][点数][2×4]
        CollisonIndex: 各轨迹碰撞点索引，-1表示无碰撞
        frame: 当前帧号
        log_flag: 是否保存多边形数据
        polygon_folder: 保存路径
    
    返回:
        DA: 可行驶区域面积(m²)，越大表示可用空间越大
    """
    import numpy as np
    import os
    
    polygon_x = []
    polygon_y = []
    collisionFlag = []
    
    for i in range(UseablePath_num):
        if CollisonIndex[i] < 0:
            collisionFlag.append(1)
            continue
        else:
            collisionFlag.append(0)
        
        if i == 0:  # 第一条轨迹
            if UseablePath_num == 1:  # 针对可用路径只有1条的特殊优化
                for j in range(CollisonIndex[i] + 1):
                    polygon_x.append(egoRect[i][j][0, 1])
                    polygon_y.append(egoRect[i][j][1, 1])
                
                polygon_x.append(egoRect[i][CollisonIndex[i]][0, 0])
                polygon_y.append(egoRect[i][CollisonIndex[i]][1, 0])
                
                for j in range(CollisonIndex[i], -1, -1):
                    polygon_x.append(egoRect[i][j][0, 3])
                    polygon_y.append(egoRect[i][j][1, 3])
                continue
            else:
                for j in range(CollisonIndex[i] + 1):
                    polygon_x.append(egoRect[i][j][0, 1])
                    polygon_y.append(egoRect[i][j][1, 1])
                
                polygon_x.append(egoRect[i][CollisonIndex[i]][0, 0])
                polygon_y.append(egoRect[i][CollisonIndex[i]][1, 0])
                
                if CollisonIndex[i] > CollisonIndex[i + 1]:
                    for j in range(CollisonIndex[i], CollisonIndex[i + 1] - 1, -1):
                        polygon_x.append(egoRect[i][j][0, 3])
                        polygon_y.append(egoRect[i][j][1, 3])
        
        if (i > 0) and (i < UseablePath_num - 1):  # 中间轨迹
            if CollisonIndex[i] > CollisonIndex[i - 1]:
                for j in range(CollisonIndex[i - 1], CollisonIndex[i] + 1):
                    polygon_x.append(egoRect[i][j][0, 0])
                    polygon_y.append(egoRect[i][j][1, 0])
                
                if CollisonIndex[i] > CollisonIndex[i + 1]:
                    for j in range(CollisonIndex[i], CollisonIndex[i + 1] - 1, -1):
                        polygon_x.append(egoRect[i][j][0, 3])
                        polygon_y.append(egoRect[i][j][1, 3])
            else:
                if CollisonIndex[i] > CollisonIndex[i + 1]:
                    for j in range(CollisonIndex[i], CollisonIndex[i + 1] - 1, -1):
                        polygon_x.append(egoRect[i][j][0, 3])
                        polygon_y.append(egoRect[i][j][1, 3])
                elif CollisonIndex[i] == CollisonIndex[i + 1]:
                    polygon_x.append(egoRect[i][CollisonIndex[i]][0, 3])
                    polygon_y.append(egoRect[i][CollisonIndex[i]][1, 3])
        
        if i == UseablePath_num - 1:  # 最后一条轨迹
            if CollisonIndex[i] > CollisonIndex[i - 1]:
                for j in range(CollisonIndex[i - 1], CollisonIndex[i] + 1):
                    polygon_x.append(egoRect[i][j][0, 0])
                    polygon_y.append(egoRect[i][j][1, 0])
            
            for j in range(CollisonIndex[i], -1, -1):
                polygon_x.append(egoRect[i][j][0, 3])
                polygon_y.append(egoRect[i][j][1, 3])
            
            polygon_x.append(egoRect[i][0][0, 2])
            polygon_y.append(egoRect[i][0][1, 2])
    
    # 如果需要记录日志，保存多边形数据
    if log_flag:
        os.makedirs(polygon_folder, exist_ok=True)
        
        with open(os.path.join(polygon_folder, f"polygon_{frame}.csv"), 'w') as file_polygon:
            for x in polygon_x:
                file_polygon.write(f"{x},")
            file_polygon.write("\n")
            for y in polygon_y:
                file_polygon.write(f"{y},")
            file_polygon.write("\n")
    
    # 计算DA值
    DA = 0
    if all(collisionFlag):
        DA = 0
        if UseablePath_num == 0:
            print("[WARNING: No Path is Available]")
        else:
            print("[WARNING: A Collision Was Happened]")
    else:
        DA = polyarea(polygon_x, polygon_y)
    
    return DA

def polyarea(polygon_x, polygon_y):
    """
    鞋带公式(Shoelace Formula) - 计算多边形面积
    
    对于n个顶点的多边形，面积公式为:
    A = 1/2 * |Σ(xᵢ*y_{i+1} - x_{i+1}*yᵢ)|
    
    这是一个高效的O(n)算法，适用于任意简单多边形(非自交)。
    
    参数:
        polygon_x: 多边形各顶点的x坐标列表
        polygon_y: 多边形各顶点的y坐标列表
    
    返回:
        area: 多边形面积(平方米)
    """
    N = len(polygon_x)
    area = 0
    
    for i in range(N):
        j = (i + 1) % N
        area += polygon_x[i] * polygon_y[j]
        area -= polygon_y[i] * polygon_x[j]
    
    area /= 2
    return abs(area)

# ==================== 风险等级计算 ====================

def Calculate_RiskLevel(UseablePath_num, AvailablePath, K_max, CollisonIndex, unitArcPath,
                       egoVeh, timeHorizon, timeStep, Path):
    """
    计算风险等级(RL) - 基于曲率和碰撞的高斯加权评估
    
    算法原理:
    1. 对每条轨迹，计算无碰撞情况下的加权长度
    2. 对每条轨迹，计算有碰撞情况下的加权长度
    3. 权重采用高斯函数，曲率越大权重越小(反映舒适性)
    4. RL = (无碰撞总长度 - 有碰撞总长度) / 无碰撞总长度
    
    高斯权重公式:
    w(k) = 1/(√(2π)σ) * exp(-k²/(2σ²))
    其中 σ = k_max/2
    
    物理意义:
    - RL接近0: 风险低，几乎无碰撞
    - RL接近1: 风险高，大部分轨迹被阻挡
    
    参数:
        UseablePath_num: 可用轨迹数量
        AvailablePath: 可用轨迹编号列表
        K_max: 各轨迹最大曲率
        CollisonIndex: 各轨迹碰撞点索引
        unitArcPath: 各轨迹单位弧长
        egoVeh: 自车对象
        timeHorizon: 预测时间范围(秒)
        timeStep: 时间步长(秒)
        Path: 轨迹信息列表
    
    返回:
        RiskLevel: 风险等级(0-1)，越大表示风险越高
    """
    import numpy as np
    
    nonCollisonIndex = [0] * UseablePath_num
    SumCurvCoeff = 0
    SumCurvCoeff_WS = 0
    weighedArea = 0
    weighedArea_WS = 0
    
    # 找出最大曲率值
    maxValue = max(K_max)
    
    for i in range(UseablePath_num):
        # 计算无碰撞情况下的索引
        nonCollisonIndex[i] = int(np.ceil((egoVeh.speed_ * timeHorizon) / unitArcPath[i]))
        
        # 计算无碰撞情况下的加权曲率和
        for j in range(nonCollisonIndex[i] + 1):
            gauss_factor = np.power(np.sqrt(2 * np.pi) * maxValue / 2, -1) * np.exp(-np.power(K_max[AvailablePath[i]], 2) / 2 / np.power(maxValue / 2, 2))
            SumCurvCoeff_WS += gauss_factor * unitArcPath[i]
        
        # 计算有碰撞情况下的加权曲率和
        for j in range(CollisonIndex[i] + 1):
            gauss_factor = np.power(np.sqrt(2 * np.pi) * maxValue / 2, -1) * np.exp(-np.power(K_max[AvailablePath[i]], 2) / 2 / np.power(maxValue / 2, 2))
            SumCurvCoeff += gauss_factor * unitArcPath[i]
        
        # 累加权重面积
        weighedArea += SumCurvCoeff
        weighedArea_WS += SumCurvCoeff_WS
        
        # 重置单轨迹曲率和
        SumCurvCoeff_WS = 0
        SumCurvCoeff = 0
    
    # 计算风险等级
    RiskLevel = (weighedArea_WS - weighedArea) / weighedArea_WS
    
    return RiskLevel